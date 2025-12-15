"""
IsoDiM - Main Model Training Script
====================================
Train the IsoDiM model (MAR Transformer + JiT Diffusion).

Usage:
    python train_IsoDiM.py \\
        --name IsoDiM \\
        --tokenizer_name IsoDiM_Tokenizer_High \\
        --tokenizer_model IsoDiM_Tokenizer_High \\
        --model IsoDiM-DiT-XL \\
        --dataset_name t2m \\
        --batch_size 64 \\
        --epoch 500 \\
        --need_evaluation

Recommended Configuration:
    - Tokenizer: IsoDiM_Tokenizer_High (64k FSQ codebook)
    - Model: IsoDiM-DiT-XL (JiT Transformer + x-prediction)
    - Batch size: 64
    - Epochs: 500
"""

import os
from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import argparse
import time
import copy
from collections import OrderedDict, defaultdict

from models.Tokenizer import Tokenizer_models
from models.IsoDiM import IsoDiM_models
from utils.evaluators import Evaluators
from utils.datasets import Text2MotionDataset, collate_fn
from utils.train_utils import update_lr_warm_up, def_value, save_isodim, print_current_loss, update_ema
from utils.eval_utils import evaluation_isodim


def main(args):
    # =========================================================================
    # Seed Configuration
    # =========================================================================
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    
    # Enable TF32 for faster training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    if args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        dim_pose = 67
    else:
        data_root = f'{args.dataset_dir}/KIT-ML/'
        dim_pose = 64
    
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    text_dir = pjoin(data_root, 'texts')
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    train_split_file = pjoin(data_root, 'train.txt')
    val_split_file = pjoin(data_root, 'val.txt')
    
    # Training data
    train_dataset = Text2MotionDataset(
        mean, std, train_split_file, args.dataset_name, motion_dir, text_dir,
        args.unit_length, args.max_motion_length, 20, evaluation=False
    )
    val_dataset = Text2MotionDataset(
        mean, std, val_split_file, args.dataset_name, motion_dir, text_dir,
        args.unit_length, args.max_motion_length, 20, evaluation=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True,
        num_workers=args.num_workers, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, drop_last=True,
        num_workers=args.num_workers, shuffle=True
    )
    
    # Evaluation data
    if args.need_evaluation:
        eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
        eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
        split_file = pjoin(data_root, 'val.txt')
        eval_dataset = Text2MotionDataset(
            eval_mean, eval_std, split_file, args.dataset_name, motion_dir, text_dir,
            4, 196, 20, evaluation=True
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=32, num_workers=args.num_workers,
            drop_last=True, collate_fn=collate_fn, shuffle=True
        )
    
    # =========================================================================
    # Model Setup
    # =========================================================================
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Load pretrained tokenizer
    tokenizer = Tokenizer_models[args.tokenizer_model](input_width=dim_pose)
    ckpt_path = pjoin(
        args.checkpoints_dir, args.dataset_name, args.tokenizer_name, 'model',
        'latest.tar' if args.dataset_name == 't2m' else 'net_best_fid.tar'
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    tokenizer.load_state_dict(ckpt['tokenizer'])
    
    print(f"\n{'='*60}")
    print(f"Loaded Tokenizer: {args.tokenizer_name}")
    print(f"  FSQ Levels: {tokenizer.fsq_levels}")
    print(f"  FSQ Dimension: {tokenizer.fsq_dim}")
    print(f"  Codebook Size: {tokenizer.codebook_size:,}")
    print(f"{'='*60}\n")
    
    # Create IsoDiM model
    isodim = IsoDiM_models[args.model](
        fsq_dim=tokenizer.fsq_dim,
        cond_mode='text',
        fsq_levels=tokenizer.fsq_levels,
    )
    
    # Create EMA model
    ema_isodim = copy.deepcopy(isodim)
    ema_isodim.eval()
    for param in ema_isodim.parameters():
        param.requires_grad_(False)
    
    # Count parameters (excluding CLIP)
    all_params = sum(
        param.numel() for name, param in isodim.named_parameters()
        if not name.startswith('clip_model.')
    )
    print(f'Total parameters (excluding CLIP): {all_params / 1_000_000:.2f}M')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.need_evaluation:
        eval_wrapper = Evaluators(args.dataset_name, device=device)
    
    # =========================================================================
    # Training Setup
    # =========================================================================
    logger = SummaryWriter(model_dir)
    
    tokenizer.eval()
    tokenizer.to(device)
    isodim.to(device)
    ema_isodim.to(device)
    
    optimizer = optim.AdamW(
        isodim.parameters(), betas=(0.9, 0.99),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_decay
    )
    
    epoch = 0
    it = 0
    if args.is_continue:
        ckpt_path = pjoin(model_dir, 'latest.tar')
        checkpoint = torch.load(ckpt_path, map_location=device)
        missing_keys, unexpected_keys = isodim.load_state_dict(checkpoint['isodim'], strict=False)
        missing_keys2, unexpected_keys2 = ema_isodim.load_state_dict(checkpoint['ema_isodim'], strict=False)
        assert len(unexpected_keys) == 0
        assert len(unexpected_keys2) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])
        assert all([k.startswith('clip_model.') for k in missing_keys2])
        optimizer.load_state_dict(checkpoint['opt_isodim'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch, it = checkpoint['ep'], checkpoint['total_it']
        print(f"Loaded checkpoint - Epoch: {epoch}, Iterations: {it}")
    
    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    print(f'Total Epochs: {args.epoch}, Total Iters: {total_iters}')
    print(f'Iters Per Epoch, Training: {len(train_loader):04d}, Validation: {len(val_loader):03d}')
    
    logs = defaultdict(def_value, OrderedDict())
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, clip_score = 1000, 0, 0, 0, 0, 100, -1
    worst_loss = 100
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    while epoch < args.epoch:
        tokenizer.eval()
        isodim.train()
        
        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter:
                update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)
            
            conds, motion, m_lens = batch_data
            motion = motion.detach().float().to(device)
            m_lens = m_lens.detach().long().to(device)
            
            # Get FSQ targets from tokenizer
            with torch.no_grad():
                fsq_target = tokenizer.encode_with_fsq_output(motion)  # [B, L//4, d]
            
            m_lens = m_lens // 4
            conds = conds.to(device).float() if torch.is_tensor(conds) else conds
            
            # Forward loss (x-prediction)
            loss = isodim.forward_loss(fsq_target, conds, m_lens)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            logs['loss'] += loss.item()
            logs['lr'] += optimizer.param_groups[0]['lr']
            
            # Update EMA
            update_ema(isodim, ema_isodim, 0.9999)
            
            if it % args.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    logger.add_scalar(f'Train/{tag}', value / args.log_every, it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
        
        # Save checkpoint
        save_isodim(
            pjoin(model_dir, 'latest.tar'), epoch, isodim, optimizer, scheduler,
            it, 'isodim', ema_isodim=ema_isodim
        )
        epoch += 1
        
        # =====================================================================
        # Validation
        # =====================================================================
        print('Validation:')
        tokenizer.eval()
        isodim.eval()
        val_loss = []
        
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                conds, motion, m_lens = batch_data
                motion = motion.detach().float().to(device)
                m_lens = m_lens.detach().long().to(device)
                
                fsq_target = tokenizer.encode_with_fsq_output(motion)
                m_lens = m_lens // 4
                conds = conds.to(device).float() if torch.is_tensor(conds) else conds
                
                loss = isodim.forward_loss(fsq_target, conds, m_lens)
                val_loss.append(loss.item())
        
        val_loss_mean = np.mean(val_loss)
        print(f"Validation loss: {val_loss_mean:.3f}")
        logger.add_scalar('Val/loss', val_loss_mean, epoch)
        
        if val_loss_mean < worst_loss:
            print(f"Improved loss from {worst_loss:.02f} to {val_loss_mean:.03f}!")
            worst_loss = val_loss_mean
        
        # Evaluation metrics
        if args.need_evaluation:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _, clip_score, writer, save_now = evaluation_isodim(
                model_dir, eval_loader, ema_isodim, tokenizer, logger, epoch-1,
                best_fid=best_fid, clip_score_old=clip_score,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper,
                device=device, train_mean=mean, train_std=std
            )
            if save_now:
                save_isodim(
                    pjoin(model_dir, 'net_best_fid.tar'), epoch-1, isodim, optimizer, scheduler,
                    it, 'isodim', ema_isodim=ema_isodim
                )
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best FID: {best_fid:.4f}")
    print(f"  Best Diversity: {best_div:.4f}")
    print(f"  Best R-Precision Top1: {best_top1:.4f}")
    print(f"  Best CLIP Score: {clip_score:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IsoDiM Model")
    
    # Model configuration
    parser.add_argument('--name', type=str, default='IsoDiM',
                        help='Experiment name')
    parser.add_argument('--tokenizer_name', type=str, default='IsoDiM_Tokenizer_High',
                        help='Tokenizer checkpoint name')
    parser.add_argument('--tokenizer_model', type=str, default='IsoDiM_Tokenizer_High',
                        choices=['IsoDiM_Tokenizer_High', 'IsoDiM_Tokenizer_Ultra', 'IsoDiM_Tokenizer_Large'],
                        help='Tokenizer model variant')
    parser.add_argument('--model', type=str, default='IsoDiM-DiT-XL',
                        choices=['IsoDiM-DiT-S', 'IsoDiM-DiT-B', 'IsoDiM-DiT-L', 'IsoDiM-DiT-XL'],
                        help='IsoDiM model variant')
    
    # Data configuration
    parser.add_argument('--dataset_name', type=str, default='t2m',
                        choices=['t2m', 'kit'])
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument("--max_motion_length", type=int, default=196)
    parser.add_argument("--unit_length", type=int, default=4)
    
    # Training configuration
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=500, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--milestones', default=[50_000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    
    # Evaluation
    parser.add_argument('--need_evaluation', action="store_true",
                        help='Enable evaluation during training')
    
    # System configuration
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_every', default=50, type=int)
    
    args = parser.parse_args()
    main(args)

