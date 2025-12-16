"""
IsoDiM - Tokenizer Training Script
==================================
Train the FSQ-based motion tokenizer.

Usage:
    python train_Tokenizer.py --name IsoDiM_Tokenizer_High --model IsoDiM_Tokenizer_High --dataset_name t2m
    python train_Tokenizer.py --name IsoDiM_Tokenizer_Small --model IsoDiM_Tokenizer_Small --gpu 0

Recommended Configuration:
    - Model: IsoDiM_Tokenizer_High (64k codebook)
    - Batch size: 256
    - Epochs: 50
"""

import os
import sys
import argparse

# Parse --gpu argument early to set CUDA_VISIBLE_DEVICES before torch import
def parse_gpu_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args.gpu

os.environ["CUDA_VISIBLE_DEVICES"] = str(parse_gpu_arg())

from os.path import join as pjoin
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import time
from collections import OrderedDict, defaultdict

from models.Tokenizer import Tokenizer_models
from utils.evaluators import Evaluators
from utils.datasets import AEDataset, Text2MotionDataset, collate_fn
from utils.train_utils import update_lr_warm_up, def_value, save, print_current_loss
from utils.eval_utils import evaluation_ae


def main(args):
    # =========================================================================
    # Seed Configuration
    # =========================================================================
    torch.backends.cudnn.benchmark = False
    os.environ["OMP_NUM_THREADS"] = "1"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    if args.dataset_name == "t2m":
        data_root = f'{args.dataset_dir}/HumanML3D/'
        joints_num = 22
        dim_pose = 67
    else:
        data_root = f'{args.dataset_dir}/KIT-ML/'
        joints_num = 21
        dim_pose = 64
    
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    text_dir = pjoin(data_root, 'texts')
    max_motion_length = 196
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    train_split_file = pjoin(data_root, 'train.txt')
    val_split_file = pjoin(data_root, 'val.txt')
    
    # Training data
    train_dataset = AEDataset(mean, std, motion_dir, args.window_size, train_split_file)
    val_dataset = AEDataset(mean, std, motion_dir, args.window_size, val_split_file)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, drop_last=True,
        num_workers=args.num_workers, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, drop_last=True,
        num_workers=args.num_workers, shuffle=True, pin_memory=True
    )
    
    # Evaluation data
    eval_mean = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_mean.npy')
    eval_std = np.load(f'utils/eval_mean_std/{args.dataset_name}/eval_std.npy')
    split_file = pjoin(data_root, 'val.txt')
    eval_dataset = Text2MotionDataset(
        eval_mean, eval_std, split_file, args.dataset_name, motion_dir, text_dir,
        4, max_motion_length, 20, evaluation=True
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
    
    tokenizer = Tokenizer_models[args.model](input_width=dim_pose)
    
    print(tokenizer)
    print(f"\n{'='*60}")
    print(f"IsoDiM Tokenizer Configuration:")
    print(f"  Model: {args.model}")
    print(f"  FSQ Levels: {tokenizer.fsq_levels}")
    print(f"  FSQ Dimension: {tokenizer.fsq_dim}")
    print(f"  Codebook Size: {tokenizer.codebook_size:,}")
    print(f"{'='*60}\n")
    
    pc_tokenizer = sum(param.numel() for param in tokenizer.parameters())
    print(f'Total parameters: {pc_tokenizer / 1_000_000:.2f}M')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_wrapper = Evaluators(args.dataset_name, device=device)
    
    # =========================================================================
    # Training Setup
    # =========================================================================
    logger = SummaryWriter(model_dir)
    
    if args.recons_loss == 'l1_smooth':
        criterion = torch.nn.SmoothL1Loss()
    else:
        criterion = torch.nn.MSELoss()
    
    tokenizer.to(device)
    optimizer = optim.AdamW(
        tokenizer.parameters(), lr=args.lr,
        betas=(0.9, 0.99), weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_decay
    )
    
    epoch = 0
    it = 0
    if args.is_continue:
        ckpt_path = pjoin(model_dir, 'latest.tar')
        checkpoint = torch.load(ckpt_path, map_location=device)
        tokenizer.load_state_dict(checkpoint['tokenizer'])
        optimizer.load_state_dict(checkpoint['opt_tokenizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch, it = checkpoint['ep'], checkpoint['total_it']
        print(f"Loaded checkpoint - Epoch: {epoch}, Iterations: {it}")
    
    start_time = time.time()
    total_iters = args.epoch * len(train_loader)
    print(f'Total Epochs: {args.epoch}, Total Iters: {total_iters}')
    print(f'Iters Per Epoch, Training: {len(train_loader):04d}, Validation: {len(val_loader):03d}')
    
    logs = defaultdict(def_value, OrderedDict())
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe = 1000, 0, 0, 0, 0, 100, 100
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    while epoch < args.epoch:
        tokenizer.train()
        
        for i, batch_data in enumerate(train_loader):
            it += 1
            if it < args.warm_up_iter:
                update_lr_warm_up(it, args.warm_up_iter, optimizer, args.lr)
            
            motions = batch_data.detach().to(device).float()
            pred_motion = tokenizer(motions)
            
            # Reconstruction loss
            loss_rec = criterion(pred_motion, motions)
            
            # Joint position auxiliary loss
            pred_local_pos = pred_motion[..., 4: (joints_num - 1) * 3 + 4]
            local_pos = motions[..., 4: (joints_num - 1) * 3 + 4]
            loss_explicit = criterion(pred_local_pos, local_pos)
            
            loss = loss_rec + args.aux_loss_joints * loss_explicit
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if it >= args.warm_up_iter:
                scheduler.step()
            
            logs['loss'] += loss.item()
            logs['loss_rec'] += loss_rec.item()
            logs['loss_joints'] += loss_explicit.item()
            logs['lr'] += optimizer.param_groups[0]['lr']
            
            if it % args.log_every == 0:
                mean_loss = OrderedDict()
                for tag, value in logs.items():
                    logger.add_scalar(f'Train/{tag}', value / args.log_every, it)
                    mean_loss[tag] = value / args.log_every
                logs = defaultdict(def_value, OrderedDict())
                print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
        
        # Save checkpoint
        save(pjoin(model_dir, 'latest.tar'), epoch, tokenizer, optimizer, scheduler, it, 'tokenizer')
        epoch += 1
        
        # =====================================================================
        # Validation
        # =====================================================================
        print('Validation:')
        tokenizer.eval()
        val_loss_rec = []
        val_loss_joints = []
        val_loss = []
        
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                motions = batch_data.detach().to(device).float()
                pred_motion = tokenizer(motions)
                
                loss_rec = criterion(pred_motion, motions)
                pred_local_pos = pred_motion[..., 4: (joints_num - 1) * 3 + 4]
                local_pos = motions[..., 4: (joints_num - 1) * 3 + 4]
                loss_explicit = criterion(pred_local_pos, local_pos)
                loss = loss_rec + args.aux_loss_joints * loss_explicit
                
                val_loss.append(loss.item())
                val_loss_rec.append(loss_rec.item())
                val_loss_joints.append(loss_explicit.item())
        
        logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
        logger.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), epoch)
        logger.add_scalar('Val/loss_joints', sum(val_loss_joints) / len(val_loss_joints), epoch)
        print(f'Validation Loss: {sum(val_loss) / len(val_loss):.5f}, '
              f'Rec: {sum(val_loss_rec) / len(val_loss_rec):.5f}, '
              f'Joints: {sum(val_loss_joints) / len(val_loss_joints):.5f}')
        
        # Evaluation metrics
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe, writer = evaluation_ae(
            model_dir, eval_loader, tokenizer, logger, epoch-1, device=device,
            num_joint=joints_num, best_fid=best_fid, best_div=best_div,
            best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
            train_mean=mean, train_std=std, best_matching=best_matching,
            eval_wrapper=eval_wrapper
        )
        print(f'Best FID: {best_fid:.4f}')
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best FID: {best_fid:.4f}")
    print(f"  Best Diversity: {best_div:.4f}")
    print(f"  Best R-Precision Top1: {best_top1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IsoDiM Tokenizer")
    
    # Model configuration
    parser.add_argument('--name', type=str, default='IsoDiM_Tokenizer_High',
                        help='Experiment name')
    parser.add_argument('--model', type=str, default='IsoDiM_Tokenizer_High',
                        choices=[
                            'IsoDiM_Tokenizer_Small',    # 3,125 codes (快速测试)
                            'IsoDiM_Tokenizer_Medium',   # 5,000 codes
                            'IsoDiM_Tokenizer_Large',    # 36,000 codes
                            'IsoDiM_Tokenizer_High',     # 64,000 codes (推荐)
                            'IsoDiM_Tokenizer_Ultra',    # 102,400 codes
                            'IsoDiM_Tokenizer_Mega',     # 163,840 codes
                            'IsoDiM_Tokenizer_HighDim7', # 109,375 codes, dim=7
                            'IsoDiM_Tokenizer_HighDim8', # 390,625 codes, dim=8
                        ],
                        help='Tokenizer model variant')
    
    # Data configuration
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--dataset_name', type=str, default='t2m',
                        choices=['t2m', 'kit'])
    parser.add_argument('--window_size', type=int, default=64)
    
    # Training configuration
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--warm_up_iter', default=2000, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    
    # Loss configuration
    parser.add_argument('--aux_loss_joints', type=float, default=1.0,
                        help='Weight for joint position auxiliary loss')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth',
                        choices=['l1_smooth', 'mse'])
    
    # System configuration
    parser.add_argument("--gpu", type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--is_continue', action="store_true")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_every', default=10, type=int)
    
    args = parser.parse_args()
    main(args)

