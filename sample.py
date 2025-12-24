"""
IsoDiM - Sampling Script
========================
Generate motion sequences from text prompts.

Usage:
    python sample.py \\
        --name IsoDiM \\
        --tokenizer_name IsoDiM_Tokenizer_High \\
        --tokenizer_model IsoDiM_Tokenizer_High \\
        --model IsoDiM-DiT-XL \\
        --text_prompt "a person walks forward and waves" \\
        --motion_length 120
"""

import os
from os.path import join as pjoin
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import random
import argparse

from models.Tokenizer import Tokenizer_models
from models.IsoDiM import IsoDiM_models
from models.LengthEstimator import LengthEstimator
from utils.motion_process import recover_from_ric, plot_3d_motion, kit_kinematic_chain, t2m_kinematic_chain


def main(args):
    # =========================================================================
    # Seed Configuration
    # =========================================================================
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # =========================================================================
    # Data Configuration
    # =========================================================================
    dim_pose = 64 if args.dataset_name == 'kit' else 67
    nb_joints = 21 if args.dataset_name == 'kit' else 22
    data_root = f'{args.dataset_dir}/KIT-ML/' if args.dataset_name == 'kit' else f'{args.dataset_dir}/HumanML3D/'
    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))
    
    # =========================================================================
    # Model Setup
    # =========================================================================
    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name, 'model')
    result_dir = pjoin('./generation', args.name)
    os.makedirs(result_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = Tokenizer_models[args.tokenizer_model](input_width=dim_pose)
    ckpt_path = pjoin(
        args.checkpoints_dir, args.dataset_name, args.tokenizer_name, 'model',
        'latest.tar' if args.dataset_name == 't2m' else 'net_best_fid.tar'
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    tokenizer.load_state_dict(ckpt['tokenizer'])
    
    # Load IsoDiM model
    ema_isodim = IsoDiM_models[args.model](
        fsq_dim=tokenizer.fsq_dim,
        cond_mode='text',
        fsq_levels=tokenizer.fsq_levels,
    )
    ckpt_path = pjoin(model_dir, 'net_best_fid.tar')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    missing_keys, unexpected_keys = ema_isodim.load_state_dict(checkpoint['ema_isodim'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    
    # Load length estimator
    length_estimator = None
    try:
        length_estimator = LengthEstimator(512, 50)
        ckpt = torch.load(
            pjoin(args.checkpoints_dir, args.dataset_name, 'length_estimator', 'model', 'finest.tar'),
            map_location='cpu'
        )
        length_estimator.load_state_dict(ckpt['estimator'])
        print("Length estimator loaded successfully.")
    except FileNotFoundError:
        print("Length estimator model not found, using fixed length instead.")
        length_estimator = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Prompt Processing
    # =========================================================================
    prompt_list = []
    length_list = []
    est_length = False

    if args.text_prompt != "":
        prompt_list.append(args.text_prompt)
        if args.motion_length == 0:
            if length_estimator is not None:
                est_length = True
            else:
                print("No length estimator available, using default length 120.")
                length_list.append(120)
        else:
            length_list.append(args.motion_length)
    elif args.text_path != "":
        with open(args.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                infos = line.split('#')
                prompt_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    if length_estimator is not None:
                        est_length = True
                        length_list = []
                    else:
                        print("No length estimator available, using default length 120.")
                        length_list.append(120)
                else:
                    length_list.append(int(infos[-1]))
    else:
        raise ValueError("A text prompt or a file of text prompts is required!")

    # =========================================================================
    # Sampling
    # =========================================================================
    tokenizer.to(device)
    ema_isodim.to(device)
    if length_estimator is not None:
        length_estimator.to(device)
        length_estimator.eval()

    tokenizer.eval()
    ema_isodim.eval()

    if est_length and length_estimator is not None:
        print("No motion length specified, using estimated lengths.")
        text_embedding = ema_isodim.encode_text(prompt_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)
        token_lens = Categorical(probs).sample()
    else:
        token_lens = torch.LongTensor(length_list) // 4
        token_lens = token_lens.to(device).long()
    
    m_length = token_lens * 4
    captions = prompt_list
    
    kinematic_chain = kit_kinematic_chain if args.dataset_name == 'kit' else t2m_kinematic_chain
    
    print(f"\n{'='*60}")
    print(f"IsoDiM Sampling")
    print(f"  Model: {args.model}")
    print(f"  Tokenizer: {args.tokenizer_name}")
    print(f"  Timesteps: {args.time_steps}")
    print(f"  CFG Scale: {args.cfg}")
    print(f"  Repeat Times: {args.repeat_times}")
    print(f"{'='*60}\n")
    
    for r in range(args.repeat_times):
        print(f"-->Repeat {r}")
        with torch.no_grad():
            # Generate FSQ codes
            pred_fsq = ema_isodim.generate(
                captions, token_lens, args.time_steps, args.cfg,
                temperature=args.temperature, hard_pseudo_reorder=args.hard_pseudo_reorder
            )
            # Decode to motion
            pred_motions = tokenizer.decode_from_fsq(pred_fsq.permute(0, 2, 1))
            pred_motions = pred_motions.detach().cpu().numpy()
            data = pred_motions * std + mean
        
        for k, (caption, joint_data) in enumerate(zip(captions, data)):
            print(f"---->Sample {k}: {caption} ({m_length[k]} frames)")
            s_path = pjoin(result_dir, str(k))
            os.makedirs(s_path, exist_ok=True)
            
            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), nb_joints).numpy()
            
            # Save motion data as numpy array
            np.save(pjoin(s_path, f"caption:{caption[:50]}_sample{k}_repeat{r}_len{m_length[k]}.npy"), joint)
            print(f"  Motion data saved to: {pjoin(s_path, f'caption:{caption[:50]}_sample{k}_repeat{r}_len{m_length[k]}.npy')}")

            # Try to save video
            save_path = pjoin(s_path, f"caption:{caption[:50]}_sample{k}_repeat{r}_len{m_length[k]}.mp4")
            try:
                plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
                # Check if files actually exist and have content
                gif_path = save_path.replace('.mp4', '.gif')
                if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                    print(f"  Video saved as GIF: {gif_path}")
                if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                    print(f"  Video saved as MP4: {save_path}")
            except Exception as e:
                print(f"  Video saving failed: {e}")
                print("  Motion data is still available as .npy file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Motion with IsoDiM")
    
    # Model configuration
    parser.add_argument('--name', type=str, default='IsoDiM',
                        help='IsoDiM checkpoint name')
    parser.add_argument('--tokenizer_name', type=str, default='IsoDiM_Tokenizer_High',
                        help='Tokenizer checkpoint name')
    parser.add_argument('--tokenizer_model', type=str, default='IsoDiM_Tokenizer_High',
                        choices=['IsoDiM_Tokenizer_High', 'IsoDiM_Tokenizer_Ultra', 'IsoDiM_Tokenizer_Large'],
                        help='Tokenizer model variant')
    parser.add_argument('--model', type=str, default='IsoDiM-DiT-XL',
                        choices=['IsoDiM-DiT-S', 'IsoDiM-DiT-B', 'IsoDiM-DiT-L', 'IsoDiM-DiT-XL','IsoDiM_test_v2'],
                        help='IsoDiM model variant')
    
    # Data configuration
    parser.add_argument('--dataset_name', type=str, default='t2m',
                        choices=['t2m', 'kit'])
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    
    # Sampling configuration
    parser.add_argument('--text_prompt', default='', type=str,
                        help='Text prompt for generation')
    parser.add_argument('--text_path', type=str, default="",
                        help='Path to file with text prompts')
    parser.add_argument("--motion_length", default=0, type=int,
                        help='Motion length in frames (0 for auto-estimation)')
    parser.add_argument("--repeat_times", default=1, type=int,
                        help='Number of generation repeats')
    parser.add_argument("--time_steps", default=18, type=int,
                        help='Number of diffusion sampling steps')
    parser.add_argument("--cfg", default=4.5, type=float,
                        help='Classifier-free guidance scale')
    parser.add_argument("--temperature", default=1.0, type=float,
                        help='Sampling temperature')
    parser.add_argument('--hard_pseudo_reorder', action="store_true",
                        help='Use hard pseudo reorder')
    
    # System configuration
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    main(args)

