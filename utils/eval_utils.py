"""
IsoDiM - Evaluation Utilities
=============================
Functions for evaluating model quality (FID, R-Precision, etc.)
"""

import os
import numpy as np
from scipy import linalg
import torch
from tqdm import tqdm

from utils.motion_process import recover_from_ric


# ============================================================================
# Decorator
# ============================================================================

def eval_decorator(fn):
    """Decorator to temporarily set model to eval mode."""
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def evaluation_ae(
    out_dir, val_loader, net, writer, ep, eval_wrapper, num_joint, device,
    best_fid=1000, best_div=0, best_top1=0, best_top2=0, best_top3=0, best_matching=100,
    train_mean=None, train_std=None, save=True, draw=True
):
    """
    Evaluate autoencoder (tokenizer) reconstruction quality.
    
    Args:
        out_dir: Output directory
        val_loader: Validation data loader
        net: Tokenizer network
        writer: TensorBoard writer
        ep: Current epoch
        eval_wrapper: Evaluator object
        num_joint: Number of joints
        device: Device to use
        best_*: Best metrics so far
        train_mean/std: Training data statistics
        save: Whether to save best model
        draw: Whether to log to TensorBoard
        
    Returns:
        Updated best metrics and writer
    """
    net.eval()
    
    motion_annotation_list = []
    motion_pred_list = []
    
    R_precision_real = 0
    R_precision = 0
    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        
        motion = motion.to(device)
        (et, em), (_, _) = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, caption, motion.clone(), m_length
        )
        bs, seq = motion.shape[0], motion.shape[1]
        
        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        motion = val_loader.dataset.transform(bgt, train_mean, train_std)
        
        pred_pose_eval = net(torch.from_numpy(motion).to(device))
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy(), train_mean, train_std)
        bpredd = val_loader.dataset.transform(bpred)
        
        (et_pred, em_pred), (_, _) = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, caption,
            torch.from_numpy(bpredd).to(device), m_length
        )
        
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            num_poses += gt.shape[0]
        
        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
        
        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match
        
        nb_sample += bs
    
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    
    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses
    
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    
    msg = "--> \t Eva. Re %d: FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, " \
          "R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), " \
          "matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f" % (
              ep, fid, diversity_real, diversity,
              R_precision_real[0], R_precision_real[1], R_precision_real[2],
              R_precision[0], R_precision[1], R_precision[2],
              matching_score_real, matching_score_pred, mpjpe
          )
    print(msg)
    
    if draw and writer is not None:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
    
    if fid < best_fid:
        if draw:
            print(f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!")
        best_fid = fid
        if save:
            torch.save({'tokenizer': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))
    
    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        if draw:
            print(f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!")
        best_div = diversity
    
    if R_precision[0] > best_top1:
        if draw:
            print(f"--> --> \t Top1 Improved from {best_top1:.5f} to {R_precision[0]:.5f} !!!")
        best_top1 = R_precision[0]
    
    if R_precision[1] > best_top2:
        if draw:
            print(f"--> --> \t Top2 Improved from {best_top2:.5f} to {R_precision[1]:.5f} !!!")
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3:
        if draw:
            print(f"--> --> \t Top3 Improved from {best_top3:.5f} to {R_precision[2]:.5f} !!!")
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching:
        if draw:
            print(f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!")
        best_matching = matching_score_pred
    
    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, mpjpe, writer


@torch.no_grad()
def evaluation_isodim(
    out_dir, val_loader, ema_isodim, tokenizer, writer, ep,
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
    eval_wrapper, device, clip_score_old,
    time_steps=None, cond_scale=None, temperature=1, cal_mm=False,
    train_mean=None, train_std=None, draw=True, hard_pseudo_reorder=False
):
    """
    Evaluate IsoDiM generation quality.
    
    Args:
        out_dir: Output directory
        val_loader: Validation data loader
        ema_isodim: EMA IsoDiM model
        tokenizer: Tokenizer for decoding
        writer: TensorBoard writer
        ep: Current epoch
        best_*: Best metrics so far
        eval_wrapper: Evaluator object
        device: Device to use
        clip_score_old: Previous best CLIP score
        time_steps: Number of sampling steps
        cond_scale: CFG scale
        temperature: Sampling temperature
        cal_mm: Calculate multimodality
        train_mean/std: Training data statistics
        draw: Whether to log to TensorBoard
        hard_pseudo_reorder: Use hard pseudo reorder
        
    Returns:
        Updated best metrics and save flag
    """
    ema_isodim.eval()
    tokenizer.eval()
    
    save = False
    
    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    
    if time_steps is None:
        time_steps = 18
    if cond_scale is None:
        cond_scale = 2.5 if "kit" in out_dir else 4.5
    
    clip_score_real = 0
    clip_score_gt = 0
    
    nb_sample = 0
    num_mm_batch = 3 if cal_mm else 0
    
    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.to(device)
        
        bs, seq = pose.shape[:2]
        
        if i < num_mm_batch:
            motion_multimodality_batch = []
            batch_clip_score_pred = 0
            for _ in tqdm(range(30), leave=False):
                pred_fsq = ema_isodim.generate(
                    clip_text, m_length // 4, time_steps, cond_scale,
                    temperature=temperature, hard_pseudo_reorder=hard_pseudo_reorder
                )
                pred_motions = tokenizer.decode_from_fsq(pred_fsq.permute(0, 2, 1))
                pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy(), train_mean, train_std)
                pred_motions = val_loader.dataset.transform(pred_motions)
                (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len, clip_text,
                    torch.from_numpy(pred_motions).to(device), m_length
                )
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)
            motion_multimodality.append(motion_multimodality_batch)
            for j in range(32):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred
        else:
            pred_fsq = ema_isodim.generate(
                clip_text, m_length // 4, time_steps, cond_scale,
                temperature=temperature, hard_pseudo_reorder=hard_pseudo_reorder
            )
            pred_motions = tokenizer.decode_from_fsq(pred_fsq.permute(0, 2, 1))
            pred_motions = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy(), train_mean, train_std)
            pred_motions = val_loader.dataset.transform(pred_motions)
            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, clip_text,
                torch.from_numpy(pred_motions).to(device), m_length
            )
            batch_clip_score_pred = 0
            for j in range(32):
                single_em = em_pred_clip[j]
                single_et = et_pred_clip[j]
                clip_score = (single_em @ single_et.T).item()
                batch_clip_score_pred += clip_score
            clip_score_real += batch_clip_score_pred
        
        pose = pose.cuda().float()
        (et, em), (et_clip, em_clip) = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, clip_text, pose.clone(), m_length
        )
        batch_clip_score = 0
        for j in range(32):
            single_em = em_clip[j]
            single_et = et_clip[j]
            clip_score = (single_em @ single_et.T).item()
            batch_clip_score += clip_score
        clip_score_gt += batch_clip_score
        
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
        
        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match
        
        nb_sample += bs
    
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    
    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    
    clip_score_real = clip_score_real / nb_sample
    clip_score_gt = clip_score_gt / nb_sample
    
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    
    msg = f"--> \t Eva. Ep/Re {ep}: FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, " \
          f"Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, " \
          f"R_precision. {R_precision}, matching_real. {matching_score_real:.4f}, " \
          f"matching_pred. {matching_score_pred:.4f}, multimodality. {multimodality:.4f}, " \
          f"clip_score_gt. {clip_score_gt:.4f}, clip_score. {clip_score_real:.4f}"
    print(msg)
    
    if draw and writer is not None:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
        writer.add_scalar('./Test/clip_score', clip_score_real, ep)
    
    if fid < best_fid:
        if draw:
            print(f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!")
        best_fid = fid
        save = True
    
    if matching_score_pred < best_matching:
        if draw:
            print(f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!")
        best_matching = matching_score_pred
    
    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        if draw:
            print(f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!")
        best_div = diversity
    
    if R_precision[0] > best_top1:
        if draw:
            print(f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!")
        best_top1 = R_precision[0]
    
    if R_precision[1] > best_top2:
        if draw:
            print(f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!")
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3:
        if draw:
            print(f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!")
        best_top3 = R_precision[2]
    
    if clip_score_real > clip_score_old:
        if draw:
            print(f"--> --> \t CLIP-score Improved from {clip_score_old:.4f} to {clip_score_real:.4f} !!!")
        clip_score_old = clip_score_real
    
    if cal_mm:
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, multimodality, clip_score_old, writer, save
    else:
        return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, 0, clip_score_old, writer, save


# ============================================================================
# Metric Calculations
# ============================================================================

def calculate_mpjpe(gt_joints: torch.Tensor, pred_joints: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Per Joint Position Error.
    
    Args:
        gt_joints: Ground truth joints [num_poses, num_joints, 3]
        pred_joints: Predicted joints [num_poses, num_joints, 3]
        
    Returns:
        MPJPE per frame
    """
    assert gt_joints.shape == pred_joints.shape
    
    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - pelvis.unsqueeze(1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - pelvis.unsqueeze(1)
    
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1)
    mpjpe_seq = mpjpe.mean(-1)
    
    return mpjpe_seq


def euclidean_distance_matrix(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """Calculate pairwise Euclidean distances."""
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    dists = np.sqrt(d1 + d2 + d3)
    return dists


def calculate_top_k(mat: np.ndarray, top_k: int) -> np.ndarray:
    """Calculate top-k accuracy matrix."""
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1: np.ndarray, embedding2: np.ndarray, top_k: int, sum_all: bool = False) -> np.ndarray:
    """Calculate R-precision metric."""
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    return top_k_mat


def calculate_activation_statistics(activations: np.ndarray):
    """Calculate mean and covariance of activations."""
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation: np.ndarray, diversity_times: int) -> float:
    """Calculate diversity metric."""
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]
    
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_multimodality(activation: np.ndarray, multimodality_times: int) -> float:
    """Calculate multimodality metric."""
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]
    
    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """Calculate Fréchet Distance between two Gaussian distributions."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    
    diff = mu1 - mu2
    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

