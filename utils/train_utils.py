"""
IsoDiM - Training Utilities
===========================
Utility functions for training IsoDiM models.
"""

import torch
import math
import time


# ============================================================================
# Mask Utilities
# ============================================================================

def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Convert length tensor to boolean mask.
    
    Args:
        lengths: Tensor of sequence lengths [B]
        max_len: Maximum sequence length
        
    Returns:
        Boolean mask [B, max_len] where True indicates valid positions
    """
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def get_mask_subset_prob(mask: torch.Tensor, prob: float) -> torch.Tensor:
    """
    Get a random subset of the mask with given probability.
    
    Args:
        mask: Boolean mask
        prob: Probability of including each masked position
        
    Returns:
        Subset mask
    """
    subset_mask = torch.bernoulli(mask.float(), p=prob).bool() & mask
    return subset_mask


def uniform(shape, device=None) -> torch.Tensor:
    """Sample uniform random values in [0, 1]."""
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    """Cosine annealing schedule: cos(t * pi/2)."""
    return torch.cos(t * math.pi * 0.5)


# ============================================================================
# EMA Utilities
# ============================================================================

def update_ema(model, ema_model, decay: float = 0.9999):
    """
    Update exponential moving average of model parameters.
    
    Args:
        model: Source model
        ema_model: EMA model to update
        decay: EMA decay rate
    """
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(model_param.data, alpha=(1 - decay))


# ============================================================================
# Logging Utilities
# ============================================================================

def def_value():
    """Default value factory for defaultdict."""
    return 0.0


def update_lr_warm_up(nb_iter: int, warm_up_iter: int, optimizer, lr: float) -> float:
    """
    Update learning rate with linear warmup.
    
    Args:
        nb_iter: Current iteration
        warm_up_iter: Total warmup iterations
        optimizer: Optimizer to update
        lr: Target learning rate
        
    Returns:
        Current learning rate
    """
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return current_lr


def save(file_name: str, ep: int, model, optimizer, scheduler, total_it: int, name: str, ema_mardm=None):
    """
    Save tokenizer checkpoint.
    
    Args:
        file_name: Path to save checkpoint
        ep: Current epoch
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        total_it: Total iterations
        name: Model name key
        ema_mardm: Optional EMA model (legacy parameter, ignored for tokenizer)
    """
    state = {
        name: model.state_dict(),
        f"opt_{name}": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        'ep': ep,
        'total_it': total_it,
    }
    torch.save(state, file_name)


def save_isodim(file_name: str, ep: int, model, optimizer, scheduler, total_it: int, name: str, ema_isodim=None):
    """
    Save IsoDiM checkpoint.
    
    Args:
        file_name: Path to save checkpoint
        ep: Current epoch
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        total_it: Total iterations
        name: Model name key
        ema_isodim: EMA model to save
    """
    state = {
        name: model.state_dict(),
        f"opt_{name}": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        'ep': ep,
        'total_it': total_it,
    }
    
    if ema_isodim is not None:
        isodim_state_dict = model.state_dict()
        ema_isodim_state_dict = ema_isodim.state_dict()
        
        # Remove CLIP weights (they are frozen and don't need to be saved)
        clip_weights = [e for e in isodim_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del isodim_state_dict[e]
            del ema_isodim_state_dict[e]
        
        state[name] = isodim_state_dict
        state["ema_isodim"] = ema_isodim_state_dict
    
    torch.save(state, file_name)


def print_current_loss(
    start_time: float,
    niter_state: int,
    total_niters: int,
    losses: dict,
    epoch: int = None,
    inner_iter: int = None,
):
    """
    Print current training loss with progress information.
    
    Args:
        start_time: Training start time
        niter_state: Current iteration
        total_niters: Total iterations
        losses: Dictionary of loss values
        epoch: Current epoch (optional)
        inner_iter: Inner iteration within epoch (optional)
    """
    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
    
    if epoch is not None:
        print('ep/it:%2d-%4d niter:%6d' % (epoch, inner_iter, niter_state), end=" ")
    
    message = ' %s completed:%3d%%)' % (
        time_since(start_time, niter_state / total_niters),
        niter_state / total_niters * 100
    )
    
    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    
    print(message)

