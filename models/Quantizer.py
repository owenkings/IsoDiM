"""
IsoDiM - Finite Scalar Quantization (FSQ) Module
================================================
Core quantization module that implements FSQ for discrete latent representation.

Key Features:
- Implicit grid-based quantization
- Topology-preserving: adjacent values remain geometrically close
- Straight-Through Estimator (STE) for gradient flow
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class FSQ(nn.Module):
    """
    Finite Scalar Quantization (FSQ) Module
    
    Implements FSQ quantization that maps continuous values to discrete grid points
    while preserving topological structure (adjacent grid points are numerically close).
    
    Args:
        levels: List of quantization levels per dimension, e.g., [8, 8, 8, 5, 5, 5]
                Implicit codebook size = prod(levels)
    
    Example:
        fsq = FSQ(levels=[8, 8, 8, 5, 5, 5])  # 64,000 implicit codes
        z_q = fsq(z)  # quantized output
    """
    
    def __init__(self, levels: List[int]):
        super().__init__()
        self._levels = levels
        self._dim = len(levels)
        self._codebook_size = np.prod(levels)
        
        # Precompute the quantization bounds for each dimension
        # Each dimension is quantized to levels[i] discrete values
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis)
        
        # Store levels as tensor for efficient computation
        _levels_tensor = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("_levels_tensor", _levels_tensor)
        
    @property
    def dim(self) -> int:
        """Return the FSQ dimension (number of quantization axes)."""
        return self._dim
    
    @property
    def codebook_size(self) -> int:
        """Return the implicit codebook size (product of all levels)."""
        return self._codebook_size
    
    @property
    def levels(self) -> List[int]:
        """Return the quantization levels per dimension."""
        return self._levels
    
    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """
        Bound the input tensor to (-1, 1) range using tanh.
        
        Args:
            z: Input tensor of shape [..., dim]
            eps: Small epsilon for numerical stability
            
        Returns:
            Bounded tensor in range (-1+eps, 1-eps)
        """
        # Use tanh to bound values, then scale slightly to avoid exact boundaries
        half_l = (self._levels_tensor - 1) / 2
        return torch.tanh(z) * half_l
    
    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantize bounded values to discrete grid points.
        
        Args:
            z: Bounded tensor from self.bound()
            
        Returns:
            Quantized tensor with values on discrete grid
        """
        # Round to nearest integer
        z_q = torch.round(z)
        # Clamp to valid range
        half_l = (self._levels_tensor - 1) / 2
        z_q = torch.clamp(z_q, -half_l, half_l)
        return z_q
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply FSQ quantization with Straight-Through Estimator (STE).
        
        The quantization preserves topology: if input value 5.1 maps to grid point 5,
        and 5.9 maps to grid point 6, the numerical relationship is preserved.
        
        Args:
            z: Input tensor of shape [..., dim]
            
        Returns:
            Quantized tensor of same shape, values on implicit grid
        """
        # Step 1: Bound to valid range using tanh
        z_bounded = self.bound(z)
        
        # Step 2: Quantize to grid points
        z_q = self.quantize(z_bounded)
        
        # Step 3: Apply STE - forward uses quantized, backward uses continuous
        z_q = z_bounded + (z_q - z_bounded).detach()
        
        # Step 4: Normalize to [-1, 1] range for consistency
        half_l = (self._levels_tensor - 1) / 2
        z_q = z_q / half_l
        
        return z_q
    
    def get_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Convert quantized values to codebook indices (for analysis).
        
        Args:
            z_q: Quantized tensor from forward()
            
        Returns:
            Integer indices into the implicit codebook
        """
        # De-normalize from [-1, 1]
        half_l = (self._levels_tensor - 1) / 2
        z_int = (z_q * half_l).round().long() + half_l.long()
        
        # Convert to flat indices
        indices = (z_int * self._basis).sum(dim=-1)
        return indices
    
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert codebook indices back to quantized values.
        
        Args:
            indices: Integer indices
            
        Returns:
            Quantized tensor values
        """
        # Convert flat indices to per-dimension indices
        z_int = torch.zeros(*indices.shape, self._dim, device=indices.device, dtype=torch.long)
        remaining = indices.clone()
        
        for i in range(self._dim - 1, -1, -1):
            z_int[..., i] = remaining % self._levels[i]
            remaining = remaining // self._levels[i]
        
        # Convert to normalized values
        half_l = (self._levels_tensor - 1) / 2
        z_q = (z_int.float() - half_l) / half_l
        
        return z_q


# ============================================================================
# FSQ Configuration Presets
# ============================================================================

FSQ_CONFIGS = {
    # Format: name -> [levels], codebook_size = prod(levels)
    # -------------------------------------------------------------------------
    # Standard configurations (dim=5 or dim=6)
    # -------------------------------------------------------------------------
    'small': [5, 5, 5, 5, 5],              # 3,125 codes, dim=5 (快速测试用)
    'medium': [8, 5, 5, 5, 5],             # 5,000 codes, dim=5
    'large': [8, 6, 6, 5, 5, 5],           # 36,000 codes, dim=6
    'high': [8, 8, 8, 5, 5, 5],            # 64,000 codes, dim=6 (RECOMMENDED)
    'ultra': [8, 8, 8, 8, 5, 5],           # 102,400 codes, dim=6
    'mega': [8, 8, 8, 8, 8, 5],            # 163,840 codes, dim=6
    
    # -------------------------------------------------------------------------
    # High-dimensional configurations (dim=7 or dim=8)
    # -------------------------------------------------------------------------
    'highdim7': [7, 5, 5, 5, 5, 5, 5],     # 109,375 codes, dim=7
    'highdim8': [5, 5, 5, 5, 5, 5, 5, 5],  # 390,625 codes, dim=8
}


def get_fsq_config(name: str) -> List[int]:
    """
    Get FSQ configuration by name.
    
    Args:
        name: Configuration name (small, medium, large, high, ultra, mega, highdim7, highdim8)
        
    Returns:
        List of quantization levels
    """
    name = name.lower()
    if name not in FSQ_CONFIGS:
        raise ValueError(f"Unknown FSQ config: {name}. Available: {list(FSQ_CONFIGS.keys())}")
    return FSQ_CONFIGS[name]

