"""
IsoDiM - FSQ Tokenizer Module
=============================
Motion tokenizer using Finite Scalar Quantization (FSQ) for discrete latent representation.

This module provides the IsoDiM_Tokenizer class which:
- Encodes motion sequences to FSQ-quantized latent codes
- Decodes FSQ codes back to motion sequences
- Preserves topological structure of motion data
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .Quantizer import FSQ, get_fsq_config


class IsoDiM_Tokenizer(nn.Module):
    """
    FSQ-based Motion Tokenizer for IsoDiM.
    
    Architecture:
        Input (67-dim) → Encoder (Conv1D) → 512-dim → Linear(512→d) → FSQ → Linear(d→512) → Decoder → Output (67-dim)
    
    Args:
        input_width: Input motion dimension (default: 67 for HumanML3D)
        output_emb_width: Encoder output dimension (default: 512)
        down_t: Number of temporal downsampling layers (default: 2)
        stride_t: Temporal stride for downsampling (default: 2)
        width: Hidden dimension width (default: 512)
        depth: Number of residual blocks (default: 3)
        dilation_growth_rate: Dilation rate growth (default: 3)
        activation: Activation function (default: 'relu')
        norm: Normalization type (default: None)
        fsq_levels: FSQ quantization levels (default: [8, 8, 8, 5, 5, 5] for 64k codebook)
    """
    
    def __init__(
        self,
        input_width: int = 67,
        output_emb_width: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = 'relu',
        norm: Optional[str] = None,
        fsq_levels: List[int] = None,
    ):
        super().__init__()
        
        # Default to FSQ-High configuration (64k codebook)
        if fsq_levels is None:
            fsq_levels = get_fsq_config('high')  # [8, 8, 8, 5, 5, 5]
        
        self.fsq_levels = fsq_levels
        self.fsq_dim = len(fsq_levels)
        self.output_emb_width = output_emb_width
        
        # Encoder: motion → latent
        self.encoder = Encoder(
            input_width, output_emb_width, down_t, stride_t, 
            width, depth, dilation_growth_rate, 
            activation=activation, norm=norm
        )
        
        # FSQ Bottleneck
        self.pre_fsq = nn.Linear(output_emb_width, self.fsq_dim)
        self.fsq = FSQ(levels=fsq_levels)
        self.post_fsq = nn.Linear(self.fsq_dim, output_emb_width)
        
        # Decoder: latent → motion
        self.decoder = Decoder(
            input_width, output_emb_width, down_t, stride_t,
            width, depth, dilation_growth_rate,
            activation=activation, norm=norm
        )
    
    @property
    def codebook_size(self) -> int:
        """Return the implicit FSQ codebook size."""
        return self.fsq.codebook_size
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input: [B, T, C] → [B, C, T]"""
        return x.permute(0, 2, 1).float()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode motion to latent representation.
        
        Args:
            x: Motion tensor [B, T, C]
            
        Returns:
            Latent tensor [B, 512, T//4]
        """
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        
        # Apply FSQ quantization
        x_encoder = x_encoder.permute(0, 2, 1)  # [B, T/4, 512]
        z = self.pre_fsq(x_encoder)              # [B, T/4, d]
        z_q = self.fsq(z)                        # [B, T/4, d] quantized
        z_out = self.post_fsq(z_q)               # [B, T/4, 512]
        
        return z_out.permute(0, 2, 1)  # [B, 512, T/4]
    
    def encode_with_fsq_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode and return FSQ quantized values (for diffusion training target).
        
        Args:
            x: Motion tensor [B, T, C]
            
        Returns:
            FSQ quantized tensor [B, T//4, d] - this is the diffusion target
        """
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        
        x_encoder = x_encoder.permute(0, 2, 1)  # [B, T/4, 512]
        z = self.pre_fsq(x_encoder)              # [B, T/4, d]
        z_q = self.fsq(z)                        # [B, T/4, d] quantized
        
        return z_q
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to motion.
        
        Args:
            x: Latent tensor [B, 512, T//4]
            
        Returns:
            Motion tensor [B, T, C]
        """
        return self.decoder(x)
    
    def decode_from_fsq(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode from FSQ quantized values directly.
        
        Args:
            z_q: FSQ quantized tensor [B, T//4, d]
            
        Returns:
            Motion tensor [B, T, C]
        """
        z_out = self.post_fsq(z_q)               # [B, T/4, 512]
        z_out = z_out.permute(0, 2, 1)           # [B, 512, T/4]
        return self.decoder(z_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode → FSQ → decode.
        
        Args:
            x: Motion tensor [B, T, C]
            
        Returns:
            Reconstructed motion tensor [B, T, C]
        """
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        
        # FSQ quantization
        x_encoder = x_encoder.permute(0, 2, 1)
        z = self.pre_fsq(x_encoder)
        z_q = self.fsq(z)
        z_out = self.post_fsq(z_q)
        z_out = z_out.permute(0, 2, 1)
        
        # Decode
        return self.decoder(z_out)


# ============================================================================
# Inner Architecture Components
# ============================================================================

class Encoder(nn.Module):
    """Convolutional encoder for motion sequences."""
    
    def __init__(
        self,
        input_emb_width: int = 3,
        output_emb_width: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = 'relu',
        norm: Optional[str] = None,
    ):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decoder(nn.Module):
    """Convolutional decoder for motion sequences."""
    
    def __init__(
        self,
        input_emb_width: int = 3,
        output_emb_width: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = 'relu',
        norm: Optional[str] = None,
    ):
        super().__init__()
        blocks = []
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x.permute(0, 2, 1)


class Resnet1D(nn.Module):
    """1D ResNet block."""
    
    def __init__(
        self,
        n_in: int,
        n_depth: int,
        dilation_growth_rate: int = 1,
        reverse_dilation: bool = True,
        activation: str = 'relu',
        norm: Optional[str] = None,
    ):
        super().__init__()
        blocks = [
            ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Nonlinearity(nn.Module):
    """SiLU-like nonlinearity: x * sigmoid(x)"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    """Residual 1D convolution block."""
    
    def __init__(
        self,
        n_in: int,
        n_state: int,
        dilation: int = 1,
        activation: str = 'silu',
        norm: Optional[str] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = dilation
        self.norm = norm
        
        # Normalization layers
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # Activation layers
        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = Nonlinearity()
            self.activation2 = Nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x
        
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)
        
        x = self.conv1(x)
        
        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)
        
        x = self.conv2(x)
        x = self.dropout(x)
        x = x + x_orig
        return x


# ============================================================================
# Model Factory Functions
# ============================================================================

def isodim_tokenizer_small(**kwargs) -> IsoDiM_Tokenizer:
    """Create IsoDiM_Tokenizer with FSQ-Small configuration (3k codebook)."""
    return IsoDiM_Tokenizer(
        output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
        dilation_growth_rate=3, activation='relu', norm=None,
        fsq_levels=get_fsq_config('small'),  # [5, 5, 5, 5, 5]
        **kwargs
    )


def isodim_tokenizer_medium(**kwargs) -> IsoDiM_Tokenizer:
    """Create IsoDiM_Tokenizer with FSQ-Medium configuration (5k codebook)."""
    return IsoDiM_Tokenizer(
        output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
        dilation_growth_rate=3, activation='relu', norm=None,
        fsq_levels=get_fsq_config('medium'),  # [8, 5, 5, 5, 5]
        **kwargs
    )


def isodim_tokenizer_large(**kwargs) -> IsoDiM_Tokenizer:
    """Create IsoDiM_Tokenizer with FSQ-Large configuration (36k codebook)."""
    return IsoDiM_Tokenizer(
        output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
        dilation_growth_rate=3, activation='relu', norm=None,
        fsq_levels=get_fsq_config('large'),  # [8, 6, 6, 5, 5, 5]
        **kwargs
    )


def isodim_tokenizer_high(**kwargs) -> IsoDiM_Tokenizer:
    """Create IsoDiM_Tokenizer with FSQ-High configuration (64k codebook) - RECOMMENDED."""
    return IsoDiM_Tokenizer(
        output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
        dilation_growth_rate=3, activation='relu', norm=None,
        fsq_levels=get_fsq_config('high'),  # [8, 8, 8, 5, 5, 5]
        **kwargs
    )


def isodim_tokenizer_ultra(**kwargs) -> IsoDiM_Tokenizer:
    """Create IsoDiM_Tokenizer with FSQ-Ultra configuration (102k codebook)."""
    return IsoDiM_Tokenizer(
        output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
        dilation_growth_rate=3, activation='relu', norm=None,
        fsq_levels=get_fsq_config('ultra'),  # [8, 8, 8, 8, 5, 5]
        **kwargs
    )


def isodim_tokenizer_mega(**kwargs) -> IsoDiM_Tokenizer:
    """Create IsoDiM_Tokenizer with FSQ-Mega configuration (164k codebook)."""
    return IsoDiM_Tokenizer(
        output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
        dilation_growth_rate=3, activation='relu', norm=None,
        fsq_levels=get_fsq_config('mega'),  # [8, 8, 8, 8, 8, 5]
        **kwargs
    )


def isodim_tokenizer_highdim7(**kwargs) -> IsoDiM_Tokenizer:
    """Create IsoDiM_Tokenizer with FSQ-HighDim7 configuration (109k codebook, dim=7)."""
    return IsoDiM_Tokenizer(
        output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
        dilation_growth_rate=3, activation='relu', norm=None,
        fsq_levels=get_fsq_config('highdim7'),  # [7, 5, 5, 5, 5, 5, 5]
        **kwargs
    )


def isodim_tokenizer_highdim8(**kwargs) -> IsoDiM_Tokenizer:
    """Create IsoDiM_Tokenizer with FSQ-HighDim8 configuration (391k codebook, dim=8)."""
    return IsoDiM_Tokenizer(
        output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
        dilation_growth_rate=3, activation='relu', norm=None,
        fsq_levels=get_fsq_config('highdim8'),  # [5, 5, 5, 5, 5, 5, 5, 5]
        **kwargs
    )


# Model registry
Tokenizer_models = {
    # Standard configurations
    'IsoDiM_Tokenizer_Small': isodim_tokenizer_small,      # 3,125 codes (快速测试)
    'IsoDiM_Tokenizer_Medium': isodim_tokenizer_medium,    # 5,000 codes
    'IsoDiM_Tokenizer_Large': isodim_tokenizer_large,      # 36,000 codes
    'IsoDiM_Tokenizer_High': isodim_tokenizer_high,        # 64,000 codes (RECOMMENDED)
    'IsoDiM_Tokenizer_Ultra': isodim_tokenizer_ultra,      # 102,400 codes
    'IsoDiM_Tokenizer_Mega': isodim_tokenizer_mega,        # 163,840 codes
    # High-dimensional configurations
    'IsoDiM_Tokenizer_HighDim7': isodim_tokenizer_highdim7,  # 109,375 codes, dim=7
    'IsoDiM_Tokenizer_HighDim8': isodim_tokenizer_highdim8,  # 390,625 codes, dim=8
}

