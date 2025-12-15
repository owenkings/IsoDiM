"""
IsoDiM - JiT-style 1D Transformer with X-Prediction
====================================================
Core diffusion backbone implementing JiT (Joint in Time) architecture.

Key Features:
- 1D Transformer with self-attention for inter-frame interactions
- adaLN (Adaptive Layer Normalization) for timestep conditioning
- Token-wise condition injection
- X-prediction (direct data prediction) instead of velocity prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from timm.models.vision_transformer import Mlp


class IsoDiM_Transformer(nn.Module):
    """
    JiT-style 1D Transformer for diffusion with X-prediction.
    
    This transformer predicts clean data x_1 directly from noisy input x_t,
    which is more suitable for FSQ's bounded discrete space.
    
    Architecture:
        Input x_t [B, L, d] + timestep t + condition z
        → Self-Attention Transformer Blocks with adaLN
        → Output x_1_pred [B, L, d]
    
    Args:
        target_channels: FSQ dimension (d), the prediction target
        z_channels: Condition dimension from MAR Transformer
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        hidden_size: Hidden dimension
        mlp_ratio: MLP expansion ratio
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        depth: int = 16,
        num_heads: int = 16,
        hidden_size: int = 1024,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.target_channels = target_channels
        self.z_channels = z_channels
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Linear(target_channels, hidden_size)
        
        # Timestep embedding
        self.time_embed = TimestepEmbedder(hidden_size)
        
        # Condition embedding
        self.cond_embed = nn.Linear(z_channels, hidden_size)
        
        # Transformer blocks with adaLN
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Output projection
        self.final_layer = FinalLayer(hidden_size, target_channels)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize model weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass predicting x_1 from x_t.
        
        Args:
            x: Noisy input x_t [B, d] or [B, L, d]
            t: Timestep [B,] in range [0, 1]
            c: Condition from MAR Transformer [B, z_channels] or [B, L, z_channels]
            mask: Optional attention mask
            
        Returns:
            Predicted clean data x_1 [B, d] or [B, L, d]
        """
        # Handle both per-token and batched inputs
        single_token = (x.dim() == 2)
        if single_token:
            x = x.unsqueeze(1)  # [B, 1, d]
            if c.dim() == 2:
                c = c.unsqueeze(1)  # [B, 1, z_channels]
        
        B, L, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # [B, L, hidden_size]
        
        # Timestep and condition embeddings
        t_emb = self.time_embed(t)  # [B, hidden_size]
        c_emb = self.cond_embed(c)  # [B, L, hidden_size]
        
        # Combined conditioning: timestep + per-token condition
        # Broadcast timestep to sequence length
        y = t_emb.unsqueeze(1) + c_emb  # [B, L, hidden_size]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, y, mask)
        
        # Final layer
        x = self.final_layer(x, y)
        
        if single_token:
            x = x.squeeze(1)  # [B, d]
        
        return x
    
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        Forward with Classifier-Free Guidance.
        
        Args:
            x: Noisy input [B*2, ...] with conditional and unconditional concatenated
            t: Timestep [B*2,]
            c: Condition [B*2, ...] with conditional and zeros concatenated
            cfg_scale: CFG scale factor
            
        Returns:
            CFG-guided output
        """
        half = x.shape[0] // 2
        combined = torch.cat([x[:half], x[:half]], dim=0)
        model_out = self.forward(combined, t, c)
        
        cond_out, uncond_out = torch.split(model_out, half, dim=0)
        cfg_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        
        return torch.cat([cfg_out, cfg_out], dim=0)


class DiffTransformer_XPred(nn.Module):
    """
    Complete Diffusion Transformer with X-Prediction Training.
    
    This module wraps IsoDiM_Transformer with:
    - Training loss computation (x-prediction MSE)
    - Sampling logic (ODE/Euler integration)
    
    Key Difference from velocity prediction:
    - Target: x_1 (clean data) instead of v = x_1 - x_0
    - More suitable for FSQ's bounded [-1, 1] space
    - Simpler target distribution
    """
    
    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        depth: int = 16,
        hidden_size: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.target_channels = target_channels
        
        self.net = IsoDiM_Transformer(
            target_channels=target_channels,
            z_channels=z_channels,
            depth=depth,
            num_heads=num_heads,
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    
    def forward(
        self, 
        target: torch.Tensor, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute x-prediction training loss.
        
        The forward interpolation is:
            x_t = t * x_1 + (1-t) * x_0
        
        where x_0 ~ N(0, I) and x_1 = target (clean data)
        
        Loss: MSE(model(x_t, t, z), x_1)
        
        Args:
            target: Clean data x_1 [B, d] or [B, L, d]
            z: Condition from MAR Transformer
            mask: Optional mask
            
        Returns:
            MSE loss scalar
        """
        B = target.shape[0]
        device = target.device
        
        # 1. Sample random timesteps uniformly from [0, 1]
        t = torch.rand(B, device=device)
        
        # 2. Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(target)
        
        # 3. Construct noisy input: x_t = t * x_1 + (1-t) * x_0
        # At t=0: x_t = x_0 (pure noise)
        # At t=1: x_t = x_1 (clean data)
        if target.dim() == 2:
            t_expand = t.view(-1, 1)
        else:
            t_expand = t.view(-1, 1, 1)
        
        x_t = t_expand * target + (1 - t_expand) * x_0
        
        # 4. Model predicts x_1 directly (NOT velocity!)
        x_1_pred = self.net(x_t, t, z)
        
        # 5. Compute x-prediction loss: MSE(pred, x_1)
        loss = (x_1_pred - target) ** 2
        
        if mask is not None:
            loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        num_steps: int = 50,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample using Euler integration of the probability flow ODE.
        
        For x-prediction, the velocity is:
            v = (x_1_pred - x_t) / (1 - t)
        
        Integration:
            x_{t+dt} = x_t + v * dt
        
        Args:
            z: Condition [B, z_channels] or [B*2, z_channels] for CFG
            num_steps: Number of integration steps
            cfg_scale: Classifier-free guidance scale
            
        Returns:
            Sampled clean data x_1
        """
        use_cfg = (cfg_scale != 1.0)
        
        if use_cfg:
            # z should be [B*2, ...] with cond and uncond concatenated
            B = z.shape[0] // 2
            noise = torch.randn(B, self.target_channels, device=z.device)
            noise = torch.cat([noise, noise], dim=0)
        else:
            B = z.shape[0]
            noise = torch.randn(B, self.target_channels, device=z.device)
        
        x = noise  # Start from pure noise (t=0)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(x.shape[0], device=x.device) * (i * dt)
            
            # Predict x_1
            if use_cfg:
                x_1_pred = self.net.forward_with_cfg(x, t, z, cfg_scale)
            else:
                x_1_pred = self.net(x, t, z)
            
            # Compute velocity: v = (x_1 - x_t) / (1 - t)
            # Avoid division by zero at t=1
            t_safe = torch.clamp(1 - t, min=1e-5)
            if x.dim() == 2:
                t_safe = t_safe.view(-1, 1)
            else:
                t_safe = t_safe.view(-1, 1, 1)
            
            v = (x_1_pred - x) / t_safe
            
            # Euler step
            x = x + v * dt
        
        if use_cfg:
            x = x[:B]  # Return only conditional samples
        
        return x


# ============================================================================
# Inner Architecture Components
# ============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN modulation."""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embed scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    Transformer block with adaptive layer norm (adaLN).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)
        
        # adaLN modulation: 6 * hidden_size for shift, scale, gate (x2 for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            c: Condition tensor [B, L, D] (timestep + per-token condition)
            mask: Optional attention mask
        """
        # Get modulation parameters (per-token)
        # c: [B, L, D] → modulation: [B, L, 6*D]
        modulation = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        
        # Self-attention with adaLN
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask=mask)
        
        # MLP with adaLN
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask[:, None, None, :]
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FinalLayer(nn.Module):
    """Final projection layer with adaLN."""
    
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        modulation = self.adaLN_modulation(c)
        shift, scale = modulation.chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ============================================================================
# Model Factory Functions
# ============================================================================

def dit_s(**kwargs) -> DiffTransformer_XPred:
    """DiT-Small: 12 layers, 384 hidden, 6 heads (~33M params)"""
    return DiffTransformer_XPred(depth=12, hidden_size=384, num_heads=6, **kwargs)


def dit_b(**kwargs) -> DiffTransformer_XPred:
    """DiT-Base: 12 layers, 768 hidden, 12 heads (~130M params)"""
    return DiffTransformer_XPred(depth=12, hidden_size=768, num_heads=12, **kwargs)


def dit_l(**kwargs) -> DiffTransformer_XPred:
    """DiT-Large: 24 layers, 1024 hidden, 16 heads (~460M params)"""
    return DiffTransformer_XPred(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def dit_xl(**kwargs) -> DiffTransformer_XPred:
    """DiT-XL: 28 layers, 1152 hidden, 16 heads (~675M params) - RECOMMENDED"""
    return DiffTransformer_XPred(depth=28, hidden_size=1152, num_heads=16, **kwargs)


# Model registry
Transformer_models = {
    'DiT-S': dit_s,
    'DiT-B': dit_b,
    'DiT-L': dit_l,
    'DiT-XL': dit_xl,
}

