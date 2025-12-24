"""
IsoDiM - Isometric Discrete Diffusion Model
============================================
Main model integrating MAR Transformer + JiT Diffusion Transformer with FSQ.

Key Features:
- Masked Autoregressive (MAR) Transformer for sequence modeling
- JiT-style Diffusion Transformer with x-prediction
- Grid Snapping for inference (snap predictions to FSQ grid)
- CLIP-based text conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import clip
from typing import Optional, List, Tuple
from functools import partial
from timm.models.vision_transformer import Mlp

from .Transformer import DiffTransformer_XPred, Transformer_models
from .Quantizer import FSQ, get_fsq_config
from utils.eval_utils import eval_decorator
from utils.train_utils import lengths_to_mask, uniform, get_mask_subset_prob, cosine_schedule


class IsoDiM(nn.Module):
    """
    Isometric Discrete Diffusion Model - Main Class
    
    Integrates:
    - MAR Transformer: Bidirectional sequence modeling with masking
    - JiT Diffusion Transformer: Per-token diffusion with x-prediction
    - FSQ Grid: Topology-preserving discrete latent space
    
    Args:
        fsq_dim: FSQ dimension (from tokenizer)
        cond_mode: Conditioning mode ('text', 'action', 'uncond')
        latent_dim: MAR Transformer hidden dimension
        ff_size: Feed-forward size
        num_layers: Number of MAR Transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        clip_dim: CLIP embedding dimension
        dit_model: Diffusion Transformer variant ('DiT-S', 'DiT-B', 'DiT-L', 'DiT-XL')
        cond_drop_prob: Condition dropout probability for CFG
        fsq_levels: FSQ quantization levels for grid snapping
        clip_version: CLIP model version
    """
    
    def __init__(
        self,
        fsq_dim: int,
        cond_mode: str = 'text',
        latent_dim: int = 1024,
        ff_size: int = 4096,
        num_layers: int = 1,
        num_heads: int = 16,
        dropout: float = 0.2,
        clip_dim: int = 512,
        dit_model: str = 'DiT-XL',
        cond_drop_prob: float = 0.1,
        fsq_levels: Optional[List[int]] = None,
        clip_version: str = 'ViT-B/32',
        **kwargs
    ):
        super().__init__()
        
        self.fsq_dim = fsq_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob
        
        # FSQ configuration for grid snapping
        if fsq_levels is None:
            fsq_levels = get_fsq_config('high')
        self.fsq_levels = fsq_levels
        self.fsq = FSQ(levels=fsq_levels)
        
        if self.cond_mode == 'action':
            assert 'num_actions' in kwargs
            self.num_actions = kwargs.get('num_actions', 1)
            self.encode_action = partial(F.one_hot, num_classes=self.num_actions)
        
        # =======================================================================
        # MAR Transformer: Bidirectional sequence modeling
        # =======================================================================
        print('Loading MAR Transformer...')
        self.input_process = InputProcess(self.fsq_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.MARTransformer = nn.ModuleList([
            MARTransBlock(self.latent_dim, num_heads, mlp_size=ff_size, drop_out=self.dropout)
            for _ in range(num_layers)
        ])
        
        # Condition embedding
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError(f"Unsupported condition mode: {self.cond_mode}")
        
        # Mask latent for masked tokens
        self.mask_latent = nn.Parameter(torch.zeros(1, 1, self.fsq_dim))
        
        # Initialize weights
        self.apply(self._init_weights)
        for block in self.MARTransformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Load CLIP for text conditioning
        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self._load_and_freeze_clip(clip_version)
        
        # =======================================================================
        # JiT Diffusion Transformer: Per-token x-prediction
        # =======================================================================
        print(f'Loading Diffusion Transformer ({dit_model})...')
        self.DiffTransformer = Transformer_models[dit_model](
            target_channels=self.fsq_dim,
            z_channels=self.latent_dim
        )
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)
    
    def _load_and_freeze_clip(self, clip_version: str):
        """Load and freeze CLIP model."""
        clip_model, _ = clip.load(clip_version, device='cpu', jit=False)
        assert torch.cuda.is_available()
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model
    
    def encode_text(self, raw_text: List[str]) -> torch.Tensor:
        """Encode text using CLIP."""
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text
    
    def mask_cond(self, cond: torch.Tensor, force_mask: bool = False) -> torch.Tensor:
        """Apply condition masking for classifier-free guidance training."""
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond
    
    def forward(
        self,
        latents: torch.Tensor,
        cond: torch.Tensor,
        padding_mask: torch.Tensor,
        force_mask: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MAR Transformer forward pass.
        
        Args:
            latents: FSQ latent codes [B, L, d]
            cond: Condition embedding [B, clip_dim]
            padding_mask: Padding mask [B, L]
            force_mask: Force masking for CFG
            mask: Token mask for hard pseudo reorder
            
        Returns:
            Transformed latents [B, L, latent_dim]
        """
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.input_process(latents)
        cond = self.cond_emb(cond)
        x = self.position_enc(x)
        x = x.permute(1, 0, 2)
        
        # Hard pseudo reorder (optional)
        if mask is not None:
            sort_indices = torch.argsort(mask.to(torch.float), dim=1)
            x = torch.gather(x, dim=1, index=sort_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            inverse_indices = torch.argsort(sort_indices, dim=1)
            padding_mask = torch.gather(padding_mask, dim=1, index=sort_indices)
        
        for block in self.MARTransformer:
            x = block(x, cond, padding_mask)
        
        if mask is not None:
            x = torch.gather(x, dim=1, index=inverse_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        
        return x
    
    def forward_loss(
        self,
        fsq_targets: torch.Tensor,
        y: List[str],
        m_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            fsq_targets: FSQ quantized codes [B, L, d] - the diffusion target
            y: Text conditions (list of strings)
            m_lens: Motion lengths [B,]
            
        Returns:
            Loss scalar
        """
        b, l, d = fsq_targets.shape
        device = fsq_targets.device
        
        non_pad_mask = lengths_to_mask(m_lens, l)
        fsq_targets = torch.where(non_pad_mask.unsqueeze(-1), fsq_targets, torch.zeros_like(fsq_targets))
        
        target = fsq_targets.clone().detach()
        input_latents = fsq_targets.clone()
        
        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.encode_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError(f"Unsupported condition mode: {self.cond_mode}")
        
        # Random masking with cosine schedule
        rand_time = uniform((b,), device=device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_masked = (l * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((b, l), device=device).argsort(dim=-1)
        mask = batch_randperm < num_masked.unsqueeze(-1)
        mask &= non_pad_mask
        
        # Apply masking: random noise + mask token
        mask_rlatents = get_mask_subset_prob(mask, 0.1)
        rand_latents = torch.randn_like(input_latents)
        input_latents = torch.where(mask_rlatents.unsqueeze(-1), rand_latents, input_latents)
        mask_mlatents = get_mask_subset_prob(mask & ~mask_rlatents, 0.88)
        input_latents = torch.where(mask_mlatents.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), input_latents)
        
        # MAR Transformer forward
        z = self.forward(input_latents, cond_vector, ~non_pad_mask, force_mask)
        
        # Prepare for diffusion: flatten masked tokens
        target_flat = target.reshape(b * l, -1)
        z_flat = z.reshape(b * l, -1)
        mask_flat = mask.reshape(b * l)
        
        target_masked = target_flat[mask_flat]
        z_masked = z_flat[mask_flat]
        
        # Diffusion loss (x-prediction)
        loss = self.DiffTransformer(target=target_masked, z=z_masked)
        
        return loss
    
    def snap_to_fsq_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Snap continuous predictions to FSQ grid points (without tanh compression).
        
        IMPORTANT: The diffusion model output is already trained to predict values
        in the FSQ quantized space [-1, 1]. We should NOT apply tanh again!
        
        This method directly scales -> rounds -> rescales, preserving the values
        that the diffusion model learned to predict.
        
        Args:
            x: Continuous predictions [B, L, d] or [B, d], already in [-1, 1] range
            
        Returns:
            Grid-snapped predictions on valid FSQ grid points
        """
        # Get half_levels for each FSQ dimension: (levels - 1) / 2
        # e.g., for levels=[8,8,8,5,5,5], half_levels=[3.5, 3.5, 3.5, 2, 2, 2]
        half_levels = (self.fsq._levels_tensor - 1) / 2  # [d]
        
        # Scale from [-1, 1] to [-half_levels, half_levels]
        z_scaled = x * half_levels
        
        # Round to nearest integer grid point
        z_quantized = torch.round(z_scaled)
        
        # Clamp to valid range (safety measure)
        z_quantized = torch.clamp(z_quantized, -half_levels, half_levels)
        
        # Scale back to [-1, 1]
        z_snapped = z_quantized / half_levels
        
        return z_snapped
    
    def forward_with_CFG(
        self,
        latents: torch.Tensor,
        cond_vector: torch.Tensor,
        padding_mask: torch.Tensor,
        cfg: float = 3.0,
        mask: Optional[torch.Tensor] = None,
        force_mask: bool = False,
        hard_pseudo_reorder: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with Classifier-Free Guidance.
        
        保留并行计算逻辑（速度快），使用 torch.where 做安全赋值。
        
        Args:
            latents: Input latents [B, L, fsq_dim]
            cond_vector: Condition embedding [B, latent_dim]
            padding_mask: Padding mask [B, L]
            cfg: CFG scale
            mask: Token mask [B, L] - True 表示需要预测的位置
            force_mask: Force unconditional
            hard_pseudo_reorder: Use hard pseudo reorder
            
        Returns:
            CFG-guided predictions [B, L, fsq_dim]
        """
        b, l, _ = latents.shape
        
        # Handle hard pseudo reorder
        if hard_pseudo_reorder:
            reorder_mask = mask.clone() if mask is not None else None
        else:
            reorder_mask = None
        
        # Force mask 模式：直接返回 unconditional 输出
        if force_mask:
            return self.forward(latents, cond_vector, padding_mask, force_mask=True, mask=reorder_mask)
        
        # =====================================================================
        # Step 1: 并行 MAR Forward (速度快)
        # =====================================================================
        if cfg != 1:
            # 构造 CFG 输入：[Conditional; Unconditional] -> [2B, L, D]
            logits = self.forward(latents, cond_vector, padding_mask, mask=reorder_mask)
            aux_logits = self.forward(latents, cond_vector, padding_mask, force_mask=True, mask=reorder_mask)
            mixed_logits = torch.cat([logits, aux_logits], dim=0)  # [2B, L, latent_dim]
        else:
            mixed_logits = self.forward(latents, cond_vector, padding_mask, mask=reorder_mask)
        
        # =====================================================================
        # Step 2: Diffusion Sampling (保持完整序列，避免 chunk 不均匀问题)
        # =====================================================================
        total_b, seq_l, latent_d = mixed_logits.size()
        
        # Reshape: [B*L, latent_dim] 或 [2B*L, latent_dim]
        mixed_logits_flat = mixed_logits.reshape(total_b * seq_l, latent_d)
        
        # DiffTransformer.sample 内部处理 CFG
        # Input:  [2B*L, latent_dim] when cfg != 1, else [B*L, latent_dim]
        # Output: [B*L, fsq_dim] (只返回 conditional 部分)
        output = self.DiffTransformer.sample(z=mixed_logits_flat, cfg_scale=cfg)
        
        # Reshape back: [B, L, fsq_dim]
        scaled_logits = output.reshape(b, seq_l, self.fsq_dim)
        
        # =====================================================================
        # Step 3: FSQ Grid Snapping
        # =====================================================================
        scaled_logits = self.snap_to_fsq_grid(scaled_logits)
        
        # =====================================================================
        # Step 4: 安全赋值 (使用 torch.where，更安全)
        # =====================================================================
        if mask is not None:
            # mask: [B, L] -> [B, L, 1]
            mask_expanded = mask.unsqueeze(-1)
            # 如果 mask=True，取预测值；如果 mask=False，保留原始 latents
            output = torch.where(mask_expanded, scaled_logits, latents)
        else:
            output = scaled_logits
        
        return output
    
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        conds: List[str],
        m_lens: torch.Tensor,
        timesteps: int = 18,
        cond_scale: float = 4.5,
        temperature: float = 1.0,
        force_mask: bool = False,
        hard_pseudo_reorder: bool = False,
    ) -> torch.Tensor:
        """
        Generate FSQ codes from text conditions.
        
        Args:
            conds: Text conditions
            m_lens: Target motion lengths (in latent space)
            timesteps: Number of MAR sampling steps
            cond_scale: CFG scale
            temperature: Sampling temperature
            force_mask: Force unconditional
            hard_pseudo_reorder: Use hard pseudo reorder
            
        Returns:
            Generated FSQ codes [B, d, L] (ready for tokenizer decode)
        """
        device = next(self.parameters()).device
        l = max(m_lens)
        b = len(m_lens)
        
        # Get condition embeddings
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.encode_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError(f"Unsupported condition mode: {self.cond_mode}")
        
        padding_mask = ~lengths_to_mask(m_lens, l)
        
        # Initialize with mask tokens
        latents = torch.where(
            padding_mask.unsqueeze(-1),
            torch.zeros(b, l, self.fsq_dim).to(device),
            self.mask_latent.repeat(b, l, 1)
        )
        masked_rand_schedule = torch.where(
            padding_mask,
            1e5,
            torch.rand_like(padding_mask, dtype=torch.float)
        )
        
        # Iterative generation
        for timestep, steps_until_x0 in zip(
            torch.linspace(0, 1, timesteps, device=device),
            reversed(range(timesteps))
        ):
            rand_mask_prob = cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))
            
            latents = torch.where(is_mask.unsqueeze(-1), self.mask_latent.repeat(b, l, 1), latents)
            logits = self.forward_with_CFG(
                latents, cond_vector, padding_mask,
                cfg=cond_scale, mask=is_mask, force_mask=force_mask,
                hard_pseudo_reorder=hard_pseudo_reorder
            )
            
            # Grid snapping already applied in forward_with_CFG
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)
            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)
        
        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
        
        # Return [B, d, L] for compatibility with tokenizer
        return latents.permute(0, 2, 1)
    
    @torch.no_grad()
    @eval_decorator
    def edit(
        self,
        conds: List[str],
        latents: torch.Tensor,
        m_lens: torch.Tensor,
        timesteps: int = 18,
        cond_scale: float = 4.5,
        temperature: float = 1.0,
        force_mask: bool = False,
        edit_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        hard_pseudo_reorder: bool = False,
    ) -> torch.Tensor:
        """
        Edit existing FSQ codes based on text conditions.
        
        Args:
            conds: Text conditions
            latents: Existing FSQ codes [B, d, L]
            m_lens: Motion lengths
            timesteps: Number of editing steps
            cond_scale: CFG scale
            temperature: Sampling temperature
            force_mask: Force unconditional
            edit_mask: Mask for regions to edit [B, L]
            padding_mask: Padding mask
            hard_pseudo_reorder: Use hard pseudo reorder
            
        Returns:
            Edited FSQ codes [B, d, L]
        """
        device = next(self.parameters()).device
        l = latents.shape[-1]
        
        # Get condition embeddings
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.encode_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(1, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError(f"Unsupported condition mode: {self.cond_mode}")
        
        if padding_mask is None:
            padding_mask = ~lengths_to_mask(m_lens, l)
        
        if edit_mask is None:
            mask_free = True
            latents = torch.where(
                padding_mask.unsqueeze(-1),
                torch.zeros(latents.shape[0], l, self.fsq_dim).to(device),
                latents.permute(0, 2, 1)
            )
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            masked_rand_schedule = torch.where(edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            latents = torch.where(
                padding_mask.unsqueeze(-1),
                torch.zeros(latents.shape[0], l, self.fsq_dim).to(device),
                latents.permute(0, 2, 1)
            )
            latents = torch.where(
                edit_mask.unsqueeze(-1),
                self.mask_latent.repeat(latents.shape[0], l, 1),
                latents
            )
            masked_rand_schedule = torch.where(edit_mask, torch.rand_like(edit_mask, dtype=torch.float), 1e5)
        
        for timestep, steps_until_x0 in zip(
            torch.linspace(0, 1, timesteps, device=device),
            reversed(range(timesteps))
        ):
            rand_mask_prob = 0.16 if mask_free else cosine_schedule(timestep)
            num_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)
            sorted_indices = masked_rand_schedule.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_masked.unsqueeze(-1))
            
            latents = torch.where(
                is_mask.unsqueeze(-1),
                self.mask_latent.repeat(latents.shape[0], latents.shape[1], 1),
                latents
            )
            logits = self.forward_with_CFG(
                latents, cond_vector, padding_mask,
                cfg=cond_scale, mask=is_mask, force_mask=force_mask,
                hard_pseudo_reorder=hard_pseudo_reorder
            )
            latents = torch.where(is_mask.unsqueeze(-1), logits, latents)
            masked_rand_schedule = masked_rand_schedule.masked_fill(~is_mask, 1e5)
        
        latents = torch.where(edit_mask.unsqueeze(-1), latents, latents)
        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(latents), latents)
        
        return latents.permute(0, 2, 1)


# ============================================================================
# Inner Architecture Components
# ============================================================================

def modulate_here(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class InputProcess(nn.Module):
    """Process input FSQ codes to latent dimension."""
    
    def __init__(self, input_feats: int, latent_dim: int):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute((1, 0, 2))
        x = self.poseEmbedding(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class Attention(nn.Module):
    """Multi-head self-attention for MAR Transformer."""
    
    def __init__(self, embed_dim: int = 512, n_head: int = 8, drop_out_rate: float = 0.2):
        super().__init__()
        assert embed_dim % 8 == 0
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            mask = mask[:, None, None, :]
            att = att.masked_fill(mask != 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_drop(self.proj(y))
        return y


class MARTransBlock(nn.Module):
    """MAR Transformer block with adaLN."""
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_size: int = 1024, drop_out: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, drop_out_rate=drop_out)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = mlp_size
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate_here(self.norm1(x), shift_msa, scale_msa), mask=padding_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate_here(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ============================================================================
# Model Factory Functions
# ============================================================================

def isodim_dit_s(**kwargs) -> IsoDiM:
    """IsoDiM with DiT-Small diffusion backbone."""
    return IsoDiM(
        latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16,
        dropout=0.2, clip_dim=512, dit_model='DiT-S',
        cond_drop_prob=0.1, **kwargs
    )


def isodim_dit_b(**kwargs) -> IsoDiM:
    """IsoDiM with DiT-Base diffusion backbone."""
    return IsoDiM(
        latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16,
        dropout=0.2, clip_dim=512, dit_model='DiT-B',
        cond_drop_prob=0.1, **kwargs
    )


def isodim_dit_l(**kwargs) -> IsoDiM:
    """IsoDiM with DiT-Large diffusion backbone."""
    return IsoDiM(
        latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16,
        dropout=0.2, clip_dim=512, dit_model='DiT-L',
        cond_drop_prob=0.1, **kwargs
    )


def isodim_dit_xl(**kwargs) -> IsoDiM:
    """IsoDiM with DiT-XL diffusion backbone - RECOMMENDED."""
    return IsoDiM(
        latent_dim=1024, ff_size=4096, num_layers=1, num_heads=16,
        dropout=0.2, clip_dim=512, dit_model='DiT-XL',
        cond_drop_prob=0.1, **kwargs
    )


# Model registry
IsoDiM_models = {
    'IsoDiM-DiT-S': isodim_dit_s,
    'IsoDiM-DiT-B': isodim_dit_b,
    'IsoDiM-DiT-L': isodim_dit_l,
    'IsoDiM-DiT-XL': isodim_dit_xl,  # RECOMMENDED
}

