"""
IsoDiM - Length Estimator Module
================================
Neural network for estimating motion length from text embeddings.
"""

import torch.nn as nn


class LengthEstimator(nn.Module):
    """
    Estimate motion length from CLIP text embeddings.
    
    Used during inference to predict appropriate motion length
    when not explicitly specified.
    
    Args:
        input_size: Input embedding dimension (default: 512 for CLIP)
        output_size: Number of possible lengths (default: 50)
    """
    
    def __init__(self, input_size: int = 512, output_size: int = 50):
        super().__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(nd // 4, output_size)
        )
        
        self.output.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, text_emb):
        """
        Predict motion length distribution.
        
        Args:
            text_emb: CLIP text embeddings [B, 512]
            
        Returns:
            Length logits [B, output_size]
        """
        return self.output(text_emb)

