"""
vision_transformer.py - Simplified Vision Transformer Model Architecture

This file implements a lightweight Vision Transformer (ViT) for the gas state classification task.
The model has been intentionally simplified to prevent overfitting, with reduced capacity and
increased regularization.

Key Components:
1. Patch Embedding: Splits 28x28 input into 4x4 patches (49 patches total)
2. Multi-Head Attention: Uses 2 attention heads for balanced feature extraction
3. Single Transformer Block: One block is sufficient for this binary task
4. Classification Head: Makes the final binary prediction

The architecture is based on the Vision Transformer paper but significantly scaled down
to match the complexity of the binary classification task.

Dependencies:
- torch: Main deep learning framework
- torch.nn: Neural network layers and utilities
- einops: For tensor reshaping operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    """Splits input image into patches and linearly embeds them.
    
    For our 28x28 input images, we use 2x2 patches, resulting in 14x14=196 patches.
    This finer granularity helps preserve subtle patterns in gas behavior.
    Each patch is then linearly projected to an increased embedding dimension.
    
    Args:
        in_channels (int): Number of input channels (default: 2 for density and recording_date)
        patch_size (int): Size of each patch (default: 4)
        emb_dim (int): Dimension of patch embeddings (default: 32, reduced to prevent overfitting)
    """
    def __init__(self, in_channels: int = 2, patch_size: int = 4, emb_dim: int = 32):
        super().__init__()
        self.patch_size = patch_size
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        # Add BatchNorm for regularization
        self.bn = nn.BatchNorm2d(emb_dim)
        # Add positional embeddings to help model understand patch positions
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # +1 for cls token
        self.pos_embedding = nn.Parameter(torch.randn(1, (28 // patch_size) ** 2 + 1, emb_dim))
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor of shape (batch_size, n_patches + 1, emb_dim)
        """
        # Project patches: (batch, channels, h, w) -> (batch, emb_dim, h', w')
        x = self.proj(x)
        # Apply batch normalization
        x = self.bn(x)
        # Rearrange to sequence: (batch, emb_dim, h', w') -> (batch, n_patches, emb_dim)
        x = rearrange(x, 'b e h w -> b (h w) e')
        
        # Add classification token
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings and dropout
        x = self.dropout(x + self.pos_embedding)
        return x


class MultiHeadAttention(nn.Module):
    """Simplified multi-head self-attention module.
    
    Uses fewer attention heads to reduce model capacity while still allowing
    the model to capture important patterns in the data.
    
    Args:
        emb_dim (int): Embedding dimension (default: 32)
        num_heads (int): Number of attention heads (default: 2)
        dropout (float): Dropout probability (default: 0.2)
    """
    def __init__(self, emb_dim: int = 32, num_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (emb_dim // num_heads) ** -0.5
        
        self.to_qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, emb_dim)
            
        Returns:
            Attended tensor of same shape as input
        """
        # Get query, key, value projections
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Simplified transformer block with increased regularization.
    
    Each block consists of:
    1. Layer normalization
    2. Multi-head self-attention (2 heads)
    3. Residual connection
    4. Layer normalization
    5. MLP with reduced dimension
    6. Residual connection
    
    Args:
        emb_dim (int): Embedding dimension (default: 32)
        num_heads (int): Number of attention heads (default: 2)
        mlp_dim (int): Hidden dimension of MLP layer (default: 64)
        dropout (float): Dropout probability (default: 0.2)
    """
    def __init__(self, emb_dim: int = 32, num_heads: int = 2, mlp_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, emb_dim)
            
        Returns:
            Processed tensor of same shape as input
        """
        # Attention block with residual
        x = x + self.attn(self.norm1(x))
        # MLP block with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Simplified Vision Transformer for gas state classification.
    
    This model processes 28x28 gas measurement images using a transformer architecture.
    The architecture has been optimized to capture subtle gas behavior differences:
    - Finer-grained 2x2 patches (196 patches instead of 49) to preserve detailed patterns
    - Increased embedding dimension (64 instead of 32) for richer feature representation
    - Two transformer blocks for hierarchical feature learning
    - Maintained dropout (0.2) for regularization
    
    Architecture Overview:
    1. Patch Embedding: Split 28x28 input into 2x2 patches (196 patches total)
    2. Two Transformer Blocks: Process patches hierarchically using attention and MLP
    3. Classification Head: Convert final representations to binary prediction
    
    Args:
        in_channels (int): Number of input channels (default: 2)
        patch_size (int): Size of image patches (default: 4)
        emb_dim (int): Embedding dimension (default: 32)
        depth (int): Number of transformer blocks (default: 1)
        num_heads (int): Number of attention heads (default: 2)
        mlp_dim (int): Hidden dimension of MLP layers (default: 64)
        num_classes (int): Number of output classes (default: 1 for binary)
        dropout (float): Dropout probability (default: 0.2)
    """
    def __init__(self,
                 in_channels: int = 1,  # Only density channel
                 patch_size: int = 2,  # Reduced from 4 to capture finer details
                 emb_dim: int = 64,    # Increased from 32 for better feature representation
                 depth: int = 2,       # Increased from 1 to allow hierarchical feature learning
                 num_heads: int = 2,
                 mlp_dim: int = 64,
                 num_classes: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        
        # Initial patch embedding
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim)
        
        # Single transformer block (reduced from 3)
        self.transformer = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """
        # Convert image to patch embeddings
        x = self.patch_embed(x)
        
        # Apply transformer blocks
        for block in self.transformer:
            x = block(x)
            
        # Use [CLS] token for classification
        x = self.norm(x[:, 0])
        x = self.fc(x)
        return x