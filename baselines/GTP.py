"""
GTP (Graph Transformer with Pooling) model adapted for 3-class classification.
Based on the exact original GTP implementation with Vision Transformer and min-cut pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import dense_mincut_pool
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import math
import numpy as np


class GCNBlock(nn.Module):
    """GCN Block from original implementation."""
    
    def __init__(self, input_dim, output_dim, bn=1, add_self=1, normalize_embedding=1,
                 dropout=0.0, relu=0, bias=True):
        super(GCNBlock, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        self.relu = relu
        self.bn = bn
        self.eps = 1e-10
        
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        if self.bn:
            self.bn_layer = torch.nn.BatchNorm1d(output_dim)

        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        torch.nn.init.xavier_normal_(self.weight)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

    def forward(self, x, adj, mask):
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            
        # Handle batch normalization
        if self.bn:
            index = mask.sum(dim=1).long().tolist()
            bn_tensor_bf = mask.new_zeros((sum(index), y.shape[2]))
            bn_tensor_af = mask.new_zeros(*y.shape)
            start_index = []
            ssum = 0
            for i in range(x.shape[0]):
                start_index.append(ssum)
                ssum += index[i]
            start_index.append(ssum)
            for i in range(x.shape[0]):
                bn_tensor_bf[start_index[i]:start_index[i+1]] = y[i, 0:index[i]]
            bn_tensor_bf = self.bn_layer(bn_tensor_bf)
            for i in range(x.shape[0]):
                bn_tensor_af[i, 0:index[i]] = bn_tensor_bf[start_index[i]:start_index[i+1]]
            y = bn_tensor_af
            
        if self.dropout > 0.001:
            y = self.dropout_layer(y)
        if self.relu == 'relu':
            y = torch.nn.functional.relu(y)
        elif self.relu == 'lrelu':
            y = torch.nn.functional.leaky_relu(y, 0.1)
        return y


class Mlp(nn.Module):
    """MLP module from Vision Transformer."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head attention from Vision Transformer."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer adapted for graph classification."""
    
    def __init__(self, num_classes=3, embed_dim=64, depth=3, num_heads=8, 
                 mlp_ratio=2., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        # Use CLS token (first token)
        x = x[:, 0]
        x = self.head(x)
        return x


class GTP(nn.Module):
    """
    GTP (Graph Transformer with Pooling) adapted for 3-class classification.
    Based on the exact original GTP implementation.
    """
    
    def __init__(self, 
                 input_dim: int = 1024,
                 hidden_dim: int = 64,     # embed_dim from original  
                 num_classes: int = 3,     # normal, luad, lscc
                 pool_size: int = 100,     # node_cluster_num from original
                 num_transformer_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super(GTP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # embed_dim
        self.num_classes = num_classes
        self.pool_size = pool_size  # node_cluster_num
        self.num_transformer_layers = num_transformer_layers
        
        # Vision Transformer
        self.transformer = VisionTransformer(
            num_classes=num_classes, 
            embed_dim=hidden_dim,
            depth=num_transformer_layers,
            num_heads=num_heads,
            drop_rate=dropout
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # GCN layer (matching original)
        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(
            input_dim, hidden_dim, 
            self.bn, self.add_self, self.normalize_embedding, 0., 0
        )
        
        # Pooling layer (matching original)
        self.pool1 = Linear(hidden_dim, pool_size)

    def forward(self, sample):
        """
        Forward pass adapted for LENS dataset format.
        
        Args:
            sample: Dictionary containing 'image' (features) and 'adj_s' (adjacency matrix)
        
        Returns:
            logits: Classification logits for 3 classes
        """
        features = sample['image']  # [num_nodes, feature_dim]
        adj_matrix = sample['adj_s']  # [num_nodes, num_nodes]
        
        # Convert to batch format (batch_size=1 for single WSI)
        X = features.unsqueeze(0)  # [1, num_nodes, feature_dim]
        adj = adj_matrix.unsqueeze(0)  # [1, num_nodes, num_nodes]
        
        # Create mask (all nodes are valid)
        num_nodes = features.size(0)
        mask = torch.ones(1, num_nodes, dtype=torch.bool, device=features.device)
        
        # Apply mask to features
        X = mask.unsqueeze(2).float() * X
        
        # GCN layer
        X = self.conv1(X, adj, mask.float())
        
        # Pooling assignment matrix
        s = self.pool1(X)  # [1, num_nodes, pool_size]
        
        # Dense min-cut pooling
        X, adj, mc_loss, ortho_loss = dense_mincut_pool(X, adj, s, mask.float())
        
        # Add CLS token
        b = X.shape[0]
        cls_token = self.cls_token.repeat(b, 1, 1)
        X = torch.cat([cls_token, X], dim=1)
        
        # Vision Transformer
        out = self.transformer(X)
        
        return out.squeeze(0)  # Remove batch dimension for consistency
    
    def get_edge_retention_rate(self, sample):
        """GTP retains 100% of edges in the pooled graph."""
        return 1.0  # 100% retention as mentioned in paper


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor
