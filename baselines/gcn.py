"""
Standard GCN baseline model adapted for 3-class classification.
Adapted from original survival prediction models to work with LENS dataset structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse


class GCN(nn.Module):
    """
    Standard Graph Convolutional Network adapted for 3-class classification.
    Based on BasicGraphConvNet from original GraphLSurv codebase.
    """
    
    def __init__(self, 
                 input_dim: int = 1024,
                 hidden_dim: int = 256, 
                 num_classes: int = 3,  # normal, luad, lscc
                 num_layers: int = 3,
                 dropout: float = 0.2):
        super(GCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.graph_encoders.append(GCNConv(hidden_dim, hidden_dim))
        
        # Classification head (adapted from survival to 3-class classification)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for max+mean pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, sample):
        """Forward pass adapted for LENS dataset format."""
        features = sample['image']  # [num_nodes, feature_dim]
        adj_matrix = sample['adj_s']  # [num_nodes, num_nodes]
        
        # Convert dense adjacency to edge_index
        edge_index, edge_weight = dense_to_sparse(adj_matrix)
        
        x = features
        
        # Apply GCN layers
        for i in range(self.num_layers):
            x = F.relu(self.graph_encoders[i](x, edge_index, edge_weight))
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling: concatenate max and mean pooling
        x_max = torch.max(x, dim=0)[0]  # [hidden_dim]
        x_mean = torch.mean(x, dim=0)   # [hidden_dim]
        graph_representation = torch.cat([x_max, x_mean], dim=0)  # [hidden_dim * 2]
        
        # Classification
        logits = self.classifier(graph_representation)
        return logits
    
    def get_edge_retention_rate(self, sample):
        """GCN uses full connectivity (100% retention)."""
        return 1.0
