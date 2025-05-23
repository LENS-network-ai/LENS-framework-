import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Constants for L0 regularization
sig = nn.Sigmoid()
hardtanh = nn.Hardtanh(0, 1)
gamma = -0.1
zeta = 1.1
beta = 0.66
eps = 1e-20
const1 = beta * np.log(-gamma / zeta + eps)

def l0_train(logAlpha, min_val, max_val):
    """L0 regularization function for training"""
    U = torch.rand(logAlpha.size()).type_as(logAlpha) + eps
    s = sig((torch.log(U / (1 - U)) + logAlpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min_val, max_val)
    return mask

def l0_test(logAlpha, min_val, max_val):
    """L0 regularization function for testing"""
    s = sig(logAlpha / beta)
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min_val, max_val)
    return mask

def get_loss2(logAlpha):
    """Compute the L0 regularization loss"""
    return sig(logAlpha - const1)

class GATLayer(nn.Module):
    """
    Graph Attention Network layer that can be integrated with the LENS framework
    
    This implementation adapts the original GAT with L0 regularization for use 
    with adjacency matrices directly rather than requiring DGL.
    """
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 num_heads=1,
                 feat_drop=0.0, 
                 attn_drop=0.0, 
                 alpha=0.2,
                 bias_l0=-0.5, 
                 residual=False, 
                 l0=0):
        super(GATLayer, self).__init__()
        
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.l0 = l0  # Whether to use L0 regularization
        
        # Feature projection
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        
        # Dropouts
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else lambda x: x
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else lambda x: x
        
        # Attention parameters
        self.attn_l = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        self.bias_l0 = nn.Parameter(torch.FloatTensor([bias_l0]))
        
        # Activation and regularization
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.residual = residual
        
        # Residual connection
        if residual:
            if in_dim != num_heads * out_dim:
                self.res_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
            else:
                self.res_fc = None
        
        # Initialize weights
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)
        
        # For tracking statistics
        self.attention_weights = None
        self.l0_loss = 0
    
    def forward(self, node_feat, adj_matrix):
        """
        Forward pass for the GAT layer
        
        Args:
            node_feat: Node features [batch_size, num_nodes, in_dim]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node features [batch_size, num_nodes, num_heads * out_dim]
            Attention weights [batch_size, num_nodes, num_nodes, num_heads]
        """
        batch_size, num_nodes, _ = node_feat.shape
        self.l0_loss = 0
        
        # Apply feature dropout
        h = self.feat_drop(node_feat)
        
        # Project features
        feat = self.fc(h).view(batch_size, num_nodes, self.num_heads, self.out_dim)
        
        # Compute attention coefficients
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)  # [batch, nodes, heads, 1]
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)  # [batch, nodes, heads, 1]
        
        attention = torch.zeros(batch_size, num_nodes, num_nodes, self.num_heads, device=node_feat.device)
        
        # For each batch
        for b in range(batch_size):
            # Only compute attention for existing edges
            src_nodes, dst_nodes = torch.where(adj_matrix[b] > 0)
            
            if len(src_nodes) > 0:
                # Get features for connected nodes
                src_el = el[b, src_nodes]  # [edges, heads, 1]
                dst_er = er[b, dst_nodes]  # [edges, heads, 1]
                
                # Compute unnormalized attention
                if self.l0 == 0:
                    # Standard GAT attention
                    edge_attention = self.leaky_relu(src_el + dst_er)
                else:
                    # L0 regularized attention
                    logits = src_el + dst_er + self.bias_l0
                    
                    if self.training:
                        edge_attention = l0_train(logits, 0, 1)
                    else:
                        edge_attention = l0_test(logits, 0, 1)
                    
                    # Compute L0 regularization loss
                    self.l0_loss += get_loss2(logits).sum()
                
                # Normalize attention coefficients (softmax over neighborhood)
                for node_idx in range(num_nodes):
                    # Find neighbors of this node
                    neighbors = (dst_nodes == node_idx)
                    if torch.any(neighbors):
                        neighbor_srcs = src_nodes[neighbors]
                        neighbor_attn = edge_attention[neighbors].squeeze(-1)  # [neighbors, heads]
                        
                        # Apply softmax over the neighborhood
                        normalized_attn = F.softmax(neighbor_attn, dim=0)
                        
                        # Store normalized attention
                        for i, src in enumerate(neighbor_srcs):
                            attention[b, src, node_idx] = normalized_attn[i]
        
        # Apply attention dropout
        attention = self.attn_drop(attention)
        self.attention_weights = attention
        
        # Apply attention to features
        output = torch.zeros(batch_size, num_nodes, self.num_heads, self.out_dim, device=node_feat.device)
        
        for b in range(batch_size):
            for h_idx in range(self.num_heads):
                # Matrix multiplication: attention @ features
                output[b, :, h_idx] = torch.mm(
                    attention[b, :, :, h_idx], 
                    feat[b, :, h_idx]
                )
        
        # Reshape output
        output = output.reshape(batch_size, num_nodes, -1)
        
        # Apply residual connection if needed
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(node_feat)
            else:
                resval = node_feat
            output = output + resval
        
        return output, self.attention_weights

class MultiLayerGAT(nn.Module):
    """
    Multi-layer Graph Attention Network that can be integrated with LENS
    """
    def __init__(self, 
                 in_dim, 
                 hidden_dim, 
                 out_dim, 
                 num_heads=8,
                 num_layers=2, 
                 dropout=0.1, 
                 attn_dropout=0.1,
                 alpha=0.2, 
                 bias_l0=-0.5, 
                 residual=True, 
                 l0=0):
        super(MultiLayerGAT, self).__init__()
        
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        
        # Input projection layer
        self.gat_layers.append(GATLayer(
            in_dim=in_dim, 
            out_dim=hidden_dim, 
            num_heads=num_heads,
            feat_drop=dropout, 
            attn_drop=attn_dropout, 
            alpha=alpha, 
            bias_l0=bias_l0, 
            residual=False, 
            l0=l0
        ))
        
        # Hidden layers
        for _ in range(1, num_layers):
            self.gat_layers.append(GATLayer(
                in_dim=hidden_dim * num_heads, 
                out_dim=hidden_dim, 
                num_heads=num_heads,
                feat_drop=dropout, 
                attn_drop=attn_dropout, 
                alpha=alpha, 
                bias_l0=bias_l0, 
                residual=residual, 
                l0=l0
            ))
        
        # Output layer
        self.gat_layers.append(GATLayer(
            in_dim=hidden_dim * num_heads, 
            out_dim=out_dim, 
            num_heads=1,
            feat_drop=dropout, 
            attn_drop=attn_dropout, 
            alpha=alpha, 
            bias_l0=bias_l0, 
            residual=residual, 
            l0=l0
        ))
        
        self.activation = F.elu
    
    def forward(self, node_feat, adj_matrix):
        """
        Forward pass through the multi-layer GAT
        
        Args:
            node_feat: Node features [batch_size, num_nodes, in_dim]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node features and attention weights from the last layer
        """
        h = node_feat
        attention_weights = []
        l0_loss = 0
        
        # Input layer
        h, attn = self.gat_layers[0](h, adj_matrix)
        attention_weights.append(attn)
        l0_loss += self.gat_layers[0].l0_loss
        h = self.activation(h)
        
        # Hidden layers
        for i in range(1, self.num_layers):
            h, attn = self.gat_layers[i](h, adj_matrix)
            attention_weights.append(attn)
            l0_loss += self.gat_layers[i].l0_loss
            h = self.activation(h)
        
        # Output layer
        h, attn = self.gat_layers[-1](h, adj_matrix)
        attention_weights.append(attn)
        l0_loss += self.gat_layers[-1].l0_loss
        
        return h, attention_weights, l0_loss
