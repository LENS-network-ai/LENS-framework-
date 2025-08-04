"""
GraphLSurv model adapted for 3-class classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter
import numpy as np

# Constants from original utils
VERY_SMALL_NUMBER = 1e-12


class AnchorGraphLearner(torch.nn.Module):
    """Anchor-based Graph Learner adapted for LENS dataset format."""
    
    def __init__(self, in_dim, hid_dim=128, ratio_anchors=0.2, epsilon=0.9, topk=None, metric_type='weighted_cosine'):
        super(AnchorGraphLearner, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.epsilon = epsilon
        self.topk = topk
        self.metric_type = metric_type
        self.ratio_anchors = ratio_anchors

        if self.metric_type == 'weighted_cosine':
            self.transformer = nn.Linear(in_dim, hid_dim, bias=False)
            nn.init.xavier_uniform_(self.transformer.weight)
        else:
            raise NotImplementedError(f'{self.metric_type} has not been implemented.')

    def forward(self, x, node_mask=None):
        """
        Adapted for LENS format where x is [num_nodes, feature_dim].
        """
        # Convert to batch format for compatibility
        x = x.unsqueeze(0)  # [1, num_nodes, feature_dim]
        if node_mask is None:
            node_mask = torch.ones(1, x.size(1), dtype=torch.bool, device=x.device)
        else:
            node_mask = node_mask.unsqueeze(0)

        # Sample anchors
        anchors_x, anchor_mask = self.sample_anchors(x, node_mask, self.ratio_anchors)

        # Compute attention
        context_fc = self.transformer(x)
        context_norm = F.normalize(context_fc, p=2, dim=-1)

        anchors_fc = self.transformer(anchors_x)
        anchors_norm = F.normalize(anchors_fc, p=2, dim=-1)

        attention = torch.matmul(context_norm, anchors_norm.transpose(-1, -2))
        
        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, 0)

        anchor_adj = self.compute_anchor_adj(attention, anchor_mask=anchor_mask)

        return attention.squeeze(0), anchors_x.squeeze(0), anchor_adj.squeeze(0), anchor_mask.squeeze(0)

    def sample_anchors(self, x, node_mask, ratio, fill_value=0):
        """Sample anchor nodes."""
        batch_size = x.size(0)
        num_nodes = node_mask.sum(1)
        max_nodes = num_nodes.max().cpu()

        sampled_num_nodes = (ratio * num_nodes).to(num_nodes.dtype)
        sampled_max_nodes = sampled_num_nodes.max().cpu()
        
        sampled_col_index = [torch.randperm(num_nodes[i])[:sampled_num_nodes[i]] for i in range(x.size(0))]
        sampled_idx = torch.cat([i*max_nodes + sampled_col_index[i] for i in range(x.size(0))])
        insert_idx = torch.cat([i*sampled_max_nodes + torch.arange(sampled_num_nodes[i]) for i in range(x.size(0))])

        x = x.view([x.size(0) * x.size(1)] + list(x.size())[2:])
        size = [batch_size * sampled_max_nodes] + list(x.size())[1:]
        
        out = x.new_full(size, fill_value)
        out[insert_idx] = x[sampled_idx]
        out = out.view([batch_size, sampled_max_nodes] + list(x.size())[1:])

        mask = torch.zeros(batch_size * sampled_max_nodes, dtype=torch.bool, device=x.device)
        mask[insert_idx] = 1
        mask = mask.view([batch_size, sampled_max_nodes])

        return out, mask

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        """Build epsilon neighbourhood."""
        mask = (attention > epsilon).detach().float()
        return attention * mask + markoff_value * (1 - mask)

    def compute_anchor_adj(self, node_anchor_adj, anchor_mask=None):
        """Compute anchor adjacency matrix."""
        node_norm = node_anchor_adj / torch.clamp(node_anchor_adj.sum(dim=-2, keepdim=True), min=VERY_SMALL_NUMBER)
        anchor_norm = node_anchor_adj / torch.clamp(node_anchor_adj.sum(dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        anchor_adj = torch.matmul(node_norm.transpose(-1, -2), anchor_norm)

        markoff_value = 0
        if anchor_mask is not None:
            anchor_adj = anchor_adj.masked_fill_(~anchor_mask.bool().unsqueeze(-1), markoff_value)
            anchor_adj = anchor_adj.masked_fill_(~anchor_mask.bool().unsqueeze(-2), markoff_value)

        return anchor_adj


class AnchorGCNLayer(nn.Module):
    """Anchor GCN layer adapted for LENS format."""
    
    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(AnchorGCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, anchor_mp=True, batch_norm=True):
        """Forward pass with anchor message passing."""
        support = torch.matmul(input, self.weight)

        if anchor_mp:
            node_anchor_adj = adj
            node_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=VERY_SMALL_NUMBER)
            anchor_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            output = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), support))
        else:
            node_adj = adj
            output = torch.matmul(node_adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        """Compute batch normalization."""
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class AnchorGCN(nn.Module):
    """Anchor GCN encoder adapted for LENS format."""
    
    def __init__(self, nfeat, nhid, graph_hops=2, ratio_init_graph=0.2, dropout_ratio=0.2, batch_norm=False):
        super(AnchorGCN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.ratio_init_graph = ratio_init_graph

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(AnchorGCNLayer(nfeat, nhid, batch_norm=batch_norm))
        for _ in range(graph_hops - 1):
            self.graph_encoders.append(AnchorGCNLayer(nhid, nhid, batch_norm=batch_norm))

    def forward(self, x, init_adj, node_anchor_adj):
        """Forward pass with hybrid message passing."""
        # Convert to batch format for compatibility
        x = x.unsqueeze(0)
        init_adj = init_adj.unsqueeze(0)
        node_anchor_adj = node_anchor_adj.unsqueeze(0)
        
        for encoder in self.graph_encoders[:-1]:
            x = self.hybrid_message_passing(encoder, x, init_adj, node_anchor_adj)
        out_x = self.hybrid_message_passing(self.graph_encoders[-1], x, init_adj, node_anchor_adj, return_raw=True)

        return out_x.squeeze(0)

    def hybrid_message_passing(self, encoder, x, init_adj, node_anchor_adj, return_raw=False):
        """Hybrid message passing combining initial and learned graphs."""
        x_from_init_graph = encoder(x, init_adj, anchor_mp=False, batch_norm=False)
        x_from_learned_graph = encoder(x, node_anchor_adj, anchor_mp=True, batch_norm=False)
        x = self.hybrid_updata_x(x_from_init_graph, x_from_learned_graph)

        if return_raw:
            return x

        if encoder.bn is not None:
            x = encoder.compute_bn(x)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout_ratio, training=self.training)

        return x
    
    def hybrid_updata_x(self, x_init_graph, x_new_graph):
        """Hybrid update combining initial and new graph representations."""
        return self.ratio_init_graph * x_init_graph + (1 - self.ratio_init_graph) * x_new_graph


class GraphLSurv(torch.nn.Module):
    """
    GraphLSurv adapted for 3-class classification.
    """
    
    def __init__(self, 
                 input_dim: int = 1024,
                 hidden_dim: int = 256, 
                 num_classes: int = 3,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 # GraphLearner parameters
                 ratio_anchors: float = 0.2,
                 epsilon: float = 0.9,
                 # GCN Encoder parameters  
                 graph_hops: int = 2,
                 ratio_init_graph: float = 0.2):
        
        super(GraphLSurv, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphLearner and GCN encoder parameters
        args_glearner = {
            'hid_dim': hidden_dim,
            'ratio_anchors': ratio_anchors,
            'epsilon': epsilon,
            'topk': None,
            'metric_type': 'weighted_cosine'
        }
        
        args_gencoder = {
            'graph_hops': graph_hops,
            'ratio_init_graph': ratio_init_graph,
            'dropout_ratio': dropout,
            'batch_norm': False
        }
        
        # Create graph learner and encoder layers
        self.net_glearners = nn.ModuleList()
        self.net_encoders = nn.ModuleList()
        
        for i in range(self.num_layers):
            if i == 0:
                self.net_glearners.append(AnchorGraphLearner(input_dim, **args_glearner))
                self.net_encoders.append(AnchorGCN(input_dim, hidden_dim, **args_gencoder))
            else:
                self.net_glearners.append(AnchorGraphLearner(hidden_dim, **args_glearner))
                self.net_encoders.append(AnchorGCN(hidden_dim, hidden_dim, **args_gencoder))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )

    def forward(self, sample):
        """Forward pass for GraphLSurv."""
        features = sample['image']  # [num_nodes, feature_dim]
        adj_matrix = sample['adj_s']  # [num_nodes, num_nodes]
        
        # Normalize adjacency matrix
        init_adj = self.batch_normalize_adj(adj_matrix.unsqueeze(0)).squeeze(0)
        
        # Create node mask (all nodes are valid in LENS format)
        num_nodes = features.size(0)
        node_mask = torch.ones(num_nodes, dtype=torch.bool, device=features.device)
        
        prev_x = features
        out_anchor_graphs = {'x': [], 'adj': [], 'mask': []}
        
        # Apply graph learning and encoding layers
        for net_glearner, net_encoder in zip(self.net_glearners, self.net_encoders):
            # Learn node-anchor adjacency
            node_anchor_adj, anchor_x, anchor_adj, anchor_mask = net_glearner(prev_x, node_mask)
            
            out_anchor_graphs['x'].append(anchor_x)
            out_anchor_graphs['adj'].append(anchor_adj)
            out_anchor_graphs['mask'].append(anchor_mask)
            
            # Update node embedding via node-anchor-node schema
            node_vec = net_encoder(prev_x, init_adj, node_anchor_adj)
            prev_x = node_vec

        # Graph pooling: max and mean pooling
        out_max = self.graph_pool(node_vec, node_mask, 'max')
        out_avg = self.graph_pool(node_vec, node_mask, 'mean')
        out = torch.cat([out_max, out_avg], dim=0)  # [hidden_dim * 2]
        
        # Classification
        logits = self.classifier(out)
        
        return logits

    @staticmethod
    def graph_pool(x, node_mask=None, pool='max'):
        """Graph pooling operation adapted for single graph."""
        if pool == 'max':
            return torch.max(x, dim=0)[0]
        elif pool == 'mean':
            return torch.mean(x, dim=0)
        else:
            raise ValueError(f"Unsupported pooling method: {pool}")

    def batch_normalize_adj(self, mx):
        """Row-normalize adjacency matrix (symmetric normalized Laplacian)."""
        mx = mx.float()
        rowsum = torch.clamp(mx.sum(-1), min=1e-12)
        r_inv_sqrt = torch.pow(rowsum, -0.5)
        return r_inv_sqrt.unsqueeze(2) * mx * r_inv_sqrt.unsqueeze(1)

    def get_edge_retention_rate(self, sample):
        """Calculate edge retention rate based on actual anchor selection."""
        features = sample['image']
        adj_matrix = sample['adj_s']
        
        # Create node mask
        num_nodes = features.size(0)
        node_mask = torch.ones(num_nodes, dtype=torch.bool, device=features.device)
        
        # Run through first graph learner to get actual retention
        with torch.no_grad():
            net_glearner = self.net_glearners[0]
            node_anchor_adj, _, _, _ = net_glearner(features, node_mask)
            
            # Calculate actual edge retention
            total_possible = adj_matrix.numel()
            retained_edges = torch.sum(node_anchor_adj > net_glearner.epsilon).item()
            retention_rate = retained_edges / total_possible
            
        return retention_rate
