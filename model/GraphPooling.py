import torch
import torch.nn.functional as F

class EdgeWeightedAttentionPooling:
    def __init__(self):
        """Initialize the edge-weighted attention pooling module"""
        pass
    
    def edge_weighted_attention_pooling(self, node_feat, edge_weights, adj_matrix, masks=None):
        """
        Edge-weighted attention pooling that uses edge importance to generate node attention scores.
        
        Args:
            node_feat: Node features tensor [batch_size, num_nodes, feature_dim]
            edge_weights: Edge weights tensor [batch_size, num_nodes, num_nodes]
            adj_matrix: Original adjacency matrix [batch_size, num_nodes, num_nodes]
            masks: Optional node masks [batch_size, num_nodes]
            
        Returns:
            Graph-level representation [batch_size, feature_dim]
        """
        batch_size = node_feat.shape[0]
        
        # Create graph-level representations with attention
        graph_rep = torch.zeros(batch_size, node_feat.shape[2], device=node_feat.device)
        
        for b in range(batch_size):
            # Get valid nodes based on mask if provided
            if masks is not None:
                valid_indices = torch.where(masks[b] > 0)[0]
            else:
                valid_indices = torch.arange(node_feat.shape[1], device=node_feat.device)
            
            if len(valid_indices) == 0:
                continue
            
            # Calculate node importance based on their edge weights
            edge_mask = (adj_matrix[b] > 0).float()
            
            # For undirected graphs, only count edges once (use incoming OR outgoing, not both)
            # Weighted degree: sum of all weighted edges connected to each node
            node_importance = torch.sum(edge_weights[b, :, valid_indices] * edge_mask[:, valid_indices], dim=0)
            
            # Apply softmax to get attention weights
            if torch.sum(node_importance) > 0:
                attention_weights = F.softmax(node_importance, dim=0)
                
                # Apply attention weights to node features
                weighted_features = node_feat[b, valid_indices] * attention_weights.unsqueeze(1)
                
                # Sum to get graph representation
                graph_rep[b] = torch.sum(weighted_features, dim=0)
            else:
                # Fallback to average pooling if no edge weights
                graph_rep[b] = torch.mean(node_feat[b, valid_indices], dim=0)
        
        return graph_rep
