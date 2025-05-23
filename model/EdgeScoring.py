import torch
import torch.nn as nn
import torch.nn.functional as F
from L0Utils import l0_train, l0_test, L0RegularizerParams

class EdgeScoringNetwork(nn.Module):
    def __init__(self, feature_dim, edge_dim, dropout=0.2, l0_params=None):
        """Initialize the edge scoring network
        
        Args:
            feature_dim: Dimension of node features
            edge_dim: Dimension of edge hidden layer
            dropout: Dropout rate
            l0_params: Optional L0RegularizerParams for customization
        """
        super().__init__()
        
        # Store L0 parameters
        self.l0_params = l0_params if l0_params is not None else L0RegularizerParams()
        
        # Edge scoring network - computes weight for each edge
        self.edge_encoder_l1 = nn.Linear(feature_dim * 2 + feature_dim, edge_dim)  # +feature_dim for difference features
        self.edge_encoder_bn1 = nn.BatchNorm1d(edge_dim)
        self.edge_encoder_l2 = nn.Linear(edge_dim, edge_dim // 2)
        self.edge_encoder_bn2 = nn.BatchNorm1d(edge_dim // 2)
        self.edge_encoder_l3 = nn.Linear(edge_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Feature attention layer to focus on important feature dimensions
        self.feature_attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Softmax(dim=1)
        )
        
        # Use Kaiming initialization for ReLU networks
        nn.init.kaiming_normal_(self.edge_encoder_l1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.edge_encoder_l2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.edge_encoder_l3.bias, 0.0)  # Start with unbiased
    
    def compute_edge_weights(self, node_feat, adj_matrix, current_epoch, warmup_epochs, 
                           temperature, graph_size_adaptation, min_edges_per_node, 
                           regularizer=None, use_l0=False, print_stats=True, l0_params=None):
        """Improved edge weight computation with feature attention and graph size adaptation
        
        Args:
            node_feat: Node features tensor [batch_size, num_nodes, feature_dim]
            adj_matrix: Original adjacency matrix [batch_size, num_nodes, num_nodes]
            current_epoch: Current training epoch
            warmup_epochs: Number of warmup epochs
            temperature: Temperature for Gumbel-Sigmoid
            graph_size_adaptation: Whether to adapt sparsity to graph size
            min_edges_per_node: Minimum edges to keep per node
            regularizer: Regularization module (for L0)
            use_l0: Whether to use L0 regularization
            print_stats: Whether to print debug stats
            l0_params: Optional L0RegularizerParams to use for this forward pass
        """
        # Update L0 parameters if provided
        if l0_params is not None:
            self.l0_params = l0_params
            
        batch_size, num_nodes, feat_dim = node_feat.shape
        edge_weights = torch.zeros_like(adj_matrix)
        
        # Clear logits storage if using L0
        if use_l0 and regularizer is not None:
            regularizer.clear_logits()
        
        # For tracking batch-level statistics
        all_weights = []
        
        # For tracking detailed per-graph statistics
        graph_details = []
        
        # Process each graph in the batch
        for b in range(batch_size):
            # Find existing edges in the original graph
            src_nodes, tgt_nodes = torch.where(adj_matrix[b] > 0)
            
            if len(src_nodes) == 0:
                continue
            
            # Determine target sparsity based on graph size if adaptation is enabled
            target_sparsity = 1.0  # Default to no pruning during warmup
            if graph_size_adaptation and current_epoch >= warmup_epochs:
                # Adaptive pruning based on graph size
                # For smaller graphs, we want less pruning
                edges_per_node = len(src_nodes) / num_nodes
                
                if edges_per_node < min_edges_per_node * 2:
                    # For very sparse graphs, prune less aggressively
                    target_sparsity = 0.8
                elif edges_per_node < min_edges_per_node * 4:
                    # For moderately sparse graphs
                    target_sparsity = 0.5
                else:
                    # For dense graphs, prune more aggressively
                    target_sparsity = 0.2
                
                # During warmup, gradually approach target sparsity
                if current_epoch < warmup_epochs + 5:
                    warmup_progress = min(1.0, (current_epoch - warmup_epochs) / 5)
                    target_sparsity = 1.0 - (1.0 - target_sparsity) * warmup_progress
            
            # Enhanced feature processing
            # Get node features
            source_feats = node_feat[b, src_nodes]
            target_feats = node_feat[b, tgt_nodes]
            
            # Normalize features for more stable learning
            source_feats = F.normalize(source_feats, p=2, dim=1)
            target_feats = F.normalize(target_feats, p=2, dim=1)
            
            # Compute feature differences (absolute) for additional signal
            feat_diff = torch.abs(source_feats - target_feats)
            
            # Concatenate features for edge representation
            edge_feats_raw = torch.cat([source_feats, target_feats], dim=1)
            
            # Apply feature attention to identify important dimensions
            if len(edge_feats_raw) > 0:  # Safety check
                attention_weights = self.feature_attention(edge_feats_raw)
                # Apply attention to difference features
                attended_diff = feat_diff * attention_weights
                
                # Final edge features - concatenate all
                edge_feats = torch.cat([source_feats, target_feats, attended_diff], dim=1)
                
                # Debug feature statistics (only for first graph)
                if b == 0 and print_stats:
                    print(f"[DEBUG] Source features - mean: {source_feats.mean().item():.4f}, std: {source_feats.std().item():.4f}")
                    print(f"[DEBUG] Feature differences - mean: {feat_diff.mean().item():.4f}, std: {feat_diff.std().item():.4f}")
                    print(f"[DEBUG] Attention weights - mean: {attention_weights.mean().item():.4f}, max: {attention_weights.max().item():.4f}")
                  
                # Get hidden representation with improved network
                # First layer with batch norm and dropout
                hidden = self.edge_encoder_l1(edge_feats)
                hidden = self.edge_encoder_bn1(hidden)
                hidden = F.relu(hidden)
                hidden = self.dropout(hidden)
                
                # Second layer with batch norm and dropout
                hidden = self.edge_encoder_l2(hidden)
                hidden = self.edge_encoder_bn2(hidden)
                hidden = F.relu(hidden)
                hidden = self.dropout(hidden)
                
                # Debug hidden layer statistics (only for first graph)
                if b == 0 and print_stats:
                    print(f"[DEBUG] Hidden layer - mean: {hidden.mean().item():.4f}, std: {hidden.std().item():.4f}")
                
                # Final layer
                logits = self.edge_encoder_l3(hidden).squeeze(-1)
                
                # If using L0 regularization, store logits
                if use_l0 and regularizer is not None:
                    regularizer.store_logits(b, logits)
                
                # Debug logits (only for first graph)
                if b == 0 and print_stats:
                    print(f"[DEBUG] Raw logits - mean: {logits.mean().item():.4f}, min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")
                
                # Apply L0 or Gumbel-Sigmoid based on use_l0 flag
                if use_l0:
                    # L0 regularization approach
                    if self.training:
                        edge_scores = l0_train(logits, 0, 1, params=self.l0_params)
                        if b == 0 and print_stats:
                            print(f"[DEBUG] Using L0 regularization (training mode)")
                    else:
                        edge_scores = l0_test(logits, 0, 1, params=self.l0_params)
                        if b == 0 and print_stats:
                            print(f"[DEBUG] Using L0 regularization (evaluation mode)")
                else:
                    # Original Gumbel-Sigmoid approach
                    if self.training:
                        # Sample from Gumbel distribution
                        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
                        
                        # Add noise to logits (stochastic during training)
                        gumbel_logits = (logits + gumbel_noise) / temperature
                        
                        # Apply sigmoid for smooth binary decision
                        edge_scores = torch.sigmoid(gumbel_logits)
                        
                        if b == 0 and print_stats:
                            print(f"[DEBUG] Using Gumbel-Sigmoid with temperature {temperature:.4f}")
                    else:
                        # During evaluation, use a fixed validation temperature without Gumbel noise
                        validation_temp = 1.0  # Lower temperature for sharper decisions
                        edge_scores = torch.sigmoid(logits / validation_temp)
                
                # If in warmup period, boost edge weights to prevent early pruning
                if current_epoch < warmup_epochs and self.training:
                    warmup_factor = 1.0 - (current_epoch / warmup_epochs)
                    edge_scores = edge_scores * 0.5 + 0.5 * warmup_factor
                
                # Apply graph size adaptation if enabled (only during training)
                if graph_size_adaptation and target_sparsity < 1.0 and self.training:
                    # Sort edge scores
                    sorted_scores, _ = torch.sort(edge_scores, descending=True)
                    # Calculate threshold based on target sparsity
                    edge_count = len(sorted_scores)
                    keep_count = max(int(edge_count * target_sparsity), int(min_edges_per_node * num_nodes))
                    keep_count = min(keep_count, edge_count)
                    
                    if keep_count < edge_count:
                        # Use threshold to encourage keeping top edges
                        threshold = sorted_scores[keep_count-1].item()
                        # Boost edges above threshold
                        edge_scores = torch.where(
                            edge_scores >= threshold,
                            edge_scores * 1.5,
                            edge_scores * 0.5
                        )
                        # Re-normalize to [0, 1]
                        edge_scores = torch.clamp(edge_scores, 0.0, 1.0)
                
                # For inference/validation, apply explicit thresholding to keep only top edges
                if not self.training:
                    # Calculate threshold value based on percentile (keep top 20% of edges)
                    percentile = 50  # 80th percentile = keep top 20%
                    sorted_scores, _ = torch.sort(edge_scores, descending=True)
                    if len(sorted_scores) > 0:
                        threshold_idx = min(int(len(sorted_scores) * percentile / 100), len(sorted_scores) - 1)
                        threshold = sorted_scores[threshold_idx]
                        
                        # Apply threshold but maintain minimum edges per node
                        thresholded_scores = torch.zeros_like(edge_scores)
                        
                        # Create index mapping to track which edges belong to which nodes
                        src_to_idx = {}
                        for i, src in enumerate(src_nodes.tolist()):
                            if src not in src_to_idx:
                                src_to_idx[src] = []
                            src_to_idx[src].append(i)
                        
                        # First, keep all edges above threshold
                        above_threshold = edge_scores >= threshold
                        thresholded_scores[above_threshold] = edge_scores[above_threshold]
                        
                        # Then ensure each node has minimum edges
                        for src in src_to_idx:
                            edges_indices = src_to_idx[src]
                            scores = edge_scores[edges_indices]
                            active_edges = above_threshold[edges_indices].sum().item()
                            
                            # If node has fewer than minimum edges, add more
                            if active_edges < min_edges_per_node:
                                # Sort by score (highest first)
                                _, sorted_idx = torch.sort(scores, descending=True)
                                # Add top edges until we reach minimum
                                need_to_add = int(min(min_edges_per_node - active_edges, len(edges_indices)))
                                for i in range(need_to_add):
                                    idx = edges_indices[sorted_idx[i]]
                                    thresholded_scores[idx] = edge_scores[idx]
                        
                        # Use thresholded scores
                        edge_scores = thresholded_scores
                        
                        # Log sparsification for first graph
                        if b == 0 and print_stats:
                            total_edges = len(edge_scores)
                            active_edges = (edge_scores > 0).sum().item()
                            pct_kept = 100 * active_edges / total_edges
                            print(f"[VALIDATION] Keeping {active_edges}/{total_edges} edges ({pct_kept:.2f}%)")
                
                # Debug edge score statistics (only for first graph)
                if b == 0 and print_stats:
                    graph_mean = edge_scores.mean().item()
                    graph_min = edge_scores.min().item() if torch.any(edge_scores > 0) else 0.0
                    graph_max = edge_scores.max().item()
                    graph_std = edge_scores.std().item()
                    
                    print(f"[DEBUG] Edge scores - min: {graph_min:.6f}, max: {graph_max:.6f}")
                    print(f"[DEBUG] Edge scores - mean: {graph_mean:.6f}, std: {graph_std:.6f}")
                    print(f"[DEBUG] Number of edges: {len(src_nodes)}")
                    if self.training:
                        print(f"[DEBUG] Target sparsity: {target_sparsity:.2f}")
                
                # Assign weights to the graph
                edge_weights[b, src_nodes, tgt_nodes] = edge_scores
        
        return edge_weights
