import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model.EGL_L0_Reg import EGLassoRegularization
from model.EdgeScoring import EdgeScoringNetwork
from model.GraphPooling import EdgeWeightedAttentionPooling
from model.StatsTracker import StatsTracker
from model.L0Utils import l0_train, l0_test, L0RegularizerParams  # Import L0 utilities with params

class ImprovedEdgeGNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes, lambda_reg=0.01, reg_mode='l0', edge_dim=32,
                 warmup_epochs=5, graph_size_adaptation=True, min_edges_per_node=2, dropout=0.2,
                 l0_gamma=-0.1, l0_zeta=1.1, l0_beta=0.66, initial_temp=5.0):
        
        super().__init__()
        self.num_classes = num_classes        
        
        # Create L0 parameters if using L0 regularization
        self.l0_params = None
        if reg_mode == 'l0':
            self.l0_params = L0RegularizerParams(gamma=l0_gamma, zeta=l0_zeta, beta_l0=l0_beta)
            print(f"Using L0 regularization with parameters: gamma={l0_gamma}, zeta={l0_zeta}, beta_l0={l0_beta}")
        
        # Create component modules with L0 parameters
        self.edge_scorer = EdgeScoringNetwork(feature_dim, edge_dim, dropout, l0_params=self.l0_params)
        self.pooling = EdgeWeightedAttentionPooling()
        self.regularizer = EGLassoRegularization(lambda_reg, reg_mode, warmup_epochs, l0_params=self.l0_params)
        self.stats_tracker = StatsTracker()
        
        # GNN layer
        self.conv = nn.Linear(feature_dim, hidden_dim)
        #classifier 
        self.classifier = nn.Sequential(
         spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
         nn.ReLU(),
         nn.LayerNorm(hidden_dim // 2),
         spectral_norm(nn.Linear(hidden_dim // 2, num_classes))
        )
        # Graph adaptation parameters
        self.graph_size_adaptation = graph_size_adaptation
        self.min_edges_per_node = min_edges_per_node
        
        # Store epoch information
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.initial_temp = initial_temp
        self.temperature = self.initial_temp
        
        # Store if using L0 regularization
        self.use_l0 = (reg_mode == 'l0')
        
        print(f"Initialized ImprovedEdgeGNN with {reg_mode} regularization (base_lambda={lambda_reg})")
        if self.use_l0:
            print(f"Using L0 regularization for sparse graph learning")
        else:
            print(f"Using warmup for {warmup_epochs} epochs")
            print(f"Using graph size adaptation: {graph_size_adaptation}")
            print(f"Using min edges per node: {min_edges_per_node}")
            print(f"Using Kaiming initialization with 3-layer edge scoring network")
            print(f"Using Gumbel-Sigmoid with initial temperature {self.temperature}")
    
    def set_print_stats(self, value):
        """Control whether to print stats during forward pass"""
        self.stats_tracker.print_stats = value
    
    def update_temperature_and_lambda(self):
        """Update temperature and lambda based on current epoch"""
        self.temperature = self.regularizer.update_temperature(
            self.current_epoch, self.warmup_epochs, self.initial_temp
        )
        self.regularizer.update_lambda(self.current_epoch, self.warmup_epochs)
        
        if self.stats_tracker.print_stats:
            print(f"Epoch {self.current_epoch}: temperature={self.temperature:.3f}, lambda={self.regularizer.current_lambda:.6f}")
    
    def set_epoch(self, epoch):
        """Set the current epoch number and update temperature/lambda"""
        self.current_epoch = epoch
        self.regularizer.current_epoch = epoch
        self.update_temperature_and_lambda()
    
    def update_l0_params(self, gamma=None, zeta=None, beta_l0=None):
        """Update L0 regularization parameters"""
        if self.l0_params is not None:
            self.l0_params.update_params(gamma, zeta, beta_l0)
            # Also update parameters in components
            if hasattr(self.edge_scorer, 'l0_params'):
                self.edge_scorer.l0_params = self.l0_params
            if hasattr(self.regularizer, 'l0_params'):
                self.regularizer.l0_params = self.l0_params
            
            print(f"Updated L0 parameters: gamma={self.l0_params.gamma}, zeta={self.l0_params.zeta}, beta_l0={self.l0_params.beta_l0}")
        else:
            print("Warning: Attempting to update L0 parameters but not using L0 regularization")
    
    def aggregate(self, node_feat, adj_matrix, edge_weights):
        """Neighborhood aggregation with learned edge weights"""
        # Apply edge weights to adjacency matrix
        weighted_adj = adj_matrix * edge_weights
        
        # Row-normalize weighted adjacency matrix
        row_sum = torch.sum(weighted_adj, dim=2, keepdim=True) + 1e-8
        norm_adj = weighted_adj / row_sum
        
        # Aggregate neighbor features (matrix form)
        return torch.bmm(norm_adj, node_feat)
    
    def forward(self, node_feat, labels, adjs, masks=None):
        """Forward pass with per-graph analysis and edge-weighted attention pooling"""
        node_feat = F.normalize(node_feat, p=2, dim=2) 
        batch_size = node_feat.shape[0]
        
        # Compute edge weights - pass regularizer if using L0
        if self.use_l0:
            edge_weights = self.edge_scorer.compute_edge_weights(
                node_feat=node_feat, 
                adj_matrix=adjs, 
                current_epoch=self.current_epoch,
                warmup_epochs=self.warmup_epochs,
                temperature=self.temperature, 
                graph_size_adaptation=self.graph_size_adaptation, 
                min_edges_per_node=self.min_edges_per_node,
                regularizer=self.regularizer,  # Pass regularizer for L0
                use_l0=True,  # Use L0 regularization
                print_stats=self.stats_tracker.print_stats,
                l0_params=self.l0_params  # Pass L0 parameters
            )
        else:
            edge_weights = self.edge_scorer.compute_edge_weights(
                node_feat=node_feat, 
                adj_matrix=adjs, 
                current_epoch=self.current_epoch,
                warmup_epochs=self.warmup_epochs,
                temperature=self.temperature, 
                graph_size_adaptation=self.graph_size_adaptation, 
                min_edges_per_node=self.min_edges_per_node,
                print_stats=self.stats_tracker.print_stats
            )
        
        # Aggregate neighbor features using learned edge weights
        h = self.aggregate(node_feat, adjs, edge_weights)
        
        # Apply GNN layer
        h = self.conv(h)
        h = F.relu(h)
        
        # Apply edge-weighted attention pooling instead of average pooling
        graph_rep = self.pooling.edge_weighted_attention_pooling(h, edge_weights, adjs, masks)
        
        # Classification
        logits = self.classifier(graph_rep)
        
        # Compute classification loss
        cls_loss = F.cross_entropy(logits, labels)
        
        # Compute regularization loss
        reg_loss = self.regularizer.compute_regularization(edge_weights, adjs)
        
        # Total loss
        total_loss = cls_loss + reg_loss
        
        # Track statistics
        self.stats_tracker.update_stats(edge_weights, adjs, cls_loss, reg_loss, 
                                        self.current_epoch, self.regularizer.current_lambda)
        
        # Return same format as expected
        return logits, labels, total_loss, adjs * edge_weights
    
    # Delegate visualization methods to the stats tracker
    def save_graph_analysis(self, epoch, batch_idx, save_dir='./'):
        return self.stats_tracker.save_graph_analysis(
            epoch, batch_idx, save_dir, self.regularizer.current_lambda, 
            self.temperature, self.warmup_epochs
        )
    
    def plot_edge_weight_distribution(self, weighted_adj, epoch, batch_idx=0, save_dir='./'):
        return self.stats_tracker.plot_edge_weight_distribution(
            weighted_adj, epoch, batch_idx, save_dir, self.regularizer.current_lambda, 
            self.temperature, self.current_epoch, self.warmup_epochs
        )
    
    def plot_stats(self, save_path='stats.png'):
        return self.stats_tracker.plot_stats(
            save_path, self.regularizer.reg_mode, self.regularizer.base_lambda, 
            self.warmup_epochs
        )
    
    def save_sparsification_report(self, epoch, save_dir='./'):
        return self.stats_tracker.save_sparsification_report(
            epoch, save_dir, self.regularizer.current_lambda, self.temperature,
            self.warmup_epochs
        )
