import torch
import torch.nn as nn
import torch.nn.functional as F
from model.L0Utils import l0_train, l0_test, L0RegularizerParams

# In your existing EdgeScoringNetwork file (e.g., edge_scorer.py)

class EdgeScoringNetwork(nn.Module):
    def __init__(self, input_dim, edge_dim, l0_method='hard-concrete', l0_params=None):
        """
        Args:
            input_dim: Node feature dimension
            edge_dim: Hidden dimension for edge scoring
            l0_method: 'hard-concrete' or 'arm'
            l0_params: L0 parameters (L0RegularizerParams or ARML0RegularizerParams)
        """
        super().__init__()
        
        self.l0_method = l0_method
        self.l0_params = l0_params
        
        # Import appropriate L0 functions based on method
        if l0_method == 'hard-concrete':
            from L0Utils import l0_train, l0_test, get_loss2
            self.l0_train_fn = l0_train
            self.l0_test_fn = l0_test
            self.l0_loss_fn = get_loss2
        elif l0_method == 'arm':
            from L0Utils_ARM import arm_sample_gates, get_expected_l0_arm
            self.arm_sample_fn = arm_sample_gates
            self.arm_loss_fn = get_expected_l0_arm
        else:
            raise ValueError(f"Unknown l0_method: {l0_method}")
        
        # Edge scoring MLP (same architecture for both methods)
        self.edge_mlp = nn.Sequential(
            spectral_norm(nn.Linear(input_dim * 2 + 1, edge_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(edge_dim, edge_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(edge_dim, 1))
        )
        
        # Initialize with Kaiming
        for m in self.edge_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def compute_edge_weights(self, node_feat, adj_matrix, training=True, 
                            regularizer=None, **kwargs):
        """
        Compute edge weights using selected L0 method
        
        Returns:
            edge_weights: Binary or soft edge weights
            logAlpha: Edge logits (for regularization)
            extra_outputs: Dict with method-specific outputs
        """
        # Normalize node features
        node_feat = node_feat / (torch.norm(node_feat, dim=-1, keepdim=True) + 1e-8)
        
        # Get edge indices
        edge_index = adj_matrix.coalesce().indices()
        src, tgt = edge_index[0], edge_index[1]
        
        # Compute distances
        distances = torch.norm(node_feat[src] - node_feat[tgt], dim=-1)
        
        # Concatenate features: [src_feat || tgt_feat || distance]
        edge_features = torch.cat([
            node_feat[src],
            node_feat[tgt],
            distances.unsqueeze(-1)
        ], dim=-1)
        
        # Compute logAlpha (edge logits)
        logAlpha = self.edge_mlp(edge_features).squeeze(-1)
        
        # Apply L0 gating based on method
        if self.l0_method == 'hard-concrete':
            if training:
                edge_weights = self.l0_train_fn(logAlpha, params=self.l0_params,temperature=temperature)
            else:
                edge_weights = self.l0_test_fn(logAlpha, params=self.l0_params)
            
            extra_outputs = {}
            
        elif self.l0_method == 'arm':
            edge_weights, edge_weights_anti = self.arm_sample_fn(
                logAlpha, self.l0_params, training=training
            )
            
            # Store antithetic samples for ARM gradient computation
            extra_outputs = {'edge_weights_anti': edge_weights_anti}
        
        return edge_weights, logAlpha, extra_outputs
