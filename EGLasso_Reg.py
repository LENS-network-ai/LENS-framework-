import torch
import numpy as np
from L0Utils import get_loss2

class EGLassoRegularization:
    def __init__(self, beta, egl_mode, warmup_epochs):
        """Initialize the regularization module
        
        Args:
            beta: Base regularization strength
            egl_mode: Regularization type ('egl', 'l1', 'l2', 'entropy', 'l0', 'none')
            warmup_epochs: Number of warmup epochs
        """
        self.base_beta = beta
        self.current_beta = 0.0  # Will increase during training
        self.egl_mode = egl_mode
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.logits_storage = {}  # For L0 regularization
    
    def clear_logits(self):
        """Clear stored logits for L0 regularization"""
        self.logits_storage = {}
    
    def store_logits(self, batch_idx, logits):
        """Store logits for a batch for L0 regularization"""
        self.logits_storage[batch_idx] = logits
    
    def update_temperature(self, current_epoch, warmup_epochs, initial_temp):
        """Update temperature based on current epoch and return the new value"""
        # Temperature annealing
        if current_epoch < warmup_epochs:
            # During warmup
            temperature = initial_temp
        else:
            # After warmup
            progress = (current_epoch - warmup_epochs) / (20 - warmup_epochs)
            progress = min(1.0, max(0.0, progress))
            temperature = max(1.0, initial_temp * (0.1 + 0.9 * (np.cos(progress * np.pi) + 1) / 2))
        
        return temperature
    
    def update_beta(self, current_epoch, warmup_epochs):
        """Update beta regularization strength based on current epoch"""
        self.current_epoch = current_epoch
        
        # During warmup
        if current_epoch < warmup_epochs:
            self.current_beta = (current_epoch / warmup_epochs) * self.base_beta * 0.1
        else:
            # After warmup
            post_warmup_epochs = current_epoch - warmup_epochs
            # Start with 10% of base_beta and gradually increase over 20 epochs
            min_beta = self.base_beta * 0.1
            max_beta = self.base_beta
            
            if post_warmup_epochs < 20:
                # Linear increase over 20 epochs
                beta_factor = min_beta + (max_beta - min_beta) * (post_warmup_epochs / 20)
            else:
                # Plateau at max value
                beta_factor = max_beta
            
            self.current_beta = beta_factor
    
    def compute_regularization(self, edge_weights, adj_matrix):
        """Compute regularization loss based on selected mode with weight stabilization
        
        Args:
            edge_weights: Edge weights tensor [batch_size, num_nodes, num_nodes]
            adj_matrix: Original adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Regularization loss (scalar)
        """
        if self.egl_mode == 'none' or self.current_beta == 0.0 or not edge_weights.requires_grad:
            return 0.0
        
        # Create mask for existing edges
        edge_mask = (adj_matrix > 0).float()
        batch_size = adj_matrix.shape[0]
        
        # L0 regularization (new addition)
        if self.egl_mode == 'l0':
            # L0 regularization based on stored logits
            reg_loss = 0.0
            
            for b in self.logits_storage:
                logits = self.logits_storage[b]
                # Calculate L0 loss
                l0_loss = get_loss2(logits).sum()
                reg_loss += l0_loss
            
            if len(self.logits_storage) > 0:
                reg_loss = reg_loss / len(self.logits_storage)
            
            return self.current_beta * reg_loss
            
        # Different regularization types
        elif self.egl_mode == 'egl':
            # Exclusive Group Lasso - group by nodes
            # For each node, calculate sum of its edge weights, then square
            reg_loss = 0.0
            
            # Sum weights per node (for each source node)
            source_sum = torch.sum(edge_weights * edge_mask, dim=2)  # [batch_size, num_nodes]
            source_reg = torch.sum(source_sum**2)
            
            # Sum weights per node (for each target node)
            target_sum = torch.sum(edge_weights * edge_mask, dim=1)  # [batch_size, num_nodes]
            target_reg = torch.sum(target_sum**2)
            
            # Average between source and target node regularization
            reg_loss = (source_reg + target_reg) / (2 * batch_size)
            
        elif self.egl_mode == 'l1':
            # L1 regularization - pushes weights toward 0
            reg_loss = torch.sum(torch.abs(edge_weights) * edge_mask) / batch_size
            
        elif self.egl_mode == 'l2':
            # L2 regularization - pushes weights toward 0 but gentler
            reg_loss = torch.sum((edge_weights**2) * edge_mask) / batch_size
            
        elif self.egl_mode == 'entropy':
            # Entropy - pushes weights toward 0.5
            masked_weights = edge_weights * edge_mask
            # Avoid log(0) issues with small epsilon
            entropy = -(masked_weights * torch.log(masked_weights + 1e-8) + 
                      (1-masked_weights) * torch.log(1-masked_weights + 1e-8))
            
            # Mask to only include existing edges
            entropy = entropy * edge_mask
            
            # Negate entropy (we're minimizing, and want to maximize entropy)
            reg_loss = -torch.sum(entropy) / batch_size
            
        else:
            raise ValueError(f"Unknown regularization mode: {self.egl_mode}")
        
        # Return combined loss
        return self.current_beta * reg_loss
