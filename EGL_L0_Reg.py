import torch
import numpy as np
from L0Utils import get_loss2, L0RegularizerParams

class EGLassoRegularization:
    def __init__(self, lambda_reg, reg_mode='l0', warmup_epochs=5, l0_params=None):
        """Initialize the regularization module
        
        Args:
            lambda_reg: Base regularization strength (λ)
            reg_mode: Regularization type ('l0' or 'egl')
            warmup_epochs: Number of warmup epochs
            l0_params: Optional L0RegularizerParams instance for customization
        """
        self.base_lambda = lambda_reg
        self.current_lambda = 0.0  # Will increase during training
        self.reg_mode = reg_mode
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.logits_storage = {}  # For L0 regularization
        
        # Store L0 regularization parameters
        self.l0_params = l0_params if l0_params is not None else L0RegularizerParams()
    
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
    
    def update_lambda(self, current_epoch, warmup_epochs):
        """Update lambda regularization strength based on current epoch"""
        self.current_epoch = current_epoch
        
        # During warmup
        if current_epoch < warmup_epochs:
            self.current_lambda = (current_epoch / warmup_epochs) * self.base_lambda * 0.1
        else:
            # After warmup
            post_warmup_epochs = current_epoch - warmup_epochs
            # Start with 10% of base_lambda and gradually increase over 20 epochs
            min_lambda = self.base_lambda * 0.1
            max_lambda = self.base_lambda
            
            if post_warmup_epochs < 20:
                # Linear increase over 20 epochs
                lambda_factor = min_lambda + (max_lambda - min_lambda) * (post_warmup_epochs / 20)
            else:
                # Plateau at max value
                lambda_factor = max_lambda
            
            self.current_lambda = lambda_factor
    
    def compute_regularization(self, edge_weights, adj_matrix):
        """Compute regularization loss based on selected mode
        
        Args:
            edge_weights: Edge weights tensor [batch_size, num_nodes, num_nodes]
            adj_matrix: Original adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Regularization loss (scalar)
        """
        if self.current_lambda == 0.0 or not edge_weights.requires_grad:
            return 0.0
        
        batch_size = adj_matrix.shape[0]
        
        # L0 regularization
        if self.reg_mode == 'l0':
            # L0 regularization based on stored logits
            reg_loss = 0.0
            
            for b in self.logits_storage:
                logits = self.logits_storage[b]
                # Calculate L0 loss with custom parameters
                l0_loss = get_loss2(logits, params=self.l0_params).sum()
                reg_loss += l0_loss
            
            if len(self.logits_storage) > 0:
                reg_loss = reg_loss / len(self.logits_storage)
        
        # Exclusive Group Lasso
        elif self.reg_mode == 'egl':
            # Create mask for existing edges
            edge_mask = (adj_matrix > 0).float()
            
            # Exclusive Group Lasso - group by nodes
            reg_loss = 0.0
            
            # Sum weights per node (for each source node)
            source_sum = torch.sum(edge_weights * edge_mask, dim=2)  # [batch_size, num_nodes]
            source_reg = torch.sum(source_sum**2)
            
            # Sum weights per node (for each target node)
            target_sum = torch.sum(edge_weights * edge_mask, dim=1)  # [batch_size, num_nodes]
            target_reg = torch.sum(target_sum**2)
            
            # Average between source and target node regularization
            reg_loss = (source_reg + target_reg) / (2 * batch_size)
        
        else:
            raise ValueError(f"Unsupported regularization mode: {self.reg_mode}")
        
        # Return combined loss
        return self.current_lambda * reg_loss
