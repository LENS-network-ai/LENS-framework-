# helper.py
"""
Helper functions for training and evaluation with ARM support
"""

import torch
from utils.metrics import ConfusionMatrix


def collate(batch):
    """Collate function for DataLoader"""
    image = [b['image'] for b in batch]
    label = [b['label'] for b in batch]
    id = [b['id'] for b in batch]
    adj_s = [b['adj_s'] for b in batch]
    return {'image': image, 'label': label, 'id': id, 'adj_s': adj_s}


def preparefeatureLabel(batch_graph, batch_label, batch_adjs, n_features: int = 512):
    """
    Prepare batched graph features and labels
    
    Args:
        batch_graph: List of node features per graph
        batch_label: List of labels
        batch_adjs: List of adjacency matrices
        n_features: Feature dimension
    
    Returns:
        node_feat: Batched node features [B, max_N, D]
        labels: Batched labels [B]
        adjs: Batched adjacency matrices [B, max_N, max_N]
        masks: Node masks [B, max_N]
    """
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
    max_node_num = 0
    
    # Find max number of nodes
    for i in range(batch_size):
        labels[i] = batch_label[i]
        max_node_num = max(max_node_num, batch_graph[i].shape[0])
    
    # Initialize tensors
    masks = torch.zeros(batch_size, max_node_num)
    adjs = torch.zeros(batch_size, max_node_num, max_node_num)
    batch_node_feat = torch.zeros(batch_size, max_node_num, n_features)
    
    # Fill in actual data
    for i in range(batch_size):
        cur_node_num = batch_graph[i].shape[0]
        batch_node_feat[i, 0:cur_node_num] = batch_graph[i]
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]
        masks[i, 0:cur_node_num] = 1
    
    # Move to GPU
    node_feat = batch_node_feat.cuda()
    labels = labels.cuda()
    adjs = adjs.cuda()
    masks = masks.cuda()
    
    return node_feat, labels, adjs, masks


class Trainer(object):
    """
    Trainer class with support for both Hard-Concrete and ARM
    """
    
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)
        self.saved_pruned_adjs = {}   # Store pruned adjacencies by WSI ID
        self.original_edges = {}      # Store original edge counts
        
    def get_scores(self):
        """Get current accuracy"""
        return self.metrics.get_scores()
    
    def reset_metrics(self):
        """Reset metrics for new epoch"""
        self.metrics.reset()
    
    def update_metrics(self, pred, labels):
        """
        Update metrics manually (for ARM training)
        
        Args:
            pred: Predictions [B, num_classes] or [B]
            labels: Ground truth labels [B]
        """
        # Detach tensors
        pred_detached = pred.detach()
        labels_detached = labels.detach()
        
        # Get predicted class
        if pred_detached.ndim == 1:
            pred_class = pred_detached
        else:
            pred_class = torch.argmax(pred_detached, dim=1)
        
        # Convert to numpy
        pred_numpy = pred_class.cpu().numpy().reshape(-1)
        label_numpy = labels_detached.cpu().numpy().reshape(-1)
        
        # Update metrics
        self.metrics.update(pred_numpy, label_numpy)
    
    def train(self, sample, model, n_features: int = 512):
        """
        Standard training step (Hard-Concrete)
        
        For ARM training, use train_arm() instead
        
        Args:
            sample: Data sample dict
            model: LENS model
            n_features: Feature dimension
        
        Returns:
            pred: Predictions
            labels: Ground truth labels
            loss: Total loss
            pruned_adj: Pruned adjacency matrix
        """
        # Prepare data
        node_feat, labels, adjs, masks = preparefeatureLabel(
            sample['image'], sample['label'], sample['adj_s'], n_features=n_features
        )
        
        # Forward pass (Hard-Concrete returns 4 values)
        model_output = model.forward(node_feat, labels, adjs, masks)
        
        # Handle different return formats
        if len(model_output) == 4:
            # Hard-Concrete: (pred, labels, loss, pruned_adj)
            pred, labels, loss, pruned_adj = model_output
        elif len(model_output) == 5:
            # ARM with return_edge_weights_anti=True: (pred, labels, loss, pruned_adj, edge_weights_anti)
            # This shouldn't happen in standard train() call, but handle it gracefully
            pred, labels, loss, pruned_adj, _ = model_output
        else:
            raise ValueError(f"Unexpected number of return values from model: {len(model_output)}")
        
        # Save pruned adjacency
        wsi_id = sample['id'][0]
        self.saved_pruned_adjs[wsi_id] = pruned_adj.cpu().detach()
        self.original_edges[wsi_id] = (adjs > 0).sum().item()
        
        # Update metrics
        self.update_metrics(pred, labels)
        
        return pred, labels, loss, pruned_adj
    
    def train_arm(self, sample, model, n_features: int = 512):
        """
        ARM-specific training step (with antithetic samples)
        
        Args:
            sample: Data sample dict
            model: LENS model
            n_features: Feature dimension
        
        Returns:
            pred: Predictions with sampled gates
            labels: Ground truth labels
            loss_b: Loss with sampled gates
            pruned_adj: Pruned adjacency with sampled gates
            edge_weights_anti: Antithetic edge weights
            node_feat: Node features (for antithetic forward pass)
            adjs: Adjacency matrices (for antithetic forward pass)
            masks: Node masks (for antithetic forward pass)
        """
        # Prepare data
        node_feat, labels, adjs, masks = preparefeatureLabel(
            sample['image'], sample['label'], sample['adj_s'], n_features=n_features
        )
        
        # Forward pass with ARM (returns 5 values)
        pred, labels_out, loss_b, pruned_adj, edge_weights_anti = model.forward(
            node_feat, labels, adjs, masks,
            return_edge_weights_anti=True
        )
        
        # Save pruned adjacency
        wsi_id = sample['id'][0]
        self.saved_pruned_adjs[wsi_id] = pruned_adj.cpu().detach()
        self.original_edges[wsi_id] = (adjs > 0).sum().item()
        
        # Update metrics
        self.update_metrics(pred, labels_out)
        
        # Return everything needed for ARM gradient computation
        return pred, labels_out, loss_b, pruned_adj, edge_weights_anti, node_feat, adjs, masks


class Evaluator(object):
    """
    Evaluator class for validation/testing
    """
    
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)
    
    def get_scores(self):
        """Get current accuracy"""
        return self.metrics.get_scores()
    
    def reset_metrics(self):
        """Reset metrics for new epoch"""
        self.metrics.reset()
    
    def update_metrics(self, pred, labels):
        """
        Update metrics manually
        
        Args:
            pred: Predictions [B, num_classes] or [B]
            labels: Ground truth labels [B]
        """
        # Get predicted class
        if pred.ndim == 1:
            pred_class = pred
        else:
            pred_class = torch.argmax(pred, dim=1)
        
        # Convert to numpy
        pred_numpy = pred_class.cpu().numpy().reshape(-1)
        label_numpy = labels.cpu().numpy().reshape(-1)
        
        # Update metrics
        self.metrics.update(pred_numpy, label_numpy)
    
    def eval_test(self, sample, model, graphcam_flag=False, n_features: int = 512):
        """
        Evaluation step (works for both Hard-Concrete and ARM)
        
        Args:
            sample: Data sample dict
            model: LENS model
            graphcam_flag: Whether to use GraphCAM (unused)
            n_features: Feature dimension
        
        Returns:
            pred: Predictions
            labels: Ground truth labels
            loss: Loss
            pruned_adj: Pruned adjacency matrix
        """
        # Prepare data
        node_feat, labels, adjs, masks = preparefeatureLabel(
            sample['image'], sample['label'], sample['adj_s'], n_features=n_features
        )
        
        # Forward pass (evaluation mode - no antithetic sampling)
        with torch.no_grad():
            model_output = model.forward(node_feat, labels, adjs, masks)
        
        # Handle different return formats
        if len(model_output) == 4:
            # Standard: (pred, labels, loss, pruned_adj)
            pred, labels, loss, pruned_adj = model_output
        elif len(model_output) == 5:
            # ARM might return 5 even in eval mode, but edge_weights_anti should be None
            pred, labels, loss, pruned_adj, _ = model_output
        else:
            raise ValueError(f"Unexpected number of return values from model: {len(model_output)}")
        
        # Update metrics
        self.update_metrics(pred, labels)
        
        return pred, labels, loss, pruned_adj
