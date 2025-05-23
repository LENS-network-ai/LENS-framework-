#!/usr/bin/env python
# coding: utf-8

import os
import torch
import argparse
from LENS import ImprovedEdgeGNN

def count_parameters(model):
    """
    Count total parameters and trainable parameters in a model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params
    }

def print_model_size_and_layers(model):
    """
    Print detailed layer-wise parameter counts and size of the model
    """
    print("\nModel Architecture:")
    print("=" * 80)
    
    # Print layer-wise parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters, Shape: {list(param.shape)}")
    
    # Count parameters
    param_counts = count_parameters(model)
    
    print("\nParameter Summary:")
    print("-" * 80)
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    
    # Calculate model size
    model_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    print(f"Model size: {model_size_mb:.2f} MB")
    
    return param_counts

def analyze_checkpoint(checkpoint_path, n_features=512, hidden_dim=512, n_class=3, 
                     edge_dim=64, beta=0.00014, egl_mode='egl', warmup_epochs=20,
                     graph_size_adaptation=False, min_edges_per_node=2.0, dropout=0.2):
    """
    Analyze model parameters from checkpoint
    """
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Try to determine parameters from checkpoint
    if 'classifier.3.weight' in checkpoint['model_state_dict']:
        # Sequential classifier with spectral norm
        n_class = checkpoint['model_state_dict']['classifier.3.weight'].shape[0]
    elif 'classifier.weight' in checkpoint['model_state_dict']:
        # Simple linear classifier
        n_class = checkpoint['model_state_dict']['classifier.weight'].shape[0]
    
    # Try to determine feature dim from checkpoint
    if 'conv.transform.weight' in checkpoint['model_state_dict']:
        n_features = checkpoint['model_state_dict']['conv.transform.weight'].shape[1]
        hidden_dim = checkpoint['model_state_dict']['conv.transform.weight'].shape[0]
    
    print(f"Detected model configuration:")
    print(f"- Number of features: {n_features}")
    print(f"- Hidden dimension: {hidden_dim}")
    print(f"- Number of classes: {n_class}")
    print(f"- Edge dimension: {edge_dim}")
    print(f"- EGL mode: {egl_mode}")
    
    # Create model
    model = ImprovedEdgeGNN(
        feature_dim=n_features,
        hidden_dim=hidden_dim,
        num_classes=n_class,
        beta=beta,
        egl_mode=egl_mode,
        edge_dim=edge_dim,
        warmup_epochs=warmup_epochs,
        graph_size_adaptation=graph_size_adaptation,
        min_edges_per_node=min_edges_per_node,
        dropout=dropout
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print model info
    param_counts = print_model_size_and_layers(model)
    
    # Print additional info from checkpoint if available
    if 'epoch' in checkpoint:
        print(f"\nCheckpoint epoch: {checkpoint['epoch']}")
    
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return model, param_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze ImprovedEdgeGNN Model Parameters')
    
    # Required parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    # Optional model parameters
    parser.add_argument('--n-features', type=int, default=512, help='Number of node features')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--n-class', type=int, default=3, help='Number of classes')
    parser.add_argument('--edge-dim', type=int, default=64, help='Edge dimension')
    parser.add_argument('--beta', type=float, default=0.00014, help='Regularization strength')
    parser.add_argument('--egl-mode', type=str, default='egl', 
                        choices=['none', 'egl', 'l1', 'l2', 'entropy'], help='EGL mode')
    parser.add_argument('--warmup-epochs', type=int, default=20, help='Warmup epochs')
    parser.add_argument('--graph-size-adaptation', action='store_true', help='Enable graph size adaptation')
    parser.add_argument('--min-edges-per-node', type=float, default=2.0, help='Minimum edges per node')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Analyze the model
    model, params = analyze_checkpoint(
        args.checkpoint,
        n_features=args.n_features,
        hidden_dim=args.hidden_dim,
        n_class=args.n_class,
        edge_dim=args.edge_dim,
        beta=args.beta,
        egl_mode=args.egl_mode,
        warmup_epochs=args.warmup_epochs,
        graph_size_adaptation=args.graph_size_adaptation,
        min_edges_per_node=args.min_edges_per_node,
        dropout=args.dropout
    )
