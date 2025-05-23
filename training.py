import os
import torch
import torch.optim as optim
import gc
import numpy as np
from torch.utils.data import DataLoader

from analysis import analyze_overfitting, plot_metrics, calculate_class_weights
from training_loop import train_and_evaluate
from helper import collate
from utils.lr_scheduler import LR_Scheduler

def train_edge_gnn(dataset, train_idx, val_idx, args, output_dir):
    """Train the ImprovedEdgeGNN model with detailed analysis"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloaders
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    
    print("\n" + "="*60)
    print("🔹 Training ImprovedEdgeGNN Model with Smart Sparsification")
    print("="*60)
    
    # Calculate class weights
    class_weights = calculate_class_weights(dataset, train_idx).to(device)
    
    # Import model here to avoid circular imports
    from LENS import ImprovedEdgeGNN
    
    # Initialize ImprovedEdgeGNN model with the specified parameters
    model = ImprovedEdgeGNN(
        feature_dim=512,
        hidden_dim=512,  # Use the first hidden dim
        num_classes=args.n_class,
        lambda_reg=args.lambda_reg if hasattr(args, 'lambda_reg') else args.beta,  # Support both parameter names
        reg_mode=args.reg_mode if hasattr(args, 'reg_mode') else args.egl_mode,    # Support both parameter names
        edge_dim=args.edge_dim,
        warmup_epochs=args.warmup_epochs,
        graph_size_adaptation=args.graph_size_adaptation,
        min_edges_per_node=args.min_edges_per_node,
        dropout=args.dropout,
        # Add new L0 parameters with defaults
        l0_gamma=args.l0_gamma if hasattr(args, 'l0_gamma') else -0.1,
        l0_zeta=args.l0_zeta if hasattr(args, 'l0_zeta') else 1.1,
        l0_beta=args.l0_beta if hasattr(args, 'l0_beta') else 0.66,
        initial_temp=args.initial_temp if hasattr(args, 'initial_temp') else 5.0
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LR_Scheduler(mode='cos', base_lr=args.lr, 
                             num_epochs=args.epochs, 
                             iters_per_epoch=len(train_loader), warmup_epochs=5)
    
    # Train and evaluate the model with detailed analysis
    results = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        n_features=args.n_features,
        output_dir=output_dir,
        warmup_epochs=args.warmup_epochs
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_accuracy': results["val_accs"][-1],
        'best_val_accuracy': results["best_val_acc"],
        'best_epoch': results["best_epoch"],
    }, final_model_path)
    
    # Also save the edge weight evolution plot
    if hasattr(model, 'plot_stats'):
        model.plot_stats(save_path=os.path.join(output_dir, 'edge_weight_evolution.png'))
    
    # Plot metrics
    plot_metrics(
        results["train_accs"],
        results["val_accs"],
        results["train_losses"],
        results["val_losses"],
        os.path.join(output_dir, 'training_metrics.png'),
        title="ImprovedEdgeGNN Training Metrics",
        warmup_epochs=args.warmup_epochs
    )
    
    # Analyze overfitting
    overfitting = analyze_overfitting(
        results["train_accs"],
        results["val_accs"],
        warmup_epochs=args.warmup_epochs
    )
    
    # Save analysis results
    save_training_results(args, output_dir, results, overfitting, model)
    
    # Print summary
    print_training_summary(results, overfitting, model)
    
    # Return results for further analysis
    return {
        "results": results,
        "overfitting": overfitting
    }

def save_training_results(args, output_dir, results, overfitting, model):
    """Save training results to a file"""
    with open(os.path.join(output_dir, 'training_results.txt'), 'w') as f:
        f.write("IMPROVED EDGE GNN MODEL RESULTS\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        
        # Lambda/regularization parameter (support both naming conventions)
        lambda_value = args.lambda_reg if hasattr(args, 'lambda_reg') else args.beta
        f.write(f"Lambda (λ): {lambda_value}\n")
        
        # Regularization mode
        reg_mode = args.reg_mode if hasattr(args, 'reg_mode') else args.egl_mode
        f.write(f"Regularization Mode: {reg_mode}\n")
        
        # If using L0, print L0-specific parameters
        if reg_mode == 'l0':
            l0_gamma = args.l0_gamma if hasattr(args, 'l0_gamma') else -0.1
            l0_zeta = args.l0_zeta if hasattr(args, 'l0_zeta') else 1.1
            l0_beta = args.l0_beta if hasattr(args, 'l0_beta') else 0.66
            f.write(f"L0 Parameters: gamma={l0_gamma}, zeta={l0_zeta}, beta={l0_beta}\n")
        
        # Other parameters
        f.write(f"Edge Dimension: {args.edge_dim}\n")
        f.write(f"Hidden Dimensions: {args.hidden_dims}\n")
        f.write(f"Warmup Epochs: {args.warmup_epochs}\n")
        f.write(f"Graph Size Adaptation: {args.graph_size_adaptation}\n")
        f.write(f"Min Edges Per Node: {args.min_edges_per_node}\n")
        f.write(f"Dropout: {args.dropout}\n\n")
        
        f.write("PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        f.write(f"Best Validation Accuracy: {results['best_val_acc']:.4f} (Epoch {results['best_epoch']})\n")
        if 'best_edge_sparsity' in results:
            f.write(f"Edge Sparsity at Best Epoch: {results['best_edge_sparsity']:.1f}% edges > 0.1\n")
        
        if hasattr(model, 'edge_stats') and len(model.edge_stats) > 0:
            f.write(f"Final Avg Edge Weight: {model.edge_stats[-1]:.6f}\n")
            
        f.write(f"Overfitting Severity: {overfitting['severity']}\n")
        f.write(f"Average Train-Val Gap (post-warmup): {overfitting['avg_post_warmup_gap']:.4f}\n")
        f.write(f"Maximum Train-Val Gap: {overfitting['max_gap']:.4f} (Epoch {overfitting['max_gap_epoch']})\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        
        if overfitting['severity'] != "None":
            f.write(f"1. Model shows {overfitting['severity'].lower()} overfitting. Consider:\n")
            
            if overfitting['severity'] == "Severe":
                f.write("   - Increasing lambda (try 2-3x current value)\n")
                f.write("   - Increasing dropout (try 0.3-0.4)\n")
                f.write("   - Increasing weight decay\n")
            elif overfitting['severity'] == "Moderate":
                f.write("   - Increasing lambda by 50-100%\n")
                f.write("   - Slightly increasing dropout\n")
            else:
                f.write("   - Minor tweaks to lambda and/or dropout\n")
        else:
            f.write("1. Model shows good generalization. Consider:\n")
            f.write("   - Experimenting with different regularization modes\n")
            f.write("   - Slightly decreasing lambda to allow more edges\n")

def print_training_summary(results, overfitting, model):
    """Print a summary of the training results"""
    print("\n" + "="*60)
    print("📊 IMPROVED EDGE GNN MODEL RESULTS")
    print("="*60)
    print(f"Best Validation Accuracy: {results['best_val_acc']:.4f} (Epoch {results['best_epoch']})")
    if 'best_edge_sparsity' in results:
        print(f"Edge Sparsity at Best Epoch: {results['best_edge_sparsity']:.1f}% edges > 0.1")
    
    if hasattr(model, 'edge_stats') and len(model.edge_stats) > 0:
        print(f"Final Avg Edge Weight: {model.edge_stats[-1]:.6f}")
        
    print(f"Overfitting: {overfitting['severity']} (Gap: {overfitting['avg_post_warmup_gap']:.4f})")
