#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.dataset import GraphDataset  # Your dataset class
from LENS import ImprovedEdgeGNN  # Your model class
from helper import Evaluator, collate, preparefeatureLabel  # Your helper functions

def plot_confusion_matrix(cm, classes, output_path, title='Confusion Matrix'):
    """Plot and save confusion matrix as an image"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    plt.close()

def test_model(args):
    """
    Test a trained ImprovedEdgeGNN model on a test dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading test data from {args.test_data}")
    # Load test dataset
    with open(args.test_data, 'r') as f:
        test_ids = f.readlines()
    
    test_dataset = GraphDataset(root=args.data_root, ids=test_ids)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    
    print(f"Loading model from {args.model_path}")
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract class count from either checkpoint or args
    if 'classifier.3.weight' in checkpoint['model_state_dict']:
        # Sequential classifier with spectral norm
        n_class = checkpoint['model_state_dict']['classifier.3.weight'].shape[0]
    elif 'classifier.weight' in checkpoint['model_state_dict']:
        # Simple linear classifier
        n_class = checkpoint['model_state_dict']['classifier.weight'].shape[0]
    else:
        # Use args if can't determine from model
        n_class = args.n_class
        print(f"Couldn't determine class count from model, using provided value: {n_class}")
    
    # Create a new model instance with the same architecture
    model = ImprovedEdgeGNN(
        feature_dim=args.n_features,
        hidden_dim=args.hidden_dim,
        num_classes=n_class,
        beta=args.beta,
        egl_mode=args.egl_mode,
        edge_dim=args.edge_dim,
        warmup_epochs=args.warmup_epochs,
        graph_size_adaptation=args.graph_size_adaptation,
        min_edges_per_node=args.min_edges_per_node,
        dropout=args.dropout
    ).to(device)
    
    # Load the weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    evaluator = Evaluator(n_class=n_class)
    
    # Variables to track test performance
    edge_sparsity_metrics = {'>0.01': [], '>0.1': [], '>0.5': []}
    confusion_matrix = np.zeros((n_class, n_class), dtype=np.int64)
    all_predictions = []
    all_labels = []
    all_graph_reps = []
    
    # Set model to not print statistics during evaluation
    if hasattr(model, 'set_print_stats'):
        model.set_print_stats(False)
    
    print("\nEvaluating model on test data...")
    
    # Process each batch
    with torch.no_grad():
        for sample in test_loader:
            try:
                # Evaluate the model
                pred, labels, loss, weighted_adj = evaluator.eval_test(sample, model, n_features=args.n_features)
                all_predictions.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Get the original adjacency matrix
                node_feat, _, adjs, masks = preparefeatureLabel(
                    sample['image'], sample['label'], sample['adj_s'], n_features=args.n_features
                )
                
                # If you want to save graph representations for visualization
                if args.save_graph_reps:
                    # Extract graph representations before classification
                    # This assumes your model has a method to get graph representations
                    # You may need to modify this based on your model's architecture
                    with torch.no_grad():
                        # Compute edge weights
                        edge_weights = model.compute_edge_weights(node_feat, adjs)
                        
                        # Aggregate neighbor features
                        h = model.aggregate(node_feat, adjs, edge_weights)
                        
                        # Apply GNN layer
                        h = model.conv(h)
                        h = F.relu(h)
                        
                        # Global pooling (use the same pooling as in your model's forward method)
                        if masks is not None:
                            graph_rep = torch.zeros(adjs.shape[0], h.size(2), device=h.device)
                            for b in range(adjs.shape[0]):
                                valid_indices = torch.where(masks[b] > 0)[0]
                                if len(valid_indices) > 0:
                                    graph_rep[b] = torch.mean(h[b, valid_indices], dim=0)
                        else:
                            graph_rep = torch.mean(h, dim=1)
                        
                        all_graph_reps.append(graph_rep.cpu().numpy())
                
                # Calculate edge sparsity metrics
                edge_mask = (adjs > 0).float()
                masked_weights = torch.zeros_like(weighted_adj)
                for i in range(adjs.shape[0]):
                    mask = adjs[i] > 0
                    if mask.sum() > 0:
                        masked_weights[i, mask] = weighted_adj[i, mask] / adjs[i, mask]
                
                # Calculate sparsity percentages
                if torch.sum(edge_mask) > 0:
                    sparsity_01 = torch.sum(masked_weights > 0.01).item() / torch.sum(edge_mask).item() * 100
                    sparsity_1 = torch.sum(masked_weights > 0.1).item() / torch.sum(edge_mask).item() * 100
                    sparsity_5 = torch.sum(masked_weights > 0.5).item() / torch.sum(edge_mask).item() * 100
                    
                    edge_sparsity_metrics['>0.01'].append(sparsity_01)
                    edge_sparsity_metrics['>0.1'].append(sparsity_1)
                    edge_sparsity_metrics['>0.5'].append(sparsity_5)
                
                # Update confusion matrix
                # Update confusion matrix
                for i in range(len(pred)):
                   # Use argmax to get the predicted class if it's a multi-class prediction
                   pred_class = pred[i].argmax().cpu().item() if pred[i].dim() > 0 else pred[i].cpu().item()
                   label_class = labels[i].argmax().cpu().item() if labels[i].dim() > 0 else labels[i].cpu().item()
                   confusion_matrix[label_class, pred_class] += 1
            except RuntimeError as e:
                print(f"Error processing sample: {e}")
    
    # Calculate final metrics
    test_accuracy = evaluator.get_scores()
    
    # Calculate precision, recall, and F1 for each class
    precision = np.zeros(n_class)
    recall = np.zeros(n_class)
    f1 = np.zeros(n_class)
    
    for i in range(n_class):
        if np.sum(confusion_matrix[:, i]) > 0:
            precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
        if np.sum(confusion_matrix[i, :]) > 0:
            recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    
    # Calculate average edge sparsity
    avg_sparsity = {k: np.mean(v) if v else 0 for k, v in edge_sparsity_metrics.items()}
    
    # Calculate macro and weighted F1 scores
    macro_f1 = np.mean(f1)
    
    # Count instances per class for weighted metrics
    class_counts = np.zeros(n_class)
    for i in range(n_class):
        class_counts[i] = np.sum(confusion_matrix[i, :])
    
    weighted_f1 = np.sum(f1 * class_counts) / np.sum(class_counts)
    
    # Print test results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Average Edge Sparsity: {avg_sparsity['>0.01']:.1f}% > 0.01, {avg_sparsity['>0.1']:.1f}% > 0.1, {avg_sparsity['>0.5']:.1f}% > 0.5")
    
    print("\nPer-Class Metrics:")
    for i in range(n_class):
        print(f"Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix)
    
    # Save results to a file
    results_path = os.path.join(args.output_dir, 'test_results.txt')
    
    with open(results_path, 'w') as f:
        f.write("TEST RESULTS\n")
        f.write("="*40 + "\n\n")
        
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Data: {args.test_data}\n\n")
        
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Macro F1 Score: {macro_f1:.4f}\n")
        f.write(f"Weighted F1 Score: {weighted_f1:.4f}\n")
        f.write(f"Average Edge Sparsity: {avg_sparsity['>0.01']:.1f}% > 0.01, {avg_sparsity['>0.1']:.1f}% > 0.1, {avg_sparsity['>0.5']:.1f}% > 0.5\n\n")
        
        f.write("Per-Class Metrics:\n")
        for i in range(n_class):
            f.write(f"Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix) + "\n")
    
    # Plot and save confusion matrix
    if args.plot_cm:
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        class_names = [f"Class {i}" for i in range(n_class)]
        plot_confusion_matrix(confusion_matrix, class_names, cm_path)
        print(f"Confusion matrix plot saved to {cm_path}")
    
    # Save all graph representations if requested
    if args.save_graph_reps and all_graph_reps:
        graph_reps = np.vstack(all_graph_reps)
        graph_reps_path = os.path.join(args.output_dir, 'graph_representations.npy')
        np.save(graph_reps_path, graph_reps)
        
        # Also save labels for visualization
        labels_path = os.path.join(args.output_dir, 'graph_labels.npy')
        np.save(labels_path, np.array(all_labels))
        
        print(f"Graph representations saved to {graph_reps_path}")
    
    print(f"\nTest results saved to {results_path}")
    
    return {
        'accuracy': test_accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'sparsity': avg_sparsity,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test ImprovedEdgeGNN Model')
    
    # Required parameters
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data file')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory for dataset')
    
    # Model parameters (should match training parameters)
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
    
    # Evaluation parameters
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--output-dir', type=str, default='test_results', help='Output directory')
    parser.add_argument('--plot-cm', action='store_true', help='Plot confusion matrix')
    parser.add_argument('--save-graph-reps', action='store_true', help='Save graph representations')
    
    args = parser.parse_args()
    
    # Run testing
    test_results = test_model(args)
