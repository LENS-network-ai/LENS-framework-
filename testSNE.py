#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# Import your existing modules
from utils.dataset import GraphDataset
from LENS import ImprovedEdgeGNN
from helper import Evaluator, collate, preparefeatureLabel

class GraphRepExtractor:
    """
    Custom extractor specifically designed for the ImprovedEdgeGNN architecture
    that directly accesses the graph representation after pooling
    """
    def __init__(self, model):
        self.model = model
        self.graph_reps = []
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks at the key points in the ImprovedEdgeGNN architecture"""
        
        def classifier_input_hook(module, input, output):
            """Capture the input to the classifier (graph representation)"""
            # Store the input to the classifier (graph representation)
            if isinstance(input, tuple) and len(input) > 0:
                self.graph_reps.append(input[0].detach().cpu())
        
        # Register hook on the classifier to capture its input
        # This is specific to the ImprovedEdgeGNN model structure
        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, torch.nn.Sequential):
            # Register hook on the first layer of the classifier
            handle = self.model.classifier[0].register_forward_hook(
                lambda module, input, output: self.graph_reps.append(input[0].detach().cpu())
            )
            self.hook_handles.append(handle)
            print("Registered hook on classifier[0] to capture graph representation")
        else:
            print("WARNING: Could not find sequential classifier in model")
            # Try to find any suitable module
            for name, module in self.model.named_modules():
                if 'classifier' in name:
                    handle = module.register_forward_hook(
                        lambda module, input, output: self.graph_reps.append(input[0].detach().cpu())
                    )
                    self.hook_handles.append(handle)
                    print(f"Registered hook on {name}")
                    break
    
    def clear_reps(self):
        """Clear stored graph representations"""
        self.graph_reps = []
    
    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

def extract_graph_representations(args):
    """
    Extract graph representations directly from the ImprovedEdgeGNN model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]
    
    print(f"Number of test IDs: {len(test_ids)}")
    
    test_dataset = GraphDataset(root=args.data_root, ids=test_ids)
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f"Test loader size: {len(test_loader)}")
    
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Determine number of classes from checkpoint
    if 'classifier.3.weight' in checkpoint['model_state_dict']:
        n_class = checkpoint['model_state_dict']['classifier.3.weight'].shape[0]
    elif 'classifier.weight' in checkpoint['model_state_dict']:
        n_class = checkpoint['model_state_dict']['classifier.weight'].shape[0]
    else:
        n_class = args.n_class
    
    # Create model
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Disable print stats for cleaner output
    if hasattr(model, 'set_print_stats'):
        model.set_print_stats(False)
    
    # Create evaluator
    evaluator = Evaluator(n_class=n_class)
    
    # Create custom graph representation extractor
    graph_rep_extractor = GraphRepExtractor(model)
    
    # List to hold all graph representations and labels
    all_graph_reps = []
    all_labels = []
    all_prob_scores = []
    
    # Process each batch
    print("\nExtracting graph representations...")
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(test_loader, desc="Processing samples")):
            # Clear previous representations
            graph_rep_extractor.clear_reps()
            
            # Use evaluator to get predictions and labels
            # This will automatically trigger the hooks to capture graph representations
            pred, labels, _, _ = evaluator.eval_test(sample, model, n_features=args.n_features)
            
            # Report hook capture status for this batch
            if graph_rep_extractor.graph_reps:
                print(f"  Batch {batch_idx+1}: Captured {len(graph_rep_extractor.graph_reps)} graph representations")
                for i, rep in enumerate(graph_rep_extractor.graph_reps):
                    print(f"    Rep {i} shape: {rep.shape}")
                    all_graph_reps.append(rep)
            else:
                print(f"  Batch {batch_idx+1}: No graph representations captured")
            
            # Store labels and probabilities
            all_labels.extend(labels.cpu().numpy())
            all_prob_scores.extend(pred.cpu().numpy())
    
    # Remove hooks
    graph_rep_extractor.remove_hooks()
    
    # Check if we captured any representations
    if not all_graph_reps:
        print("ERROR: No graph representations were captured. Visualization is not possible.")
        return None, None, None, n_class
    
    # Process the graph representations
    print("\nProcessing graph representations...")
    processed_reps = []
    
    # Check each representation and reshape if needed
    for rep in all_graph_reps:
        # Print the shape for debugging
        print(f"  Original shape: {rep.shape}")
        
        # Handle different dimension cases
        if len(rep.shape) == 1:
            # Already a vector, use as is
            processed_reps.append(rep)
        elif len(rep.shape) == 2:
            # 2D tensor - could be [batch_size, features] or [nodes, features]
            # If first dimension is 1, it's likely [batch_size=1, features]
            if rep.shape[0] == 1:
                processed_reps.append(rep.squeeze(0))  # Remove batch dimension
            else:
                # Otherwise, average across first dimension (nodes)
                processed_reps.append(torch.mean(rep, dim=0))
        elif len(rep.shape) == 3:
            # 3D tensor - likely [batch_size, nodes, features]
            # Average across nodes dimension, then remove batch dimension
            avg_rep = torch.mean(rep, dim=1)  # Result: [batch_size, features]
            if avg_rep.shape[0] == 1:
                processed_reps.append(avg_rep.squeeze(0))  # Remove batch dimension
            else:
                # If somehow we have multiple batches, add each separately
                for i in range(avg_rep.shape[0]):
                    processed_reps.append(avg_rep[i])
        else:
            # Higher dimensions - flatten to 1D
            processed_reps.append(rep.reshape(-1))
    
    # Convert to numpy arrays
    processed_reps_np = torch.stack(processed_reps).numpy()
    labels_np = np.array(all_labels)
    prob_scores_np = np.array(all_prob_scores)
    
    print(f"\nFinal processed representation shape: {processed_reps_np.shape}")
    print(f"Labels shape: {labels_np.shape}")
    print(f"Probability scores shape: {prob_scores_np.shape}")
    
    # Convert one-hot labels to indices if needed
    if len(labels_np.shape) > 1 and labels_np.shape[1] > 1:
        labels_np = np.argmax(labels_np, axis=1)
    
    return processed_reps_np, labels_np, prob_scores_np, n_class

def plot_pca(embeddings, labels, n_classes, output_path):
    """Plot PCA visualization of graph representations"""
    print("\nComputing PCA projection...")
    
    # Ensure embeddings are 2D (samples, features)
    if len(embeddings.shape) > 2:
        print(f"  Reshaping embeddings from {embeddings.shape} to 2D")
        num_samples = embeddings.shape[0]
        embeddings = embeddings.reshape(num_samples, -1)
        print(f"  New shape: {embeddings.shape}")
    
    # Check if we have enough dimensions for PCA
    n_components = min(2, embeddings.shape[1])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Generate colors for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    
    # Create a scatter plot for each class
    for i in range(n_classes):
        class_mask = labels == i
        if np.any(class_mask):
            # For 2D PCA
            if embeddings_2d.shape[1] == 2:
                plt.scatter(
                    embeddings_2d[class_mask, 0], 
                    embeddings_2d[class_mask, 1],
                    c=[colors[i]],
                    label=f'Class {i}',
                    alpha=0.7,
                    edgecolors='w',
                    s=80
                )
            # For 1D PCA (add zero as Y coordinate)
            else:
                plt.scatter(
                    embeddings_2d[class_mask], 
                    np.zeros(np.sum(class_mask)),
                    c=[colors[i]],
                    label=f'Class {i}',
                    alpha=0.7,
                    edgecolors='w',
                    s=80
                )
    
    # Calculate variance explained
    var_exp = pca.explained_variance_ratio_
    
    # Add title and labels
    plt.title("PCA Visualization of Graph Representations", fontsize=16)
    if embeddings_2d.shape[1] == 2:
        plt.xlabel(f"PC1 ({var_exp[0]:.1%} variance explained)", fontsize=14)
        plt.ylabel(f"PC2 ({var_exp[1]:.1%} variance explained)", fontsize=14)
    else:
        plt.xlabel(f"PC1 ({var_exp[0]:.1%} variance explained)", fontsize=14)
        plt.ylabel("No 2nd component", fontsize=14)
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"PCA visualization saved to {output_path}")

def plot_heatmap(embeddings, labels, output_path):
    """Plot a heatmap of the embedding features"""
    print("\nGenerating embedding feature heatmap...")
    
    # Ensure embeddings are 2D (samples, features)
    if len(embeddings.shape) > 2:
        print(f"  Reshaping embeddings from {embeddings.shape} to 2D")
        num_samples = embeddings.shape[0]
        embeddings = embeddings.reshape(num_samples, -1)
        print(f"  New shape: {embeddings.shape}")
    
    # Sort samples by class for better visualization
    sort_indices = np.argsort(labels)
    sorted_embeddings = embeddings[sort_indices]
    sorted_labels = labels[sort_indices]
    
    # Create class boundaries for visualization
    unique_labels = np.unique(sorted_labels)
    boundaries = [np.where(sorted_labels == label)[0][0] for label in unique_labels]
    boundaries.append(len(sorted_labels))
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Generate the heatmap
    sns.heatmap(sorted_embeddings, cmap='viridis', cbar_kws={'label': 'Feature Value'})
    
    # Add horizontal lines to mark class boundaries
    for boundary in boundaries[:-1]:
        plt.axhline(y=boundary, color='r', linestyle='-', linewidth=2)
    
    # Add class labels
    for i, label in enumerate(unique_labels):
        mid_point = (boundaries[i] + boundaries[i+1]) // 2
        plt.text(-0.05 * embeddings.shape[1], mid_point, f'Class {label}', 
                 fontsize=12, va='center', ha='right', weight='bold')
    
    plt.title("Embedding Feature Heatmap by Class", fontsize=16)
    plt.xlabel("Feature Dimension", fontsize=14)
    plt.ylabel("Samples (sorted by class)", fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Embedding heatmap saved to {output_path}")

def plot_confidence_distribution(prob_scores, labels, n_classes, output_path):
    """Plot the confidence distribution for each class"""
    print("\nGenerating confidence distribution plot...")
    
    # Calculate confidence scores (max probability)
    confidence_scores = np.max(prob_scores, axis=1)
    predictions = np.argmax(prob_scores, axis=1)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot confidence histograms for each class
    for i in range(n_classes):
        class_mask = labels == i
        
        # Calculate true positives and false negatives
        true_positives = (class_mask & (predictions == i))
        false_negatives = (class_mask & (predictions != i))
        
        # Plot true positives
        if np.any(true_positives):
            plt.hist(confidence_scores[true_positives], alpha=0.7, bins=20,
                     label=f'Class {i} (Correct)', color=plt.cm.rainbow(i/n_classes),
                     range=(0, 1))
        
        # Plot false negatives (misclassified samples of this class)
        if np.any(false_negatives):
            plt.hist(confidence_scores[false_negatives], alpha=0.4, bins=20,
                     label=f'Class {i} (Misclassified)', color=plt.cm.rainbow(i/n_classes),
                     hatch='///', range=(0, 1))
    
    plt.title("Confidence Score Distribution by Class", fontsize=16)
    plt.xlabel("Confidence Score (max probability)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Confidence distribution plot saved to {output_path}")

def plot_confusion_matrix(prob_scores, labels, n_classes, output_path):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    print("\nGenerating confusion matrix...")
    
    # Get predictions
    predictions = np.argmax(prob_scores, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=[f"Class {i}" for i in range(n_classes)],
                yticklabels=[f"Class {i}" for i in range(n_classes)])
    
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")

def plot_feature_importance(embeddings, labels, n_classes, output_path):
    """Plot feature importance based on class separability"""
    print("\nComputing feature importance...")
    
    # Ensure embeddings are 2D (samples, features)
    if len(embeddings.shape) > 2:
        print(f"  Reshaping embeddings from {embeddings.shape} to 2D")
        num_samples = embeddings.shape[0]
        embeddings = embeddings.reshape(num_samples, -1)
        print(f"  New shape: {embeddings.shape}")
    
    # Number of features
    n_features = embeddings.shape[1]
    
    # Calculate feature importance scores
    importance_scores = np.zeros(n_features)
    
    for feature_idx in range(n_features):
        # Calculate class means for this feature
        class_means = np.array([
            np.mean(embeddings[labels == i, feature_idx]) 
            for i in range(n_classes)
        ])
        
        # Calculate class standard deviations for this feature
        class_stds = np.array([
            np.std(embeddings[labels == i, feature_idx]) 
            for i in range(n_classes)
        ])
        
        # Calculate separability score (similar to F-statistic)
        # Higher variance between classes, lower variance within classes = better feature
        between_class_var = np.var(class_means)
        within_class_var = np.mean(class_stds**2)
        
        # Avoid division by zero
        if within_class_var > 0:
            importance_scores[feature_idx] = between_class_var / within_class_var
        else:
            importance_scores[feature_idx] = 0
    
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_scores = importance_scores[sorted_indices]
    
    # Select top 30 features (or less if fewer features)
    top_n = min(30, n_features)
    top_indices = sorted_indices[:top_n]
    top_scores = sorted_scores[:top_n]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot feature importance
    bars = plt.bar(range(top_n), top_scores, color='skyblue')
    
    # Highlight top 5 features
    for i in range(min(5, top_n)):
        bars[i].set_color('navy')
    
    plt.title("Top Feature Importance for Class Separability", fontsize=16)
    plt.xlabel("Feature Index", fontsize=14)
    plt.ylabel("Importance Score (Between/Within Variance Ratio)", fontsize=14)
    plt.xticks(range(top_n), top_indices, rotation=90 if top_n > 10 else 0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")
    
    return top_indices, top_scores

def plot_class_separation(embeddings, labels, feature_indices, n_classes, output_path):
    """Plot how the top features separate the classes"""
    print("\nGenerating class separation plots...")
    
    # Ensure embeddings are 2D (samples, features)
    if len(embeddings.shape) > 2:
        print(f"  Reshaping embeddings from {embeddings.shape} to 2D")
        num_samples = embeddings.shape[0]
        embeddings = embeddings.reshape(num_samples, -1)
        print(f"  New shape: {embeddings.shape}")
    
    # Select top 2 features for visualization
    if len(feature_indices) < 2:
        print("Not enough features for separation plot")
        return
    
    # Get the top 2 features
    feature1 = feature_indices[0]
    feature2 = feature_indices[1]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot each class with different color
    for i in range(n_classes):
        class_mask = labels == i
        plt.scatter(
            embeddings[class_mask, feature1],
            embeddings[class_mask, feature2],
            c=[plt.cm.rainbow(i/n_classes)],
            label=f'Class {i}',
            alpha=0.7,
            edgecolors='w',
            s=70
        )
    
    plt.title(f"Class Separation using Top 2 Features", fontsize=16)
    plt.xlabel(f"Feature {feature1}", fontsize=14)
    plt.ylabel(f"Feature {feature2}", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Class separation plot saved to {output_path}")

def generate_model_report(embeddings, labels, prob_scores, n_classes, output_dir):
    """Generate a comprehensive model performance report"""
    print("\nGenerating model performance report...")
    
    # Ensure embeddings are 2D (samples, features)
    if len(embeddings.shape) > 2:
        print(f"  Reshaping embeddings from {embeddings.shape} to 2D")
        num_samples = embeddings.shape[0]
        embeddings = embeddings.reshape(num_samples, -1)
        print(f"  New shape: {embeddings.shape}")
    
    # Calculate overall accuracy
    predictions = np.argmax(prob_scores, axis=1)
    accuracy = np.mean(predictions == labels)
    
    # Calculate per-class metrics
    class_metrics = {}
    for i in range(n_classes):
        class_mask = labels == i
        class_pred = predictions == i
        
        # True positives, false positives, false negatives
        tp = np.sum(class_mask & class_pred)
        fp = np.sum(~class_mask & class_pred)
        fn = np.sum(class_mask & ~class_pred)
        
        # Precision, recall, F1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        
        class_metrics[i] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': np.sum(class_mask)
        }
    
    # Calculate average confidence for correct and incorrect predictions
    correct_mask = predictions == labels
    avg_confidence_correct = np.mean(np.max(prob_scores[correct_mask], axis=1))
    avg_confidence_incorrect = np.mean(np.max(prob_scores[~correct_mask], axis=1)) if np.any(~correct_mask) else 0
    
    # Create the report
    report_path = os.path.join(output_dir, 'model_performance_report.txt')
    with open(report_path, 'w') as f:
        f.write("MODEL PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Number of samples: {len(labels)}\n")
        f.write(f"Number of classes: {n_classes}\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Average confidence (correct predictions): {avg_confidence_correct:.4f}\n")
        if np.any(~correct_mask):
            f.write(f"Average confidence (incorrect predictions): {avg_confidence_incorrect:.4f}\n")
        f.write("\n")
        
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Class':<10}{'Precision':<15}{'Recall':<15}{'F1':<15}{'Support':<15}\n")
        
        weighted_f1 = 0
        total_samples = len(labels)
        
        for class_idx, metrics in class_metrics.items():
            f.write(f"{class_idx:<10}{metrics['precision']:<15.4f}{metrics['recall']:<15.4f}" +
                   f"{metrics['f1']:<15.4f}{metrics['support']:<15}\n")
            weighted_f1 += metrics['f1'] * metrics['support'] / total_samples
        
        f.write("\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        
        # Class distribution
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 30 + "\n")
        for i in range(n_classes):
            count = np.sum(labels == i)
            percentage = 100 * count / len(labels)
            f.write(f"Class {i}: {count} samples ({percentage:.1f}%)\n")
        
        f.write("\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 30 + "\n")
        
        # Simple text-based confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true_label, pred_label in zip(labels, predictions):
            cm[true_label, pred_label] += 1
        
        # Header row
        f.write(f"{'True\\Pred':<10}")
        for i in range(n_classes):
            f.write(f"{i:<10}")
        f.write("\n")
        
        # Content rows
        for i in range(n_classes):
            f.write(f"{i:<10}")
            for j in range(n_classes):
                f.write(f"{cm[i, j]:<10}")
            f.write("\n")
        
        f.write("\n")
        f.write("VISUALIZATION FILES\n")
        f.write("-" * 30 + "\n")
        f.write("1. PCA Visualization: pca_visualization.png\n")
        f.write("2. Embedding Heatmap: embedding_heatmap.png\n")
        f.write("3. Confidence Distribution: confidence_distribution.png\n")
        f.write("4. Confusion Matrix: confusion_matrix.png\n")
        f.write("5. Feature Importance: feature_importance.png\n")
        f.write("6. Class Separation Plot: class_separation.png\n")
    
    print(f"Model performance report saved to {report_path}")
    return report_path

def main(args):
    """Main function to run visualizations"""
    print(f"Starting visualization for LENS model...")
    
    # Extract graph representations
    graph_reps, labels, prob_scores, n_classes = extract_graph_representations(args)
    
    if graph_reps is None:
        print("Failed to extract graph representations. Visualization aborted.")
        return
    
    # Print shapes and statistics
    print(f"\nExtracted {graph_reps.shape[0]} graph representations of dimension {graph_reps.shape[1:]}")
    print(f"Number of classes: {n_classes}")
    
    # Generate visualizations
    plot_pca(graph_reps, labels, n_classes, os.path.join(args.output_dir, 'pca_visualization.png'))
    plot_heatmap(graph_reps, labels, os.path.join(args.output_dir, 'embedding_heatmap.png'))
    plot_confidence_distribution(prob_scores, labels, n_classes, os.path.join(args.output_dir, 'confidence_distribution.png'))
    plot_confusion_matrix(prob_scores, labels, n_classes, os.path.join(args.output_dir, 'confusion_matrix.png'))
    top_features, _ = plot_feature_importance(graph_reps, labels, n_classes, os.path.join(args.output_dir, 'feature_importance.png'))
    plot_class_separation(graph_reps, labels, top_features, n_classes, os.path.join(args.output_dir, 'class_separation.png'))
    
    # Generate comprehensive report
    generate_model_report(graph_reps, labels, prob_scores, n_classes, args.output_dir)
    
    # Save the extracted representations and labels for later use
    np.save(os.path.join(args.output_dir, 'graph_representations.npy'), graph_reps)
    np.save(os.path.join(args.output_dir, 'graph_labels.npy'), labels)
    np.save(os.path.join(args.output_dir, 'graph_probabilities.npy'), prob_scores)
    
    print("\nVisualization complete! All outputs saved to", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LENS Model Visualization')
    
    # Required parameters
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data file')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory for dataset')
    
    # Model parameters
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
    
    # Other parameters
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--output-dir', type=str, default='lens_visualization', help='Output directory')
    
    args = parser.parse_args()
    
    # Run visualization
    main(args)
