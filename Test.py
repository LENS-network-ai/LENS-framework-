#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, roc_auc_score, 
                           precision_recall_curve, average_precision_score,
                           confusion_matrix, classification_report,
                           accuracy_score, f1_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize
from scipy import stats
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# Import your existing modules
from utils.dataset import GraphDataset
from LENS import ImprovedEdgeGNN
from helper import Evaluator, collate, preparefeatureLabel

def save_weighted_adjacencies(weighted_adjs, sample_ids, output_dir):
    """
    Save weighted adjacency matrices for later visualization
    
    Args:
        weighted_adjs: List of weighted adjacency matrices
        sample_ids: List of sample IDs corresponding to each matrix
        output_dir: Directory to save the matrices
    """
    adj_output_dir = os.path.join(output_dir, 'weighted_adjacencies')
    os.makedirs(adj_output_dir, exist_ok=True)
    
    print(f"Saving {len(weighted_adjs)} weighted adjacency matrices...")
    
    for i, (weighted_adj, sample_id) in enumerate(zip(weighted_adjs, sample_ids)):
        # Clean the sample ID to make it a valid filename
        if isinstance(sample_id, torch.Tensor):
            sample_id = sample_id.item() if sample_id.numel() == 1 else str(sample_id.tolist())
        
        clean_id = ''.join(c for c in str(sample_id) if c.isalnum() or c in '._- ')
        
        # Save in torch format
        adj_path = os.path.join(adj_output_dir, f"{clean_id}_weighted_adj.pt")
        torch.save({
            'weighted_adj': weighted_adj.cpu() if torch.is_tensor(weighted_adj) else weighted_adj,
            'sample_id': sample_id,
            'matrix_index': i
        }, adj_path)
    
    print(f"Weighted adjacencies saved to {adj_output_dir}")

def bootstrap_roc_pr_analysis(y_true, y_score_probs, n_bootstrap=10000, confidence_level=0.95):
    """
    Compute bootstrap confidence intervals for ROC AUC, PR AUC, and curves
    
    Args:
        y_true: True labels (one-hot encoded for multiclass)
        y_score_probs: Predicted probabilities for each class
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
    
    Returns:
        Dictionary containing ROC and PR statistics and confidence intervals
    """
    n_classes = y_score_probs.shape[1]
    n_samples = len(y_true)
    
    # Convert one-hot to class labels if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true
    
    # Initialize storage for ROC
    roc_auc_scores = {i: [] for i in range(n_classes)}
    roc_auc_scores['macro'] = []
    roc_auc_scores['weighted'] = []
    
    # Initialize storage for PR
    pr_auc_scores = {i: [] for i in range(n_classes)}
    pr_auc_scores['macro'] = []
    pr_auc_scores['weighted'] = []
    
    # Store TPR values at fixed FPR points for ROC
    fpr_grid = np.linspace(0, 1, 100)
    tpr_values = {i: np.zeros((n_bootstrap, len(fpr_grid))) for i in range(n_classes)}
    
    # Store Precision values at fixed Recall points for PR
    recall_grid = np.linspace(0, 1, 100)
    precision_values = {i: np.zeros((n_bootstrap, len(recall_grid))) for i in range(n_classes)}
    
    print(f"Running bootstrap with {n_bootstrap} iterations...")
    
    for b in tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Get bootstrap sample
        y_true_boot = y_true_labels[indices]
        y_score_boot = y_score_probs[indices]
        
        # Binarize labels for current bootstrap sample
        y_true_bin = label_binarize(y_true_boot, classes=range(n_classes))
        if n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        # Calculate metrics for each class
        for i in range(n_classes):
            try:
                # ROC AUC
                roc_auc = roc_auc_score(y_true_bin[:, i], y_score_boot[:, i])
                roc_auc_scores[i].append(roc_auc)
                
                # ROC curve
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_boot[:, i])
                tpr_interp = np.interp(fpr_grid, fpr, tpr)
                tpr_values[i][b] = tpr_interp
                
                # PR AUC
                pr_auc = average_precision_score(y_true_bin[:, i], y_score_boot[:, i])
                pr_auc_scores[i].append(pr_auc)
                
                # PR curve
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score_boot[:, i])
                # Interpolate precision at fixed recall points (reversed for proper interpolation)
                precision_interp = np.interp(recall_grid[::-1], recall[::-1], precision[::-1])[::-1]
                precision_values[i][b] = precision_interp
                
            except ValueError:
                # Handle case where a class might not be present in bootstrap sample
                roc_auc_scores[i].append(np.nan)
                pr_auc_scores[i].append(np.nan)
                tpr_values[i][b] = np.nan
                precision_values[i][b] = np.nan
        
        # Calculate macro and weighted averages
        valid_roc_aucs = [auc for auc in [roc_auc_scores[i][-1] for i in range(n_classes)] if not np.isnan(auc)]
        valid_pr_aucs = [auc for auc in [pr_auc_scores[i][-1] for i in range(n_classes)] if not np.isnan(auc)]
        
        if valid_roc_aucs:
            # Macro averages
            macro_roc_auc = np.mean(valid_roc_aucs)
            macro_pr_auc = np.mean(valid_pr_aucs)
            
            # Weighted averages
            class_counts = np.bincount(y_true_boot, minlength=n_classes)
            weighted_roc_auc = np.average(valid_roc_aucs, weights=class_counts[:len(valid_roc_aucs)])
            weighted_pr_auc = np.average(valid_pr_aucs, weights=class_counts[:len(valid_pr_aucs)])
            
            roc_auc_scores['macro'].append(macro_roc_auc)
            roc_auc_scores['weighted'].append(weighted_roc_auc)
            pr_auc_scores['macro'].append(macro_pr_auc)
            pr_auc_scores['weighted'].append(weighted_pr_auc)
        else:
            roc_auc_scores['macro'].append(np.nan)
            roc_auc_scores['weighted'].append(np.nan)
            pr_auc_scores['macro'].append(np.nan)
            pr_auc_scores['weighted'].append(np.nan)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    results = {}
    
    # Process results for each class
    for i in range(n_classes):
        valid_roc_scores = [s for s in roc_auc_scores[i] if not np.isnan(s)]
        valid_pr_scores = [s for s in pr_auc_scores[i] if not np.isnan(s)]
        valid_tprs = tpr_values[i][~np.isnan(tpr_values[i]).any(axis=1)]
        valid_precisions = precision_values[i][~np.isnan(precision_values[i]).any(axis=1)]
        
        if valid_roc_scores and valid_pr_scores:
            results[f'class_{i}'] = {
                # ROC metrics
                'roc_auc_mean': np.mean(valid_roc_scores),
                'roc_auc_std': np.std(valid_roc_scores),
                'roc_auc_ci_low': np.percentile(valid_roc_scores, 100 * alpha / 2),
                'roc_auc_ci_high': np.percentile(valid_roc_scores, 100 * (1 - alpha / 2)),
                'tpr_mean': np.mean(valid_tprs, axis=0),
                'tpr_std': np.std(valid_tprs, axis=0),
                'tpr_ci_low': np.percentile(valid_tprs, 100 * alpha / 2, axis=0),
                'tpr_ci_high': np.percentile(valid_tprs, 100 * (1 - alpha / 2), axis=0),
                'fpr_grid': fpr_grid,
                
                # PR metrics
                'pr_auc_mean': np.mean(valid_pr_scores),
                'pr_auc_std': np.std(valid_pr_scores),
                'pr_auc_ci_low': np.percentile(valid_pr_scores, 100 * alpha / 2),
                'pr_auc_ci_high': np.percentile(valid_pr_scores, 100 * (1 - alpha / 2)),
                'precision_mean': np.mean(valid_precisions, axis=0),
                'precision_std': np.std(valid_precisions, axis=0),
                'precision_ci_low': np.percentile(valid_precisions, 100 * alpha / 2, axis=0),
                'precision_ci_high': np.percentile(valid_precisions, 100 * (1 - alpha / 2), axis=0),
                'recall_grid': recall_grid
            }
    
    # Process macro and weighted results
    for avg_type in ['macro', 'weighted']:
        valid_roc_scores = [s for s in roc_auc_scores[avg_type] if not np.isnan(s)]
        valid_pr_scores = [s for s in pr_auc_scores[avg_type] if not np.isnan(s)]
        
        if valid_roc_scores and valid_pr_scores:
            results[avg_type] = {
                # ROC metrics
                'roc_auc_mean': np.mean(valid_roc_scores),
                'roc_auc_std': np.std(valid_roc_scores),
                'roc_auc_ci_low': np.percentile(valid_roc_scores, 100 * alpha / 2),
                'roc_auc_ci_high': np.percentile(valid_roc_scores, 100 * (1 - alpha / 2)),
                
                # PR metrics
                'pr_auc_mean': np.mean(valid_pr_scores),
                'pr_auc_std': np.std(valid_pr_scores),
                'pr_auc_ci_low': np.percentile(valid_pr_scores, 100 * alpha / 2),
                'pr_auc_ci_high': np.percentile(valid_pr_scores, 100 * (1 - alpha / 2))
            }
    
    return results

def plot_combined_curves(results, n_classes, output_path):
    """
    Plot combined ROC and PR curves with confidence bands
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Use a colormap for different classes
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    
    # Plot ROC curves
    for i in range(n_classes):
        class_key = f'class_{i}'
        if class_key in results:
            res = results[class_key]
            
            # Plot mean ROC curve
            ax1.plot(res['fpr_grid'], res['tpr_mean'], 
                    color=colors[i], linewidth=2,
                    label=f'Class {i} (AUC = {res["roc_auc_mean"]:.3f} ± {res["roc_auc_std"]:.3f})')
            
            # Plot confidence bands
            ax1.fill_between(res['fpr_grid'], 
                           res['tpr_ci_low'], 
                           res['tpr_ci_high'],
                           color=colors[i], alpha=0.2)
    
    # ROC plot formatting
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title('ROC Curves with 95% Confidence Intervals', fontsize=16)
    ax1.legend(loc="lower right", fontsize=12)
    
    # Plot PR curves
    for i in range(n_classes):
        class_key = f'class_{i}'
        if class_key in results:
            res = results[class_key]
            
            # Plot mean PR curve
            ax2.plot(res['recall_grid'], res['precision_mean'], 
                    color=colors[i], linewidth=2,
                    label=f'Class {i} (AP = {res["pr_auc_mean"]:.3f} ± {res["pr_auc_std"]:.3f})')
            
            # Plot confidence bands
            ax2.fill_between(res['recall_grid'], 
                           res['precision_ci_low'], 
                           res['precision_ci_high'],
                           color=colors[i], alpha=0.2)
    
    # PR plot formatting
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    ax2.set_xlabel('Recall', fontsize=14)
    ax2.set_ylabel('Precision', fontsize=14)
    ax2.set_title('Precision-Recall Curves with 95% Confidence Intervals', fontsize=16)
    ax2.legend(loc="lower left", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_individual_curves(results, n_classes, output_dir):
    """
    Plot individual ROC and PR curves for each class
    """
    for i in range(n_classes):
        class_key = f'class_{i}'
        if class_key in results:
            res = results[class_key]
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
            
            # ROC curve
            ax1.plot(res['fpr_grid'], res['tpr_mean'], 
                    color='blue', linewidth=3,
                    label=f'Mean ROC (AUC = {res["roc_auc_mean"]:.3f})')
            
            ax1.fill_between(res['fpr_grid'], 
                           res['tpr_ci_low'], 
                           res['tpr_ci_high'],
                           color='blue', alpha=0.2,
                           label=f'95% CI [{res["roc_auc_ci_low"]:.3f}, {res["roc_auc_ci_high"]:.3f}]')
            
            ax1.fill_between(res['fpr_grid'], 
                           res['tpr_mean'] - res['tpr_std'], 
                           res['tpr_mean'] + res['tpr_std'],
                           color='red', alpha=0.1,
                           label=f'±1 SD (σ = {res["roc_auc_std"]:.3f})')
            
            ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
            ax1.set_xlim([-0.01, 1.01])
            ax1.set_ylim([-0.01, 1.01])
            ax1.set_xlabel('False Positive Rate', fontsize=14)
            ax1.set_ylabel('True Positive Rate', fontsize=14)
            ax1.set_title(f'ROC Curve - Class {i}', fontsize=16)
            ax1.legend(loc="lower right", fontsize=12)
            
            # PR curve
            ax2.plot(res['recall_grid'], res['precision_mean'], 
                    color='green', linewidth=3,
                    label=f'Mean PR (AP = {res["pr_auc_mean"]:.3f})')
            
            ax2.fill_between(res['recall_grid'], 
                           res['precision_ci_low'], 
                           res['precision_ci_high'],
                           color='green', alpha=0.2,
                           label=f'95% CI [{res["pr_auc_ci_low"]:.3f}, {res["pr_auc_ci_high"]:.3f}]')
            
            ax2.fill_between(res['recall_grid'], 
                           res['precision_mean'] - res['precision_std'], 
                           res['precision_mean'] + res['precision_std'],
                           color='red', alpha=0.1,
                           label=f'±1 SD (σ = {res["pr_auc_std"]:.3f})')
            
            ax2.set_xlim([-0.01, 1.01])
            ax2.set_ylim([-0.01, 1.01])
            ax2.set_xlabel('Recall', fontsize=14)
            ax2.set_ylabel('Precision', fontsize=14)
            ax2.set_title(f'Precision-Recall Curve - Class {i}', fontsize=16)
            ax2.legend(loc="lower left", fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'roc_pr_class_{i}.png'), dpi=300, bbox_inches='tight')
            plt.close()

def calculate_additional_metrics(y_true, y_pred_probs):
    """
    Calculate additional metrics like accuracy, F1, precision, recall, confusion matrix
    """
    # Convert probabilities to predictions
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Convert one-hot encoded labels to class labels if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_labels, y_pred)
    
    # Per-class and average metrics
    precision_macro = precision_score(y_true_labels, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true_labels, y_pred, average='weighted', zero_division=0)
    precision_per_class = precision_score(y_true_labels, y_pred, average=None, zero_division=0)
    
    recall_macro = recall_score(y_true_labels, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true_labels, y_pred, average='weighted', zero_division=0)
    recall_per_class = recall_score(y_true_labels, y_pred, average=None, zero_division=0)
    
    f1_macro = f1_score(y_true_labels, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true_labels, y_pred, average='weighted', zero_division=0)
    f1_per_class = f1_score(y_true_labels, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred)
    
    # Classification report
    report = classification_report(y_true_labels, y_pred, output_dict=True, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'precision_per_class': precision_per_class,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'recall_per_class': recall_per_class,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report': report
    }

def test_model_with_bootstrap(args):
    """
    Test model and perform bootstrap ROC/PR analysis
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_ids = f.readlines()
    
    test_dataset = GraphDataset(root=args.data_root, ids=test_ids)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Determine number of classes from checkpoint
    if 'classifier.3.weight' in checkpoint['model_state_dict']:
        n_class = checkpoint['model_state_dict']['classifier.3.weight'].shape[0]
    elif 'classifier.weight' in checkpoint['model_state_dict']:
        n_class = checkpoint['model_state_dict']['classifier.weight'].shape[0]
    else:
        n_class = args.n_class
    
    # Handle parameter name conversion for backward compatibility
    lambda_reg = args.lambda_reg if hasattr(args, 'lambda_reg') else getattr(args, 'beta', 0.00014)
    reg_mode = args.reg_mode if hasattr(args, 'reg_mode') else getattr(args, 'egl_mode', 'l0')
    
    # Create model with new parameter structure
    model = ImprovedEdgeGNN(
        feature_dim=args.n_features,
        hidden_dim=args.hidden_dim,
        num_classes=n_class,
        lambda_reg=lambda_reg,
        reg_mode=reg_mode,
        edge_dim=args.edge_dim,
        warmup_epochs=args.warmup_epochs,
        graph_size_adaptation=args.graph_size_adaptation,
        min_edges_per_node=args.min_edges_per_node,
        dropout=args.dropout,
        # Add L0 parameters
        l0_gamma=getattr(args, 'l0_gamma', -0.1),
        l0_zeta=getattr(args, 'l0_zeta', 1.1),
        l0_beta=getattr(args, 'l0_beta', 0.66),
        initial_temp=getattr(args, 'initial_temp', 5.0)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    evaluator = Evaluator(n_class=n_class)
    
    # Collect predictions, labels, and weighted adjacencies
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_weighted_adjs = []
    all_sample_ids = []
    
    print("\nCollecting predictions and weighted adjacencies...")
    
    with torch.no_grad():
        for sample_idx, sample in enumerate(tqdm(test_loader, desc="Processing samples")):
            try:
                # Get the data for this sample
                node_feat, labels, adjs, masks = preparefeatureLabel(
                    sample['image'], sample['label'], sample['adj_s'], n_features=args.n_features
                )
                
                # Forward pass through the model to get weighted adjacencies
                logits, _, _, weighted_adj = model(node_feat, labels, adjs, masks)
                
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=1)
                
                all_predictions.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                
                # Store weighted adjacencies and sample IDs
                for b in range(weighted_adj.shape[0]):
                    all_weighted_adjs.append(weighted_adj[b])
                    # Try to get sample ID from the sample
                    if 'id' in sample:
                        sample_id = sample['id'][b] if isinstance(sample['id'], (list, torch.Tensor)) else sample['id']
                    else:
                        sample_id = f"sample_{sample_idx}_{b}"
                    all_sample_ids.append(sample_id)
                
            except Exception as e:
                print(f"Warning: Error processing sample {sample_idx}: {str(e)}")
                continue
    
    # Save weighted adjacency matrices
    print(f"\nSaving {len(all_weighted_adjs)} weighted adjacency matrices...")
    save_weighted_adjacencies(all_weighted_adjs, all_sample_ids, args.output_dir)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate additional metrics
    print("\nCalculating additional metrics...")
    additional_metrics = calculate_additional_metrics(all_labels, all_probabilities)
    
    # Perform bootstrap analysis
    print(f"\nPerforming bootstrap analysis with {args.n_bootstrap} iterations...")
    bootstrap_results = bootstrap_roc_pr_analysis(
        all_labels, 
        all_probabilities, 
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level
    )
    
    # Plot curves
    print("\nGenerating plots...")
    
    # Combined ROC and PR plot
    plot_combined_curves(
        bootstrap_results, 
        n_class, 
        os.path.join(args.output_dir, 'roc_pr_curves_combined.png')
    )
    
    # Individual plots
    plot_individual_curves(bootstrap_results, n_class, args.output_dir)
    
    # Save comprehensive results
    results_file = os.path.join(args.output_dir, 'comprehensive_results.txt')
    with open(results_file, 'w') as f:
        f.write("Comprehensive Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic information
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Data: {args.test_data}\n")
        f.write(f"Number of test samples: {len(all_labels)}\n")
        f.write(f"Number of classes: {n_class}\n")
        f.write(f"Number of bootstrap iterations: {args.n_bootstrap}\n")
        f.write(f"Confidence level: {args.confidence_level * 100}%\n")
        f.write(f"Regularization strength (λ): {lambda_reg}\n")
        f.write(f"Regularization mode: {reg_mode}\n")
        
        # Add L0 parameters if using L0 regularization
        if reg_mode == 'l0':
            f.write(f"L0 gamma: {getattr(args, 'l0_gamma', -0.1)}\n")
            f.write(f"L0 zeta: {getattr(args, 'l0_zeta', 1.1)}\n")
            f.write(f"L0 beta: {getattr(args, 'l0_beta', 0.66)}\n")
        f.write("\n")
        
        # Overall metrics
        f.write("Overall Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {additional_metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1 Score: {additional_metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted F1 Score: {additional_metrics['f1_weighted']:.4f}\n")
        f.write(f"Macro Precision: {additional_metrics['precision_macro']:.4f}\n")
        f.write(f"Weighted Precision: {additional_metrics['precision_weighted']:.4f}\n")
        f.write(f"Macro Recall: {additional_metrics['recall_macro']:.4f}\n")
        f.write(f"Weighted Recall: {additional_metrics['recall_weighted']:.4f}\n")
        f.write("\n")
        
        # ROC and PR AUC metrics with bootstrap CI
        f.write("ROC and PR AUC Metrics (with Bootstrap CI):\n")
        f.write("-" * 50 + "\n")
        
        # Class-specific results
        for i in range(n_class):
            class_key = f'class_{i}'
            if class_key in bootstrap_results:
                res = bootstrap_results[class_key]
                f.write(f"\nClass {i}:\n")
                f.write(f"  ROC AUC: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.4f}\n")
                f.write(f"  ROC AUC 95% CI: [{res['roc_auc_ci_low']:.4f}, {res['roc_auc_ci_high']:.4f}]\n")
                f.write(f"  PR AUC: {res['pr_auc_mean']:.4f} ± {res['pr_auc_std']:.4f}\n")
                f.write(f"  PR AUC 95% CI: [{res['pr_auc_ci_low']:.4f}, {res['pr_auc_ci_high']:.4f}]\n")
                f.write(f"  Precision: {additional_metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall: {additional_metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1 Score: {additional_metrics['f1_per_class'][i]:.4f}\n")
        
        # Macro and weighted averages
        f.write("\nAverage Metrics:\n")
        for avg_type in ['macro', 'weighted']:
            if avg_type in bootstrap_results:
                res = bootstrap_results[avg_type]
                f.write(f"\n{avg_type.capitalize()} Average:\n")
                f.write(f"  ROC AUC: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.4f}\n")
                f.write(f"  ROC AUC 95% CI: [{res['roc_auc_ci_low']:.4f}, {res['roc_auc_ci_high']:.4f}]\n")
                f.write(f"  PR AUC: {res['pr_auc_mean']:.4f} ± {res['pr_auc_std']:.4f}\n")
                f.write(f"  PR AUC 95% CI: [{res['pr_auc_ci_low']:.4f}, {res['pr_auc_ci_high']:.4f}]\n")
        
        # Confusion matrix
        f.write("\nConfusion Matrix:\n")
        f.write("-" * 30 + "\n")
        f.write(str(additional_metrics['confusion_matrix']) + "\n")
        
        # Classification report
        f.write("\nDetailed Classification Report:\n")
        f.write("-" * 50 + "\n")
        for class_name, metrics in additional_metrics['classification_report'].items():
            if isinstance(metrics, dict):
                f.write(f"\n{class_name}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Weighted adjacency matrices saved to {os.path.join(args.output_dir, 'weighted_adjacencies')}")
    
    return bootstrap_results, additional_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bootstrap ROC/PR Analysis for ImprovedEdgeGNN Model')
    
    # Required parameters
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, required=True, help='Path to test data file')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory for dataset')
    
    # Model parameters - New naming with backward compatibility
    parser.add_argument('--n-features', type=int, default=512, help='Number of node features')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--n-class', type=int, default=3, help='Number of classes')
    parser.add_argument('--edge-dim', type=int, default=64, help='Edge dimension')
    
    # Regularization parameters - New naming
    parser.add_argument('--lambda-reg', type=float, default=0.000182, help='Regularization strength (λ)')
    parser.add_argument('--reg-mode', type=str, default='l0', 
                        choices=['l0', 'egl', 'none'], help='Regularization mode')
    
    # Backward compatibility parameters
    parser.add_argument('--beta', type=float, help='[DEPRECATED] Use --lambda-reg instead')
    parser.add_argument('--egl-mode', type=str, 
                        choices=['none', 'egl', 'l1', 'l2', 'entropy', 'l0'], 
                        help='[DEPRECATED] Use --reg-mode instead')
    
    # L0 regularization specific parameters
    parser.add_argument('--l0-gamma', type=float, default=-0.12, help='L0 regularization gamma parameter')
    parser.add_argument('--l0-zeta', type=float, default=1.09, help='L0 regularization zeta parameter')
    parser.add_argument('--l0-beta', type=float, default=0.72, help='L0 regularization beta parameter')
    parser.add_argument('--initial-temp', type=float, default=4.35, help='Initial temperature for edge gating')
    
    # Other model parameters
    parser.add_argument('--warmup-epochs', type=int, default=35, help='Warmup epochs')
    parser.add_argument('--graph-size-adaptation', action='store_true', help='Enable graph size adaptation')
    parser.add_argument('--min-edges-per-node', type=float, default=2.0, help='Minimum edges per node')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Bootstrap parameters
    parser.add_argument('--n-bootstrap', type=int, default=10000, help='Number of bootstrap iterations')
    parser.add_argument('--confidence-level', type=float, default=0.95, help='Confidence level for intervals')
    
    # Other parameters
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--output-dir', type=str, default='bootstrap_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Run bootstrap analysis
    bootstrap_results, additional_metrics = test_model_with_bootstrap(args)
