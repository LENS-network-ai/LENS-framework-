#!/usr/bin/env python
# coding: utf-8

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import gc
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
from model.LENS import ImprovedEdgeGNN
from helper import Evaluator, collate, preparefeatureLabel

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
    
    # Initialize storage for additional metrics
    accuracy_scores = []
    precision_scores = {i: [] for i in range(n_classes)}
    recall_scores = {i: [] for i in range(n_classes)}
    f1_scores = {i: [] for i in range(n_classes)}
    specificity_scores = {i: [] for i in range(n_classes)}
    macro_precision_scores = []
    macro_recall_scores = []
    macro_f1_scores = []
    macro_specificity_scores = []
    weighted_precision_scores = []
    weighted_recall_scores = []
    weighted_f1_scores = []
    weighted_specificity_scores = []
    
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
        
        # Calculate predicted classes from probabilities
        y_pred_boot = np.argmax(y_score_boot, axis=1)
        
        # Calculate additional metrics
        accuracy_scores.append(accuracy_score(y_true_boot, y_pred_boot))
        
        # Calculate specificity for each class in this bootstrap sample
        for i in range(n_classes):
            # True negatives: not in class i and not predicted as class i
            tn = np.sum(np.logical_and(y_true_boot != i, y_pred_boot != i))
            # False positives: not in class i but predicted as class i
            fp = np.sum(np.logical_and(y_true_boot != i, y_pred_boot == i))
            
            if tn + fp > 0:
                specificity = tn / (tn + fp)
            else:
                specificity = 0
            
            specificity_scores[i].append(specificity)
        
        # Macro specificity
        macro_specificity_scores.append(np.mean([specificity_scores[i][-1] for i in range(n_classes)]))
        
        # Weighted specificity
        class_counts = np.bincount(y_true_boot, minlength=n_classes)
        total_samples = np.sum(class_counts)
        weights = [(total_samples - count) / total_samples for count in class_counts]
        weighted_specificity_scores.append(np.average([specificity_scores[i][-1] for i in range(n_classes)], weights=weights))
        
        # Per-class metrics
        for i in range(n_classes):
            mask = (y_true_boot == i) | (y_pred_boot == i)
            if np.sum(mask) > 0:
                precision_scores[i].append(precision_score(y_true_boot, y_pred_boot, labels=[i], average=None, zero_division=0)[0])
                recall_scores[i].append(recall_score(y_true_boot, y_pred_boot, labels=[i], average=None, zero_division=0)[0])
                f1_scores[i].append(f1_score(y_true_boot, y_pred_boot, labels=[i], average=None, zero_division=0)[0])
            else:
                precision_scores[i].append(np.nan)
                recall_scores[i].append(np.nan)
                f1_scores[i].append(np.nan)
        
        # Macro and weighted averages
        macro_precision_scores.append(precision_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        macro_recall_scores.append(recall_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        macro_f1_scores.append(f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        
        weighted_precision_scores.append(precision_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
        weighted_recall_scores.append(recall_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
        weighted_f1_scores.append(f1_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
        
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
        
        # Calculate stats for additional metrics
        valid_precision = [s for s in precision_scores[i] if not np.isnan(s)]
        valid_recall = [s for s in recall_scores[i] if not np.isnan(s)]
        valid_f1 = [s for s in f1_scores[i] if not np.isnan(s)]
        valid_specificity = [s for s in specificity_scores[i] if not np.isnan(s)]
        
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
                'recall_grid': recall_grid,
                
                # Additional metrics with bootstrap stats
                'precision_score_mean': np.mean(valid_precision) if valid_precision else np.nan,
                'precision_score_std': np.std(valid_precision) if valid_precision else np.nan,
                'recall_score_mean': np.mean(valid_recall) if valid_recall else np.nan,
                'recall_score_std': np.std(valid_recall) if valid_recall else np.nan,
                'f1_score_mean': np.mean(valid_f1) if valid_f1 else np.nan,
                'f1_score_std': np.std(valid_f1) if valid_f1 else np.nan,
                'specificity_score_mean': np.mean(valid_specificity) if valid_specificity else np.nan,
                'specificity_score_std': np.std(valid_specificity) if valid_specificity else np.nan
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
    
    # Add overall accuracy stats
    results['accuracy'] = {
        'mean': np.mean(accuracy_scores),
        'std': np.std(accuracy_scores),
        'ci_low': np.percentile(accuracy_scores, 100 * alpha / 2),
        'ci_high': np.percentile(accuracy_scores, 100 * (1 - alpha / 2))
    }
    
    # Add both macro average metrics and macro AUC
    results['macro_metrics'] = {
        'precision_mean': np.mean(macro_precision_scores),
        'precision_std': np.std(macro_precision_scores),
        'recall_mean': np.mean(macro_recall_scores),
        'recall_std': np.std(macro_recall_scores),
        'f1_mean': np.mean(macro_f1_scores),
        'f1_std': np.std(macro_f1_scores)
    }
    
    # Add weighted average metrics
    results['weighted_metrics'] = {
        'precision_mean': np.mean(weighted_precision_scores),
        'precision_std': np.std(weighted_precision_scores),
        'recall_mean': np.mean(weighted_recall_scores),
        'recall_std': np.std(weighted_recall_scores),
        'f1_mean': np.mean(weighted_f1_scores),
        'f1_std': np.std(weighted_f1_scores)
    }
    
    # Add macro specificity metrics
    results['macro_specificity'] = {
        'mean': np.mean(macro_specificity_scores),
        'std': np.std(macro_specificity_scores)
    }
    
    # Add weighted specificity metrics
    results['weighted_specificity'] = {
        'mean': np.mean(weighted_specificity_scores),
        'std': np.std(weighted_specificity_scores)
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

def plot_specificity_curve(results, n_classes, output_path):
    """
    Plot specificity vs sensitivity (ROC) curve
    """
    plt.figure(figsize=(12, 10))
    
    # Use a colormap for different classes
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    
    # Plot specificity-sensitivity curve for each class
    for i in range(n_classes):
        class_key = f'class_{i}'
        if class_key in results:
            res = results[class_key]
            
            # Since specificity = 1 - FPR, we can calculate it from the ROC curve
            specificity = 1 - res['fpr_grid']
            sensitivity = res['tpr_mean']
            
            # Calculate specificity confidence intervals
            specificity_ci_low = 1 - res['fpr_grid']  # High FPR -> Low specificity
            specificity_ci_high = 1 - res['fpr_grid']  # Low FPR -> High specificity
            
            # Plot mean curve
            plt.plot(specificity, sensitivity, 
                    color=colors[i], linewidth=2,
                    label=f'Class {i} (Spec = {results[class_key]["specificity_score_mean"]:.3f} ± {results[class_key]["specificity_score_std"]:.3f})')
            
            # Plot confidence bands (just for sensitivity, as specificity is derived from FPR)
            plt.fill_between(specificity, 
                           res['tpr_ci_low'], 
                           res['tpr_ci_high'],
                           color=colors[i], alpha=0.2)
    
    # Plot formatting
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])
    plt.xlabel('Specificity (1 - False Positive Rate)', fontsize=14)
    plt.ylabel('Sensitivity (True Positive Rate)', fontsize=14)
    plt.title('Specificity vs Sensitivity Curves with 95% Confidence Intervals', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    
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
    Calculate additional metrics like accuracy, F1, precision, recall, specificity, confusion matrix
    """
    # Convert probabilities to predictions
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Convert one-hot encoded labels to class labels if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true
    
    # Calculate basic metrics
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
    
    # Calculate specificity for each class
    n_classes = len(np.unique(y_true_labels))
    specificity_per_class = []
    
    for i in range(n_classes):
        # True negatives are all the samples that are not in class i and were not predicted as class i
        tn = np.sum(np.logical_and(y_true_labels != i, y_pred != i))
        # False positives are samples that are not in class i but were predicted as class i
        fp = np.sum(np.logical_and(y_true_labels != i, y_pred == i))
        
        # Specificity = TN / (TN + FP)
        if tn + fp > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0
        
        specificity_per_class.append(specificity)
    
    # Calculate macro and weighted specificity
    specificity_macro = np.mean(specificity_per_class)
    
    # Weighted specificity (by class support)
    class_counts = np.bincount(y_true_labels, minlength=n_classes)
    total_samples = np.sum(class_counts)
    weights = [(total_samples - count) / total_samples for count in class_counts]  # Weight by negative samples
    specificity_weighted = np.average(specificity_per_class, weights=weights)
    
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
        'specificity_per_class': specificity_per_class,
        'specificity_macro': specificity_macro,
        'specificity_weighted': specificity_weighted,
        'confusion_matrix': cm,
        'classification_report': report
    }

def test_model_with_bootstrap(args):
    """
    Test model and perform bootstrap ROC/PR analysis
    """
    # Track memory usage
    memory_tracker = {
        'initial': psutil.Process().memory_info().rss / (1024 * 1024),  # MB
        'peak': 0,
        'after_model_load': 0,
        'after_testing': 0,
        'after_bootstrap': 0,
        'final': 0
    }
    
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
    
    # Update memory usage after model load
    memory_tracker['after_model_load'] = psutil.Process().memory_info().rss / (1024 * 1024)
    memory_tracker['peak'] = max(memory_tracker['peak'], memory_tracker['after_model_load'])
    
    evaluator = Evaluator(n_class=n_class)
    
    # Collect predictions and labels
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    # Track inference time
    inference_times = []
    
    print("\nCollecting predictions...")
    
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Processing samples"):
            # Measure inference time
            start_time = time.time()
            
            # Use evaluator to get predictions and labels
            pred, labels, _, _ = evaluator.eval_test(sample, model, n_features=args.n_features)
            
            # Record inference time
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # The predictions are already softmax probabilities from the evaluator
            probs = pred
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            
            # Update peak memory
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_tracker['peak'] = max(memory_tracker['peak'], current_memory)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate average inference time
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    # Update memory after testing
    memory_tracker['after_testing'] = psutil.Process().memory_info().rss / (1024 * 1024)
    memory_tracker['peak'] = max(memory_tracker['peak'], memory_tracker['after_testing'])
    
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
    
    # Update memory after bootstrap
    memory_tracker['after_bootstrap'] = psutil.Process().memory_info().rss / (1024 * 1024)
    memory_tracker['peak'] = max(memory_tracker['peak'], memory_tracker['after_bootstrap'])
    
    # Add inference time to results
    bootstrap_results['inference_time'] = {
        'mean': avg_inference_time,
        'std': std_inference_time,
        'per_sample': inference_times
    }
    
    # Add memory usage to results
    bootstrap_results['memory_usage'] = memory_tracker
    
    # Try to free up some memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    
    # Plot curves
    print("\nGenerating plots...")
    
    # Combined ROC and PR plot
    plot_combined_curves(
        bootstrap_results, 
        n_class, 
        os.path.join(args.output_dir, 'roc_pr_curves_combined.png')
    )
    
    # Specificity curve
    plot_specificity_curve(
        bootstrap_results,
        n_class,
        os.path.join(args.output_dir, 'specificity_curves.png')
    )
    
    # Individual plots
    plot_individual_curves(bootstrap_results, n_class, args.output_dir)
    
    # Update final memory
    memory_tracker['final'] = psutil.Process().memory_info().rss / (1024 * 1024)
    
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
        f.write("\n")
        
        # Write performance metrics with standard deviations
        f.write("\nPerformance Metrics with Standard Deviations:\n")
        f.write("-" * 60 + "\n")
        
        # Global metrics
        f.write(f"Accuracy: {bootstrap_results['accuracy']['mean']:.4f} ± {bootstrap_results['accuracy']['std']:.4f}\n")
        f.write(f"  95% CI: [{bootstrap_results['accuracy']['ci_low']:.4f}, {bootstrap_results['accuracy']['ci_high']:.4f}]\n\n")
        
        f.write("Macro Averages:\n")
        f.write(f"  F1 Score: {bootstrap_results['macro_metrics']['f1_mean']:.4f} ± {bootstrap_results['macro_metrics']['f1_std']:.4f}\n")
        f.write(f"  Precision: {bootstrap_results['macro_metrics']['precision_mean']:.4f} ± {bootstrap_results['macro_metrics']['precision_std']:.4f}\n")
        f.write(f"  Recall: {bootstrap_results['macro_metrics']['recall_mean']:.4f} ± {bootstrap_results['macro_metrics']['recall_std']:.4f}\n")
        f.write(f"  Specificity: {bootstrap_results['macro_specificity']['mean']:.4f} ± {bootstrap_results['macro_specificity']['std']:.4f}\n\n")
        
        f.write("Weighted Averages:\n")
        f.write(f"  F1 Score: {bootstrap_results['weighted_metrics']['f1_mean']:.4f} ± {bootstrap_results['weighted_metrics']['f1_std']:.4f}\n")
        f.write(f"  Precision: {bootstrap_results['weighted_metrics']['precision_mean']:.4f} ± {bootstrap_results['weighted_metrics']['precision_std']:.4f}\n")
        f.write(f"  Recall: {bootstrap_results['weighted_metrics']['recall_mean']:.4f} ± {bootstrap_results['weighted_metrics']['recall_std']:.4f}\n")
        f.write(f"  Specificity: {bootstrap_results['weighted_specificity']['mean']:.4f} ± {bootstrap_results['weighted_specificity']['std']:.4f}\n\n")
        
        # ROC and PR AUC metrics with bootstrap CI
        f.write("Per-Class Metrics with Bootstrap Statistics:\n")
        f.write("-" * 60 + "\n")
        
        # Class-specific table for better readability
        f.write(f"{'Metric':<15}{'Class 0':<20}{'Class 1':<20}{'Class 2':<20}\n")
        f.write("-" * 75 + "\n")
        
        metrics_rows = [
            ("ROC AUC", "roc_auc_mean", "roc_auc_std"),
            ("PR AUC", "pr_auc_mean", "pr_auc_std"),
            ("Precision", "precision_score_mean", "precision_score_std"),
            ("Recall", "recall_score_mean", "recall_score_std"),
            ("F1 Score", "f1_score_mean", "f1_score_std"),
            ("Specificity", "specificity_score_mean", "specificity_score_std")
        ]
        
        for metric_name, mean_key, std_key in metrics_rows:
            row = f"{metric_name:<15}"
            for i in range(n_class):
                class_key = f'class_{i}'
                if class_key in bootstrap_results and mean_key in bootstrap_results[class_key]:
                    mean_val = bootstrap_results[class_key][mean_key]
                    std_val = bootstrap_results[class_key][std_key]
                    row += f"{mean_val:.4f} ± {std_val:.4f}{'  ' if i < n_class-1 else ''}"
                else:
                    row += f"{'N/A':<20}"
            f.write(row + "\n")
        
        # Add inference time information
        f.write("\nInference Time:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average time per instance: {bootstrap_results['inference_time']['mean'] * 1000:.2f} ± {bootstrap_results['inference_time']['std'] * 1000:.2f} ms\n")
        f.write(f"Total inference time for {len(bootstrap_results['inference_time']['per_sample'])} samples: {np.sum(bootstrap_results['inference_time']['per_sample']):.4f} seconds\n")
        
        # Add memory usage information
        f.write("\nMemory Usage:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Initial: {bootstrap_results['memory_usage']['initial']:.2f} MB\n")
        f.write(f"After model load: {bootstrap_results['memory_usage']['after_model_load']:.2f} MB\n")
        f.write(f"After testing: {bootstrap_results['memory_usage']['after_testing']:.2f} MB\n")
        f.write(f"After bootstrap: {bootstrap_results['memory_usage']['after_bootstrap']:.2f} MB\n")
        f.write(f"Final: {bootstrap_results['memory_usage']['final']:.2f} MB\n")
        f.write(f"Peak memory usage: {bootstrap_results['memory_usage']['peak']:.2f} MB\n")
        f.write(f"Memory growth: {bootstrap_results['memory_usage']['final'] - bootstrap_results['memory_usage']['initial']:.2f} MB\n")
        
        # Confusion matrix
        f.write("\nConfusion Matrix:\n")
        f.write("-" * 30 + "\n")
        f.write(str(additional_metrics['confusion_matrix']) + "\n")
        
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
                f.write(f"    95% CI: [{res['roc_auc_ci_low']:.4f}, {res['roc_auc_ci_high']:.4f}]\n")
                f.write(f"  PR AUC: {res['pr_auc_mean']:.4f} ± {res['pr_auc_std']:.4f}\n")
                f.write(f"    95% CI: [{res['pr_auc_ci_low']:.4f}, {res['pr_auc_ci_high']:.4f}]\n")
                f.write(f"  Precision: {res['precision_score_mean']:.4f} ± {res['precision_score_std']:.4f}\n")
                f.write(f"  Recall: {res['recall_score_mean']:.4f} ± {res['recall_score_std']:.4f}\n")
                f.write(f"  F1 Score: {res['f1_score_mean']:.4f} ± {res['f1_score_std']:.4f}\n")
        
        # Macro and weighted averages for AUC metrics
        f.write("\nAverage AUC Metrics:\n")
        for avg_type in ['macro', 'weighted']:
            if avg_type in bootstrap_results:
                res = bootstrap_results[avg_type]
                f.write(f"\n{avg_type.capitalize()} Average:\n")
                f.write(f"  ROC AUC: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.4f}\n")
                f.write(f"    95% CI: [{res['roc_auc_ci_low']:.4f}, {res['roc_auc_ci_high']:.4f}]\n")
                f.write(f"  PR AUC: {res['pr_auc_mean']:.4f} ± {res['pr_auc_std']:.4f}\n")
                f.write(f"    95% CI: [{res['pr_auc_ci_low']:.4f}, {res['pr_auc_ci_high']:.4f}]\n")
        
        # Confusion matrix
        f.write("\nConfusion Matrix:\n")
        f.write("-" * 30 + "\n")
        f.write(str(additional_metrics['confusion_matrix']) + "\n")
        
        # Classification report (original without bootstrap)
        f.write("\nDetailed Classification Report (Single Test Set):\n")
        f.write("-" * 50 + "\n")
        for class_name, metrics in additional_metrics['classification_report'].items():
            if isinstance(metrics, dict):
                f.write(f"\n{class_name}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        # Summary table of all metrics with standard deviations
        f.write("\n\nSummary Table (Mean ± StdDev):\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Metric':<20} {'Class 0':<15} {'Class 1':<15} {'Class 2':<15}\n")
        f.write("-" * 70 + "\n")
        
        # Per-class metrics
        metrics_to_show = ['roc_auc', 'pr_auc', 'precision_score', 'recall_score', 'f1_score']
        for metric in metrics_to_show:
            metric_name = metric.replace('_', ' ').title()
            if metric == 'pr_auc':
                metric_name = 'PR AUC'
            elif metric == 'roc_auc':
                metric_name = 'ROC AUC'
                
            row = f"{metric_name:<20}"
            for i in range(n_class):
                class_key = f'class_{i}'
                if class_key in bootstrap_results:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    if mean_key in bootstrap_results[class_key] and std_key in bootstrap_results[class_key]:
                        mean_val = bootstrap_results[class_key][mean_key]
                        std_val = bootstrap_results[class_key][std_key]
                        row += f"{mean_val:.4f} ± {std_val:.4f}  "
                    else:
                        row += "N/A  "
                else:
                    row += "N/A  "
            f.write(row.strip() + "\n")
    
    print(f"\nResults saved to {args.output_dir}")
    
    return bootstrap_results, additional_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bootstrap ROC/PR Analysis for ImprovedEdgeGNN Model')
    
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
    
    # Bootstrap parameters
    parser.add_argument('--n-bootstrap', type=int, default=10000, help='Number of bootstrap iterations')
    parser.add_argument('--confidence-level', type=float, default=0.95, help='Confidence level for intervals')
    
    # Other parameters
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--output-dir', type=str, default='bootstrap_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Run bootstrap analysis
    bootstrap_results, additional_metrics = test_model_with_bootstrap(args)
