import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(dataset, train_idx):
    """
    Calculate class weights based on the training set class distribution
    """
    # Get all labels from training data
    train_labels = [dataset[i]['label'] for i in train_idx]
    
    # Count occurrences of each class
    unique_classes = np.unique(train_labels)
    class_counts = np.bincount(train_labels)
    
    print(f"Class distribution in training set:")
    for cls in unique_classes:
        count = class_counts[cls] if cls < len(class_counts) else 0
        pct = 100 * count / len(train_labels)
        print(f"  Class {cls}: {count} samples ({pct:.2f}%)")
    
    # Compute class weights - inversely proportional to class frequencies
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_labels
    )
    
    # Convert to PyTorch tensor
    weights = torch.FloatTensor(class_weights)
    
    print(f"Calculated class weights: {weights}")
    
    return weights

def plot_metrics(train_accs, val_accs, train_losses, val_losses, save_path, title="Training Metrics", warmup_epochs=5):
    """Plot training and validation metrics"""
    epochs = range(1, len(train_accs) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy
    ax1.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax1.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax1.set_title('Accuracy vs. Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Add warmup period shading
    if warmup_epochs > 0:
        ax1.axvspan(1, min(warmup_epochs, len(epochs)), alpha=0.2, color='green', label='Warmup Period')
    
    # Plot loss
    ax2.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax2.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax2.set_title('Loss vs. Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Add warmup period shading
    if warmup_epochs > 0:
        ax2.axvspan(1, min(warmup_epochs, len(epochs)), alpha=0.2, color='green', label='Warmup Period')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Also plot train-val accuracy gap
    fig, ax = plt.subplots(figsize=(10, 6))
    gaps = [train - val for train, val in zip(train_accs, val_accs)]
    ax.plot(epochs, gaps, 'g-', label='Train-Val Accuracy Gap')
    ax.set_title('Train-Val Accuracy Gap (Higher = More Overfitting)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap')
    ax.grid(True)
    ax.legend()
    
    # Add horizontal line at "acceptable" gap threshold
    ax.axhline(y=0.1, color='r', linestyle='--', label='Threshold (0.1)')
    
    # Add warmup period shading
    if warmup_epochs > 0:
        ax.axvspan(1, min(warmup_epochs, len(epochs)), alpha=0.2, color='green', label='Warmup Period')
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_gap.png'))
    plt.close()

def analyze_overfitting(train_accs, val_accs, warmup_epochs=5):
    """Analyze training and validation accuracy for signs of overfitting"""
    # Calculate train-val accuracy gaps
    gaps = [train - val for train, val in zip(train_accs, val_accs)]
    
    # Basic statistics
    avg_gap = sum(gaps) / len(gaps)
    max_gap = max(gaps)
    max_gap_epoch = gaps.index(max_gap) + 1
    
    # Look at trend in last third of training (excluding warmup)
    post_warmup_gaps = gaps[warmup_epochs:]
    if len(post_warmup_gaps) > 0:
        avg_post_warmup_gap = sum(post_warmup_gaps) / len(post_warmup_gaps)
        
        # Look at last third
        last_third = len(post_warmup_gaps) // 3
        if last_third > 0:
            recent_gaps = post_warmup_gaps[-last_third:]
            recent_avg_gap = sum(recent_gaps) / len(recent_gaps)
            gap_trend = recent_avg_gap - avg_post_warmup_gap
        else:
            recent_avg_gap = avg_post_warmup_gap
            gap_trend = 0
    else:
        avg_post_warmup_gap = avg_gap
        recent_avg_gap = avg_gap
        gap_trend = 0
    
    # Check if validation accuracy improves over time (post-warmup)
    post_warmup_val = val_accs[warmup_epochs:]
    if len(post_warmup_val) > 1:
        val_improvements = [post_warmup_val[i] - post_warmup_val[i-1] for i in range(1, len(post_warmup_val))]
        positive_improvements = sum(1 for imp in val_improvements if imp > 0)
        improvement_ratio = positive_improvements / len(val_improvements) if val_improvements else 0
    else:
        improvement_ratio = 0
    
    # Determine overfitting severity - focusing on post-warmup
    is_overfitting = False
    severity = "None"
    
    if avg_post_warmup_gap > 0.2 or max_gap > 0.3:
        is_overfitting = True
        severity = "Severe"
    elif avg_post_warmup_gap > 0.1 or max_gap > 0.2:
        is_overfitting = True
        severity = "Moderate"
    elif avg_post_warmup_gap > 0.05 or max_gap > 0.1:
        is_overfitting = True
        severity = "Mild"
    
    results = {
        "avg_gap": avg_gap,
        "avg_post_warmup_gap": avg_post_warmup_gap,
        "max_gap": max_gap,
        "max_gap_epoch": max_gap_epoch,
        "recent_avg_gap": recent_avg_gap,
        "gap_trend": gap_trend,
        "improvement_ratio": improvement_ratio,
        "is_overfitting": is_overfitting,
        "severity": severity
    }
    
    return results
