import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv
from typing import List, Dict, Any, Optional, Tuple

class StatsTracker:
    def __init__(self):
        """Initialize tracking for edge weights and other statistics"""
        # For tracking statistics
        self.edge_stats: List[float] = []
        self.reg_losses: List[float] = []
        self.cls_losses: List[float] = []
        self.edge_mins: List[float] = []
        self.edge_maxs: List[float] = []
        self.edge_stds: List[float] = []
        self.graph_stats: List[List[Dict[str, Any]]] = []  # Will store detailed per-graph statistics
        
        # For printing control
        self.print_stats: bool = True
    
    def update_stats(self, edge_weights, adjs, cls_loss, reg_loss, current_epoch, current_beta):
        """Update tracking statistics after a forward pass"""
        # Calculate average edge weight for monitoring
        edge_mask = (adjs > 0).float()
        num_edges = torch.sum(edge_mask).item()
        avg_edge_weight = torch.sum(edge_weights * edge_mask).item() / max(1, num_edges)
        
        # Track min, max, std of edge weights
        weights = edge_weights * edge_mask
        positive_weights = weights[weights > 0]
        if len(positive_weights) > 0:
            min_weight = positive_weights.min().item()
            max_weight = positive_weights.max().item()
            std_weight = positive_weights.std().item()
        else:
            min_weight = 0.0
            max_weight = 0.0
            std_weight = 0.0
        
        # Track statistics
        self.edge_stats.append(avg_edge_weight)
        self.cls_losses.append(cls_loss.item())
        self.reg_losses.append(reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss)
        self.edge_mins.append(min_weight)
        self.edge_maxs.append(max_weight)
        self.edge_stds.append(std_weight)
        
        # Print current status (if enabled)
        if self.print_stats:
            print(f"Epoch {current_epoch} - "
                  f"Avg edge weight: {avg_edge_weight:.6f}, "
                  f"Min: {min_weight:.6f}, Max: {max_weight:.6f}, Std: {std_weight:.6f}, "
                  f"Classification loss: {cls_loss.item():.4f}, "
                  f"Regularization loss: {reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss:.4f}, "
                  f"Beta: {current_beta:.6f}")
    
    def plot_edge_weight_distribution(self, weighted_adj, epoch, batch_idx=0, save_dir='./', 
                                    current_beta=0.0, temperature=1.0, current_epoch=0, warmup_epochs=0):
        """Plot distribution of edge weights"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Flatten edge weights for all edges that exist
        edge_mask = (weighted_adj > 0).float()
        weights = (weighted_adj * edge_mask).detach().cpu().numpy().flatten()
        weights = weights[weights > 0]  # Only consider positive weights
        
        if len(weights) == 0:
            print("No edge weights to plot.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create a more detailed histogram with many bins
        counts, bins, _ = plt.hist(weights, bins=100, alpha=0.7, color='skyblue')
        
        # Add a title with statistics
        plt.title(f"Edge Weight Distribution - Epoch {epoch}, Batch {batch_idx}\n" + 
                 f"Min: {np.min(weights):.6f}, Mean: {np.mean(weights):.6f}, Max: {np.max(weights):.6f}, Std: {np.std(weights):.6f}\n" +
                 f"Beta: {current_beta:.6f}, Temperature: {temperature:.4f}",
                 fontsize=14)
        
        plt.xlabel("Edge Weight", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for key values
        plt.axvline(x=np.min(weights), color='r', linestyle='--', label=f'Min: {np.min(weights):.6f}')
        plt.axvline(x=np.mean(weights), color='g', linestyle='--', label=f'Mean: {np.mean(weights):.6f}')
        plt.axvline(x=np.max(weights), color='b', linestyle='--', label=f'Max: {np.max(weights):.6f}')
        
        # Add thresholds
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
        colors = ['purple', 'orange', 'brown', 'pink', 'cyan']
        for threshold, color in zip(thresholds, colors):
            if threshold > np.min(weights) and threshold < np.max(weights):
                plt.axvline(x=threshold, color=color, linestyle=':', 
                           label=f'Threshold {threshold:.2f}: {(weights > threshold).sum()} edges')
        
        # Add percentage annotations
        total_edges = len(weights)
        plt.text(0.02, 0.95, f"Total edges: {total_edges}", transform=plt.gca().transAxes, fontsize=10)
        
        for threshold in thresholds:
            count = (weights > threshold).sum()
            pct = 100 * count / total_edges
            plt.text(0.02, 0.95 - (thresholds.index(threshold) + 1) * 0.05, 
                    f"Edges > {threshold:.2f}: {count} ({pct:.1f}%)", 
                    transform=plt.gca().transAxes, fontsize=10)
        
        # Add warmup status
        if current_epoch < warmup_epochs:
            warmup_status = f"Warmup: {current_epoch}/{warmup_epochs} epochs"
            plt.text(0.02, 0.7, warmup_status, transform=plt.gca().transAxes, 
                    fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
        
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        # Save the figure
        filename = f"{save_dir}/edge_dist_epoch{epoch}_batch{batch_idx}.png"
        plt.savefig(filename)
        plt.close()
        
        return filename
    
    def plot_stats(self, save_path='stats.png', egl_mode='egl', base_beta=0.01, warmup_epochs=5):
        """Plot edge weight and loss evolution"""
        epochs = range(1, len(self.edge_stats) + 1)
        
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot average edge weight
        axs[0, 0].plot(epochs, self.edge_stats, 'b-')
        axs[0, 0].set_title('Average Edge Weight vs. Epoch')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Average Edge Weight')
        axs[0, 0].set_ylim([0, 1])
        axs[0, 0].grid(True)
        
        # Add warmup region shading
        if warmup_epochs > 0:
            axs[0, 0].axvspan(1, min(warmup_epochs, len(epochs)), alpha=0.2, color='green', label='Warmup Period')
        
        # Plot min/max edge weights
        axs[0, 1].plot(epochs, self.edge_mins, 'r-', label='Min')
        axs[0, 1].plot(epochs, self.edge_maxs, 'g-', label='Max')
        axs[0, 1].plot(epochs, self.edge_stds, 'b--', label='Std')
        axs[0, 1].set_title('Edge Weight Range vs. Epoch')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Edge Weight')
        axs[0, 1].set_ylim([0, 1])
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Add warmup region shading
        if warmup_epochs > 0:
            axs[0, 1].axvspan(1, min(warmup_epochs, len(epochs)), alpha=0.2, color='green', label='Warmup Period')
        
        # Plot losses
        axs[1, 0].plot(epochs, self.cls_losses, 'b-', label='Classification Loss')
        axs[1, 0].set_title('Classification Loss vs. Epoch')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Add warmup region shading
        if warmup_epochs > 0:
            axs[1, 0].axvspan(1, min(warmup_epochs, len(epochs)), alpha=0.2, color='green', label='Warmup Period')
        
        # Plot regularization loss
        axs[1, 1].plot(epochs, self.reg_losses, 'r-', label='Regularization Loss')
        axs[1, 1].set_title('Regularization Loss vs. Epoch')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        # Add warmup region shading
        if warmup_epochs > 0:
            axs[1, 1].axvspan(1, min(warmup_epochs, len(epochs)), alpha=0.2, color='green', label='Warmup Period')
        
        plt.suptitle(f'Training Evolution with {egl_mode.upper()} (beta={base_beta}, warmup={warmup_epochs})')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        # Also save data as CSV for further analysis
        csv_path = save_path.replace('.png', '.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Avg_Weight', 'Min_Weight', 'Max_Weight', 'Std_Weight', 'Cls_Loss', 'Reg_Loss'])
            for i in range(len(epochs)):
                writer.writerow([
                    epochs[i],
                    self.edge_stats[i],
                    self.edge_mins[i],
                    self.edge_maxs[i],
                    self.edge_stds[i],
                    self.cls_losses[i],
                    self.reg_losses[i]
                ])
                
        return save_path
    
    def save_graph_analysis(self, epoch, batch_idx, save_dir='./', current_beta=0.0, 
                           temperature=1.0, warmup_epochs=5):
        """Save detailed per-graph analysis to a file"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have any graph statistics
        if not self.graph_stats or len(self.graph_stats) == 0:
            print("No graph statistics available to save.")
            return
        
        # Get the most recent batch statistics
        latest_batch = self.graph_stats[-1]
        
        # Choose evenly spaced samples
        sample_count = min(10, len(latest_batch))
        sample_indices = np.linspace(0, len(latest_batch)-1, sample_count, dtype=int)
        sample_graphs = [latest_batch[i] for i in sample_indices]
        
        # Create a detailed report
        report = {
            'epoch': epoch,
            'current_beta': current_beta,
            'temperature': temperature,
            'batch_idx': batch_idx,
            'batch_size': len(latest_batch),
            'avg_original_edges': sum(g['original_edges'] for g in latest_batch) / len(latest_batch),
            'avg_mean_weight': sum(g['mean_weight'] for g in latest_batch) / len(latest_batch),
            'sample_graphs': sample_graphs
        }
        
        # Save the report to a JSON file
        filename = f"{save_dir}/graph_analysis_epoch{epoch}_batch{batch_idx}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save a human-readable text report
        txt_filename = f"{save_dir}/graph_analysis_epoch{epoch}_batch{batch_idx}.txt"
        with open(txt_filename, 'w') as f:
            f.write(f"GRAPH ANALYSIS - Epoch {epoch}, Batch {batch_idx}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Current beta: {current_beta:.6f}\n")
            f.write(f"Current temperature: {temperature:.4f}\n")
            f.write(f"Warmup status: {'In progress' if epoch < warmup_epochs else 'Complete'}\n\n")
            
            f.write(f"Batch size: {len(latest_batch)} graphs\n")
            f.write(f"Average original edges per graph: {report['avg_original_edges']:.1f}\n")
            f.write(f"Average mean edge weight: {report['avg_mean_weight']:.6f}\n\n")
            
            # Calculate average effective sparsity
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
            f.write("Average effective edges after thresholding:\n")
            for threshold in thresholds:
                key = f">{threshold:.2f}"
                if key in sample_graphs[0]['effective_edges']:
                    avg_effective = sum(g['effective_edges'][key] for g in latest_batch) / len(latest_batch)
                    avg_effective_pct = 100 * avg_effective / report['avg_original_edges']
                    f.write(f"  Edges with weight {key}: {avg_effective:.1f} ({avg_effective_pct:.1f}% of original)\n")
            
            # Write detailed statistics for sample graphs
            f.write("\nDETAILED SAMPLE GRAPHS:\n")
            f.write("-" * 50 + "\n")
            
            for i, g in enumerate(sample_graphs):
                f.write(f"\nGraph {i+1} of {len(sample_graphs)} (ID: {g['graph_idx']}):\n")
                f.write(f"  Nodes: {g['total_nodes']}, Edges: {g['original_edges']}, Density: {g['edge_density']:.4f}\n")
                f.write(f"  Target sparsity: {g.get('target_sparsity', 1.0):.2f}\n")
                f.write(f"  Edge weights: mean={g['mean_weight']:.6f}, min={g['min_weight']:.6f}, max={g['max_weight']:.6f}, std={g['std_weight']:.6f}\n")
                
                # Weight distribution
                f.write("  Edge weight distribution:\n")
                for range_key, count in g['weight_ranges'].items():
                    pct = 100 * count / g['original_edges']
                    f.write(f"    {range_key}: {count} edges ({pct:.1f}%)\n")
                
                # Effective edge counts
                f.write("  Effective edge counts:\n")
                for threshold in thresholds:
                    key = f">{threshold:.2f}"
                    if key in g['effective_edges']:
                        count = g['effective_edges'][key]
                        pct = 100 * count / g['original_edges']
                        f.write(f"    Weight {key}: {count} edges ({pct:.1f}% of original)\n")
        
        print(f"Graph analysis saved to {txt_filename} and {filename}")
        
        return txt_filename
    
    def save_sparsification_report(self, epoch, save_dir='./', current_beta=0.0, 
                                  temperature=1.0, warmup_epochs=5):
        """Generate a comprehensive sparsification report across all processed batches"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Check if we have any graph statistics
        if not self.graph_stats or len(self.graph_stats) == 0:
            print("No graph statistics available to generate report.")
            return
        
        # Collect all graph statistics from all batches
        all_graphs = []
        for batch in self.graph_stats:
            all_graphs.extend(batch)
        
        # Skip if no graphs were collected
        if len(all_graphs) == 0:
            print("No graphs collected for sparsification report.")
            return
        
        # Calculate overall statistics
        total_graphs = len(all_graphs)
        total_original_edges = sum(g['original_edges'] for g in all_graphs)
        avg_density = sum(g['edge_density'] for g in all_graphs) / total_graphs
        
        # Edge weight statistics
        all_mean_weights = [g['mean_weight'] for g in all_graphs]
        avg_mean_weight = sum(all_mean_weights) / total_graphs
        min_mean_weight = min(all_mean_weights)
        max_mean_weight = max(all_mean_weights)
        
        # Calculate effective sparsity at different thresholds
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
        threshold_stats = {}
        
        for threshold in thresholds:
            key = f">{threshold:.2f}"
            if key in all_graphs[0]['effective_edges']:
                total_effective = sum(g['effective_edges'][key] for g in all_graphs)
                avg_effective = total_effective / total_graphs
                pct_remaining = 100 * total_effective / total_original_edges
                
                threshold_stats[key] = {
                    'total_edges': total_effective,
                    'avg_per_graph': avg_effective,
                    'pct_of_original': pct_remaining
                }
        
        # Weight distribution across all graphs
        weight_ranges = list(all_graphs[0]['weight_ranges'].keys())
        distribution_stats = {}
        
        for weight_range in weight_ranges:
            total_in_range = sum(g['weight_ranges'][weight_range] for g in all_graphs)
            pct_in_range = 100 * total_in_range / total_original_edges
            
            distribution_stats[weight_range] = {
                'total_edges': total_in_range,
                'pct_of_original': pct_in_range
            }
        
        # Group graphs by size for more detailed analysis
        size_ranges = [(0, 500), (500, 1000), (1000, 2000), (2000, float('inf'))]
        size_group_stats = {}
        
        for low, high in size_ranges:
            group_key = f"{low}-{int(high) if high != float('inf') else 'inf'}"
            graphs_in_group = [g for g in all_graphs if low <= g['total_nodes'] < high]
            
            if graphs_in_group:
                group_count = len(graphs_in_group)
                group_avg_weight = sum(g['mean_weight'] for g in graphs_in_group) / group_count
                
                # Effective edges stats for this size group
                group_thresholds = {}
                for threshold in thresholds:
                    key = f">{threshold:.2f}"
                    if key in graphs_in_group[0]['effective_edges']:
                        total_in_group = sum(g['effective_edges'][key] for g in graphs_in_group)
                        avg_in_group = total_in_group / group_count
                        total_original = sum(g['original_edges'] for g in graphs_in_group)
                        pct_remaining = 100 * total_in_group / total_original if total_original > 0 else 0
                        
                        group_thresholds[key] = {
                            'total_edges': total_in_group,
                            'avg_per_graph': avg_in_group,
                            'pct_of_original': pct_remaining
                        }
                
                size_group_stats[group_key] = {
                    'count': group_count,
                    'avg_nodes': sum(g['total_nodes'] for g in graphs_in_group) / group_count,
                    'avg_edges': sum(g['original_edges'] for g in graphs_in_group) / group_count,
                    'avg_density': sum(g['edge_density'] for g in graphs_in_group) / group_count,
                    'avg_weight': group_avg_weight,
                    'threshold_stats': group_thresholds
                }
        
        # Build the report
        report = {
            'epoch': epoch,
            'current_beta': current_beta,
            'temperature': temperature,
            'warmup_status': 'Active' if epoch < warmup_epochs else 'Complete',
            'total_graphs': total_graphs,
            'total_original_edges': total_original_edges,
            'avg_edges_per_graph': total_original_edges / total_graphs,
            'avg_edge_density': avg_density,
            'edge_weight_stats': {
                'avg_mean_weight': avg_mean_weight,
                'min_mean_weight': min_mean_weight,
                'max_mean_weight': max_mean_weight
            },
            'threshold_stats': threshold_stats,
            'distribution_stats': distribution_stats,
            'size_group_stats': size_group_stats
        }
        
        # Save the JSON report
        json_path = f"{save_dir}/sparsification_report_epoch{epoch}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create a human-readable text report
        txt_path = f"{save_dir}/sparsification_report_epoch{epoch}.txt"
        with open(txt_path, 'w') as f:
            f.write(f"GRAPH SPARSIFICATION REPORT - Epoch {epoch}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MODEL PARAMETERS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Current beta: {current_beta:.6f}\n")
            f.write(f"Current temperature: {temperature:.4f}\n")
            f.write(f"Warmup status: {'In progress' if epoch < warmup_epochs else 'Complete'}\n")
            if epoch < warmup_epochs:
                progress = 100 * (epoch / warmup_epochs)
                f.write(f"Warmup progress: {progress:.1f}%\n")
            f.write("\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total graphs analyzed: {total_graphs}\n")
            f.write(f"Total original edges: {total_original_edges}\n")
            f.write(f"Average edges per graph: {total_original_edges / total_graphs:.1f}\n")
            f.write(f"Average edge density: {avg_density:.6f}\n\n")
            
            f.write("EDGE WEIGHT STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average mean edge weight: {avg_mean_weight:.6f}\n")
            f.write(f"Minimum mean edge weight: {min_mean_weight:.6f}\n")
            f.write(f"Maximum mean edge weight: {max_mean_weight:.6f}\n\n")
            
            f.write("EDGE WEIGHT DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            for weight_range, stats in distribution_stats.items():
                f.write(f"Weight range {weight_range}: {stats['total_edges']} edges ({stats['pct_of_original']:.1f}% of original)\n")
            f.write("\n")
            
            f.write("SPARSIFICATION EFFECTIVENESS\n")
            f.write("-" * 30 + "\n")
            for threshold, stats in threshold_stats.items():
                f.write(f"Edges with weight {threshold}: {stats['total_edges']} ({stats['pct_of_original']:.1f}% of original)\n")
                f.write(f"  Average per graph: {stats['avg_per_graph']:.1f}\n")
            f.write("\n")
            
            f.write("GRAPH SIZE GROUP ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for group_name, stats in size_group_stats.items():
                f.write(f"Size group {group_name} nodes ({stats['count']} graphs):\n")
                f.write(f"  Average nodes: {stats['avg_nodes']:.1f}\n")
                f.write(f"  Average edges: {stats['avg_edges']:.1f}\n")
                f.write(f"  Average density: {stats['avg_density']:.6f}\n")
                f.write(f"  Average edge weight: {stats['avg_weight']:.6f}\n")
                
                f.write("  Effective edges after thresholding:\n")
                for threshold, th_stats in stats['threshold_stats'].items():
                    f.write(f"    {threshold}: {th_stats['avg_per_graph']:.1f} edges/graph ({th_stats['pct_of_original']:.1f}% retained)\n")
                f.write("\n")
            
            # Add 10 sample graphs for detailed examination
            f.write("\n\nDETAILED SAMPLE GRAPHS\n")
            f.write("-" * 30 + "\n")
            
            # Select representative samples
            if total_graphs > 10:
                sample_indices = np.linspace(0, total_graphs-1, 10, dtype=int)
                samples = [all_graphs[i] for i in sample_indices]
            else:
                samples = all_graphs
            
            for i, g in enumerate(samples):
                f.write(f"\nGraph {i+1} of {len(samples)} (ID: {g['graph_idx']}):\n")
                f.write(f"  Nodes: {g['total_nodes']}, Edges: {g['original_edges']}, Density: {g['edge_density']:.6f}\n")
                if 'target_sparsity' in g:
                    f.write(f"  Target sparsity: {g['target_sparsity']:.2f}\n")
                f.write(f"  Edge weights: mean={g['mean_weight']:.6f}, min={g['min_weight']:.6f}, max={g['max_weight']:.6f}\n")
                
                # Weight distribution
                f.write("  Weight distribution:\n")
                for weight_range, count in g['weight_ranges'].items():
                    pct = 100 * count / g['original_edges'] if g['original_edges'] > 0 else 0
                    f.write(f"    {weight_range}: {count} edges ({pct:.1f}%)\n")
                
                # Effective edges after thresholding
                f.write("  Effective edges after thresholding:\n")
                for threshold in thresholds:
                    key = f">{threshold:.2f}"
                    if key in g['effective_edges']:
                        count = g['effective_edges'][key]
                        pct = 100 * count / g['original_edges'] if g['original_edges'] > 0 else 0
                        f.write(f"    {key}: {count} edges ({pct:.1f}%)\n")
        
        print(f"Sparsification report saved to {txt_path}")
        return txt_path
