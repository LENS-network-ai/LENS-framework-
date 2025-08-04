"""
Computational profiler for measuring GFLOPs using PyTorch Profiler.
"""

import torch
from torch.profiler import profile, ProfilerActivity, record_function
import numpy as np
from typing import Dict, Any
import json


class ComputationalProfiler:
    """Measures computational complexity of graph models."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results = {}
        
    def create_representative_sample(self, features_path=None, adj_path=None, num_patches: int = 29546, feature_dim: int = 1024):
        """
        Create a representative WSI sample from real data or synthetic data.
        
        Args:
            features_path: Path to real features.pt file
            adj_path: Path to real adj_s.pt file  
            num_patches: Number of patches (only used if no real data provided)
            feature_dim: Feature dimension (only used if no real data provided)
        """
        if features_path and adj_path:
            # Load real WSI data
            features = torch.load(features_path, map_location=self.device)
            adj_matrix = torch.load(adj_path, map_location=self.device)
            
            print(f"Loaded real WSI data:")
            print(f"  Features shape: {features.shape}")
            print(f"  Adjacency shape: {adj_matrix.shape}")
            
        else:
            # Create synthetic data (fallback)
            print("No real data provided, creating synthetic representative sample")
            features = torch.randn(num_patches, feature_dim, device=self.device)
            
            # Create sparse adjacency matrix
            sparsity = 0.02  # 2% sparsity
            num_edges = int(num_patches * num_patches * sparsity)
            
            adj_matrix = torch.zeros(num_patches, num_patches, device=self.device)
            edge_indices = torch.randint(0, num_patches, (2, num_edges), device=self.device)
            edge_weights = torch.rand(num_edges, device=self.device)
            
            adj_matrix[edge_indices[0], edge_indices[1]] = edge_weights
            # Make symmetric
            adj_matrix = (adj_matrix + adj_matrix.T) / 2
        
        sample = {
            'image': features,
            'adj_s': adj_matrix,
            'id': 'representative_wsi',
            'label': 1
        }
        
        return sample
    
    def measure_model_flops(self, model: torch.nn.Module, sample: Dict[str, Any], 
                           model_name: str = "model") -> float:
        """Measure GFLOPs for a single forward pass."""
        model.eval()
        model.to(self.device)
        
        # Move sample to device
        sample_device = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample_device[key] = value.to(self.device)
            else:
                sample_device[key] = value
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(sample_device)
        
        # Profile the model
        total_flops = 0
        num_runs = 5
        
        for _ in range(num_runs):
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                with_modules=True
            ) as prof:
                with record_function(f"{model_name}_forward"):
                    with torch.no_grad():
                        _ = model(sample_device)
            
            # Extract FLOP count
            flops = sum([
                int(evt.flops) for evt in prof.key_averages() 
                if evt.flops is not None and evt.flops > 0
            ])
            total_flops += flops
        
        # Average over runs and convert to GFLOPs
        avg_flops = total_flops / num_runs
        gflops = avg_flops / 1e9
        
        self.results[model_name] = {
            'gflops': gflops,
            'num_patches': sample_device['image'].shape[0],
            'feature_dim': sample_device['image'].shape[1],
            'total_flops': avg_flops
        }
        
        return gflops
    
    def measure_all_baselines(self, models_dict: Dict[str, torch.nn.Module], 
                             sample: Dict[str, Any] = None,
                             features_path: str = None,
                             adj_path: str = None) -> Dict[str, float]:
        """
        Measure GFLOPs for all baseline models.
        
        Args:
            models_dict: Dictionary of models to profile
            sample: Pre-created sample (optional)
            features_path: Path to real features.pt file
            adj_path: Path to real adj_s.pt file
        """
        if sample is None:
            sample = self.create_representative_sample(features_path=features_path, adj_path=adj_path)
        
        results = {}
        
        print(f"Measuring computational complexity:")
        print(f"- Patches: {sample['image'].shape[0]:,}")
        print(f"- Feature dim: {sample['image'].shape[1]}")
        print(f"- Device: {self.device}")
        print()
        
        for model_name, model in models_dict.items():
            print(f"Profiling {model_name}...")
            try:
                gflops = self.measure_model_flops(model, sample, model_name)
                results[model_name] = gflops
                print(f"  {model_name}: {gflops:.2f} GFLOPs")
            except Exception as e:
                print(f"  Error profiling {model_name}: {str(e)}")
                results[model_name] = None
            print()
        
        return results
    
    def calculate_efficiency_metrics(self, results: Dict[str, float], 
                                   accuracies: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate efficiency ratios (Accuracy/GFLOPs) for all models."""
        metrics = {}
        
        for model_name in results.keys():
            if results[model_name] is not None and model_name in accuracies:
                gflops = results[model_name]
                accuracy = accuracies[model_name]
                efficiency_ratio = accuracy / gflops
                
                metrics[model_name] = {
                    'accuracy': accuracy,
                    'gflops': gflops,
                    'efficiency_ratio': efficiency_ratio
                }
        
        return metrics
    
    def save_results(self, filepath: str = 'computational_analysis.json'):
        """Save profiling results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")
    
    def print_comparison_table(self, metrics: Dict[str, Dict[str, float]]):
        """Print formatted comparison table."""
        print("\n" + "="*80)
        print("COMPUTATIONAL EFFICIENCY COMPARISON")
        print("="*80)
        print(f"{'Model':<12} {'Accuracy (%)':<12} {'GFLOPs':<8} {'Efficiency Ratio':<15}")
        print("-"*80)
        
        # Sort by efficiency ratio (descending)
        sorted_models = sorted(metrics.items(), key=lambda x: x[1]['efficiency_ratio'], reverse=True)
        
        for model_name, data in sorted_models:
            print(f"{model_name:<12} {data['accuracy']:<12.1f} {data['gflops']:<8.1f} {data['efficiency_ratio']:<15.2f}")
        
        print("="*80)
