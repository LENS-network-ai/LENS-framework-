import numpy as np
import cv2
import openslide
import torch
import matplotlib.pyplot as plt
import os
import networkx as nx
import argparse

def create_jet_style_colormap():
    """Create a custom colormap similar to COLORMAP_JET but with more subdued colors."""
    # Create array for BGR format (OpenCV uses BGR)
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    
    # Define control points for a jet-like colormap (BGR format)
    # Format: [position, [B, G, R]]
    control_points = [
        [0.0, [128, 0, 0]],     # Dark blue
        [0.15, [255, 0, 0]],    # Blue
        [0.3, [255, 128, 0]],   # Cyan
        [0.5, [0, 255, 0]],     # Green
        [0.7, [0, 128, 255]],   # Yellow
        [0.85, [0, 0, 255]],    # Red
        [1.0, [0, 0, 128]]      # Dark red
    ]
    
    # Apply interpolation between control points
    for i in range(256):
        normalized = i / 255.0
        
        # Find the two control points to interpolate between
        idx = 0
        while idx < len(control_points) - 1 and normalized > control_points[idx + 1][0]:
            idx += 1
        
        if idx == len(control_points) - 1:
            colormap[i, 0, :] = control_points[-1][1]
        else:
            # Get the two control points
            t1, c1 = control_points[idx]
            t2, c2 = control_points[idx + 1]
            
            # Calculate interpolation factor
            t = (normalized - t1) / (t2 - t1) if t2 != t1 else 0
            
            # Apply linear interpolation
            for c in range(3):
                colormap[i, 0, c] = int(c1[c] * (1 - t) + c2[c] * t)
    
    return colormap

def show_importance_on_image(img, mask, alpha=0.7):
    """Apply red colormap with smoother blending for better visualization."""
    # Create a custom red colormap
    red_colormap = create_smooth_red_colormap()
    
    # Apply Gaussian blur to smooth the mask
    smoothed_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    # Normalize the smoothed mask to ensure it's in [0,1] range
    if np.max(smoothed_mask) > 0:
        smoothed_mask = smoothed_mask / np.max(smoothed_mask)
    
    # Create RGB heatmap directly (instead of using applyColorMap)
    heatmap = np.zeros_like(img)
    # Apply red gradient based on intensity
    heatmap[:,:,2] = np.uint8(smoothed_mask * 255)  # Red channel
    
    # Create alpha blending for more natural overlay
    overlay = img.copy()
    # Apply alpha blending: result = img * (1-alpha) + heatmap * alpha
    cv2.addWeighted(img, 1, heatmap, alpha, 0, overlay)
    
    return overlay

def importance_to_mask(gray, patches, importance_values, w, h, w_s, h_s):
    """Create a mask from importance values with smoother transitions."""
    # Initialize an empty mask
    mask = np.zeros_like(gray).astype(np.float32)
    
    # First pass: fill in the patches
    for ind1, patch in enumerate(patches):
        if ind1 >= len(importance_values):
            continue
        x, y = patch.split('.')[0].split('_')
        x, y = int(x), int(y)
        if y < 5 or x > w-5 or y > h-5:
            continue
        mask[int(y*h_s):int((y+1)*h_s), int(x*w_s):int((x+1)*w_s)].fill(importance_values[ind1])
    
    # Apply a very strong Gaussian blur for smoother results (matching reference images)
    # Increase kernel size for more extensive smoothing
    final_mask = cv2.GaussianBlur(mask, (51, 51), 15)
    
    return final_mask

def get_adj_from_file(adj_path, key='pruned_adj'):
    """Load adjacency matrix from your specific dictionary format."""
    loaded_data = torch.load(adj_path)
    
    if isinstance(loaded_data, dict):
        if key in loaded_data:
            print(f"Found key '{key}' in loaded data")
            return loaded_data[key].cpu().numpy()
        else:
            print(f"Key '{key}' not found. Available keys: {loaded_data.keys()}")
            for k, v in loaded_data.items():
                if torch.is_tensor(v):
                    print(f"Using tensor from key: {k}")
                    return v.cpu().numpy()
    else:
        return loaded_data.cpu().numpy()

def calculate_edge_weight_importance(adj_matrix):
    """Calculate node importance based on the sum of edge weights for each node."""
    if adj_matrix.ndim == 3:
        adj_matrix = adj_matrix[0]
    
    # Sum the weights of all edges for each node
    weight_sum = np.sum(adj_matrix, axis=1)
    
    # Normalize to [0, 1]
    if np.max(weight_sum) > 0:
        weight_sum = weight_sum / np.max(weight_sum)
    
    return weight_sum

def visualize_edge_weight_distribution(adj_matrix, output_file="edge_weight_distribution.png"):
    """Visualize the distribution of edge weights."""
    if adj_matrix.ndim == 3:
        adj_matrix = adj_matrix[0]
    
    # Get non-zero weights
    weights = adj_matrix[adj_matrix > 0]
    
    plt.figure(figsize=(10, 6))
    # Use red color for histogram
    plt.hist(weights, bins=30, color='red', alpha=0.7)
    plt.title("Edge Weight Distribution")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file)
    plt.close()
    
    print(f"Edge weight statistics:")
    print(f"  Min: {np.min(weights):.6f}")
    print(f"  Max: {np.max(weights):.6f}")
    print(f"  Mean: {np.mean(weights):.6f}")
    print(f"  Median: {np.median(weights):.6f}")
    
    return weights

def compare_original_pruned(original_adj_path, pruned_adj_path, 
                          original_key='adj_s', pruned_key='pruned_adj', 
                          output_dir="edge_comparison"):
    """Compare original and pruned adjacency matrices to analyze pruning effect."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load matrices
    original_adj = get_adj_from_file(original_adj_path, key=original_key)
    pruned_adj = get_adj_from_file(pruned_adj_path, key=pruned_key)
    
    # Ensure they're 2D matrices
    if original_adj.ndim == 3:
        original_adj = original_adj[0]
    if pruned_adj.ndim == 3:
        pruned_adj = pruned_adj[0]
    
    # Get the first file basename (without extension)
    file_name = os.path.basename(original_adj_path).split('.')[0]
    
    # Calculate edge statistics
    orig_edges = np.count_nonzero(original_adj)
    pruned_edges = np.count_nonzero(pruned_adj)
    edge_retention = pruned_edges / orig_edges if orig_edges > 0 else 0
    
    print(f"Original edges: {orig_edges}")
    print(f"Pruned edges: {pruned_edges}")
    print(f"Edge retention: {edge_retention:.2%}")
    print(f"Edges removed: {orig_edges - pruned_edges} ({(1-edge_retention):.2%})")
    
    # Visualize edge weight distribution for both
    plt.figure(figsize=(12, 6))
    
    # Original weights - using red color
    orig_weights = original_adj[original_adj > 0]
    plt.subplot(1, 2, 1)
    plt.hist(orig_weights, bins=30, alpha=0.7, color='darkred')
    plt.title("Original Edge Weights")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    # Pruned weights - using lighter red color
    pruned_weights = pruned_adj[pruned_adj > 0]
    plt.subplot(1, 2, 2)
    plt.hist(pruned_weights, bins=30, alpha=0.7, color='red')
    plt.title("Pruned Edge Weights")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_weight_comparison.png"))
    plt.close()
    
    # Calculate node importance based on weight change
    orig_weight_sum = np.sum(original_adj, axis=1)
    pruned_weight_sum = np.sum(pruned_adj, axis=1)
    
    # Normalize both for comparison
    if np.max(orig_weight_sum) > 0:
        orig_weight_sum = orig_weight_sum / np.max(orig_weight_sum)
    if np.max(pruned_weight_sum) > 0:
        pruned_weight_sum = pruned_weight_sum / np.max(pruned_weight_sum)
    
    # Difference in importance (negative means loss of importance)
    weight_diff = pruned_weight_sum - orig_weight_sum
    
    # Visualize difference in node importance with red gradients
    plt.figure(figsize=(10, 6))
    plt.hist(weight_diff, bins=30, color='red', alpha=0.7)
    plt.title("Change in Node Importance (Pruned - Original)")
    plt.xlabel("Weight Difference")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{file_name}_importance_diff.png"))
    plt.close()
    
    print(f"✅ Comparison saved in {output_dir}")
    return weight_diff

def overlay_edge_weight_heatmap(wsi_path, patch_info_path, adj_path, 
                              output_dir="edge_weight_heatmap", adj_key='pruned_adj'):
    """
    Overlay graph edge weights as a heatmap on WSI with JET colormap.
    
    Parameters:
    - wsi_path: Path to the whole slide image
    - patch_info_path: Path to file containing patch coordinates
    - adj_path: Path to adjacency matrix
    - output_dir: Directory to save output visualizations
    - adj_key: Key to use when loading adjacency matrix from file
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filename from WSI path
    file_name = os.path.basename(wsi_path).split('.')[0]
    
    # Load adjacency matrix
    adj_matrix = get_adj_from_file(adj_path, key=adj_key)
    
    # Visualize edge weight distribution
    visualize_edge_weight_distribution(adj_matrix, 
                                     os.path.join(output_dir, f"{file_name}_weight_distribution.png"))
    
    # Calculate importance values based on edge weights
    print("Calculating importance based on edge weights...")
    importance_values = calculate_edge_weight_importance(adj_matrix)
    
    # Read patch information
    patch_info = []
    with open(patch_info_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                x, y = line.split('\t')
            else:
                # Handle other potential formats
                parts = line.split()
                if len(parts) >= 2:
                    x, y = parts[0], parts[1]
                else:
                    continue
            patch_info.append(f"{x}_{y}.jpeg")
    
    # Load WSI with higher resolution for better visualization
    ori = openslide.OpenSlide(wsi_path)
    width, height = ori.dimensions
    
    # Calculate dimensions for visualization - higher resolution for better detail
    w, h = int(width/512), int(height/512)
    w_r, h_r = int(width/10), int(height/10)  # Higher resolution for better detail
    resized_img = ori.get_thumbnail((w_r, h_r))
    resized_img = resized_img.resize((w_r, h_r))
    w_s, h_s = float(512/10), float(512/10)  # Adjusted accordingly
    
    # Convert to numpy array and make sure it's uint8
    output_img = np.asarray(resized_img)[:,:,::-1].copy()
    output_img_uint8 = np.uint8(output_img)
    
    # Create grayscale version for mask
    gray = cv2.cvtColor(output_img_uint8, cv2.COLOR_BGR2GRAY)
    
    # Create importance mask with enhanced values
    print(f"Creating mask with {len(importance_values)} importance values for {len(patch_info)} patches...")
    mask = importance_to_mask(gray, patch_info, importance_values, w, h, w_s, h_s)
    
    # Apply Gaussian blur to smooth the mask - more aggressive smoothing for better results
    # Use a very large kernel for the smooth gradient effect
    smoothed_mask = cv2.GaussianBlur(mask, (101, 101), 30)
    
    # Normalize the mask to 0-255 range
    if np.max(smoothed_mask) > 0:
        smoothed_mask_norm = smoothed_mask / np.max(smoothed_mask) * 255
    else:
        smoothed_mask_norm = smoothed_mask * 255
        
    # Convert to uint8
    smoothed_mask_uint8 = np.uint8(smoothed_mask_norm)
    
    # Apply the JET colormap like in the original code, but with our smoother mask
    jet_colormap = create_jet_style_colormap()
    heatmap = cv2.applyColorMap(smoothed_mask_uint8, jet_colormap)
    
    # Create a mask for the slide area (non-background)
    _, tissue_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    tissue_mask = cv2.dilate(tissue_mask, np.ones((5,5), np.uint8), iterations=3)
    
    # Use addWeighted for proper alpha blending
    alpha = 0.6  # Opacity of the overlay
    beta = 1.0 - alpha  # Weight of the original image
    
    # Apply the blending
    vis_uint8 = cv2.addWeighted(output_img_uint8, beta, heatmap, alpha, 0)
    
    # Then apply the tissue mask to only keep the heatmap on tissue areas
    tissue_mask_float = tissue_mask.astype(float) / 255
    tissue_mask_float_3ch = np.stack([tissue_mask_float]*3, axis=2)
    
    # Create a dark blue background for non-tissue areas
    bg = np.zeros_like(output_img_uint8)
    bg[:,:,0] = 100  # Blue channel
    
    # Blend tissue with heatmap and background
    final_vis = vis_uint8 * tissue_mask_float_3ch + bg * (1 - tissue_mask_float_3ch)
    final_vis = np.uint8(final_vis)
    
    # Save individual images
    cv2.imwrite(os.path.join(output_dir, f"{file_name}_original.png"), output_img_uint8, 
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(os.path.join(output_dir, f"{file_name}_edge_weight_heatmap.png"), final_vis, 
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # Save the colormap as a reference
    colormap_vis = np.zeros((50, 256, 3), dtype=np.uint8)
    for i in range(256):
        colormap_vis[:, i, :] = jet_colormap[i, 0, :]
    cv2.imwrite(os.path.join(output_dir, f"{file_name}_colormap.png"), colormap_vis,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # For visualization of both images side by side, create a simple combination
    h, w, c = output_img_uint8.shape
    
    # Create a new image large enough to hold both
    # If image is wider than tall, place side by side
    if w > h:
        combined = np.zeros((h, 2*w, c), dtype=np.uint8)
        combined[:, 0:w, :] = output_img_uint8
        combined[:, w:2*w, :] = final_vis
    # If image is taller than wide, place one above the other
    else:
        combined = np.zeros((2*h, w, c), dtype=np.uint8)
        combined[0:h, :, :] = output_img_uint8
        combined[h:2*h, :, :] = final_vis
    
    # Save the combined visualization
    cv2.imwrite(os.path.join(output_dir, f"{file_name}_combined.png"), combined, 
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    print(f"✅ JET colormap heatmap saved in {output_dir}")
    return importance_values

def main():
    parser = argparse.ArgumentParser(description='LENS Edge Weight Heatmap Visualization')
    
    # Required paths
    parser.add_argument('--wsi-path', type=str, required=True,
                        help='Path to whole slide image (.svs file)')
    parser.add_argument('--patch-info-path', type=str, required=True,
                        help='Path to patch coordinates file (c_idx.txt)')
    parser.add_argument('--pruned-adj-path', type=str, required=True,
                        help='Path to pruned adjacency matrix file (.pt)')
    
    # Optional paths
    parser.add_argument('--original-adj-path', type=str, default=None,
                        help='Path to original adjacency matrix file (.pt) for comparison')
    
    # Output directories
    parser.add_argument('--output-dir', type=str, default='heatmap_results',
                        help='Base output directory for all results')
    parser.add_argument('--comparison-dir', type=str, default=None,
                        help='Output directory for comparison plots (default: output-dir/comparison)')
    parser.add_argument('--pruned-heatmap-dir', type=str, default=None,
                        help='Output directory for pruned heatmap (default: output-dir/pruned_heatmap)')
    parser.add_argument('--original-heatmap-dir', type=str, default=None,
                        help='Output directory for original heatmap (default: output-dir/original_heatmap)')
    
    # Matrix keys
    parser.add_argument('--original-key', type=str, default='adj_s',
                        help='Key for original adjacency matrix in .pt file')
    parser.add_argument('--pruned-key', type=str, default='pruned_adj',
                        help='Key for pruned adjacency matrix in .pt file')
    
    # Analysis options
    parser.add_argument('--skip-comparison', action='store_true',
                        help='Skip comparison between original and pruned matrices')
    parser.add_argument('--skip-original-heatmap', action='store_true',
                        help='Skip generating heatmap for original adjacency matrix')
    
    args = parser.parse_args()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default subdirectories if not specified
    if args.comparison_dir is None:
        args.comparison_dir = os.path.join(args.output_dir, 'comparison')
    if args.pruned_heatmap_dir is None:
        args.pruned_heatmap_dir = os.path.join(args.output_dir, 'pruned_heatmap')
    if args.original_heatmap_dir is None:
        args.original_heatmap_dir = os.path.join(args.output_dir, 'original_heatmap')
    
    print(f"Starting LENS edge weight visualization...")
    print(f"WSI: {args.wsi_path}")
    print(f"Patch info: {args.patch_info_path}")
    print(f"Pruned adjacency: {args.pruned_adj_path}")
    if args.original_adj_path:
        print(f"Original adjacency: {args.original_adj_path}")
    
    # Step 1: Compare original vs pruned adjacency matrices (if both provided)
    if args.original_adj_path and not args.skip_comparison:
        print("\n--- Comparing original and pruned adjacency matrices ---")
        weight_diff = compare_original_pruned(
            original_adj_path=args.original_adj_path,
            pruned_adj_path=args.pruned_adj_path,
            original_key=args.original_key,
            pruned_key=args.pruned_key,
            output_dir=args.comparison_dir
        )
    
    # Step 2: Generate edge weight heatmap for pruned adjacency
    print("\n--- Creating edge weight heatmap for pruned adjacency ---")
    pruned_importance = overlay_edge_weight_heatmap(
        wsi_path=args.wsi_path,
        patch_info_path=args.patch_info_path,
        adj_path=args.pruned_adj_path,
        output_dir=args.pruned_heatmap_dir,
        adj_key=args.pruned_key
    )
    
    # Step 3: Generate edge weight heatmap for original adjacency (if provided)
    if args.original_adj_path and not args.skip_original_heatmap:
        print("\n--- Creating edge weight heatmap for original adjacency ---")
        original_importance = overlay_edge_weight_heatmap(
            wsi_path=args.wsi_path,
            patch_info_path=args.patch_info_path,
            adj_path=args.original_adj_path,
            output_dir=args.original_heatmap_dir,
            adj_key=args.original_key
        )
    
    print(f"\n✅ Analysis complete! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
