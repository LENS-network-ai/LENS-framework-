import os
import torch
import gc
from helper import Trainer, Evaluator, preparefeatureLabel

def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, device, 
                      num_epochs, n_features, output_dir, warmup_epochs=5):
    """Train and evaluate a model with detailed graph analysis"""
    # Initialize trainer and evaluator
    trainer = Trainer(n_class=model.num_classes)
    evaluator = Evaluator(n_class=model.num_classes)
    
    # Add an output directory for pruned adjacencies
    adj_output_dir = os.path.join(output_dir, 'pruned_adjacencies')
    os.makedirs(adj_output_dir, exist_ok=True)
    
    # Track metrics
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    best_val_acc = 0.0
    best_epoch = 0
    best_edge_sparsity = 0.0
    
    # Create directories for analysis outputs
    analysis_dir = os.path.join(output_dir, 'graph_analysis')
    edge_dist_dir = os.path.join(output_dir, 'edge_distributions')
    report_dir = os.path.join(output_dir, 'sparsification_reports')
    
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(edge_dist_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        train_acc, train_loss = train_epoch(
            epoch, model, train_loader, optimizer, scheduler, 
            trainer, n_features, num_epochs, warmup_epochs,
            analysis_dir, edge_dist_dir
        )
        
        train_accs.append(float(train_acc))
        train_losses.append(float(train_loss))
        
        torch.cuda.empty_cache()
        
        # Validation phase
        val_acc, val_loss, edge_sparsity = validate_epoch(
            model, val_loader, evaluator, n_features
        )
        
        val_accs.append(float(val_acc))
        val_losses.append(float(val_loss))
        
        print(f"   🔹 Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")
        
        # Save comprehensive sparsification report at end of epoch
        if hasattr(model, 'save_sparsification_report'):
            try:
                report_path = model.save_sparsification_report(
                    epoch + 1,
                    save_dir=report_dir
                )
                if report_path:  # Check if report_path is not None
                    print(f"   📊 Sparsification report saved: {os.path.basename(report_path)}")
                else:
                    print(f"   📊 Sparsification report generated")
            except Exception as e:
                print(f"   ⚠️ Error generating sparsification report: {str(e)}")

        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_edge_sparsity = edge_sparsity
            best_epoch = epoch + 1
            
            # Save model
            model_path = os.path.join(output_dir, f'best_model_epoch{best_epoch}.pt')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy': best_val_acc,
                'edge_sparsity': best_edge_sparsity,
            }, model_path)
            
            print(f"   ✅ New Best Model! Accuracy: {best_val_acc:.4f}, Sparsity: {best_edge_sparsity:.1f}% edges > 0.1")
    
    # Save pruned adjacencies
    save_pruned_adjacencies(trainer, adj_output_dir)
    
    return {
        "train_accs": train_accs,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "val_losses": val_losses,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_edge_sparsity": best_edge_sparsity
    }

def train_epoch(epoch, model, train_loader, optimizer, scheduler, trainer, 
               n_features, num_epochs, warmup_epochs, analysis_dir, edge_dist_dir):
    """Run one epoch of training"""
    print(f"\n🚀 Epoch {epoch+1}/{num_epochs}")
    
    # Set the current epoch for the model
    model.set_epoch(epoch)
    
    # Print current temperature and lambda/beta (check both attributes for backward compatibility)
    if hasattr(model, 'temperature'):
        temp_str = f"   🌡️ Temperature: {model.temperature:.4f}"
        
        # Check for lambda parameter (new name)
        if hasattr(model.regularizer, 'current_lambda'):
            lambda_val = model.regularizer.current_lambda
            print(f"{temp_str}, λ: {lambda_val:.6f}")
        # Fallback to beta parameter (old name) for backward compatibility
        elif hasattr(model.regularizer, 'current_beta'):
            beta_val = model.regularizer.current_beta
            print(f"{temp_str}, β: {beta_val:.6f}")
        else:
            print(temp_str)
    
    # Print warmup status
    if epoch < warmup_epochs:
        progress = 100 * (epoch / warmup_epochs)
        print(f"   🔹 Warmup: {progress:.1f}% complete ({epoch+1}/{warmup_epochs} epochs)")
    else:
        print(f"   🔹 Warmup completed. Full sparsification active.")
    
    # Print the current average edge weight
    if hasattr(model, 'edge_stats') and len(model.edge_stats) > 0:
        print(f"   📊 Current avg edge weight: {model.edge_stats[-1]:.6f}")
    
    # Training phase
    model.train()
    train_loss = 0.0
    trainer.reset_metrics()
    
    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Only print stats occasionally
        if hasattr(model, 'set_print_stats'):
            model.set_print_stats(batch_idx % 50 == 0)
        
        try:
            # Use the Trainer's train method (which calls the model's forward method)
            pred, labels, loss, weighted_adj = trainer.train(sample, model, n_features=n_features)
            
            # Save detailed graph analysis and edge weight distribution periodically
            if batch_idx % 50 == 0:
                # Save graph analysis
                if hasattr(model, 'save_graph_analysis'):
                    model.save_graph_analysis(
                        epoch + 1, 
                        batch_idx, 
                        save_dir=analysis_dir
                    )
                
                # Plot edge weight distribution
                if hasattr(model, 'plot_edge_weight_distribution'):
                    model.plot_edge_weight_distribution(
                        weighted_adj,
                        epoch + 1,
                        batch_idx,
                        save_dir=edge_dist_dir
                    )
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Warning: Loss is {loss.item()} in batch {batch_idx}, skipping")
                continue
            
            if batch_idx % 20 == 0:
                print(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler(optimizer, batch_idx, epoch, 0)
            train_loss += loss.item()
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"⚠️ CUDA OOM in batch {batch_idx}. Skipping and clearing cache.")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e
    
    train_acc = trainer.get_scores()
    avg_train_loss = train_loss / max(1, len(train_loader))
    print(f"   🔹 Training Accuracy: {train_acc:.4f}, Loss: {avg_train_loss:.4f}")
    
    return train_acc, avg_train_loss

def validate_epoch(model, val_loader, evaluator, n_features):
    """Run one epoch of validation"""
    model.eval()
    val_loss = 0.0
    evaluator.reset_metrics()

    # Make sure we don't print stats during validation
    if hasattr(model, 'set_print_stats'):
        model.set_print_stats(False)

    # Track edge sparsity during validation
    all_edge_weights = []

    with torch.no_grad():
        for sample in val_loader:
            try:
                # Use the Evaluator's eval_test method (which calls preparefeatureLabel)
                pred, labels, loss, weighted_adj = evaluator.eval_test(sample, model, n_features=n_features)
                val_loss += loss.item()
                
                # Get the original node_feat, labels, adjs, masks used by the model
                node_feat, labels, adjs, masks = preparefeatureLabel(
                    sample['image'], sample['label'], sample['adj_s'], n_features=n_features
                )
                
                # Now calculate edge sparsity using the tensor versions
                edge_mask = (adjs > 0).float()
                # Only look at the weighted adjacency where there are actual edges
                masked_weights = torch.zeros_like(weighted_adj)
                for i in range(adjs.shape[0]):
                    mask = adjs[i] > 0
                    if mask.sum() > 0:
                        masked_weights[i, mask] = weighted_adj[i, mask] / adjs[i, mask]
                
                all_edge_weights.append(masked_weights.detach().cpu())
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️ CUDA OOM in validation. Skipping and clearing cache.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
                    
    val_acc = evaluator.get_scores()
    avg_val_loss = val_loss / max(1, len(val_loader))
    
    # Calculate edge sparsity metrics
    edge_sparsity = calculate_edge_sparsity(all_edge_weights)
    
    return val_acc, avg_val_loss, edge_sparsity

def calculate_edge_sparsity(all_edge_weights):
    """Calculate edge sparsity from edge weights"""
    if all_edge_weights:
        # Concatenate all edge weights
        all_weights = torch.cat([w.flatten() for w in all_edge_weights])
        all_weights = all_weights[all_weights > 0]  # Only consider positive weights
        
        if len(all_weights) > 0:
            # Calculate various sparsity metrics
            avg_weight = all_weights.mean().item()
            median_weight = all_weights.median().item()
            
            # Count weights above thresholds
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
            sparsity_metrics = {}
            for t in thresholds:
                sparsity_metrics[f">{t}"] = (all_weights > t).float().mean().item() * 100.0
            
            # Calculate sparsity score (percentage of weights > 0.1)
            edge_sparsity = sparsity_metrics[">0.1"]
            
            # Print sparsity metrics
            print(f"   📊 Edge weight stats - Mean: {avg_weight:.6f}, Median: {median_weight:.6f}")
            print(f"   📊 Edge sparsity - {sparsity_metrics['>0.01']:.1f}% > 0.01, {sparsity_metrics['>0.1']:.1f}% > 0.1, {sparsity_metrics['>0.5']:.1f}% > 0.5")
            
            return edge_sparsity
    
    return 0.0

def save_pruned_adjacencies(trainer, adj_output_dir):
    """Save pruned adjacency matrices"""
    if hasattr(trainer, 'saved_pruned_adjs') and trainer.saved_pruned_adjs:
        print(f"Saving {len(trainer.saved_pruned_adjs)} pruned adjacency matrices...")
        
        # Create the directory for adjacencies if it doesn't exist
        os.makedirs(adj_output_dir, exist_ok=True)
        
        # Save each WSI's pruned adjacency matrix separately
        for wsi_id, pruned_adj in trainer.saved_pruned_adjs.items():
            # Clean the WSI ID to make it a valid filename
            clean_id = ''.join(c for c in wsi_id if c.isalnum() or c in '._- ')
            
            # Save in torch format
            adj_path = os.path.join(adj_output_dir, f"{clean_id}_pruned_adj.pt")
            torch.save({
                'pruned_adj': pruned_adj,
                'original_edges': trainer.original_edges.get(wsi_id, 0),
                'wsi_id': wsi_id
            }, adj_path)
        
        print(f"Pruned adjacencies saved to {adj_output_dir}")
