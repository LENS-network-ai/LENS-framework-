# L0Utils_ARM.py
"""
ARM (Augment-REINFORCE-Merge) implementation for L0 regularization
Reference: Yin & Zhou (2019) - ARM: Augment-REINFORCE-Merge Gradient
"""

import torch
import torch.nn as nn
import numpy as np


class ARML0RegularizerParams:
    """Parameters for ARM-based L0 regularization"""
    
    def __init__(self, baseline_ema=0.9, eps=1e-20):
        """
        Args:
            ]
            baseline_ema: Exponential moving average coefficient for baseline
            eps: Small constant for numerical stability
        """
        
        self.baseline_ema = baseline_ema
        self.eps = eps
        
        # Running baseline for variance reduction
        self.baseline = None
        
        # Storage for backward pass
        self.last_pi = None
        self.last_b = None
        self.last_b_anti = None


def arm_sample_gates(logAlpha, params, training=True):
    """
    Sample binary gates using ARM method
    
    Args:
        logAlpha: Edge logits (log of alpha)
        params: ARML0RegularizerParams instance
        training: If True, sample stochastically; else deterministic
    
    Returns:
        gates: Binary edge gates (0 or 1)
        gates_anti: Antithetic samples (for ARM gradient)
    """
    # Compute keep probability
    pi = torch.sigmoid(logAlpha)
    
    if training:
        # Sample binary gates
        u = torch.rand_like(pi)
        b = (u < pi).float()
        
        # Antithetic sample (control variate for variance reduction)
        b_anti = 1.0 - b
        
        # Store for gradient computation
        params.last_pi = pi
        params.last_b = b
        params.last_b_anti = b_anti
        
        return b, b_anti
    else:
        # Deterministic at test time
        b = (pi > 0.5).float()
        return b, None


def compute_arm_loss(loss_b, loss_b_anti, logAlpha, params):
    """
    Compute ARM gradient estimate
    
    Args:
        loss_b: Task loss with sampled gates
        loss_b_anti: Task loss with antithetic gates
        logAlpha: Edge logits
        params: ARML0RegularizerParams instance
    
    Returns:
        arm_loss: Differentiable loss for ARM gradient
    """
    pi = params.last_pi
    b = params.last_b
    
    # Difference between loss with b and b_anti (control variate)
    f_diff = loss_b - loss_b_anti
    
    # Update running baseline (exponential moving average)
    if params.baseline is None:
        params.baseline = f_diff.detach()
    else:
        params.baseline = (params.baseline_ema * params.baseline + 
                          (1 - params.baseline_ema) * f_diff.detach())
    
    # Center the difference using baseline
    centered_diff = f_diff - params.baseline
    
    # ARM gradient: ∇_θ E[f(b)] ≈ (f(b) - f(b̃) - baseline) * (b - π)
    # We create a "pseudo-loss" that gives correct gradients when differentiated
    pseudo_loss = (centered_diff.detach() * (b - pi.detach()) * logAlpha).sum()
    
    return pseudo_loss


def get_expected_l0_arm(logAlpha, params):
    """
    Compute expected L0 penalty (same formula as Hard-Concrete)
    
    Args:
        logAlpha: Edge logits
        params: ARML0RegularizerParams instance
    
    Returns:
        Expected number of non-zero edges
    """
    pi = torch.sigmoid(logAlpha)
    return pi.sum()


# Create default global instance
arm_params = ARML0RegularizerParams()
