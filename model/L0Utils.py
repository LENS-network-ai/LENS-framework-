"""
L0 Regularization with Hard-Concrete Gating and Straight-Through Estimator (STE)

This module provides L0 regularization functions for neural networks with support for:
1. Standard Hard-Concrete (stochastic, continuous outputs)
2. Hard-Concrete + STE (stochastic sampling, binary outputs)
3. Hard-Concrete + STE Deterministic (no sampling, binary outputs)

Reference:
    Louizos et al. (2018) "Learning Sparse Neural Networks through L0 Regularization"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class L0RegularizerParams:
    """
    Class to hold configurable parameters for L0 regularization
    
    Parameters:
        gamma (float): Lower stretch parameter (default: -0.1)
        zeta (float): Upper stretch parameter (default: 1.1)
        beta_l0 (float): Temperature parameter (default: 0.66)
        beta_ste (float): Threshold for STE hard decision (default: 0.5)
        eps (float): Small constant for numerical stability (default: 1e-20)
    """
    def __init__(self, gamma=-0.1, zeta=1.1, beta_l0=0.66, beta_ste=0.5, eps=1e-20):
        self.sig = nn.Sigmoid()
        self.gamma = gamma
        self.zeta = zeta
        self.beta_l0 = beta_l0
        self.beta_ste = beta_ste  # Threshold for STE
        self.eps = eps
        self._update_const1()
    
    def _update_const1(self):
        """Update const1 when parameters change"""
        self.const1 = self.beta_l0 * np.log(-self.gamma / self.zeta + self.eps)
    
    def update_params(self, gamma=None, zeta=None, beta_l0=None, beta_ste=None):
        """Update parameters and recalculate dependent values"""
        if gamma is not None:
            self.gamma = gamma
        if zeta is not None:
            self.zeta = zeta
        if beta_l0 is not None:
            self.beta_l0 = beta_l0
        if beta_ste is not None:
            self.beta_ste = beta_ste
        self._update_const1()
        return self
    
    def to_dict(self):
        """Convert parameters to dictionary"""
        return {
            'gamma': self.gamma,
            'zeta': self.zeta,
            'beta_l0': self.beta_l0,
            'beta_ste': self.beta_ste,
            'eps': self.eps
        }


# Create a global default instance
l0_params = L0RegularizerParams()


# ============================================================
# STANDARD HARD-CONCRETE (Original)
# ============================================================

def l0_train(logAlpha, min_val=0, max_val=1, params=None):
    """
    L0 regularization function for training - uses stochastic gates
    Standard Hard-Concrete with continuous outputs in [0, 1]
    
    Args:
        logAlpha (torch.Tensor): Log-odds of gate being open
        min_val (float): Minimum value for clipping (default: 0)
        max_val (float): Maximum value for clipping (default: 1)
        params (L0RegularizerParams): Parameter object (default: global l0_params)
        
    Returns:
        torch.Tensor: Continuous gate values in [min_val, max_val]
    """
    if params is None:
        params = l0_params
    
    # Sample uniform noise
    U = torch.rand(logAlpha.size()).type_as(logAlpha) + params.eps
    
    # Add Gumbel noise and pass through sigmoid
    s = params.sig((torch.log(U / (1 - U)) + logAlpha) / params.beta_l0)
    
    # Stretch to [gamma, zeta]
    s_bar = s * (params.zeta - params.gamma) + params.gamma
    
    # Clip to [min_val, max_val]
    mask = F.hardtanh(s_bar, min_val, max_val)
    
    return mask


def l0_test(logAlpha, min_val=0, max_val=1, params=None):
    """
    L0 regularization function for testing - deterministic version
    Uses mean (expected value) without sampling
    
    Args:
        logAlpha (torch.Tensor): Log-odds of gate being open
        min_val (float): Minimum value for clipping (default: 0)
        max_val (float): Maximum value for clipping (default: 1)
        params (L0RegularizerParams): Parameter object (default: global l0_params)
        
    Returns:
        torch.Tensor: Deterministic gate values in [min_val, max_val]
    """
    if params is None:
        params = l0_params
    
    # Use mean (no sampling)
    s = params.sig(logAlpha / params.beta_l0)
    
    # Stretch to [gamma, zeta]
    s_bar = s * (params.zeta - params.gamma) + params.gamma
    
    # Clip to [min_val, max_val]
    mask = F.hardtanh(s_bar, min_val, max_val)
    
    return mask


# ============================================================
# HARD-CONCRETE + STE (Stochastic)
# ============================================================

def l0_train_ste(logAlpha, min_val=0, max_val=1, params=None, threshold=None):
   
    if params is None:
        params = l0_params
    
    if threshold is None:
        threshold = params.beta_ste
    
    # Step 1-4: Sample continuous gates (same as l0_train)
    U = torch.rand(logAlpha.size()).type_as(logAlpha) + params.eps
    s = params.sig((torch.log(U / (1 - U)) + logAlpha) / params.beta_l0)
    s_bar = s * (params.zeta - params.gamma) + params.gamma
    z_soft = F.hardtanh(s_bar, min_val, max_val)
    
    # Step 5: Hard threshold to binary
    z_hard = (z_soft > threshold).float()
    
    # Step 6: Straight-Through Estimator trick
    # Forward: uses z_hard (binary)
    # Backward: gradients flow through z_soft (continuous)
    mask = z_hard.detach() + z_soft - z_soft.detach()
    
    return mask


def l0_train_ste_deterministic(logAlpha, min_val=0, max_val=1, params=None, threshold=None):

    if params is None:
        params = l0_params
    
    if threshold is None:
        threshold = params.beta_ste
    
    # Deterministic (no sampling) - use mean
    s = params.sig(logAlpha / params.beta_l0)
    s_bar = s * (params.zeta - params.gamma) + params.gamma
    z_soft = F.hardtanh(s_bar, min_val, max_val)
    
    # Hard threshold to binary
    z_hard = (z_soft > threshold).float()
    
    # Straight-Through Estimator trick
    mask = z_hard.detach() + z_soft - z_soft.detach()
    
    return mask


def l0_test_ste(logAlpha, min_val=0, max_val=1, params=None, threshold=None):
   
    if params is None:
        params = l0_params
    
    if threshold is None:
        threshold = params.beta_ste
    
    # Deterministic evaluation
    s = params.sig(logAlpha / params.beta_l0)
    s_bar = s * (params.zeta - params.gamma) + params.gamma
    z_soft = F.hardtanh(s_bar, min_val, max_val)
    
    # Hard threshold (no STE needed in eval mode)
    mask = (z_soft > threshold).float()
    
    return mask


def get_loss2(logAlpha, params=None):
    """
    Calculate the L0 regularization penalty
    
    Computes the expected number of non-zero parameters:
        L0 = sum(σ(logAlpha - log(-γ/ζ)))
    
    This is the same for all variants (standard, STE, etc.)
    
    Args:
        logAlpha (torch.Tensor): Log-odds of gate being open
        params (L0RegularizerParams): Parameter object (default: global l0_params)
        
    Returns:
        torch.Tensor: L0 penalty (expected number of non-zero gates)
    """
    if params is None:
        params = l0_params
    
    return params.sig(logAlpha - params.const1)



def l0_forward(logAlpha, training=True, use_ste=False, deterministic_ste=False, 
               min_val=0, max_val=1, params=None, threshold=None):
   
    if params is None:
        params = l0_params
    
    if use_ste:
        # STE modes
        if training:
            if deterministic_ste:
                return l0_train_ste_deterministic(logAlpha, min_val, max_val, params, threshold)
            else:
                return l0_train_ste(logAlpha, min_val, max_val, params, threshold)
        else:
            return l0_test_ste(logAlpha, min_val, max_val, params, threshold)
    else:
        # Standard Hard-Concrete
        if training:
            return l0_train(logAlpha, min_val, max_val, params)
        else:
            return l0_test(logAlpha, min_val, max_val, params)


def get_expected_l0_sparsity(logAlpha, params=None):
    """
    Compute expected L0 sparsity (proportion of gates that are open)
    
    Args:
        logAlpha (torch.Tensor): Log-odds of gate being open
        params (L0RegularizerParams): Parameter object
        
    Returns:
        float: Expected sparsity (0-1, where 1 means all gates open)
    """
    if params is None:
        params = l0_params
    
    prob_open = params.sig(logAlpha - params.const1)
    return prob_open.mean().item()


def get_hard_sparsity(logAlpha, threshold=0.5, params=None):
    
    if params is None:
        params = l0_params
    
    # Get deterministic gate values
    s = params.sig(logAlpha / params.beta_l0)
    s_bar = s * (params.zeta - params.gamma) + params.gamma
    z_soft = torch.clamp(s_bar, 0, 1)
    
    # Count gates above threshold
    gates_open = (z_soft > threshold).float().mean().item()
    
    return gates_open


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

# For backward compatibility - exported constants with original default values
gamma = l0_params.gamma
zeta = l0_params.zeta
beta_l0 = l0_params.beta_l0
beta_ste = l0_params.beta_ste
eps = l0_params.eps
sig = l0_params.sig
const1 = l0_params.const1
