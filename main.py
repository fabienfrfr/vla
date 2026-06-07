"""
Variational Linear Attention (VLA)
==================================
Author: Fabien Furfaro

This module implements VLA, a memory-efficient attention mechanism that treats
latent state updates as a Recursive Least Squares (RLS) problem rather than 
simple stochastic gradient descent.

Comparison with DeltaNet:
-------------------------
1. Delta Rule (Widrow-Hoff / DeltaNet):
   Formula: ΔS = η * (y - S*k_hat) * k_hat^T
   - Uses a fixed learning rate 'η'. 
   - Simple but rigid; prone to information crosstalk.

2. VLA (Recursive Least Squares / Adaptive):
   Formula: ΔS = residual * (A * k_hat)^T
   - Uses an adaptive matrix learning rate 'A' (inverse covariance).
   - Dynamically orthogonalizes memory; optimizes for stable retention.
"""

import torch
import torch.nn.functional as F

def vla_sequential_forward(x, weight_q, weight_k, weight_v, weight_u, 
                           lambda0=0.1, epsilon=1e-4, period=20, eta=1e-3):
    """
    Sequential forward pass of Variational Linear Attention (VLA) v3.
    x: [B, T, D] - Input sequence
    weight_q, weight_k, weight_v, weight_u: Projection weight matrices
    """
    B, T, D = x.shape
    dh = weight_q.shape[0]
    device = x.device
    
    # Memory state S [B, dh, dh] and penalty matrix A [B, dh, dh]
    memory_s = torch.zeros(B, dh, dh, device=device)
    penalty_a = (1.0 / lambda0) * torch.eye(dh, device=device).unsqueeze(0).repeat(B, 1, 1)
    key_accumulator = torch.zeros(B, dh, device=device)
    
    outputs = []
    
    for t in range(T):
        xt = x[:, t, :] # [B, D]
        
        # --- Feature Mapping ---
        k_raw = xt @ weight_k.T # [B, dh]
        k_feat = F.elu(k_raw) + 1.0
        k_hat = k_feat / (k_feat.norm(dim=1, keepdim=True) + 1e-6) # [B, dh]
        
        # Penalty direction (Wu * k_raw)
        u = F.normalize(k_raw @ weight_u.T, p=2, dim=1) / (dh**0.5) # [B, dh]
        
        # --- Sherman-Morrison update for A ---
        # A acts as the inverse covariance matrix, tracking the 
        # 'uncertainty' of the latent memory directions (Kalman Gain logic).
        z_sm = torch.einsum('bij, bj -> bi', penalty_a, u) # [B, dh]
        delta = torch.clamp(1.0 + torch.einsum('bi, bi -> b', u, z_sm), min=epsilon) # [B]
        penalty_a = penalty_a - torch.einsum('bi, bj -> bij', z_sm, z_sm) / delta.view(-1, 1, 1)
        
        # Periodic refresh to prevent eigenvalue drift
        if (t + 1) % period == 0:
            penalty_a = penalty_a + eta * torch.eye(dh, device=device).unsqueeze(0)
            
        # --- Residual Memory S update ---
        alpha = torch.einsum('bij, bj -> bi', penalty_a, k_hat) # [B, dh]
        alpha_hat = alpha / (alpha.norm(dim=1, keepdim=True) + 1e-6)
        
        residual = (xt @ weight_v.T) - torch.einsum('bij, bj -> bi', memory_s, k_hat)
        memory_s = memory_s + torch.einsum('bi, bj -> bij', residual, alpha_hat)
        
        # --- Output Calculation ---
        query = F.elu(xt @ weight_q.T) + 1.0 # [B, dh]
        key_accumulator = key_accumulator + k_feat
        
        output = (
            torch.einsum('bij, bj -> bi', memory_s, query) / 
            torch.clamp(torch.einsum('bi, bi -> b', key_accumulator, query), min=epsilon).unsqueeze(1)
        )
        outputs.append(output)
        
    return torch.stack(outputs, dim=1) # [B, T, dh]
    

def vla_semi_parallel_s_update(hat_k, hat_alpha, value_raw):
    """
    Computes the associative scan for the Memory state S.
    hat_k: [T, dh]
    hat_alpha: [T, dh]
    value_raw: [T, dh]
    """
    seq_len, hidden_dim = hat_k.shape
    
    # 1. Calculate local operators (F_t, G_t)
    # F_t = I - hat_alpha @ hat_k.T
    # G_t = residual_t @ hat_alpha.T
    identity = torch.eye(hidden_dim, device=hat_k.device)
    f_operators = identity - torch.einsum('ti, tj -> tij', hat_alpha, hat_k)
    g_operators = torch.einsum('ti, tj -> tij', value_raw, hat_alpha)
    
    # 2. Iterative scan (Associative property: (Fr, Gr) o (Fl, Gl) = (Fr Fl, Fr Gl + Gr))
    s_states = torch.zeros(seq_len, hidden_dim, hidden_dim, device=hat_k.device)
    curr_f, curr_g = f_operators[0], g_operators[0]
    s_states[0] = curr_g
    
    for t in range(1, seq_len):
        curr_f = f_operators[t] @ curr_f
        curr_g = f_operators[t] @ curr_g + g_operators[t]
        s_states[t] = curr_g
        
    return s_states