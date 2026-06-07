import torch
import torch.nn.functional as F

def vla_sequential_forward(x, weight_q, weight_k, weight_v, weight_u, 
                           lambda0=0.1, epsilon=1e-4, period=20, eta=1e-3):
    """
    Sequential forward pass of Variational Linear Attention (VLA) v3.
    x: [T, D] - Input sequence
    weight_q, weight_k, weight_v, weight_u: Projection weight matrices
    """
    seq_len, dim = x.shape
    hidden_dim = weight_q.shape[0]  # Hidden dimension (dh)
    
    # Memory state S and penalty matrix A
    memory_s = torch.zeros(hidden_dim, hidden_dim, device=x.device)
    penalty_a = (1.0 / lambda0) * torch.eye(hidden_dim, device=x.device)
    key_accumulator = torch.zeros(hidden_dim, device=x.device)
    
    outputs = []
    
    for t in range(seq_len):
        xt = x[t]
        
        # --- Feature Mapping ---
        k_raw = weight_k @ xt
        k_feat = F.elu(k_raw) + 1.0
        k_hat = k_feat / (k_feat.norm() + 1e-6)
        
        # Penalty direction (Wu * k_raw)
        u = F.normalize(weight_u @ k_raw, p=2, dim=0) / (hidden_dim**0.5)
        
        # --- Sherman-Morrison update for A ---
        z_sm = penalty_a @ u
        delta = torch.clamp(1.0 + u.T @ z_sm, min=epsilon)
        penalty_a = penalty_a - (z_sm.ger(z_sm) / delta)
        
        # Periodic refresh to prevent eigenvalue drift
        if (t + 1) % period == 0:
            penalty_a = penalty_a + eta * torch.eye(hidden_dim, device=x.device)
            
        # --- Residual Memory S update ---
        alpha = penalty_a @ k_hat
        alpha_hat = alpha / (alpha.norm() + 1e-6)
        
        residual = (weight_v @ xt) - (memory_s @ k_hat)
        memory_s = memory_s + residual.ger(alpha_hat)
        
        # --- Output Calculation ---
        query = F.elu(weight_q @ xt) + 1.0
        key_accumulator = key_accumulator + k_feat
        
        output = (memory_s @ query) / torch.clamp(key_accumulator.T @ query, min=epsilon)
        outputs.append(output)
        
    return torch.stack(outputs)


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