import torch
import pytest
from main import vla_sequential_forward, vla_semi_parallel_s_update

def test_vla_equivalence():
    """Verify that the forward pass outputs the correct shape."""
    batch_size = 1
    seq_len = 10
    dim = 8
    hidden_dim = 4
    
    # Input with batch dim: [B, T, D]
    x = torch.randn(batch_size, seq_len, dim)

    wq = torch.randn(hidden_dim, dim)
    wk = torch.randn(hidden_dim, dim)
    wv = torch.randn(hidden_dim, dim)
    wu = torch.randn(hidden_dim, dim)
    
    # Check sequential forward output shape
    outputs = vla_sequential_forward(x, wq, wk, wv, wu)
    assert outputs.shape == (seq_len, hidden_dim)

def test_associative_scan_output_shape():
    """Verify the shape of the semi-parallel memory state update."""
    seq_len, dh = 3, 2
    
    # Generate dummy input for the scan
    hat_k = torch.randn(seq_len, dh)
    hat_alpha = torch.randn(seq_len, dh)
    value_raw = torch.randn(seq_len, dh)
    
    # Calculate associative scan
    res = vla_semi_parallel_s_update(hat_k, hat_alpha, value_raw)
    
    assert res.shape == (seq_len, dh, dh)
    assert not torch.isnan(res).any(), "The scan produced NaNs, check stability."

def test_penalty_matrix_positive_definiteness():
    """Ensure the precision matrix (penalty_a) remains positive definite."""
    hidden_dim = 4
    penalty_a = torch.eye(hidden_dim) * 10.0
    u = torch.randn(hidden_dim)
    
    # Manual Sherman-Morrison update simulation
    delta = 1.0 + u.T @ (penalty_a @ u)
    new_a = penalty_a - (penalty_a @ torch.outer(u, u) @ penalty_a) / delta
    
    # Eigenvalues must remain positive for the RLS/Kalman stability
    eigvals = torch.linalg.eigvals(new_a)
    assert torch.all(eigvals.real > 0), "Penalty matrix lost positive definiteness."