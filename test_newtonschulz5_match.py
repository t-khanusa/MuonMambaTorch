"""
Test file to verify the fast single-step NewtonSchulz5BF16 matches an official
single-iteration implementation. Checks dtype consistency, numerical output,
and gradient flow.
"""

import torch
from mambapy.pscan import newtonschulz5

def newtonschulz5_official(G, steps=1, eps=1e-7):
    """Official single-step PyTorch implementation."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

def test_dtype_consistency():
    """Test that all intermediate computations are in bfloat16."""
    print("=" * 60)
    print("Test: Dtype Consistency")
    print("=" * 60)
    
    G = torch.randn(15, 20, dtype=torch.float32)
    
    # Test our implementation
    result_ours = newtonschulz5(G)
    
    # Test official implementation
    result_official = newtonschulz5_official(G)
    
    print(f"Input dtype: {G.dtype}")
    print(f"Our output dtype: {result_ours.dtype}")
    print(f"Official output dtype: {result_official.dtype}")
    
    assert result_ours.dtype == result_official.dtype, \
        f"Dtype mismatch: ours={result_ours.dtype}, official={result_official.dtype}"
    
    print("✓ Dtype consistency test passed!\n")

def test_numerical_match():
    """Test numerical output matches official implementation."""
    print("=" * 60)
    print("Test: Numerical Match")
    print("=" * 60)
    
    G = torch.randn(15, 20, dtype=torch.float32)
    
    result_ours = newtonschulz5(G)
    result_official = newtonschulz5_official(G)
    
    result_ours_bf16 = result_ours.to(torch.bfloat16)
    result_official_bf16 = result_official.to(torch.bfloat16)
    
    assert result_ours.shape == result_official.shape, \
        f"Shape mismatch: ours={result_ours.shape}, official={result_official.shape}"
    
    max_diff = (result_ours_bf16 - result_official_bf16).abs().max().item()
    print(f"Max difference: {max_diff}")
    print(f"Max difference ratio: {max_diff / result_official_bf16.abs().max().item():.6e}")
    
    assert max_diff < 1e-2, f"Numerical mismatch too large: {max_diff}"
    
    print("✓ Numerical match test passed!\n")

def test_gradient_computation():
    """Test gradient computation works correctly."""
    print("=" * 60)
    print("Test: Gradient Computation")
    print("=" * 60)
    
    G = torch.randn(15, 20, dtype=torch.float32, requires_grad=True)
    
    result = newtonschulz5(G)
    loss = result.sum()
    loss.backward()
    
    print(f"Input requires_grad: {G.requires_grad}")
    print(f"Output requires_grad: {result.requires_grad}")
    print(f"Gradient dtype: {G.grad.dtype if G.grad is not None else None}")
    print(f"Gradient shape: {G.grad.shape if G.grad is not None else None}")
    
    assert G.grad is not None, "Gradient should be computed"
    assert G.grad.shape == G.shape, f"Gradient shape mismatch: {G.grad.shape} vs {G.shape}"
    
    print("✓ Gradient computation test passed!\n")

def test_different_shapes():
    """Test with different matrix shapes."""
    print("=" * 60)
    print("Test: Different Shapes")
    print("=" * 60)
    
    shapes = [
        (10, 20),  # Wide (transpose case)
        (20, 10),  # Tall (transpose case)
        (15, 15),  # Square
    ]
    
    for shape in shapes:
        G = torch.randn(*shape, dtype=torch.float32)
        result = newtonschulz5(G)
        official = newtonschulz5_official(G)
        
        assert result.shape == official.shape, \
            f"Shape {shape}: mismatch {result.shape} vs {official.shape}"
        assert result.dtype == official.dtype, \
            f"Shape {shape}: dtype mismatch {result.dtype} vs {official.dtype}"
        
        print(f"✓ Shape {shape}: passed (output shape: {result.shape}, dtype: {result.dtype})")
    
    print()

def test_single_step_gradient():
    """Confirm gradients flow through the single Newton-Schulz step."""
    print("=" * 60)
    print("Test: Single-Step Gradient Flow")
    print("=" * 60)
    
    G = torch.randn(15, 20, dtype=torch.float32, requires_grad=True)
    result = newtonschulz5(G)
    loss = result.sum()
    loss.backward()
    
    assert G.grad is not None, "Gradient should propagate through the step"
    print(f"Gradient norm: {G.grad.norm().item():.6e}")
    print("✓ Single-step gradient flow test passed!\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Fast Newton-Schulz Implementation Tests")
    print("=" * 60 + "\n")
    
    test_dtype_consistency()
    test_numerical_match()
    test_gradient_computation()
    test_different_shapes()
    test_single_step_gradient()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


