"""
Test file to check dtype at each computation step in the single-step Newton-Schulz
routine. This verifies that all matrix operations are performed in bfloat16.
"""

import torch

def newtonschulz5_official(G, steps=1, eps=1e-7):
    """
    Official PyTorch implementation with dtype checks at each step.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    print(f"Input G dtype: {G.dtype}, shape: {G.shape}")
    
    # Step 1: Cast to bfloat16
    X = G.bfloat16()
    assert X.dtype == torch.bfloat16, f"Step 1: X must be bfloat16, got {X.dtype}"
    print(f"Step 1: X = G.bfloat16() -> dtype: {X.dtype}")
    
    # Step 2: Normalize
    norm_val = X.norm() + eps
    print(f"Step 2: norm computation (scalar) -> dtype: {norm_val.dtype if isinstance(norm_val, torch.Tensor) else type(norm_val)}")
    X /= norm_val
    assert X.dtype == torch.bfloat16, f"Step 2: X after normalization must be bfloat16, got {X.dtype}"
    print(f"Step 2: X /= norm -> dtype: {X.dtype}")
    
    # Step 3: Transpose if needed
    if G.size(0) > G.size(1):
        X = X.T
        assert X.dtype == torch.bfloat16, f"Step 3: X after transpose must be bfloat16, got {X.dtype}"
        print(f"Step 3: X = X.T -> dtype: {X.dtype}")
    
    # Step 4: Iterations (single step by default)
    for i in range(steps):
        print(f"\n--- Iteration {i+1} ---")
        
        A = X @ X.T
        assert A.dtype == torch.bfloat16, f"Step 4.{i+1}.1: A must be bfloat16, got {A.dtype}"
        print(f"  A = X @ X.T -> dtype: {A.dtype}")
        
        A_squared = A @ A
        assert A_squared.dtype == torch.bfloat16, f"Step 4.{i+1}.2: A @ A must be bfloat16, got {A_squared.dtype}"
        print(f"  A @ A -> dtype: {A_squared.dtype}")
        
        B = b * A + c * A_squared
        assert B.dtype == torch.bfloat16, f"Step 4.{i+1}.3: B must be bfloat16, got {B.dtype}"
        print(f"  B = b*A + c*A@A -> dtype: {B.dtype}")
        
        B_X = B @ X
        assert B_X.dtype == torch.bfloat16, f"Step 4.{i+1}.4: B @ X must be bfloat16, got {B_X.dtype}"
        print(f"  B @ X -> dtype: {B_X.dtype}")
        
        X = a * X + B_X
        assert X.dtype == torch.bfloat16, f"Step 4.{i+1}.5: X after update must be bfloat16, got {X.dtype}"
        print(f"  X = a*X + B@X -> dtype: {X.dtype}")
    
    # Step 5: Transpose back if needed
    if G.size(0) > G.size(1):
        X = X.T
        assert X.dtype == torch.bfloat16, f"Step 5: X after transpose back must be bfloat16, got {X.dtype}"
        print(f"Step 5: X = X.T (back) -> dtype: {X.dtype}")
    
    print(f"\nFinal output dtype: {X.dtype}")
    return X

def test_newtonschulz5_dtype():
    """Test Newton-Schulz with different input shapes and dtypes."""
    
    print("=" * 60)
    print("Test 1: Wide matrix (transpose case)")
    print("=" * 60)
    G1 = torch.randn(10, 20, dtype=torch.float32)
    result1 = newtonschulz5_official(G1)
    print(f"Result dtype: {result1.dtype}, shape: {result1.shape}\n")
    
    print("=" * 60)
    print("Test 2: Tall matrix (no transpose case)")
    print("=" * 60)
    G2 = torch.randn(20, 10, dtype=torch.float32)
    result2 = newtonschulz5_official(G2)
    print(f"Result dtype: {result2.dtype}, shape: {result2.shape}\n")
    
    print("=" * 60)
    print("Test 3: Square matrix")
    print("=" * 60)
    G3 = torch.randn(15, 15, dtype=torch.float32)
    result3 = newtonschulz5_official(G3)
    print(f"Result dtype: {result3.dtype}, shape: {result3.shape}\n")
    
    print("=" * 60)
    print("All tests passed! All operations are in bfloat16.")
    print("=" * 60)

if __name__ == "__main__":
    test_newtonschulz5_dtype()

