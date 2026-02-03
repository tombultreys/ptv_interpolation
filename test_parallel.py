
import numpy as np
import pandas as pd
from interpolator import create_grid, interpolate_field

def test_parallel_rbf():
    print("Testing Parallel RBF...")
    # Create fake data
    df = pd.DataFrame({
        'x': [0, 10, 0, 10, 5],
        'y': [0, 0, 10, 10, 5],
        'z': [0, 0, 0, 0, 5],
        'u': [1, 1, 1, 1, 2],
        'v': [0, 0, 0, 0, 0],
        'w': [0, 0, 0, 0, 0]
    })
    
    bounds = ((0, 10), (0, 10), (0, 10))
    res = 10
    grid, _ = create_grid(bounds, res)
    
    # Run parallel
    print("Starting parallel RBF (n_jobs=2)...")
    U, V, W = interpolate_field(df, grid, method='rbf', n_jobs=2)
    print("Parallel RBF check: Success if no crash.")
    print(f"Result shape: {U.shape}")
    assert U.shape == (10, 10, 10)
    print("Verified.")

if __name__ == "__main__":
    test_parallel_rbf()
