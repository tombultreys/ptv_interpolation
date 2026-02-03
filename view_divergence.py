import numpy as np
import argparse
import os
from visualizer import compare_scalars
from physics import compute_consistent_divergence

def main():
    parser = argparse.ArgumentParser(description="Visualize flow field divergence before and after cleaning.")
    parser.add_argument("file", nargs="?", default="sinteredGlass_interpolated.npz", help="Path to the .npz result file.")
    args = parser.parse_args()

    print(f"Loading data from {args.file}...")
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return
    
    data = np.load(args.file)
    
    # Extract coordinates and spacing
    x, y, z = data['x'], data['y'], data['z']
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    mask = data['mask'] if 'mask' in data else None
    
    print(f"Grid Spacing: dx={dx:.4e}, dy={dy:.4e}, dz={dz:.4e}")

    # Extract velocity fields
    if 'u_init' not in data:
        print("Error: No initial velocity field found in NPZ. Only 'u', 'v', 'w' present.")
        print("This script requires both 'u_init' and the 'u' (cleaned) field.")
        return

    u_init, v_init, w_init = data['u_init'], data['v_init'], data['w_init']
    u_clean, v_clean, w_clean = data['u'], data['v'], data['w']

    print("Computing divergence for Initial field...")
    div_init = compute_consistent_divergence(u_init, v_init, w_init, mask, dx, dy, dz)
    
    print("Computing divergence for Cleaned field...")
    div_clean = compute_consistent_divergence(u_clean, v_clean, w_clean, mask, dx, dy, dz)

    # Statistics
    m_init = np.mean(np.abs(div_init[mask])) if mask is not None else np.mean(np.abs(div_init))
    m_clean = np.mean(np.abs(div_clean[mask])) if mask is not None else np.mean(np.abs(div_clean))
    
    print("\nDivergence Statistics (Mean Absolute):")
    print(f"  Initial: {m_init:.6e}")
    print(f"  Cleaned: {m_clean:.6e}")
    print(f"  Reduction: {m_init/m_clean:.2f}x")

    print("\nLaunching Side-by-Side Divergence Viewer...")
    compare_scalars(div_init, div_clean, x, y, z, mask=mask, 
                    labels=("Initial Divergence", "Cleaned Divergence"),
                    title="Flow Field Divergence Comparison")

if __name__ == "__main__":
    main()
