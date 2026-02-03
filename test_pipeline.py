
import numpy as np
import pandas as pd
import tifffile
import os
import subprocess

def generate_synthetic_data(filename="synthetic_ptv.csv", n_points=1000):
    # Domain [-1, 1]^3
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    z = np.random.uniform(-1, 1, n_points)
    
    # Flow field: Solid body rotation around Z axis
    # u = -y, v = x, w = 0
    u = -y
    v = x
    w = 0.1 * z # Slight divergence to test cleaning (div u = 0.1)
    
    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'u': u, 'v': v, 'w': w})
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {n_points} points.")
    return df

def generate_mask(filename="mask.tif", size=32):
    # Create a sphere mask
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Sphere radius 0.5
    mask = (X**2 + Y**2 + Z**2) < 0.25 # Solid sphere in center
    
    # Convert to uint8 (0=fluid, 1=solid) -> Invert logic?
    # My code says: "load_mask": >0 is solid.
    # So if sphere is solid obstacle -> 1 inside sphere.
    mask_uint8 = mask.astype(np.uint8)
    
    tifffile.imwrite(filename, mask_uint8)
    print(f"Generated {filename} with shape {mask.shape}.")

def run_pipeline():
    # Run main.py
    cmd = [
        "python", "main.py",
        "--input", "synthetic_ptv.csv",
        "--mask", "mask.tif",
        "--resolution", "32",
        "--divergence-free",
        "--output-tif", "output.tif",
        "--output-npz", "output.npz",
        "--no-plot" # Headless
    ]
    
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    
    # Verify output
    if os.path.exists("output.tif"):
        print("Success: output.tif created.")
        data = tifffile.imread("output.tif")
        print("Output shape:", data.shape)
        # Expected: (32, 3, 32, 32) or (32, 32, 32, 3) dep on stack
        # Main code stacks as (Z, C, Y, X).
        # Tifffile might interpret C as samples per pixel if contiguous?
        # Let's see.
    else:
        print("Error: output.tif not found.")

if __name__ == "__main__":
    generate_synthetic_data()
    generate_mask()
    run_pipeline()
