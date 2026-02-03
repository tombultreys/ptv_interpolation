
import argparse
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.optimize import minimize
from interpolator import load_ptv_data, load_mask

def find_best_offset(df, mask, initial_offset=(0, 0, 0), invert=False):
    """
    Finds the (dx, dy, dz) offset that minimizes points in solid regions.
    Uses Distance Transform of the fluid region for a smooth gradient.
    
    Assumption: Default (invert=False) mask is True for Fluid, False for Solid.
    """
    if invert:
        # Input was True=Solid, False=Fluid.
        solid_mask = mask
    else:
        # Default: Input was True=Fluid, False=Solid.
        solid_mask = ~mask
    
    # Distance transform: distance from True (solid) to nearest False (fluid) voxel
    print("Computing Distance Transform...")
    dt = ndimage.distance_transform_edt(solid_mask) 
    
    nz, ny, nx = mask.shape
    points = df[['x', 'y', 'z']].values
    
    def objective(offset):
        dx, dy, dz = offset
        # Shift points
        shifted = points + [dx, dy, dz]
        
        # Check bounds
        # Convert to indices
        ix = np.round(shifted[:, 0]).astype(int)
        iy = np.round(shifted[:, 1]).astype(int)
        iz = np.round(shifted[:, 2]).astype(int)
        
        # Only keep points inside volume bounds
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        
        if not np.any(valid):
            return 1e9 # Penalty for being completely outside
        
        # Get distances at point locations
        # Note: dt is indexed (z, y, x)
        distances = dt[iz[valid], iy[valid], ix[valid]]
        
        # Penalty for points outside the volume
        outside_count = np.sum(~valid)
        penalty = outside_count * np.max(dt)
        
        return np.sum(distances) + penalty

    print(f"Starting optimization from initial offset {initial_offset}...")
    # Use Nelder-Mead or Powell for non-gradient based optimization on discrete grid
    res = minimize(objective, initial_offset, method='Powell', tol=1e-1)
    
    return res.x, res.fun

def main():
    parser = argparse.ArgumentParser(description="Find best alignment offset between PTV points and mask.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--mask", "-m", required=True, help="Input Mask TIFF")
    parser.add_argument("--invert-mask", action="store_true", help="Invert mask")
    parser.add_argument("--initial", type=int, nargs=3, default=[0, 0, 0], help="Initial guess (x y z)")
    parser.add_argument("--sample", type=int, default=5000, help="Number of points to sample for speed")
    parser.add_argument("--swap-xy", action="store_true", help="Swap X and Y coordinates")
    parser.add_argument("--mask-transpose", type=int, nargs=3, help="Transpose mask axes: e.g., 2 1 0")

    args = parser.parse_args()
    
    print("Loading data...")
    df = load_ptv_data(args.input)
    
    if args.swap_xy:
        print("Swapping X and Y coordinates...")
        df[['x', 'y']] = df[['y', 'x']]
        if 'u' in df.columns and 'v' in df.columns:
            df[['u', 'v']] = df[['v', 'u']]

    if len(df) > args.sample:
        print(f"Sampling {args.sample} points for faster optimization...")
        df = df.sample(args.sample)
        
    print("Loading mask...")
    mask = load_mask(args.mask)
    
    if args.mask_transpose:
        print(f"Transposing mask with axes {args.mask_transpose}...")
        mask = np.transpose(mask, axes=args.mask_transpose)
    
    best_offset, score = find_best_offset(df, mask, initial_offset=args.initial, invert=args.invert_mask)
    
    print("\n" + "="*30)
    print("OPTIMIZATION COMPLETE")
    print("="*30)
    print(f"Best Offset (x, y, z): {best_offset}")
    print(f"Rounded Offset: {np.round(best_offset).astype(int)}")
    print(f"Final Score (Sum of distances): {score:.2f}")
    print("="*30)
    print("\nYou can now copy these values into your run scripts.")

if __name__ == "__main__":
    main()
