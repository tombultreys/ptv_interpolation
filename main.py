
import argparse
import numpy as np
import tifffile
import os
import sys

try:
    from interpolator import load_ptv_data, load_mask, create_grid, interpolate_field
    from physics import clean_divergence
    from visualizer import show
except ImportError:
    # Allow running from parent directory or installed package
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from interpolator import load_ptv_data, load_mask, create_grid, interpolate_field
    from physics import clean_divergence
    from visualizer import show

def main():
    parser = argparse.ArgumentParser(description="Interpolate 3D PTV velocity field.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file with columns x, y, z, u, v, w")
    parser.add_argument("--mask", "-m", help="Optional 3D mask TIFF file (0=fluid, >0=solid)")
    parser.add_argument("--downscale", "-s", type=float, default=1.0, help="Downscale factor relative to mask (default 1.0)")
    parser.add_argument("--divergence-free", "-d", action="store_true", help="Apply iterative divergence cleaning.")
    parser.add_argument("--iter", type=int, default=3, help="Number of iterations for divergence cleaning.")
    parser.add_argument("--output-tif", "-o", help="Output TIFF filename (will save as multi-channel or separate files)")
    parser.add_argument("--output-npz", help="Output NPZ filename for raw data")
    parser.add_argument("--crop", type=int, nargs=6, help="Crop region: xmin xmax ymin ymax zmin zmax")
    parser.add_argument("--method", default="linear", choices=["linear", "nearest", "cubic", "rbf"], help="Interpolation method")
    parser.add_argument("--rbf-neighbors", type=int, default=50, help="Number of neighbors for local RBF (3D)")
    parser.add_argument("--rbf-kernel", default="thin_plate_spline", help="RBF kernel (thin_plate_spline, cubic, quintic, etc.)")
    parser.add_argument("--no-plot", action="store_true", help="Don't show the plot.")
    parser.add_argument("--invert-mask", action="store_true", help="Invert mask logic (swap fluid/solid)")
    parser.add_argument("--data-offset", type=int, nargs=3, help="Offset to align data to mask: x y z")
    parser.add_argument("--swap-xy", action="store_true", help="Swap X and Y coordinates and velocities")
    parser.add_argument("--mask-transpose", type=int, nargs=3, help="Transpose mask axes: e.g., 2 1 0")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of processes for parallel RBF evaluation")

    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Loading data from {args.input}...")
    df = load_ptv_data(args.input)
    
    # Apply data offset if provided
    if args.data_offset:
        ox, oy, oz = args.data_offset
        print(f"Applying coordinate offset: x+={ox}, y+={oy}, z+={oz}")
        df.x += ox
        df.y += oy
        df.z += oz
        
    # Swap XY if requested
    if args.swap_xy:
        print("Swapping X and Y coordinates and velocities...")
        df[['x', 'y']] = df[['y', 'x']]
        df[['u', 'v']] = df[['v', 'u']]
    
    # 2. Mask Handling & Domain Definition
    mask = None
    bounds = None
    
    if args.mask:
        print(f"Loading mask from {args.mask}...")
        mask_raw = load_mask(args.mask) 
        print(f"Loaded Mask Shape: {mask_raw.shape}")
        
        if args.mask_transpose:
            print(f"Transposing mask with axes {args.mask_transpose}...")
            mask_raw = np.transpose(mask_raw, axes=args.mask_transpose)
            print(f"Mask Shape after transposition: {mask_raw.shape}")

        if args.invert_mask:
            print("Inverting mask...")
            mask_raw = ~mask_raw
        
        if args.crop:
            xs, xe, ys, ye, zs, ze = args.crop
            print(f"Cropping mask to X[{xs}:{xe}], Y[{ys}:{ye}], Z[{zs}:{ze}]...")
            mask_raw = mask_raw[zs:ze, ys:ye, xs:xe]
            bounds = ((xs, xe), (ys, ye), (zs, ze))
            
            # Filter Dataframes to crop region
            print(f"Filtering PTV data to crop region...")
            msg_before = len(df)
            df = df[
                (df.x >= xs) & (df.x < xe) &
                (df.y >= ys) & (df.y < ye) &
                (df.z >= zs) & (df.z < ze)
            ]
            print(f"Points: {msg_before} -> {len(df)}")
            
        else:
            # If no crop is specified, the domain is the full mask
            nz, ny, nx = mask_raw.shape
            bounds = ((0, nx), (0, ny), (0, nz))
            
        # Calculate resolution based on mask and downscale
        nz, ny, nx = mask_raw.shape
        resolution = (
            max(1, int(round(nx / args.downscale))),
            max(1, int(round(ny / args.downscale))),
            max(1, int(round(nz / args.downscale)))
        )
            
    if bounds is None:
        # Fallback to data bounds if no mask was provided
        # We use inclusive max (max + 1) because create_grid uses xmax-1
        xmin, xmax = df.x.min(), df.x.max()
        ymin, ymax = df.y.min(), df.y.max()
        zmin, zmax = df.z.min(), df.z.max()
        
        bounds = (
            (xmin, xmax + 1),
            (ymin, ymax + 1),
            (zmin, zmax + 1)
        )
        # Default resolution for no-mask case
        base_res = 64
        resolution = max(1, int(round(base_res / args.downscale)))

    # 3. Create Grid
    print(f"Creating grid with resolution {resolution}...")
    (X, Y, Z), (x, y, z) = create_grid(bounds, resolution)
    
    dx_grid = x[1] - x[0] if len(x)>1 else 1.0
    dy_grid = y[1] - y[0] if len(y)>1 else 1.0
    dz_grid = z[1] - z[0] if len(z)>1 else 1.0
    
    # Process Mask: Sample it onto the new grid coordinates
    if args.mask:
         from interpolator import sample_mask_on_grid
         print("Sampling mask onto interpolation grid...")
         mask = sample_mask_on_grid(mask_raw, (X, Y, Z), bounds_raw=bounds)
    else:
         mask = np.zeros(X.shape, dtype=bool)

    # 4. Interpolate
    print(f"Interpolating using {args.method} method...")
    U, V, W = interpolate_field(df, (X, Y, Z), 
                                method=args.method, 
                                rbf_neighbors=args.rbf_neighbors,
                                rbf_kernel=args.rbf_kernel,
                                n_jobs=args.n_jobs)
    
    # Replace NaNs
    if np.isnan(U).any():
        print("Warning: NaNs in interpolation (outside convex hull). Filling with 0.")
        U = np.nan_to_num(U)
        V = np.nan_to_num(V)
        W = np.nan_to_num(W)
        
    # Apply Mask (Hard zero in solid regions)
    if args.mask:
        print("Applying mask zeroes (enforcing zero velocity in solid regions)...")
        solid = ~mask
        U[solid] = 0
        V[solid] = 0
        W[solid] = 0
    
    # Store initial interpolation result
    U_init, V_init, W_init = U.copy(), V.copy(), W.copy()
        
    # 5. Divergence Cleaning
    if args.divergence_free:
        print(f"Applying divergence cleaning ({args.iter} iterations)...")
        U, V, W = clean_divergence(U, V, W, mask, dx_grid, dy_grid, dz_grid, iterations=args.iter)
        
    # 6. Save Output
    if args.output_npz:
        print(f"Saving npz to {args.output_npz}...")
        save_dict = {'x':x, 'y':y, 'z':z, 'u':U, 'v':V, 'w':W, 'mask':mask}
        if args.divergence_free:
            save_dict.update({'u_init':U_init, 'v_init':V_init, 'w_init':W_init})
        np.savez(args.output_npz, **save_dict)
        
    if args.output_tif:
        print(f"Saving TIFF to {args.output_tif}...")
        stack = np.stack([U.astype(np.float32), V.astype(np.float32), W.astype(np.float32)], axis=1)
        tifffile.imwrite(args.output_tif, stack, imagej=True, metadata={'axes': 'ZCYX'})
        
    # 7. Visualize
    if not args.no_plot:
        print("Opening visualizer (interactive)...")
        # If divergence free was applied, we pass both initial and cleaned fields
        u_data = (U, U_init) if args.divergence_free else U
        v_data = (V, V_init) if args.divergence_free else V
        w_data = (W, W_init) if args.divergence_free else W
        
        show(u_data, v_data, w_data, x, y, z, mask=mask, input_df=df)
        
    print("Done.")

if __name__ == "__main__":
    main()
