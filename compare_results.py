import numpy as np
import tifffile
import os
import argparse
from visualizer import side_by_side

def main():
    parser = argparse.ArgumentParser(description="Side-by-side comparison of PTV results against simulation.")
    parser.add_argument("--npz", default="sinteredGlass_interpolated_linear_downsample2.npz", help="PTV results NPZ")
    parser.add_argument("--ref-dir", default=r"C:\Users\tbultrey\Documents\XPTV_porousGlass\3_simulatedVelocityFields", help="Directory for Reference TIFs")
    parser.add_argument("--upscale", action="store_true", help="Upscale PTV data by factor of 2 (no interpolation)")
    parser.add_argument("--downscale-ref", action="store_true", help="Downscale simulation data by factor of 2 (using [::2, ::2, ::2])")
    parser.add_argument("--normalize", action="store_true", default=True, help="Divide each field by its own mean speed (default: True)")
    parser.add_argument("--no-norm", action="store_false", dest="normalize", help="Disable normalization")
    args = parser.parse_args()

    # 1. Load PTV Data
    print(f"Loading PTV results from {args.npz}...")
    if not os.path.exists(args.npz):
        print(f"Error: {args.npz} not found.")
        return
    data = np.load(args.npz)

    # Handle dual fields (Cleaned vs Original) - we use Cleaned for primary comparison
    u1, v1, w1 = data['u'].astype(float), data['v'].astype(float), data['w'].astype(float)
    x, y, z = data['x'], data['y'], data['z']
    mask = data['mask'] if 'mask' in data else None

    # Upscaling if requested
    if args.upscale:
        print("Upscaling PTV field by factor of 2...")
        u1 = u1.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
        v1 = v1.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
        w1 = w1.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
        if mask is not None:
            mask = mask.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
            
        def upscale_coords(c):
            if len(c) < 2: return c
            dc = c[1] - c[0]
            new_c = np.zeros(len(c) * 2)
            new_c[0::2] = c
            new_c[1::2] = c + dc/2
            return new_c
            
        x, y, z = upscale_coords(x), upscale_coords(y), upscale_coords(z)

    # 2. Load Reference TIFs
    print(f"Loading reference TIFs from {args.ref_dir}...")
    try:
        u2 = tifffile.imread(os.path.join(args.ref_dir, "Ufx_resliced.tif")).astype(float)
        v2 = tifffile.imread(os.path.join(args.ref_dir, "Ufy_resliced.tif")).astype(float)
        w2 = tifffile.imread(os.path.join(args.ref_dir, "Ufz_resliced.tif")).astype(float)
        
        if args.downscale_ref:
            print("Downscaling Reference field by factor of 2...")
            u2, v2, w2 = u2[::2, ::2, ::2], v2[::2, ::2, ::2], w2[::2, ::2, ::2]
            
    except Exception as e:
        print(f"Error loading reference TIFs: {e}")
        return

    # Check for shape mismatch and truncate if necessary
    ptv_shape = u1.shape
    ref_shape = u2.shape
    
    if ptv_shape != ref_shape:
        print(f"Warning: Shape mismatch! PTV {ptv_shape} vs Ref {ref_shape}.")
        print("Truncating to smallest common dimensions...")
        nz, ny, nx = min(ptv_shape[0], ref_shape[0]), min(ptv_shape[1], ref_shape[1]), min(ptv_shape[2], ref_shape[2])
        u1, v1, w1 = u1[:nz,:ny,:nx], v1[:nz,:ny,:nx], w1[:nz,:ny,:nx]
        u2, v2, w2 = u2[:nz,:ny,:nx], v2[:nz,:ny,:nx], w2[:nz,:ny,:nx]
        x, y, z = x[:nx], y[:ny], z[:nz]
        if mask is not None: mask = mask[:nz,:ny,:nx]

    # 3. Normalization
    if args.normalize:
        print("Normalizing fields by their own mean speed...")
        def normalize_field(u, v, w, m=None):
            speed = np.sqrt(u**2 + v**2 + w**2)
            if m is not None:
                mean_val = np.nanmean(speed[m])
            else:
                # Fallback to non-zero values if no mask
                mean_val = np.nanmean(speed[speed > 1e-6])
            
            if np.isnan(mean_val) or mean_val == 0:
                print("Warning: Mean speed is zero or NaN, skipping normalization for this field.")
                return u, v, w
            print(f"  Normalization factor: {mean_val:.4e}")
            return u / mean_val, v / mean_val, w / mean_val

        u1, v1, w1 = normalize_field(u1, v1, w1, mask)
        u2, v2, w2 = normalize_field(u2, v2, w2, None) # Ref usually doesn't have mask here

    print("Opening Side-by-Side Comparison...")
    side_by_side(u1, v1, w1, u2, v2, w2, x, y, z, mask=mask, labels=("PTV Result", "Simulation Ref"))

if __name__ == "__main__":
    main()
