import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def calculate_flux_xy(w_field, dx, dy):
    """Calculates flux through XY planes (W component integral)."""
    return np.sum(w_field, axis=(1, 2)) * dx * dy

def calculate_flux_xz(v_field, dx, dz):
    """Calculates flux through XZ planes (V component integral)."""
    return np.sum(v_field, axis=(0, 2)) * dx * dz

def calculate_flux_yz(u_field, dy, dz):
    """Calculates flux through YZ planes (U component integral)."""
    return np.sum(u_field, axis=(0, 1)) * dy * dz

def main():
    parser = argparse.ArgumentParser(description="Compare volumetric flux of original and cleaned velocity fields across all planes.")
    parser.add_argument("file", nargs="?", default="sinteredGlass_interpolated.npz", help="Path to the .npz result file.")
    parser.add_argument("--output", "-o", default="flux_comparison.png", help="Output plot filename.")
    parser.add_argument("--no-show", action="store_true", help="Don't show the plot window.")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return

    print(f"Loading data from {args.file}...")
    data = np.load(args.file)
    
    x, y, z = data['x'], data['y'], data['z']
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    
    u_cleaned, v_cleaned, w_cleaned = data['u'], data['v'], data['w']
    has_dual = 'u_init' in data
    
    if has_dual:
        u_init, v_init, w_init = data['u_init'], data['v_init'], data['w_init']
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Volumetric Flux Comparison: {os.path.basename(args.file)}', fontsize=14)
    
    planes = [
        ('XY (Z-flux)', z, w_cleaned, calculate_flux_xy, dx, dy, 'Z Position', (w_init if has_dual else None)),
        ('XZ (Y-flux)', y, v_cleaned, calculate_flux_xz, dx, dz, 'Y Position', (v_init if has_dual else None)),
        ('YZ (X-flux)', x, u_cleaned, calculate_flux_yz, dy, dz, 'X Position', (u_init if has_dual else None))
    ]
    
    print("\nFlux Statistics:")
    
    for i, (title, coords, field, func, h1, h2, xlabel, field_init) in enumerate(planes):
        ax = axs[i]
        flux_c = func(field, h1, h2)
        
        ax.plot(coords, flux_c, 'b-', label='Cleaned', linewidth=2)
        
        c_mean, c_std = np.mean(flux_c), np.std(flux_c)
        c_var = (c_std / abs(c_mean) * 100) if abs(c_mean) > 1e-12 else 0
        print(f"  {title} Cleaned: Mean={c_mean:.4e}, Std={c_std:.4e} ({c_var:.2f}% variation)")
        
        if field_init is not None:
            flux_i = func(field_init, h1, h2)
            ax.plot(coords, flux_i, 'r--', label='Original', alpha=0.7)
            
            i_mean, i_std = np.mean(flux_i), np.std(flux_i)
            i_var = (i_std / abs(i_mean) * 100) if abs(i_mean) > 1e-12 else 0
            print(f"  {title} Original: Mean={i_mean:.4e}, Std={i_std:.4e} ({i_var:.2f}% variation)")
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if i == 0: ax.set_ylabel('Volumetric Flux (Q)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    print(f"\nSaving plot to {args.output}...")
    plt.savefig(args.output, dpi=150)
    
    if not args.no_show:
        plt.show()

if __name__ == "__main__":
    main()
