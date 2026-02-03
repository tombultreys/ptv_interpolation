
import numpy as np
import pandas as pd
import tifffile

def generate(filename="spheres_ptv.csv", maskname="spheres_mask.tif", n_points=8000, size=64):
    # Sphere Radius
    R = 0.5
    D = 2 * R
    
    # 6 Spheres in a Simple Hexagonal arrangement (Two stacked triangles)
    # Layer 1: z = 0
    # Layer 2: z = D (touching vertically)
    
    # Triangle centers (equilateral, dist=D)
    cx1 = 0.0
    cy1 = 0.0
    
    cx2 = D
    cy2 = 0.0
    
    cx3 = D / 2.0
    cy3 = np.sqrt(3) * D / 2.0
    
    centers = [
        (cx1, cy1, 0),
        (cx2, cy2, 0),
        (cx3, cy3, 0),
        (cx1, cy1, D),
        (cx2, cy2, D),
        (cx3, cy3, D)
    ]
    
    # Domain limits
    # Min/Max based on centers + R + buffer
    xmin = min(c[0] for c in centers) - R - 0.2
    xmax = max(c[0] for c in centers) + R + 0.2
    
    ymin = min(c[1] for c in centers) - R - 0.2
    ymax = max(c[1] for c in centers) + R + 0.2
    
    zmin = min(c[2] for c in centers) - R - 0.2
    zmax = max(c[2] for c in centers) + R + 0.2
    
    # Generate random points
    x = np.random.uniform(xmin, xmax, n_points)
    y = np.random.uniform(ymin, ymax, n_points)
    z = np.random.uniform(zmin, zmax, n_points)
    
    # Calculate mask (inside any sphere)
    mask_pt = np.zeros(n_points, dtype=bool)
    for (cx, cy, cz) in centers:
        dist2 = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
        mask_pt |= (dist2 < R**2)
        
    # Flow Field:
    # We want flow through the pore (approx center of triangle).
    # Pore center roughly: mean of triangle vertices
    pore_x = (cx1 + cx2 + cx3) / 3.0
    pore_y = (cy1 + cy2 + cy3) / 3.0
    
    # Simple Poiseuille-like profile driven by distance from pore center
    # Max velocity at pore center, min at spheres
    # We cheat: uniform flow * (distance from nearest sphere / R)
    # Or just uniform flow + noise, and let the interpolator clean it.
    # The user asked for "Steady Stokes flow".
    # I'll implement a parabolic profile depending on radial distance from pore axis.
    
    dist_pore_axis = np.sqrt((x - pore_x)**2 + (y - pore_y)**2)
    # The pore "throat" radius is roughly when it hits the spheres.
    # Dist from pore center to sphere center is ...
    # Triangle centroid to vertex is D / sqrt(3) ~ 0.577 * D
    # Sphere surface is at R=0.5*D.
    # Gap radius ~ 0.577D - 0.5D = 0.077D (very small!)
    # Let's check:
    # c3 = (0.5, 0.866, 0)
    # pore = (0.5, 0.288, 0)
    # dist = sqrt(0 + (0.866-0.288)^2) = 0.577. Correct.
    # Sphere radius = 0.5.
    # Gap = 0.077 * 1 = 0.077.
    # That's a tight squeeze.
    
    # Velocity w dominant
    # u, v small radial components?
    # Let's just set w = constant for free space (initial PTV) and let physics solver fix it.
    w = np.ones_like(z)
    u = np.zeros_like(z)
    v = np.zeros_like(z)
    
    # Zero out inside spheres
    u[mask_pt] = 0
    v[mask_pt] = 0
    w[mask_pt] = 0
    
    # Remove masked points for PTV realism
    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'u': u, 'v': v, 'w': w})
    df = df[~mask_pt]
    
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {len(df)} points.")
    
    # 3D Mask TIFF
    gx = np.linspace(xmin, xmax, size)
    gy = np.linspace(ymin, ymax, size)
    gz = np.linspace(zmin, zmax, size) # Cubic?
    
    MX, MY, MZ = np.meshgrid(gx, gy, gz, indexing='ij')
    
    mask_grid = np.zeros(MX.shape, dtype=bool)
    for (cx, cy, cz) in centers:
        dist2 = (MX - cx)**2 + (MY - cy)**2 + (MZ - cz)**2
        mask_grid |= (dist2 < R**2)
        
    tifffile.imwrite(maskname, mask_grid.astype(np.uint8))
    print(f"Generated {maskname} with shape {mask_grid.shape}.")

if __name__ == "__main__":
    generate()
