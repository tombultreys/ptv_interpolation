
import numpy as np
import pandas as pd
import tifffile

def flow_past_cylinder(x, y, U0, R, xc, yc):
    """
    Potential flow past a cylinder at (xc, yc) with radius R.
    Superposition of uniform flow + doublet.
    """
    # Shift coordinates
    X = x - xc
    Y = y - yc
    
    # Polar coordinates
    r2 = X**2 + Y**2
    r = np.sqrt(r2)
    theta = np.arctan2(Y, X)
    
    # Radial and tangential velocities
    # Vr = U0 * (1 - R^2/r^2) * cos(theta)
    # Vt = -U0 * (1 + R^2/r^2) * sin(theta)
    
    # Inside cylinder, velocity is 0 (physically) or undefined (potential)
    # We will mark it 0 later using mask.
    
    # Cartesian components
    # u = Vr cos - Vt sin
    # v = Vr sin + Vt cos
    
    # Simplified potential function psi = U0 * y * (1 - R^2/r^2)
    # u = dpsi/dy, v = -dpsi/dx
    
    # Potential flow solution:
    # u = U0 * (1 - R^2/r^2 * cos(2*theta))
    # v = -U0 * R^2/r^2 * sin(2*theta)
    
    # Let's use the doublet formula directly
    # Doublet strength mu = 2*pi*U0*R^2 ? No
    # psi = U0 * r * sin(theta) * (1 - R^2/r^2)
    
    # u = U0 * (1 - (R^2/r^2) * cos(2*theta))
    # v = -U0 * (R^2/r^2) * sin(2*theta)
    
    cos2t = np.cos(2*theta)
    sin2t = np.sin(2*theta)
    
    u = U0 * (1 - (R**2 / r2) * cos2t)
    v = -U0 * (R**2 / r2) * sin2t
    
    return u, v

def generate(filename="cylinders_ptv.csv", maskname="cylinders_mask.tif", n_points=5000, size=64):
    # Domain: X [-2, 6], Y [-2, 2], Z [0, 1] (Thin slice for 2D effect)
    x = np.random.uniform(-2, 6, n_points)
    y = np.random.uniform(-2, 2, n_points)
    z = np.random.uniform(0, 1, n_points) # Quasi-2D
    
    # Cylinder parameters
    md = 0.5 # diameter
    R = md / 2
    
    c1_pos = (0, 0)
    c2_pos = (3, 0) # Downstream
    U0 = 1.0
    
    # Calculate velocity as superposition?
    # Potential flow is linear for Phi, so velocities add.
    # But boundary conditions are non-linear (velocity tangent to BOTH surfaces).
    # Simple addition of doublets is an approximation that violates the other cylinder's boundary slightly.
    # For a demo/test, superposition of two single-cylinder solutions on top of U0 is okay-ish.
    # U_total = U_freestream + U_perturbation1 + U_perturbation2
    # U_perturbation = U_cylinder - U_freestream
    
    # Perturbation 1
    u1, v1 = flow_past_cylinder(x, y, U0, R, c1_pos[0], c1_pos[1])
    u_pert1 = u1 - U0
    v_pert1 = v1
    
    # Perturbation 2
    u2, v2 = flow_past_cylinder(x, y, U0, R, c2_pos[0], c2_pos[1])
    u_pert2 = u2 - U0
    v_pert2 = v2
    
    u = U0 + u_pert1 + u_pert2
    v = 0 + v_pert1 + v_pert2
    w = np.zeros_like(u) # 2D flow
    
    # Mask out points inside cylinders (set to 0)
    # Check distance to C1
    dist1 = np.sqrt((x - c1_pos[0])**2 + (y - c1_pos[1])**2)
    dist2 = np.sqrt((x - c2_pos[0])**2 + (y - c2_pos[1])**2)
    
    mask_idx = (dist1 < R) | (dist2 < R)
    u[mask_idx] = 0
    v[mask_idx] = 0
    
    # Save CSV
    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'u': u, 'v': v, 'w': w})
    # Remove points inside cylinders so interpolation has to do the work?
    # Or keep them as zero? Visualizer expects them.
    # Let's remove them to simulate real PTV where you don't get vectors inside solids.
    df = df[~mask_idx]
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {len(df)} points.")
    
    # Generate 3D Mask TIFF
    grid_x = np.linspace(-2, 6, size)
    grid_y = np.linspace(-2, 2, size) # Aspect ratio?
    # We should keep dy uniform with dx?
    # If X range is 8, Y range is 4.
    # If size=64 for X, then Y should be 32.
    nx = size
    ny = size // 2
    nz = 16 # Thin Z
    
    grid_y = np.linspace(-2, 2, ny)
    grid_z = np.linspace(0, 1, nz)
    
    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    
    D1 = np.sqrt((X - c1_pos[0])**2 + (Y - c1_pos[1])**2)
    D2 = np.sqrt((X - c2_pos[0])**2 + (Y - c2_pos[1])**2)
    
    mask_grid = (D1 < R) | (D2 < R)
    tifffile.imwrite(maskname, mask_grid.astype(np.uint8))
    print(f"Generated {maskname} with shape {mask_grid.shape}.")

if __name__ == "__main__":
    generate()
