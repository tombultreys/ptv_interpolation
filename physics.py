
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, lsqr, cg

def compute_consistent_divergence(u, v, w, mask, dx, dy, dz):
    """
    Computes divergence using finite volume logic consistent with the Laplacian.
    div_i = [(u_R - u_L)/dx + (v_U - v_D)/dy + (w_F - w_B)/dz]
    where face velocities are averaged for fluid-fluid and taken from cell for fluid-solid.
    """
    nz, ny, nx = u.shape
    div = np.zeros_like(u)
    
    # X direction (axis 2)
    u_face_r = np.zeros((nz, ny, nx))
    # Face i+1/2 is average of i and i+1 if i+1 is fluid
    # If i+1 is solid, face i+1/2 has velocity u_i (Neumann)
    # Actually, u_new_face = 0. So u_correction_face = u_old_face.
    
    # Let's use simpler: 
    # du/dx at cell i = (u_{i+1/2} - u_{i-1/2})/dx
    # u_{i+1/2} = (u_i + u_{i+1})/2 if both fluid.
    # u_{i+1/2} = u_i if i fluid, i+1 solid. (This implies u_new = 0 at wall)
    
    def get_face_vel(vel, axis, mask):
        # Shifted velocities
        v_prev = np.roll(vel, shift=1, axis=axis)
        v_next = np.roll(vel, shift=-1, axis=axis)
        m_prev = np.roll(mask, shift=1, axis=axis)
        m_next = np.roll(mask, shift=-1, axis=axis)
        
        # Right face of cell i
        # If next is fluid: (v_i + v_{i+1})/2
        # If next is solid: 0.0 (No-penetration)
        f_next = np.where(m_next, (vel + v_next)/2.0, 0.0)
        # Boundary handling for domain edges (Neumann: flow carries through)
        slices = [slice(None)] * 3
        slices[axis] = -1
        f_next[tuple(slices)] = vel[tuple(slices)] 
        
        # Left face of cell i
        f_prev = np.roll(f_next, shift=1, axis=axis)
        slices[axis] = 0
        f_prev[tuple(slices)] = vel[tuple(slices)] # Neumann at edge
        
        return f_next, f_prev

    ufn, ufp = get_face_vel(u, 2, mask)
    vfn, vfp = get_face_vel(v, 1, mask)
    wfn, wfp = get_face_vel(w, 0, mask)
    
    return (ufn - ufp)/dx + (vfn - vfp)/dy + (wfn - wfp)/dz

def build_laplacian_matrix(mask, dx, dy, dz):
    """
    Builds a sparse Laplacian matrix for the fluid domain.
    Consistent with compute_consistent_divergence.
    """
    nz, ny, nx = mask.shape
    fluid_mask = mask
    n_fluid = np.sum(fluid_mask)
    
    idx_map = np.full(mask.shape, -1, dtype=np.int32)
    idx_map[fluid_mask] = np.arange(n_fluid)
    
    rows, cols, data = [], [], []
    dx2_inv, dy2_inv, dz2_inv = 1.0/(dx**2), 1.0/(dy**2), 1.0/(dz**2)
    
    I, J, K = np.where(fluid_mask)
    curr = idx_map[I, J, K]
    
    # Laplacian: L_{ii}*phi_i + \sum L_{ij}*phi_j = div_i
    # Sum of (phi_j - phi_i)/h^2 for fluid neighbors.
    
    for axis, h2_inv, dim_size in [(2, dx2_inv, nx), (1, dy2_inv, ny), (0, dz2_inv, nz)]:
        for offset in [-1, 1]:
            In, Jn, Kn = I, J, K
            if axis == 2: Kn = K + offset
            elif axis == 1: Jn = J + offset
            else: In = I + offset
            
            valid_b = (In >= 0) & (In < nz) & (Jn >= 0) & (Jn < ny) & (Kn >= 0) & (Kn < nx)
            
            neigh = np.full_like(curr, -1)
            neigh[valid_b] = idx_map[In[valid_b], Jn[valid_b], Kn[valid_b]]
            
            connected = neigh != -1
            
            # Off-diagonal: 1/h^2 (Note: spsolve solves Ax=b, where A is -Laplacian usually. 
            # But let's stay consistent: div = Lap phi.
            # (phi_{i+1} - 2phi_i + phi_{i-1})/h^2
            rows.append(curr[connected])
            cols.append(neigh[connected])
            data.append(np.full(np.sum(connected), h2_inv))
            
            # Diagonal: -1/h^2 for each fluid neighbor
            rows.append(curr[connected])
            cols.append(curr[connected])
            data.append(np.full(np.sum(connected), -h2_inv))

    # Flatten the lists of arrays
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_fluid, n_fluid))
    return A.tocsr(), idx_map

def apply_consistent_correction(u, v, w, phi, mask, dx, dy, dz):
    """
    Corrects cell-centered velocity by averaging staggered face gradients.
    Consistent with compute_consistent_divergence and build_laplacian_matrix.
    """
    nz, ny, nx = u.shape
    phi_grid = np.zeros_like(u)
    phi_grid[mask] = phi
    
    def get_cell_grad(p, axis, m, h):
        # Staggered face gradient: g_{i+1/2} = (phi_{i+1} - phi_i)/h
        # If neighbor is solid, face gradient is 0 (Neumann BC)
        p_next = np.roll(p, shift=-1, axis=axis)
        m_next = np.roll(m, shift=-1, axis=axis)
        
        # Right face gradient
        g_next = np.where(m_next & m, (p_next - p)/h, 0.0)
        # Handle domain boundaries
        slices = [slice(None)] * 3
        slices[axis] = -1
        g_next[tuple(slices)] = 0.0
        
        # Left face gradient is right face gradient of previous cell
        g_prev = np.roll(g_next, shift=1, axis=axis)
        slices[axis] = 0
        g_prev[tuple(slices)] = 0.0
        
        # Cell-centered correction is average of face corrections
        return (g_next + g_prev) / 2.0

    u_new = u - get_cell_grad(phi_grid, 2, mask, dx)
    v_new = v - get_cell_grad(phi_grid, 1, mask, dy)
    w_new = w - get_cell_grad(phi_grid, 0, mask, dz)
    
    u_new[~mask] = 0
    v_new[~mask] = 0
    w_new[~mask] = 0
    return u_new, v_new, w_new

def clean_divergence(u, v, w, mask, dx, dy, dz, iterations=3):
    """
    Main driver for divergence cleaning with iterative project to handle 
    collocated grid smearing.
    """
    u_c, v_c, w_c = u.copy(), v.copy(), w.copy()
    fluid_mask = mask
    
    print(f"Starting Iterative Divergence Cleaning ({iterations} iterations)...")
    
    # 0. Initial Flux Balance (Validation)
    def report_flux(u_field, label):
        # Net flux through middle YZ plane
        nx = u_field.shape[2]
        mid_x = nx // 2
        flux = np.sum(u_field[:, :, mid_x]) * dy * dz
        print(f"  [{label}] Net X-Flux (mid-plane): {flux:.4e}")

    report_flux(u_c, "Initial")

    for i in range(iterations):
        print(f"\n--- Iteration {i+1}/{iterations} ---")
        
        # 1. Compute Divergence
        div = compute_consistent_divergence(u_c, v_c, w_c, mask, dx, dy, dz)
        m_div = np.mean(np.abs(div[fluid_mask]))
        print(f"  Current Mean Abs Div: {m_div:.6e}")
        
        # 2. Build/Solve Poisson
        # We REBUILD the matrix only once if mask doesn't change
        if i == 0:
            A, idx_map = build_laplacian_matrix(mask, dx, dy, dz)
        
        b = div[fluid_mask]
        b = b - np.mean(b)
        
        print(f"  Solving Poisson (A: {A.shape[0]}x{A.shape[1]})...")
        res = lsqr(A, b, damp=1e-8, atol=1e-10, btol=1e-10, iter_lim=5000, show=(i==0))
        phi = res[0]
        
        if np.isnan(phi).any():
            print("  Warning: Solve failed. Stopping iterations.")
            break

        # 3. Apply Correction
        u_c, v_c, w_c = apply_consistent_correction(u_c, v_c, w_c, phi, mask, dx, dy, dz)

    # Final Report
    div_final = compute_consistent_divergence(u_c, v_c, w_c, mask, dx, dy, dz)
    m_div_final = np.mean(np.abs(div_final[fluid_mask]))
    m_div_init = np.mean(np.abs(compute_consistent_divergence(u, v, w, mask, dx, dy, dz)[fluid_mask]))
    
    print("\n" + "="*40)
    print("DIVERGENCE CLEANING COMPLETE")
    print(f"Initial Mean Abs Div: {m_div_init:.6e}")
    print(f"Final Mean Abs Div:   {m_div_final:.6e}")
    print(f"Total Reduction:      {m_div_init/m_div_final:.2f}x")
    report_flux(u_c, "Final")
    print("="*40 + "\n")

    return u_c, v_c, w_c
