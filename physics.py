
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

def clean_divergence_projection(u, v, w, mask, dx, dy, dz, iterations=3):
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
        res = lsqr(A, b, damp=1e-8, atol=1e-10, btol=1e-10, iter_lim=3000, show=(i==0))
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

def compute_force_divergence(fx, fy, fz, mask, dx, dy, dz, wall_bc='zero-neumann'):
    """
    Computes divergence of a force field f, including boundary fluxes for 
    Neumann BCs in a Poisson problem.
    """
    def get_flux_gradient(field, axis, mask, h):
        div_comp = np.zeros_like(field)
        
        # Face fluxes between cell i and i+1
        f_face = np.zeros_like(field)
        
        s_curr = [slice(None)] * 3
        s_next = [slice(None)] * 3
        s_curr[axis] = slice(0, -1)
        s_next[axis] = slice(1, None)
        
        # Internal faces: Average where both are fluid
        m_curr = mask[tuple(s_curr)]
        m_next = mask[tuple(s_next)]
        both_fluid = m_curr & m_next
        
        f_face_view = f_face[tuple(s_curr)]
        field_curr = field[tuple(s_curr)]
        field_next = field[tuple(s_next)]
        
        # Average fluid-fluid
        f_face_view[both_fluid] = 0.5 * (field_curr[both_fluid] + field_next[both_fluid])
        
        # Grains
        if wall_bc == 'inhomogeneous':
            curr_fluid_next_solid = m_curr & (~m_next)
            next_fluid_curr_solid = m_next & (~m_curr)
            f_face_view[curr_fluid_next_solid] = field_curr[curr_fluid_next_solid]
            f_face_view[next_fluid_curr_solid] = field_next[next_fluid_curr_solid]
        
        # CRITICAL: For a Zero-Neumann Laplacian matrix, any physics-based boundary flux 
        # (f.n) must be treated as ZAYO in the divergence operator to "inject" it into 
        # the RHS. If we included it in the divergence, we would be subtracting it
        # from the cell, which cancels the driving force.
        # Thus, f_face at domain edges and solid boundaries (if zero-neumann) stays 0.
        
        # Divergence = (F_face[i] - F_face[i-1]) / h
        f_face_prev = np.zeros_like(f_face)
        f_face_prev[tuple(s_next)] = f_face[tuple(s_curr)]
        
        return (f_face - f_face_prev) / h

    # Sum contributions from all 3 axes
    div = get_flux_gradient(fx, 2, mask, dx) + \
          get_flux_gradient(fy, 1, mask, dy) + \
          get_flux_gradient(fz, 0, mask, dz)
    return div

def solve_poisson(source, mask, dx, dy, dz, force_field=None, wall_bc='inhomogeneous', 
                  dirichlet_mask=None, dirichlet_values=None):
    """
    Solves Lap(p) = source on the fluid domain designated by the mask.
    
    force_field: (fx, fy, fz) to calculate consistent RHS with boundary fluxes.
    wall_bc: Treatment of Neumann boundaries ('inhomogeneous' or 'zero-neumann').
    dirichlet_mask: Boolean mask of nodes to fix (Dirichlet BC).
    dirichlet_values: Values or array of values for fixed nodes.
    """
    n_fluid = np.sum(mask)
    if n_fluid == 0:
        return np.zeros_like(mask, dtype=float)
        
    if force_field is not None:
        fx, fy, fz = force_field
        rhs_field = compute_force_divergence(fx, fy, fz, mask, dx, dy, dz, wall_bc=wall_bc)
    else:
        rhs_field = source
        
    A, idx_map = build_laplacian_matrix(mask, dx, dy, dz)
    b = rhs_field[mask].copy()
    
    # Handle Dirichlet BCs
    if dirichlet_mask is not None:
        d_mask_fluid = dirichlet_mask[mask]
        d_indices = np.where(d_mask_fluid)[0]
        
        if len(d_indices) > 0:
            if np.isscalar(dirichlet_values):
                vals = np.ones(len(d_indices)) * dirichlet_values
            else:
                vals = dirichlet_values[mask][d_indices]
            
            # Efficient Dirichlet implementation for sparse matrix
            # 1. Modify RHS: for each neighbor j of Dirichlet node i, b_j -= A_ji * val_i
            # Get columns of A corresponding to Dirichlet nodes
            A = A.tocsc()
            for i_idx, val in zip(d_indices, vals):
                col = A.getcol(i_idx)
                # Find rows where this column is non-zero (neighbors)
                rows = col.indices
                if len(rows) > 0:
                    b[rows] -= col.data * val
            
            # 2. Zero out rows and columns of Dirichlet nodes and set diagonal to 1
            A = A.tocsr()
            for i_idx, val in zip(d_indices, vals):
                # Zero out the row
                row_start = A.indptr[i_idx]
                row_end = A.indptr[i_idx+1]
                A.data[row_start:row_end] = 0.0
                # Set diagonal to 1 and RHS to target value
                # (We need to find the diagonal element index)
                # Or easier: A[i_idx, i_idx] = 1.0 (slower than direct data access but safer)
                b[i_idx] = val
            
            # After zeroing data, we must re-set the diagonal to 1
            # To be efficient, we can do it after the loop
            diag_indices = A.diagonal().nonzero()[0] # Incorrect if we zeroed them
            # Correct way: use A.eliminate_zeros() then set diagonals
            A.eliminate_zeros()
            A = A.tolil()
            for i_idx in d_indices:
                A[i_idx, i_idx] = 1.0
            A = A.tocsr()
    else:
        # Pure Neumann compatibility (singular system)
        b = b - np.mean(b)
    
    # Solve system (CG or LSQR for robust handling)
    if dirichlet_mask is not None:
        # Symmetric Positive Definite if anchored by Dirichlet
        p_vec, info = cg(A, b, tol=1e-10, maxiter=3000)
    else:
        res = lsqr(A, b, damp=1e-8, atol=1e-10, btol=1e-10, iter_lim=3000)
        p_vec = res[0]
    
    p_grid = np.zeros_like(rhs_field)
    p_grid[mask] = p_vec
    
    return p_grid

def clean_divergence(u, v, w, mask, dx, dy, dz, iterations=3, method='projection', lambda_reg=1e3):
    """
    Dispatcher for divergence cleaning methods.
    """
    if method == 'variational':
        return clean_divergence_variational(u, v, w, mask, dx, dy, dz, lambda_reg=lambda_reg)
    else:
        return clean_divergence_projection(u, v, w, mask, dx, dy, dz, iterations=iterations)

def build_divergence_operators(mask, dx, dy, dz):
    """
    Builds sparse matrices Dx, Dy, Dz such that div = Dx*u + Dy*v + Dz*w.
    Consistent with compute_consistent_divergence face averaging.
    """
    nz, ny, nx = mask.shape
    n_fluid = np.sum(mask)
    idx_map = np.full(mask.shape, -1, dtype=np.int32)
    idx_map[mask] = np.arange(n_fluid)
    
    I, J, K = np.where(mask)
    curr = idx_map[I, J, K]
    
    def get_op(axis, h):
        rows, cols, data = [], [], []
        # For each cell, we need (f_next - f_prev) / h
        # f_next = (v_i + v_{i+1})/2 if i+1 fluid, else 0
        # f_prev = (v_{i-1} + v_i)/2 if i-1 fluid, else 0
        
        # Positive shift (next)
        In, Jn, Kn = I, J, K
        if axis == 2: Kn = K + 1
        elif axis == 1: Jn = J + 1
        else: In = I + 1
        
        valid_n = (In >= 0) & (In < nz) & (Jn >= 0) & (Jn < ny) & (Kn >= 0) & (Kn < nx)
        neigh_n = np.full_like(curr, -1)
        neigh_n[valid_n] = idx_map[In[valid_n], Jn[valid_n], Kn[valid_n]]
        
        # Negative shift (prev)
        Ip, Jp, Kp = I, J, K
        if axis == 2: Kp = K - 1
        elif axis == 1: Jp = J - 1
        else: Ip = I - 1
        
        valid_p = (Ip >= 0) & (Ip < nz) & (Jp >= 0) & (Jp < ny) & (Kp >= 0) & (Kp < nx)
        neigh_p = np.full_like(curr, -1)
        neigh_p[valid_p] = idx_map[Ip[valid_p], Jp[valid_p], Kp[valid_p]]

        # Contribution of v_i to div_i:
        # Through f_next: +1/(2h) if next is fluid, else 0 (if next is domain edge, it's Neumann so +1/h?)
        # Current logic has Neumann at edges: f_next = v_i if at edge.
        # Let's simplify and follow consistent_divergence exactly.
        
        # We'll just build it by adding contributions
        # Coeffs for v_i:
        c_curr = np.zeros(n_fluid)
        
        # ufn part: if m_next: (v_i + v_next)/2h, else 0
        # Contribution to div_i: v_i * (0.5/h) if fluid else 0
        c_curr += np.where(valid_n & (neigh_n != -1), 0.5/h, 0.0)
        # Handle domain edge Neumann: f_next = v_curr
        c_curr += np.where(~valid_n, 1.0/h, 0.0)
        
        # ufp part: if m_prev: (v_prev + v_i)/2h, else 0
        # Contribution to -div_i: v_i * (-0.5/h) if fluid else 0
        c_curr -= np.where(valid_p & (neigh_p != -1), 0.5/h, 0.0)
        # Handle domain edge Neumann: f_prev = v_curr
        c_curr -= np.where(~valid_p, 1.0/h, 0.0)
        
        rows.append(curr)
        cols.append(curr)
        data.append(c_curr)
        
        # Coeffs for v_next:
        valid_next_fluid = (neigh_n != -1)
        rows.append(curr[valid_next_fluid])
        cols.append(neigh_n[valid_next_fluid])
        data.append(np.full(np.sum(valid_next_fluid), 0.5/h))
        
        # Coeffs for v_prev:
        valid_prev_fluid = (neigh_p != -1)
        rows.append(curr[valid_prev_fluid])
        cols.append(neigh_p[valid_prev_fluid])
        data.append(np.full(np.sum(valid_prev_fluid), -0.5/h))
        
        return sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), 
                                 shape=(n_fluid, n_fluid)).tocsr()

    Dx = get_op(2, dx)
    Dy = get_op(1, dy)
    Dz = get_op(0, dz)
    return Dx, Dy, Dz, idx_map

def clean_divergence_variational(u, v, w, mask, dx, dy, dz, iterations=1, lambda_reg=1e3):
    """
    Divergence cleaning using variational cost-function optimization.
    Minimizes J = ||U - U0||^2 + lambda * ||div(U)||^2
    """
    print(f"Starting Variational Divergence Cleaning (lambda={lambda_reg})...")
    
    nz, ny, nx = u.shape
    fluid_mask = mask
    n_fluid = np.sum(fluid_mask)
    
    # 1. Build Operators
    Dx, Dy, Dz, idx_map = build_divergence_operators(mask, dx, dy, dz)
    
    # 2. Form RHS
    u0 = u[fluid_mask]
    v0 = v[fluid_mask]
    w0 = w[fluid_mask]
    rhs = np.concatenate([u0, v0, w0])
    
    # 3. Form System: (I + lambda * D^T * D) U = U0
    # D = [Dx, Dy, Dz]
    # D^T * D is a 3x3 block matrix
    from scipy.sparse import bmat, eye
    
    I = eye(n_fluid)
    
    # Pre-multiply D^T parts to build blocks
    # This might be faster than building D and doing D^T * D
    Dxx = Dx.T @ Dx
    Dxy = Dx.T @ Dy
    Dxz = Dx.T @ Dz
    Dyy = Dy.T @ Dy
    Dyz = Dy.T @ Dz
    Dzz = Dz.T @ Dz
    
    A = bmat([
        [I + lambda_reg*Dxx,     lambda_reg*Dxy,     lambda_reg*Dxz],
        [lambda_reg*Dxy.T,   I + lambda_reg*Dyy,     lambda_reg*Dyz],
        [lambda_reg*Dxz.T,   lambda_reg*Dyz.T,   I + lambda_reg*Dzz]
    ], format='csr')

    print(f"  Solving Variational System (A: {A.shape[0]}x{A.shape[1]})...")
    
    # Use CG for the symmetric positive definite system
    sol, info = cg(A, rhs, tol=1e-8, maxiter=2000)
    
    if info > 0:
        print(f"  Warning: CG did not converge after {info} iterations.")
    elif info < 0:
        print("  Error: CG failed.")
        return u, v, w

    # 4. Reconstruct
    u_new = np.zeros_like(u)
    v_new = np.zeros_like(v)
    w_new = np.zeros_like(w)
    
    u_new[fluid_mask] = sol[:n_fluid]
    v_new[fluid_mask] = sol[n_fluid:2*n_fluid]
    w_new[fluid_mask] = sol[2*n_fluid:]
    
    # Final Report
    m_div_init = np.mean(np.abs(compute_consistent_divergence(u, v, w, mask, dx, dy, dz)[fluid_mask]))
    m_div_final = np.mean(np.abs(compute_consistent_divergence(u_new, v_new, w_new, mask, dx, dy, dz)[fluid_mask]))
    
    print("\n" + "="*40)
    print("VARIATIONAL CLEANING COMPLETE")
    print(f"Mean Abs Div (Initial): {m_div_init:.6e}")
    print(f"Mean Abs Div (Final):   {m_div_final:.6e}")
    reduction = m_div_init / m_div_final if m_div_final > 0 else float('inf')
    print(f"Total Reduction:        {reduction:.2f}x")
    print("="*40 + "\n")
    
    return u_new, v_new, w_new
