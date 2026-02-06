"""
Velocity field analysis functions.

This module provides functions for analyzing interpolated velocity fields,
including strain rate tensor and viscous dissipation calculations.
"""

import numpy as np

def compute_strain_rate(u, v, w, dx, dy, dz, mask=None):
    """
    Compute local strain rate magnitude from velocity field.
    
    The strain rate tensor (rate-of-strain tensor) is the symmetric part of 
    the velocity gradient tensor:
        ε̇_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
    
    In rheology, the "shear rate" often refers to the magnitude of the strain rate:
        γ̇ = sqrt(2 * ε̇_ij * ε̇_ij) = sqrt(2 * sum(ε̇_ij^2))
    
    This implementation computes the shear rate magnitude, which is the standard
    measure for analyzing flow in porous media and non-Newtonian fluids.
    
    Args:
        u, v, w: Velocity components (3D arrays)
        dx, dy, dz: Grid spacing
        mask: Boolean mask (True = fluid, False = solid)
    
    Returns:
        strain_rate_magnitude: Scalar field (shear rate)
    """
    # Compute gradients using central differences
    # np.gradient returns [dF/dz, dF/dy, dF/dx] for 3D arrays
    du_dz, du_dy, du_dx = np.gradient(u, dz, dy, dx)
    dv_dz, dv_dy, dv_dx = np.gradient(v, dz, dy, dx)
    dw_dz, dw_dy, dw_dx = np.gradient(w, dz, dy, dx)
    
    # Strain rate tensor components (symmetric part of velocity gradient)
    # ε̇_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
    # For convenience, we compute 2*ε̇_ij to avoid the 0.5 factor:
    epsilon_xx = 2 * du_dx  # 2 * ε̇_xx
    epsilon_yy = 2 * dv_dy  # 2 * ε̇_yy
    epsilon_zz = 2 * dw_dz  # 2 * ε̇_zz
    epsilon_xy = du_dy + dv_dx  # 2 * ε̇_xy
    epsilon_xz = du_dz + dw_dx  # 2 * ε̇_xz
    epsilon_yz = dv_dz + dw_dy  # 2 * ε̇_yz
    
    # Shear rate magnitude (second invariant of strain rate tensor) 
    # γ̇ = sqrt(2 * ε̇_ij * ε̇_ij)
    # Using the pre-computed 2*ε̇_ij (epsilon_xx etc.):
    # γ̇^2 = 2 * [ ε̇_xx^2 + ε̇_yy^2 + ε̇_zz^2 + 2*(ε̇_xy^2 + ε̇_xz^2 + ε̇_yz^2) ]
    # γ̇^2 = 0.5*(2ε̇_xx)^2 + 0.5*(2ε̇_yy)^2 + 0.5*(2ε̇_zz)^2 + (2ε̇_xy)^2 + (2ε̇_xz)^2 + (2ε̇_yz)^2
    strain_rate_magnitude = np.sqrt(
        0.5 * (epsilon_xx**2 + epsilon_yy**2 + epsilon_zz**2) +
        epsilon_xy**2 + epsilon_xz**2 + epsilon_yz**2
    )
    
    # Apply mask (set strain rate to zero in solid regions)
    if mask is not None:
        solid = ~mask
        strain_rate_magnitude[solid] = 0.0
    
    return strain_rate_magnitude

def compute_viscous_dissipation(strain_rate, viscosity, dx=1.0, dy=1.0, dz=1.0, mask=None):
    """
    Compute local viscous dissipation rate using Pilotti (2002) approach.
    
    For Newtonian fluids:
        Φ = 2μ ε̇_ij ε̇_ij = (μ/2) γ̇²
    
    where γ̇ is the shear rate magnitude (strain_rate input).
    
    Args:
        strain_rate: Shear rate magnitude field (scalar)
        viscosity: Dynamic viscosity (Pa·s)
        dx, dy, dz: Grid spacing for volume integration
        mask: Boolean mask (True = fluid, False = solid)
    
    Returns:
        dissipation: Local viscous dissipation rate (W/m³)
    """
    # Standard definition: Φ = 2μ ε̇_ij ε̇_ij = μ γ̇²
    # where γ̇ is the shear rate magnitude (strain_rate input).
    dissipation = viscosity * strain_rate**2
    
    # Apply mask (set dissipation to zero in solid regions)
    if mask is not None:
        solid = ~mask
        dissipation[solid] = 0.0
    
    return dissipation

def compute_vorticity(u, v, w, dx, dy, dz, mask=None):
    """
    Compute vorticity magnitude from velocity field.
    
    Vorticity ω = ∇ × u
    ω_x = ∂w/∂y - ∂v/∂z
    ω_y = ∂u/∂z - ∂w/∂x
    ω_z = ∂v/∂x - ∂u/∂y
    
    Returns:
        vorticity_magnitude: Scalar field
    """
    # np.gradient returns [dF/dz, dF/dy, dF/dx]
    du_dz, du_dy, du_dx = np.gradient(u, dz, dy, dx)
    dv_dz, dv_dy, dv_dx = np.gradient(v, dz, dy, dx)
    dw_dz, dw_dy, dw_dx = np.gradient(w, dz, dy, dx)
    
    vort_x = dw_dy - dv_dz
    vort_y = du_dz - dw_dx
    vort_z = dv_dx - du_dy
    
    vorticity_magnitude = np.sqrt(vort_x**2 + vort_y**2 + vort_z**2)
    
    if mask is not None:
        vorticity_magnitude[~mask] = 0.0
        
    return vorticity_magnitude

def compute_permeability(u, v, w, dissipation, viscosity, dx, dy, dz, mask=None):
    """
    Compute permeability using the energy dissipation approach (Pilotti 2002).
    
    Relationship: k = μ * U0^2 / <Φ>_V
    where:
        k = permeability (m^2)
        μ = dynamic viscosity (Pa·s)
        U0 = Darcy velocity (average velocity over TOTAL volume)
        <Φ>_V = mean dissipation rate over TOTAL volume
    """
    
    # 1. Compute Darcy Velocity (average velocity over TOTAL domain volume)
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    w_mean = np.mean(w)
    darcy_velocity_mag = np.sqrt(u_mean**2 + v_mean**2 + w_mean**2)
    
    # 2. Compute Mean Dissipation over TOTAL domain volume
    mean_dissipation = np.mean(dissipation)
    
    if mean_dissipation == 0:
        return 0.0
        
    # 3. Compute Permeability: k = μ * U0^2 / <Φ>_V
    k = (viscosity * darcy_velocity_mag**2) / mean_dissipation
    
    return k

def compute_astarita_flow_type(strain_rate, vorticity_mag, mask=None):
    """
    Compute Astarita flow type parameter ξ (Astarita 1979).
    
    The parameter is defined as:
        ξ = (γ̇ - |ω|) / (γ̇ + |ω|)
    
    where γ̇ is the shear rate magnitude and |ω| is the vorticity magnitude.
    
    Classification:
        ξ = 1: Pure extensional flow
        ξ = 0: Pure shear flow
        ξ = -1: Solid-body rotation
    
    Args:
        strain_rate: Shear rate magnitude field (scalar)
        vorticity_mag: Vorticity magnitude field (scalar)
        mask: Boolean mask (True = fluid, False = solid)
        
    Returns:
        flow_type: Astarita parameter field (-1 to 1)
    """
    
    numerator = strain_rate - vorticity_mag
    denominator = strain_rate + vorticity_mag
    
    # Initialize with 0 (shear-like) or NaN? 
    # Usually 0 is a safe bet for indeterminate regions in porous media.
    flow_type = np.zeros_like(strain_rate)
    
    # Avoid division by zero
    safe_mask = denominator > 1e-15
    flow_type[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
    
    if mask is not None:
        flow_type[~mask] = 0.0
        
    return flow_type

def compute_pressure_field(u, v, w, dx, dy, dz, mu, rho=0, mask=None, wall_bc='zero-neumann', anchor='outlet', flow_direction='auto'):
    """
    Recover relative pressure field by solving the pressure Poisson equation.
    
    Args:
        u, v, w: Velocity components (3D arrays)
        dx, dy, dz: Grid spacing
        mu: Dynamic viscosity (Pa·s)
        rho: Density (kg/m³)
        mask: Boolean mask (True = fluid, False = solid)
        wall_bc: 'zero-neumann' (more stable) or 'inhomogeneous' (physically exact but noisy)
        anchor: 'outlet', 'inlet', or 'none'. 
        flow_direction: 'auto', 'positive', or 'negative' relative to Z-axis.
        
    Returns:
        pressure: 3D pressure field (Pa)
    """
    print(f"Computing pressure field source term (mu={mu}, rho={rho}, wall_bc={wall_bc}, flow={flow_direction})...")
    
    # helper for Laplacian
    def laplacian_mask_aware(f, dx, dy, dz, mask=None):
        """
        Computes Laplacian with boundary protection.
        1. Computes laplacian only where stencil is fully within fluid.
        2. Extrapolates/fills boundary nodes to avoid artifacts.
        """
        nz, ny, nx = f.shape
        lap = np.zeros_like(f)
        
        # 1. Standard Laplacian
        for axis, h2 in [(0, dz**2), (1, dy**2), (2, dx**2)]:
            f_next = np.roll(f, -1, axis=axis)
            f_prev = np.roll(f, 1, axis=axis)
            
            # Array bounds
            s_last = [slice(None)] * 3; s_last[axis] = -1
            s_first = [slice(None)] * 3; s_first[axis] = 0
            f_next[tuple(s_last)] = f[tuple(s_last)]
            f_prev[tuple(s_first)] = f[tuple(s_first)]
            
            lap += (f_next - 2*f + f_prev) / h2

        if mask is not None:
            # 2. Identify 'bulk' nodes (where all neighbors are fluid)
            # Use binary erosion to find nodes far from the wall
            from scipy.ndimage import binary_erosion, binary_dilation
            bulk_mask = binary_erosion(mask, iterations=1)
            
            # 3. For non-bulk fluid nodes, the Laplacian might be spiky.
            # Fill them using the values from nearby bulk nodes.
            boundary_mask = mask & (~bulk_mask)
            
            if np.any(bulk_mask):
                # Simple extrapolation: Use dilation to fill boundary with bulk values
                # We iteratively fill from bulk to boundary
                lap_filled = lap.copy()
                for _ in range(2):
                    # Find boundary nodes adjacent to bulk
                    to_fill = boundary_mask & (~bulk_mask)
                    if not np.any(to_fill): break
                    
                    # Compute local average of neighbors that HAVE valid values (bulk)
                    sum_val = np.zeros_like(lap)
                    count = np.zeros_like(lap)
                    for axis in [0, 1, 2]:
                        for shift in [-1, 1]:
                            l_shift = np.roll(lap_filled, shift, axis=axis)
                            m_shift = np.roll(bulk_mask, shift, axis=axis)
                            valid = to_fill & m_shift
                            sum_val[valid] += l_shift[valid]
                            count[valid] += 1
                    
                    # Update lap_filled at to_fill
                    mask_upd = to_fill & (count > 0)
                    lap_filled[mask_upd] = sum_val[mask_upd] / count[mask_upd]
                    bulk_mask[mask_upd] = True
                
                lap = lap_filled

        return lap

    # mu * Laplacian(u)
    # Using mask-aware Laplacian to handle no-slip properly at boundary cells
    fx = mu * laplacian_mask_aware(u, dx, dy, dz, mask)
    fy = mu * laplacian_mask_aware(v, dx, dy, dz, mask)
    fz = mu * laplacian_mask_aware(w, dx, dy, dz, mask)
    
    # - rho * (u . grad)u
    if rho > 0:
        grad_u_z, grad_u_y, grad_u_x = np.gradient(u, dz, dy, dx)
        grad_v_z, grad_v_y, grad_v_x = np.gradient(v, dz, dy, dx)
        grad_w_z, grad_w_y, grad_w_x = np.gradient(w, dz, dy, dx)
        
        advection_u = u * grad_u_x + v * grad_u_y + w * grad_u_z
        advection_v = u * grad_v_x + v * grad_v_y + w * grad_v_z
        advection_w = u * grad_w_x + v * grad_w_y + w * grad_w_z
        
        fx -= rho * advection_u
        fy -= rho * advection_v
        fz -= rho * advection_w

    print(f"  Force field stats (SI):")
    print(f"    Fx: mean={np.mean(np.abs(fx[mask])): .4e}")
    print(f"    Fy: mean={np.mean(np.abs(fy[mask])): .4e}")
    print(f"    Fz: mean={np.mean(np.abs(fz[mask])): .4e}")

    # Solve Poisson
    try:
        from physics import solve_poisson
        
        dirichlet_mask = None
        dirichlet_values = 0.0
        
        # Determine flow direction for inlet/outlet identification
        w_m = np.mean(w[mask] if mask is not None else w)
        
        if flow_direction == 'positive':
            plane_inlet, plane_outlet = 0, -1
        elif flow_direction == 'negative':
            plane_inlet, plane_outlet = -1, 0
        else: # auto
            if w_m >= 0:
                plane_inlet, plane_outlet = 0, -1
            else:
                plane_inlet, plane_outlet = -1, 0
            
        if anchor != 'none':
            dirichlet_mask = np.zeros_like(mask, dtype=bool)
            if anchor == 'outlet':
                dirichlet_mask[plane_outlet, :, :] = True
            elif anchor == 'inlet':
                dirichlet_mask[plane_inlet, :, :] = True
            dirichlet_mask = dirichlet_mask & mask

        print(f"Solving pressure Poisson equation (anchor={anchor} at Z-plane, dir={flow_direction})...")
        p = solve_poisson(None, mask, dx, dy, dz, force_field=(fx, fy, fz), 
                          wall_bc=wall_bc, dirichlet_mask=dirichlet_mask, dirichlet_values=0.0)
        return p
    except ImportError:
        print("Warning: could not import solve_poisson from physics.py. Pressure recovery failed.")
        return np.zeros_like(u)

def compute_interface_drag(u, v, w, pressure, viscosity, dx, dy, dz, mask, labels=None, method='staircase', mesh_step=1, volume=None, background_mask=None):
    """
    Compute total force (drag) on fluid-solid or fluid-fluid interfaces.
    
    Methods:
        'staircase': Sums forces over discrete voxel faces (fast, staircase area).
        'mesh': Triangulates surface using Marching Cubes and integrates stresses (accurate).
    """
    if method == 'mesh':
        return compute_interface_drag_mesh(u, v, w, pressure, viscosity, dx, dy, dz, mask, labels, 
                                         mesh_step=mesh_step, volume=volume, background_mask=background_mask)
    
    if labels is None:
        labels = np.unique(mask)
        labels = labels[labels > 0]
    
    results = {}
    for label in labels:
        results[label] = {
            'Fx_v': 0.0, 'Fy_v': 0.0, 'Fz_v': 0.0,
            'Fx_v_tan': 0.0, 'Fy_v_tan': 0.0, 'Fz_v_tan': 0.0,
            'Fx_v_nor': 0.0, 'Fy_v_nor': 0.0, 'Fz_v_nor': 0.0,
            'Fx_p': 0.0, 'Fy_p': 0.0, 'Fz_p': 0.0,
            'Area': 0.0
        }
    
    # Unit areas for faces normal to Z, Y, X
    dA = [dy*dx, dz*dx, dz*dy] 
    h = [dz, dy, dx]
    
    # 3D arrays to helper for face gradients
    # Axis 0: Z, Axis 1: Y, Axis 2: X
    for axis in range(3):
        # Shift mask and fields to find faces
        s_curr = [slice(None)] * 3
        s_next = [slice(None)] * 3
        s_curr[axis] = slice(0, -1)
        s_next[axis] = slice(1, None)
        
        m_curr = mask[tuple(s_curr)]
        m_next = mask[tuple(s_next)]
        
        # Area of one face
        area = dA[axis]
        step = h[axis]
        
        # Interface type A: Fluid(curr) -> Solid(next)
        # Normal n points from curr to next: e.g. n=(0,0,1) for axis 0
        for label in labels:
            if label not in results: continue
            
            # 1. Fluid -> Solid
            idx = (m_curr == 0) & (m_next == label)
            if np.any(idx):
                results[label]['Area'] += np.sum(idx) * area
                
                # Pressure drag: F_p = -p * n * dA
                if pressure is not None:
                    p_face = 0.5 * (pressure[tuple(s_curr)][idx] + pressure[tuple(s_next)][idx])
                    # Traction t = -p*n. Force on solid = t*area?
                    # If n points INTO solid, pushing ON solid is in direction n.
                    # so F_p = p * n * area.
                    p_drag = p_face * area
                    if axis == 0: results[label]['Fz_p'] += np.sum(p_drag)
                    elif axis == 1: results[label]['Fy_p'] += np.sum(p_drag)
                    elif axis == 2: results[label]['Fx_p'] += np.sum(p_drag)

                # Normal gradient components (using distance from cell center to face = step/2)
                # du/dn into solid is (u_solid - u_fluid) / (step/2) = -2*u_fluid / step
                du_dn = -2.0 * u[tuple(s_curr)][idx] / step
                dv_dn = -2.0 * v[tuple(s_curr)][idx] / step
                dw_dn = -2.0 * w[tuple(s_curr)][idx] / step
                
                # Force on solid is WITH flow. du/dn is negative, so use negative sign.
                if axis == 0: # n = (0,0,1)
                    dfz_v = viscosity * 2 * dw_dn * area
                    dfx_v = viscosity * du_dn * area
                    dfy_v = viscosity * dv_dn * area
                    results[label]['Fz_v'] -= np.sum(dfz_v)
                    results[label]['Fz_v_nor'] -= np.sum(dfz_v)
                    results[label]['Fx_v'] -= np.sum(dfx_v)
                    results[label]['Fx_v_tan'] -= np.sum(dfx_v)
                    results[label]['Fy_v'] -= np.sum(dfy_v)
                    results[label]['Fy_v_tan'] -= np.sum(dfy_v)
                elif axis == 1: # n = (0,1,0)
                    dfy_v = viscosity * 2 * dv_dn * area
                    dfx_v = viscosity * du_dn * area
                    dfz_v = viscosity * dw_dn * area
                    results[label]['Fy_v'] -= np.sum(dfy_v)
                    results[label]['Fy_v_nor'] -= np.sum(dfy_v)
                    results[label]['Fx_v'] -= np.sum(dfx_v)
                    results[label]['Fx_v_tan'] -= np.sum(dfx_v)
                    results[label]['Fz_v'] -= np.sum(dfz_v)
                    results[label]['Fz_v_tan'] -= np.sum(dfz_v)
                elif axis == 2: # n = (1,0,0)
                    dfx_v = viscosity * 2 * du_dn * area
                    dfy_v = viscosity * dv_dn * area
                    dfz_v = viscosity * dw_dn * area
                    results[label]['Fx_v'] -= np.sum(dfx_v)
                    results[label]['Fx_v_nor'] -= np.sum(dfx_v)
                    results[label]['Fy_v'] -= np.sum(dfy_v)
                    results[label]['Fy_v_tan'] -= np.sum(dfy_v)
                    results[label]['Fz_v'] -= np.sum(dfz_v)
                    results[label]['Fz_v_tan'] -= np.sum(dfz_v)

            # 2. Solid -> Fluid
            idx = (m_curr == label) & (m_next == 0)
            if np.any(idx):
                results[label]['Area'] += np.sum(idx) * area
                
                if pressure is not None:
                    p_face = 0.5 * (pressure[tuple(s_curr)][idx] + pressure[tuple(s_next)][idx])
                    # n is negative (out of solid?) No, we want n pointing INTO solid.
                    # staircase search m_curr=solid -> m_next=fluid.
                    # n should point curr->next? No, solid->fluid is OUT of solid.
                    # So n = -(curr->next) = -axis.
                    # F_p = p * n * area = -p * area.
                    p_drag = -p_face * area
                    if axis == 0: results[label]['Fz_p'] += np.sum(p_drag)
                    elif axis == 1: results[label]['Fy_p'] += np.sum(p_drag)
                    elif axis == 2: results[label]['Fx_p'] += np.sum(p_drag)

                # Viscous drag. t = mu * du/dn. n is -axis.
                # du/dn (along -axis) is (u[curr]-u[next])/step ?
                # (u_solid - u_fluid) / (step/2) = -2*u_fluid / step.
                # n is -1. So t = mu * (-2*u_fluid/step) * (-1) = 2*mu*u_fluid/step.
                # wait, let's just use the same logic: F = mu * du/dn * area.
                # du/dn into solid is (u_solid - u_fluid) / (step/2) = -2*u_next / step.
                # Since n is -1 (for axis 0), force is t*n*area? No.
                # Let's just be consistent: du/dn (into solid) * mu * area.
                du_dn = 2.0 * (0.0 - u[tuple(s_next)][idx]) / step
                dv_dn = 2.0 * (0.0 - v[tuple(s_next)][idx]) / step
                dw_dn = 2.0 * (0.0 - w[tuple(s_next)][idx]) / step
                
                # du/dn into solid is (u_solid - u_fluid) / (step/2) = -2*u_fluid / step.
                # Since u_fluid is in u_next:
                du_dn = -2.0 * u[tuple(s_next)][idx] / step
                dv_dn = -2.0 * v[tuple(s_next)][idx] / step
                dw_dn = -2.0 * w[tuple(s_next)][idx] / step
                
                if axis == 0:
                    dfz_v = viscosity * 2 * dw_dn * area
                    dfx_v = viscosity * du_dn * area
                    dfy_v = viscosity * dv_dn * area
                    results[label]['Fz_v'] -= np.sum(dfz_v)
                    results[label]['Fz_v_nor'] -= np.sum(dfz_v)
                    results[label]['Fx_v'] -= np.sum(dfx_v)
                    results[label]['Fx_v_tan'] -= np.sum(dfx_v)
                    results[label]['Fy_v'] -= np.sum(dfy_v)
                    results[label]['Fy_v_tan'] -= np.sum(dfy_v)
                elif axis == 1:
                    dfy_v = viscosity * 2 * dv_dn * area
                    dfx_v = viscosity * du_dn * area
                    dfz_v = viscosity * dw_dn * area
                    results[label]['Fy_v'] -= np.sum(dfy_v)
                    results[label]['Fy_v_nor'] -= np.sum(dfy_v)
                    results[label]['Fx_v'] -= np.sum(dfx_v)
                    results[label]['Fx_v_tan'] -= np.sum(dfx_v)
                    results[label]['Fz_v'] -= np.sum(dfz_v)
                    results[label]['Fz_v_tan'] -= np.sum(dfz_v)
                elif axis == 2:
                    dfx_v = viscosity * 2 * du_dn * area
                    dfy_v = viscosity * dv_dn * area
                    dfz_v = viscosity * dw_dn * area
                    results[label]['Fx_v'] -= np.sum(dfx_v)
                    results[label]['Fx_v_nor'] -= np.sum(dfx_v)
                    results[label]['Fy_v'] -= np.sum(dfy_v)
                    results[label]['Fy_v_tan'] -= np.sum(dfy_v)
                    results[label]['Fz_v'] -= np.sum(dfz_v)
                    results[label]['Fz_v_tan'] -= np.sum(dfz_v)

    # Normalize by volume if requested
    if volume:
        for label in results:
            r = results[label]
            r['Mx'] = r['Fx'] / volume
            r['My'] = r['Fy'] / volume
            r['Mz'] = r['Fz'] / volume

    return results

def compute_interface_drag_mesh(u, v, w, pressure, viscosity, dx, dy, dz, mask, labels=None, mesh_step=1, volume=None, background_mask=None):
    """
    Compute drag force by triangulating the interface (Marching Cubes).
    Uses the "offset velocity" method for accurate stress recovery.
    
    If background_mask is provided, interfaces are classified as:
    - 'water': Neighbor in background_mask is 1 (pore space).
    - 'solid': Neighbor in background_mask is 0 (solid matrix).
    """
    try:
        from skimage import measure
        from scipy.ndimage import map_coordinates
    except ImportError:
        print("Warning: skimage or scipy not found. Falling back to staircase drag.")
        return compute_interface_drag(u, v, w, pressure, viscosity, dx, dy, dz, mask, labels, method='staircase')

    if labels is None:
        labels = np.unique(mask)
        labels = labels[labels > 0]
    
    results = {}
    for label in labels:
        # Extract isosurface for this label (level=0.5 on 0->1 mask)
        label_mask = (mask == label).astype(float)
        if not np.any(label_mask): continue
        
        try:
            # verts/normals are in voxel-index coordinates [z, y, x]
            # Level 0.5: normals point in direction of increasing mask (into solid)
            # step_size controls the coarseness of the mesh
            verts, faces, normals, _ = measure.marching_cubes(label_mask, level=0.5, step_size=mesh_step)
        except (ValueError, RuntimeError):
            continue
            
        tri_verts = verts[faces]
        centroids = tri_verts.mean(axis=1) # [N_tri, 3] in [z, y, x]
        
        # Triangle area and unit normals in physical space
        # e1 = v1 - v0, e2 = v2 - v0
        v0 = tri_verts[:, 0, :]
        v1 = tri_verts[:, 1, :]
        v2 = tri_verts[:, 2, :]
        e1 = (v1 - v0) * [dz, dy, dx]
        e2 = (v2 - v0) * [dz, dy, dx]
        n_scaled = 0.5 * np.cross(e1, e2) 
        tri_areas = np.linalg.norm(n_scaled, axis=1)
        
        # Normalize n vectors to get units (pointing into solid) [z, y, x]
        # We also need them in voxel units for offset sampling
        n_unit_physical = n_scaled / np.maximum(tri_areas[:, np.newaxis], 1e-20)
        n_unit_voxel = n_unit_physical / [dz, dy, dx]
        n_unit_voxel /= np.linalg.norm(n_unit_voxel, axis=1)[:, np.newaxis]

        # Offset distance (voxel units)
        delta_voxel = 0.25
        delta_phys = delta_voxel * np.sqrt((n_unit_voxel[:,0]*dz)**2 + (n_unit_voxel[:,1]*dy)**2 + (n_unit_voxel[:,2]*dx)**2)
        
        # User specified: Velocity is always INSIDE the labeled phase (the oil)
        # n_unit_voxel points INTO the labeled phase (mask 0->1)
        sample_coords = (centroids + delta_voxel * n_unit_voxel).T
        outer_coords = (centroids - delta_voxel * n_unit_voxel).T
        
        # 1. Stress Sampling
        u_inner = map_coordinates(u, sample_coords, order=3, mode='nearest')
        v_inner = map_coordinates(v, sample_coords, order=3, mode='nearest')
        w_inner = map_coordinates(w, sample_coords, order=3, mode='nearest')
        
        u_interface = map_coordinates(u, centroids.T, order=1, mode='nearest')
        v_interface = map_coordinates(v, centroids.T, order=1, mode='nearest')
        w_interface = map_coordinates(w, centroids.T, order=1, mode='nearest')
        
        # Traction ON the labeled phase (resistive drag)
        tx_v = viscosity * (u_interface - u_inner) / delta_phys
        ty_v = viscosity * (v_interface - v_inner) / delta_phys
        tz_v = viscosity * (w_interface - w_inner) / delta_phys
        
        # 2. Pressure Traction: p * n
        p_tri = map_coordinates(pressure, centroids.T, order=1) if pressure is not None else np.zeros(len(centroids))
        
        # n_unit_physical components (nx, ny, nz)
        nz_phys, ny_phys, nx_phys = n_unit_physical[:, 0], n_unit_physical[:, 1], n_unit_physical[:, 2]
        
        tx_p = p_tri * nx_phys
        ty_p = p_tri * ny_phys
        tz_p = p_tri * nz_phys
        
        # Decompose Viscous Traction into Tangential (Shear) and Normal parts
        t_v_dot_n = tx_v * nx_phys + ty_v * ny_phys + tz_v * nz_phys
        
        tx_v_nor = t_v_dot_n * nx_phys
        ty_v_nor = t_v_dot_n * ny_phys
        tz_v_nor = t_v_dot_n * nz_phys
        
        tx_v_tan = tx_v - tx_v_nor
        ty_v_tan = ty_v - ty_v_nor
        tz_v_tan = tz_v - tz_v_nor

        # Classification
        if background_mask is not None:
            bg_near = map_coordinates(background_mask.astype(float), outer_coords, order=0, mode='nearest')
            is_water = bg_near > 0.5
            is_solid = ~is_water
        else:
            is_water = np.ones(len(tri_areas), dtype=bool)
            is_solid = np.zeros(len(tri_areas), dtype=bool)
        
        # Integrate over area
        results[label] = {
            'Fx_v': np.sum(tx_v * tri_areas),
            'Fy_v': np.sum(ty_v * tri_areas),
            'Fz_v': np.sum(tz_v * tri_areas),
            'Fx_v_tan': np.sum(tx_v_tan * tri_areas),
            'Fy_v_tan': np.sum(ty_v_tan * tri_areas),
            'Fz_v_tan': np.sum(tz_v_tan * tri_areas),
            'Fx_v_nor': np.sum(tx_v_nor * tri_areas),
            'Fy_v_nor': np.sum(ty_v_nor * tri_areas),
            'Fz_v_nor': np.sum(tz_v_nor * tri_areas),
            'Fx_p': np.sum(tx_p * tri_areas),
            'Fy_p': np.sum(ty_p * tri_areas),
            'Fz_p': np.sum(tz_p * tri_areas),
            'Area': np.sum(tri_areas),
            
            # Phase-split components
            'Fx_water': np.sum((tx_v[is_water] + tx_p[is_water]) * tri_areas[is_water]),
            'Fy_water': np.sum((ty_v[is_water] + ty_p[is_water]) * tri_areas[is_water]),
            'Fz_water': np.sum((tz_v[is_water] + tz_p[is_water]) * tri_areas[is_water]),
            'Fx_solid': np.sum((tx_v[is_solid] + tx_p[is_solid]) * tri_areas[is_solid]),
            'Fy_solid': np.sum((ty_v[is_solid] + ty_p[is_solid]) * tri_areas[is_solid]),
            'Fz_solid': np.sum((tz_v[is_solid] + tz_p[is_solid]) * tri_areas[is_solid]),
            'Area_water': np.sum(tri_areas[is_water]),
            'Area_solid': np.sum(tri_areas[is_solid])
        }
        
        # Combine
        r = results[label]
        r['Fx'] = r['Fx_v'] + r['Fx_p']
        r['Fy'] = r['Fy_v'] + r['Fy_p']
        r['Fz'] = r['Fz_v'] + r['Fz_p']
        
        if volume:
            r['Mx'] = r['Fx'] / volume
            r['My'] = r['Fy'] / volume
            r['Mz'] = r['Fz'] / volume
            
    return results

def compute_permeability_from_pressure(u, v, w, pressure, viscosity, dx, dy, dz):
    """
    Compute permeability using Darcy's Law: k = -μ * (U0 · ∇P) / |∇P|^2
    where U0 is the bulk Darcy velocity and ∇P is the bulk pressure gradient.
    
    Args:
        u, v, w: Velocity components (SI units)
        pressure: Pressure field (Pa)
        viscosity: Dynamic viscosity (Pa·s)
        dx, dy, dz: Grid spacing
        
    Returns:
        k: Permeability (m²)
    """
    # 1. Darcy velocity (average over TOTAL domain)
    u0 = np.mean(u)
    v0 = np.mean(v)
    w0 = np.mean(w)
    U0_vec = np.array([u0, v0, w0])
    
    # 2. Bulk pressure gradient
    # Macroscopic gradient across the sample.
    dp_dz, dp_dy, dp_dx = np.gradient(pressure, dz, dy, dx)
    
    grad_p_x = np.mean(dp_dx)
    grad_p_y = np.mean(dp_dy)
    grad_p_z = np.mean(dp_dz)
    grad_P_vec = np.array([grad_p_x, grad_p_y, grad_p_z])
    
    grad_P_mag2 = np.sum(grad_P_vec**2)
    
    if grad_P_mag2 == 0:
        return 0.0
        
    # 3. Apply Darcy's Law: U0 = -(k/μ) * grad_P
    # k = -μ * (U0 · grad_P) / |grad_P|^2
    k = -viscosity * np.dot(U0_vec, grad_P_vec) / grad_P_mag2
    
    return k
