#!/usr/bin/env python3
"""
Unified flow analysis script.

Analyzes interpolated velocity fields with selectable calculations:
- Strain rate (shear rate) tensor
- Viscous dissipation (Pilotti 2002)
"""

import argparse
import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from velocity_analysis import (compute_strain_rate, compute_viscous_dissipation, 
                               compute_vorticity, compute_permeability, 
                               compute_astarita_flow_type, compute_pressure_field,
                               compute_permeability_from_pressure, compute_interface_drag)

BASENAME = "reference_downscaled"#"sinteredGlass_interpolated_sibson100_var"#"reference_downscaled"#"sinteredGlass_interpolated_sibson100_var"#"sinteredGlass_interpolated_sibson50_var"
INPUTDATA= BASENAME + ".npz"
OUTPUTDATA= BASENAME + "_analysis.npz"
VOXEL_SIZE = 2.36e-5 # voxel size in meters
DT = 35.0 # time per frame in seconds

def load_velocity_field(npz_path):
    """
    Load velocity field from NPZ file.
    Returns: (u, v, w, x, y, z, mask)
    """
    data = np.load(npz_path)
    
    # Check for required fields
    required = ['u', 'v', 'w', 'x', 'y', 'z']
    for field in required:
        if field not in data:
            raise ValueError(f"NPZ file missing required field: {field}")
    
    u = data['u']
    v = data['v']
    w = data['w']
    x = data['x']
    y = data['y']
    z = data['z']
    mask = data.get('mask', np.ones(u.shape, dtype=bool))
    
    print(f"Loaded velocity field:")
    print(f"  Shape: {u.shape}")
    print(f"  Grid: x[{x.min():.2f}, {x.max():.2f}], y[{y.min():.2f}, {y.max():.2f}], z[{z.min():.2f}, {z.max():.2f}]")
    
    return u, v, w, x, y, z, mask

def show_scalar_field(scalar_field, x, y, z, mask=None, field_name="Scalar Field", log_scale=False, fig=None, interactive=True, cmap=None, clim=None):
    """
    Interactive or Static 3D scalar field viewer with slice navigation and optional mask overlay.
    """
    nz, ny, nx = scalar_field.shape
    
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [nx, nx, ny]})
        show_at_end = True
    else:
        if len(fig.axes) >= 3:
            axes = fig.axes[:3]
        else:
            fig.clf()
            axes = fig.subplots(1, 3, gridspec_kw={'width_ratios': [nx, nx, ny]})
        show_at_end = False
        
    fig.suptitle(f'{field_name}', fontsize=14)
    
    # Initial slice indices
    iz = nz // 2
    iy = ny // 2
    ix = nx // 2
    
    # Determine colormap limits
    if mask is not None:
        valid_data = scalar_field[mask]
    else:
        valid_data = scalar_field[scalar_field > 0]
    
    if log_scale and len(valid_data) > 0:
        plot_data = np.log10(scalar_field + 1e-20)
        vmin = np.log10(np.percentile(valid_data, 1) + 1e-20)
        vmax = np.log10(np.percentile(valid_data, 99) + 1e-20)
        curr_cmap = 'hot' if cmap is None else cmap
        label = f"log10({field_name})"
    else:
        plot_data = scalar_field
        if clim is not None:
            vmin, vmax = clim
        else:
            vmin = np.percentile(valid_data, 1) if len(valid_data) > 0 else 0
            vmax = np.percentile(valid_data, 99) if len(valid_data) > 0 else scalar_field.max()
        curr_cmap = 'viridis' if cmap is None else cmap
        label = field_name
    
    # Helper to generate RGBA mask (black for solid, transparent for fluid)
    def get_mask_rgba(mask_3d, axis, idx):
        if mask_3d is None: return None
        if axis == 0: m_slice = mask_3d[idx, :, :]
        elif axis == 1: m_slice = mask_3d[:, idx, :]
        else: m_slice = mask_3d[:, :, idx]
        
        rgba = np.zeros(m_slice.shape + (4,))
        rgba[~m_slice] = [0, 0, 0, 1] # Black for solid
        return rgba

    # Create initial images
    im0 = axes[0].imshow(plot_data[iz, :, :], cmap=curr_cmap, vmin=vmin, vmax=vmax, origin='lower')
    mask_im0 = None
    if mask is not None:
        mask_im0 = axes[0].imshow(get_mask_rgba(mask, 0, iz), origin='lower')
    axes[0].set_title(f'XY plane (Z={z[iz]:.1f})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    im1 = axes[1].imshow(plot_data[:, iy, :], cmap=curr_cmap, vmin=vmin, vmax=vmax, origin='lower')
    mask_im1 = None
    if mask is not None:
        mask_im1 = axes[1].imshow(get_mask_rgba(mask, 1, iy), origin='lower')
    axes[1].set_title(f'XZ plane (Y={y[iy]:.1f})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    
    im2 = axes[2].imshow(plot_data[:, :, ix], cmap=curr_cmap, vmin=vmin, vmax=vmax, origin='lower')
    mask_im2 = None
    if mask is not None:
        mask_im2 = axes[2].imshow(get_mask_rgba(mask, 2, ix), origin='lower')
    axes[2].set_title(f'YZ plane (X={x[ix]:.1f})')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    
    # Unified colorbar on the right
    fig.colorbar(im2, ax=axes.ravel().tolist(), label=label, aspect=30, pad=0.08)
    
    if interactive:
        # Add sliders
        plt.subplots_adjust(bottom=0.25)
        
        ax_z = fig.add_axes([0.15, 0.15, 0.2, 0.03])
        ax_y = fig.add_axes([0.15, 0.10, 0.2, 0.03])
        ax_x = fig.add_axes([0.15, 0.05, 0.2, 0.03])
        
        slider_z = Slider(ax_z, 'Z slice', 0, nz-1, valinit=iz, valstep=1)
        slider_y = Slider(ax_y, 'Y slice', 0, ny-1, valinit=iy, valstep=1)
        slider_x = Slider(ax_x, 'X slice', 0, nx-1, valinit=ix, valstep=1)
        
        def update(val):
            iz = int(slider_z.val)
            iy = int(slider_y.val)
            ix = int(slider_x.val)
            
            im0.set_data(plot_data[iz, :, :])
            if mask_im0: mask_im0.set_data(get_mask_rgba(mask, 0, iz))
            axes[0].set_title(f'XY plane (Z={z[iz]:.1f})')
            
            im1.set_data(plot_data[:, iy, :])
            if mask_im1: mask_im1.set_data(get_mask_rgba(mask, 1, iy))
            axes[1].set_title(f'XZ plane (Y={y[iy]:.1f})')
            
            im2.set_data(plot_data[:, :, ix])
            if mask_im2: mask_im2.set_data(get_mask_rgba(mask, 2, ix))
            axes[2].set_title(f'YZ plane (X={x[ix]:.1f})')
            
            fig.canvas.draw_idle()
        
        slider_z.on_changed(update)
        slider_y.on_changed(update)
        slider_x.on_changed(update)
        
        # Store slider references
        fig._sliders = [slider_z, slider_y, slider_x]
    
    if show_at_end:
        plt.show()

    return fig

def main():
    parser = argparse.ArgumentParser(description="Analyze interpolated velocity fields.")
    parser.add_argument("--input", "-i", default=INPUTDATA, help="Input NPZ file with velocity field (default: velocity_field.npz)")
    
    # Analysis options
    parser.add_argument("--strain-rate", action="store_true", default=True, help="Compute strain rate tensor (default: True)")
    parser.add_argument("--dissipation", action="store_true", default=True, help="Compute viscous dissipation (default: True)")
    parser.add_argument("--vorticity", action="store_true", default=True, help="Compute vorticity magnitude (default: True)")
    parser.add_argument("--permeability_dissipation", action="store_true", default=True, help="Estimate dissipation-based permeability (default: True)")
    parser.add_argument("--permeability_pressure", action="store_true", default=True, help="Estimate pressure-based permeability (default: True)")
    parser.add_argument("--pressure", action="store_true", default=True, help="Recover pressure field (default: True)")
    parser.add_argument("--pressure-wall-bc", choices=['zero-neumann', 'inhomogeneous'], default='zero-neumann', help="Wall BC for pressure: 'zero-neumann' (stable) or 'inhomogeneous' (exact but noisy)")
    parser.add_argument("--pressure-anchor", choices=['inlet', 'outlet', 'none'], default='outlet', help="Pressure anchor location (default: outlet)")
    parser.add_argument("--viscosity", type=float, default=0.001, help="Dynamic viscosity (Pa·s) for dissipation and pressure (default: 0.001)")
    parser.add_argument("--rho", type=float, default=0.0, help="Fluid density (kg/m³) for inertial pressure terms (default: 0.0 for Stokes)")
    parser.add_argument("--flow-direction", choices=['auto', 'positive', 'negative'], default='auto', help="Flow direction along Z-axis (default: auto)")
    parser.add_argument("--drag", action="store_true", default=True, help="Compute interface drag force (default: True)")
    parser.add_argument("--drag-labels", type=int, nargs='*', help="Specific mask labels for drag calculation (default: all solid)")
    parser.add_argument("--drag-method", choices=['staircase', 'mesh'], default='mesh', help="Method for drag calculation: 'staircase' (discrete faces) or 'mesh' (smooth triangulation, default: mesh)")
    parser.add_argument("--drag-mesh-step", type=int, default=1, help="Mesh coarseness (step size) for marching cubes (default: 1)")
    parser.add_argument("--pore-mask", help="TIFF file containing the background pore geometry (0: solid, 1: pore) for interface classification")
    
    # Physical scaling options
    parser.add_argument("--voxel-size", type=float, default=VOXEL_SIZE, help="Voxel size in physical units (e.g. meters/voxel, default: 1.0)")
    parser.add_argument("--dt", type=float, default=DT, help="Time step in physical units (e.g. seconds, default: 1.0)")
    
    # Output options
    parser.add_argument("--output-npz", default=OUTPUTDATA, help="Output NPZ file for analysis results (default: analysis.npz)")
    parser.add_argument("--output-tif-strain", default=BASENAME + "_strain.tif", help="Output TIFF file for strain rate field")
    parser.add_argument("--output-tif-dissipation", default=BASENAME + "_dissipation.tif", help="Output TIFF file for dissipation field")
    parser.add_argument("--output-tif-vorticity", default=BASENAME + "_vorticity.tif", help="Output TIFF file for vorticity Magnitude field")
    parser.add_argument("--output-tif-pressure", default=BASENAME + "_pressure.tif", help="Output TIFF file for pressure field")
    
    # Visualization options
    parser.add_argument("--plot-strain", action="store_true", default=False, help="Show strain rate plot (default: True)")
    parser.add_argument("--plot-dissipation", action="store_true", default=False, help="Show dissipation plot (default: False)")
    parser.add_argument("--plot-vorticity", action="store_true", default=False, help="Show vorticity plot (default: False)")
    parser.add_argument("--plot-pressure", action="store_true", default=False, help="Show pressure plot (default: False)")
    parser.add_argument("--plot-velocity", action="store_true", default=False, help="Show velocity magnitude plot (default: False)")
    parser.add_argument("--plot-flowtype", action="store_true", default=False, help="Show Astarita flow type plot (default: False)")
    parser.add_argument("--log-scale", action="store_true", default=True, help="Use log scale for dissipation visualization (default: True)")
    parser.add_argument("--interactive", action="store_true", default=True, help="Enable interactive slice navigation (default: True)")
    
    # Allow disabling defaults
    parser.add_argument("--no-strain-rate", action="store_false", dest="strain_rate", help="Disable strain rate computation")
    parser.add_argument("--no-dissipation", action="store_false", dest="dissipation", help="Disable dissipation computation")
    parser.add_argument("--no-vorticity", action="store_false", dest="vorticity", help="Disable vorticity computation")
    parser.add_argument("--no-permeability_dissipation", action="store_false", dest="permeability_dissipation", help="Disable dissipation permeability computation")
    parser.add_argument("--no-permeability_pressure", action="store_false", dest="permeability_pressure", help="Disable pressure permeability computation")
    parser.add_argument("--no-pressure", action="store_false", dest="pressure", help="Disable pressure computation")
    parser.add_argument("--no-plot-dissipation", action="store_false", dest="plot_dissipation", help="Disable dissipation plot")
    parser.add_argument("--no-plot-strain", action="store_false", dest="plot_strain", help="Disable strain rate plot")
    parser.add_argument("--no-plot-vorticity", action="store_false", dest="plot_vorticity", help="Disable vorticity plot")
    parser.add_argument("--no-plot-pressure", action="store_false", dest="plot_pressure", help="Disable pressure plot")
    parser.add_argument("--no-plot-velocity", action="store_false", dest="plot_velocity", help="Disable velocity magnitude plot")
    parser.add_argument("--no-plot-flowtype", action="store_false", dest="plot_flowtype", help="Disable Astarita flow type plot")
    parser.add_argument("--no-drag", action="store_false", dest="drag", help="Disable interface drag computation")
    parser.add_argument("--no-log-scale", action="store_false", dest="log_scale", help="Disable log scale")
    parser.add_argument("--no-interactive", action="store_false", dest="interactive", help="Disable interactive slice navigation")
    parser.add_argument("--no-output-npz", action="store_const", const=None, dest="output_npz", help="Disable NPZ output")
    
    args = parser.parse_args()
    
    # If no analysis selected, default to strain rate
    if not args.strain_rate and not args.dissipation and not args.vorticity and not args.pressure:
        print("No analysis selected. Computing strain rate by default.")
        args.strain_rate = True
    
    # Initialize stats log
    stats_log = []
    def log_print(msg):
        print(msg)
        stats_log.append(msg)
    
    # Load velocity field
    log_print(f"Loading velocity field from {args.input}...")
    u, v, w, x, y, z, mask = load_velocity_field(args.input)
    
    # ENFORCE MASK: Ensure velocity is zero in solid regions
    if mask is not None:
        log_print("Enforcing zero velocity in solid regions of the mask...")
        u[~mask] = 0.0
        v[~mask] = 0.0
        w[~mask] = 0.0
        porosity = np.mean(mask)
        log_print(f"  Calculated porosity: {porosity:.4e}")
    
    # Compute raw statistics
    speed_raw = np.sqrt(u**2 + v**2 + w**2)
    valid_speed_raw = speed_raw[mask] if mask is not None else speed_raw
    
    log_print("\n--- Flow Field Statistics (Raw Scan Units) ---")
    log_print(f"  Velocity Magnitude (voxel/frame):")
    log_print(f"    Mean: {np.mean(valid_speed_raw):.4e}")
    log_print(f"    Max:  {np.max(valid_speed_raw):.4e}")
    log_print(f"    Std:  {np.std(valid_speed_raw):.4e}")
    
    # Apply physical scaling
    if args.voxel_size != 1.0 or args.dt != 1.0:
        log_print(f"Applying physical scaling: voxel_size={args.voxel_size}, dt={args.dt}...")
        scaling_velocity = args.voxel_size / args.dt
        u *= scaling_velocity
        v *= scaling_velocity
        w *= scaling_velocity
        
        x *= args.voxel_size
        y *= args.voxel_size
        z *= args.voxel_size
    
    # Compute grid spacing
    dx = x[1] - x[0] if len(x) > 1 else args.voxel_size
    dy = y[1] - y[0] if len(y) > 1 else args.voxel_size
    dz = z[1] - z[0] if len(z) > 1 else args.voxel_size
    
    # Compute physical statistics
    speed_phys = np.sqrt(u**2 + v**2 + w**2)
    valid_speed_phys = speed_phys[mask] if mask is not None else speed_phys
    
    log_print("\n--- Flow Field Statistics (Physical SI Units) ---")
    v_scal = 1e6  # m/s to um/s
    log_print(f"  Velocity Magnitude (um/s):")
    log_print(f"    Mean: {np.mean(valid_speed_phys)*v_scal:.4e}")
    log_print(f"    Max:  {np.max(valid_speed_phys)*v_scal:.4e}")
    log_print(f"    Std:  {np.std(valid_speed_phys)*v_scal:.4e}")
    
    # Flux and Volumetric Flow Rate along Z-axis
    dA = (x[1]-x[0])*(y[1]-y[0]) if len(x)>1 and len(y)>1 else args.voxel_size**2
    Q_z = np.sum(w, axis=(1, 2)) * dA  # Volumetric flow rate through each XY slice
    
    nz, ny, nx = w.shape
    A_total = nx * ny * dA  # Total cross-sectional area
    q_z = Q_z / A_total     # Darcy flux (superficial velocity)
    
    log_print("\n--- Z-Axis Flow Rates & Fluxes (SI Units) ---")
    Q_conv = 6e10  # m³/s to uL/min
    log_print(f"  Volumetric Flow Rate (Q):")
    log_print(f"    Average: {np.mean(Q_z):.4e} m³/s ({np.mean(Q_z)*Q_conv:.4e} uL/min)")
    log_print(f"    Range:   [{np.min(Q_z):.4e}, {np.max(Q_z):.4e}] m³/s")
    log_print(f"             ([{np.min(Q_z)*Q_conv:.4e}, {np.max(Q_z)*Q_conv:.4e}] uL/min)")
    log_print(f"  Darcy Flux (q = Q/A_total):")
    log_print(f"    Average: {np.mean(q_z):.4e} m/s")
    log_print(f"    Range:   [{np.min(q_z):.4e}, {np.max(q_z):.4e}] m/s")
    
    # Results dictionary
    results = {}
    
    # Compute strain rate if requested
    strain_rate = None
    if args.strain_rate or args.dissipation:  # Dissipation needs strain rate
        log_print("\n=== Computing Strain Rate ===")
        strain_rate = compute_strain_rate(u, v, w, dx, dy, dz, mask)
        results['strain_rate'] = strain_rate
        
        log_print(f"  Mean: {np.mean(strain_rate[mask] if mask is not None else strain_rate):.4e} 1/s")
        log_print(f"  Max:  {np.max(strain_rate):.4e} 1/s")
        log_print(f"  Min:  {np.min(strain_rate):.4e} 1/s")
        
        if args.output_tif_strain:
            log_print(f"Saving strain rate TIFF to {args.output_tif_strain}...")
            tifffile.imwrite(args.output_tif_strain, strain_rate.astype(np.float32), imagej=True)
    
    # Compute dissipation if requested
    dissipation = None
    if args.dissipation:
        log_print(f"\n=== Computing Viscous Dissipation ===")
        dissipation = compute_viscous_dissipation(strain_rate, args.viscosity, dx, dy, dz, mask)
        results['dissipation'] = dissipation
        results['viscosity'] = args.viscosity
        
        valid_diss = dissipation[mask] if mask is not None else dissipation
        log_print(f"  Mean: {np.mean(valid_diss):.6e} W/m³")
        log_print(f"  Max:  {np.max(dissipation):.6e} W/m³")
        log_print(f"  Min:  {np.min(dissipation):.6e} W/m³")
        log_print(f"  Total dissipation: {np.sum(valid_diss) * dx * dy * dz:.6e} W")
        
        if args.output_tif_dissipation:
            log_print(f"Saving dissipation TIFF to {args.output_tif_dissipation}...")
            tifffile.imwrite(args.output_tif_dissipation, dissipation.astype(np.float32), imagej=True)
            
    # Compute vorticity if requested
    vorticity_magnitude = None
    if args.vorticity:
        log_print(f"\n=== Computing Vorticity ===")
        vorticity_magnitude = compute_vorticity(u, v, w, dx, dy, dz, mask)
        results['vorticity_magnitude'] = vorticity_magnitude
        
        log_print(f"  Mean: {np.mean(vorticity_magnitude[mask] if mask is not None else vorticity_magnitude):.4e} 1/s")
        log_print(f"  Max:  {np.max(vorticity_magnitude):.4e} 1/s")
        
        if args.output_tif_vorticity:
            log_print(f"Saving vorticity TIFF to {args.output_tif_vorticity}...")
            tifffile.imwrite(args.output_tif_vorticity, vorticity_magnitude.astype(np.float32), imagej=True)

    # Compute pressure if requested
    pressure = None
    if args.pressure:
        log_print("\n=== Recovering Pressure Field ===")
        pressure = compute_pressure_field(u, v, w, dx, dy, dz, args.viscosity, args.rho, mask, 
                                        wall_bc=args.pressure_wall_bc, 
                                        anchor=args.pressure_anchor,
                                        flow_direction=args.flow_direction)
        results['pressure'] = pressure
        
        valid_p = pressure[mask] if mask is not None else pressure
        log_print(f"  Pressure Range: [{np.min(valid_p):.4e}, {np.max(valid_p):.4e}] Pa")
        
        # Global Pressure Drops for Diagnostic
        log_print(f"\n--- Global Pressure Drops ---")
        for ax_name, (m_start, m_end, p_start, p_end) in [
            ('Z (axial)', (mask[0], mask[-1], pressure[0], pressure[-1])),
            ('Y (trans)', (mask[:, 0], mask[:, -1], pressure[:, 0], pressure[:, -1])),
            ('X (trans)', (mask[:, :, 0], mask[:, :, -1], pressure[:, :, 0], pressure[:, :, -1]))
        ]:
            if np.any(m_start) and np.any(m_end):
                dp = np.mean(p_start[m_start]) - np.mean(p_end[m_end])
                log_print(f"  ΔP_{ax_name}: {dp: .4e} Pa")
            else:
                log_print(f"  ΔP_{ax_name}: N/A (Solid boundary)")
        
        if args.output_tif_pressure:
            log_print(f"Saving pressure TIFF to {args.output_tif_pressure}...")
            tifffile.imwrite(args.output_tif_pressure, pressure.astype(np.float32), imagej=True)
            
    # Compute permeability if requested
    if args.permeability_dissipation or args.permeability_pressure:
        log_print(f"\n=== Estimating Permeability ===")
        if args.permeability_dissipation and dissipation is not None:
            k_diss = compute_permeability(u, v, w, dissipation, args.viscosity, dx, dy, dz, mask)
            results['permeability_dissipation'] = k_diss
            log_print(f"  From Energy Dissipation (k_diss): {k_diss:.6e} m²")
        
        if args.permeability_pressure and pressure is not None:
            k_press = compute_permeability_from_pressure(u, v, w, pressure, args.viscosity, dx, dy, dz)
            results['permeability_pressure'] = k_press
            log_print(f"  From Pressure Gradient (k_press):  {k_press:.6e} m²")
            
            if args.permeability_dissipation and dissipation is not None and k_diss > 0:
                ratio = k_press / k_diss
                log_print(f"  Ratio (k_press/k_diss): {ratio:.4f}")
    
    # Compute Interface Drag if requested
    if args.drag:
        log_print("\n=== Computing Interface Drag Force ===")
        # Note: mask must be numeric labels for multi-grain drag
        drag_mask = mask.astype(int) if mask is not None else np.zeros_like(u, dtype=int)
        
        # Calculate domain volume for normalization (Eq. 6)
        nz, ny, nx = u.shape
        total_volume = nz * dz * ny * dy * nx * dx
        
        # Load background pore mask for classification if provided
        background_mask = None
        if args.pore_mask and os.path.exists(args.pore_mask):
            log_print(f"Loading background pore mask from {args.pore_mask}...")
            background_mask = tifffile.imread(args.pore_mask)
            if background_mask.shape != u.shape:
                log_print(f"  Warning: Pore mask shape {background_mask.shape} does not match velocity field {u.shape}. Skipping classification.")
                background_mask = None
            else:
                # Ensure binary (pore=1, solid=0)
                background_mask = (background_mask > 0)
        
        drag_results = compute_interface_drag(u, v, w, pressure, args.viscosity, dx, dy, dz, drag_mask, 
                                             labels=args.drag_labels, method=args.drag_method, 
                                             mesh_step=args.drag_mesh_step, volume=total_volume,
                                             background_mask=background_mask)
        results['drag'] = drag_results
        
        if not drag_results:
            log_print("  No interfaces found or labels not present.")
        else:
            for label, d in drag_results.items():
                log_print(f"  Grain/Phase Label {label}:")
                log_print(f"    Total Drag Force (N):       [{d['Fx']:.4e}, {d['Fy']:.4e}, {d['Fz']:.4e}]")
                log_print(f"    Force Density M (N/m³):     [{d['Mx']:.4e}, {d['My']:.4e}, {d['Mz']:.4e}]")
                log_print(f"    Surface Area (m²):           {d['Area']:.4e}")
                
                if background_mask is not None:
                    log_print(f"    --- Phase-Split Analysis ---")
                    log_print(f"    Water-Oil Drag (N):        [{d['Fx_water']:.4e}, {d['Fy_water']:.4e}, {d['Fz_water']:.4e}]")
                    log_print(f"    Oil-Solid Friction (N):    [{d['Fx_solid']:.4e}, {d['Fy_solid']:.4e}, {d['Fz_solid']:.4e}]")
                    log_print(f"    Water Area (m²):           {d['Area_water']:.4e}")
                    log_print(f"    Solid Area (m²):           {d['Area_solid']:.4e}")
                
                log_print(f"    --- Stress Components ---")
                log_print(f"    Viscous Force (Shear) (N):  [{d['Fx_v_tan']:.4e}, {d['Fy_v_tan']:.4e}, {d['Fz_v_tan']:.4e}]")
                log_print(f"    Viscous Force (Normal) (N): [{d['Fx_v_nor']:.4e}, {d['Fy_v_nor']:.4e}, {d['Fz_v_nor']:.4e}]")
                if pressure is not None:
                    log_print(f"    Pressure Force (N):         [{d['Fx_p']:.4e}, {d['Fy_p']:.4e}, {d['Fz_p']:.4e}]")
    
    # Save results
    if args.output_npz:
        log_print(f"\nSaving results to {args.output_npz}...")
        np.savez(args.output_npz, 
                 x=x, y=y, z=z, mask=mask,
                 **results)
    
    # Save statistics log
    stats_file = BASENAME + "_stats.txt"
    log_print(f"Saving statistics to {stats_file}...")
    with open(stats_file, 'w') as f:
        f.write("\n".join(stats_log))
    
    # Visualizations
    if args.plot_strain and strain_rate is not None:
        log_print("\nOpening strain rate viewer...")
        fig1 = plt.figure(figsize=(14, 7))
        show_scalar_field(strain_rate, x, y, z, mask, field_name="Strain Rate (Shear Rate) (1/s)", fig=fig1, interactive=args.interactive, cmap='viridis')
        if not args.interactive:
            fig1.savefig(BASENAME + "_strain.png", dpi=150)
    
    if args.plot_dissipation and dissipation is not None:
        log_print("\nOpening dissipation viewer...")
        fig2 = plt.figure(figsize=(14, 7))
        show_scalar_field(dissipation, x, y, z, mask, 
                         field_name="Viscous Dissipation (W/m³)",
                         log_scale=args.log_scale, fig=fig2, interactive=args.interactive, cmap='viridis')
        if not args.interactive:
            fig2.savefig(BASENAME + "_dissipation.png", dpi=150)
            
    if args.plot_vorticity and vorticity_magnitude is not None:
        log_print("\nOpening vorticity viewer...")
        fig3 = plt.figure(figsize=(14, 7))
        show_scalar_field(vorticity_magnitude, x, y, z, mask, field_name="Vorticity Magnitude (1/s)", fig=fig3, interactive=args.interactive, cmap='viridis')
        if not args.interactive:
            fig3.savefig(BASENAME + "_vorticity.png", dpi=150)

    if args.plot_velocity:
        log_print("\nOpening velocity magnitude viewer...")
        speed = np.sqrt(u**2 + v**2 + w**2)
        fig4 = plt.figure(figsize=(14, 7))
        show_scalar_field(speed, x, y, z, mask, field_name="Velocity Magnitude (m/s)", fig=fig4, interactive=args.interactive, cmap='viridis')
        if not args.interactive:
            fig4.savefig(BASENAME + "_velocity.png", dpi=150)

    if args.plot_pressure and pressure is not None:
        log_print("\nOpening pressure viewer...")
        fig_p = plt.figure(figsize=(14, 7))
        show_scalar_field(pressure, x, y, z, mask, field_name="Pressure Field (Pa)", fig=fig_p, interactive=args.interactive, cmap='RdBu_r')
        if not args.interactive:
            fig_p.savefig(BASENAME + "_pressure.png", dpi=150)

    if args.plot_flowtype:
        # Need strain rate and vorticity
        sr = strain_rate if strain_rate is not None else compute_strain_rate(u, v, w, dx, dy, dz, mask)
        vm = vorticity_magnitude if vorticity_magnitude is not None else compute_vorticity(u, v, w, dx, dy, dz, mask)
        
        log_print("\nOpening Astarita flow type viewer...")
        log_print("Computing Astarita flow type classification...")
        xi = compute_astarita_flow_type(sr, vm, mask)
        results['flow_type'] = xi
        
        valid_xi = xi[mask] if mask is not None else xi
        log_print("Astarita flow type statistics:")
        log_print(f"  Mean ξ: {np.mean(valid_xi):.4e}")
        log_print(f"  Max ξ (extensional): {np.max(xi):.4e}")
        log_print(f"  Min ξ (rotational): {np.min(xi):.4e}")
        
        fig5 = plt.figure(figsize=(14, 7))
        show_scalar_field(xi, x, y, z, mask, field_name="Astarita Flow Type ξ (-1:Rot, 0:Shear, 1:Ext)", 
                         fig=fig5, interactive=args.interactive, cmap='RdBu_r', clim=(-1, 1))
        if not args.interactive:
            fig5.savefig(BASENAME + "_flowtype.png", dpi=150)

    if args.plot_strain or args.plot_dissipation or args.plot_vorticity or args.plot_velocity or args.plot_flowtype or args.plot_pressure:
        plt.show()

    log_print("\nDone.")

if __name__ == "__main__":
    main()
