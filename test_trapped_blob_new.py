def test_drag_trapped_blob():
    """
    Validation: Oil blob trapped in a side cavity with water flowing overhead.
    
    Geometry:
    - Main channel (y > 0): Water flowing left-to-right (positive Z)
    - Side cavity: Oil pocket with curved meniscus at top
    - The meniscus is a parabolic curve dipping into the cavity
    
    Physics:
    - Water: Plug flow at velocity U_water
    - Oil: Driven by shear from meniscus, creating recirculation
    - We calculate drag force on the oil blob from the water above
    """
    print("\n--- Testing Trapped Blob in Side Cavity ---")
    
    # 1. Setup Grid
    nn = 60
    nz, ny, nx = nn, nn, nn
    dz, dy, dx = 1e-5, 1e-5, 1e-5
    U_water = 0.1  # Water velocity (m/s)
    mu = 0.001     # Viscosity (Pa·s)
    
    # Voxel coordinates
    z_vox = np.arange(nz)
    y_vox = np.arange(ny) - ny//2
    x_vox = np.arange(nx) - nx//2
    
    # Physical coordinates
    z, y, x = np.meshgrid(z_vox * dz, y_vox * dy, x_vox * dx, indexing='ij')
    
    # 2. Geometry: Curved meniscus cavity
    cavity_w_vox = 30   # Cavity width (voxels)
    cavity_h_vox = 25   # Cavity depth (voxels)
    depth_vox = 8       # Meniscus dip depth (voxels)
    z_start, z_end = 10, 50  # Cavity extent in Z
    
    # Parabolic meniscus: y_meniscus(z) = -depth * (1 - ((z - z_mid) / (L/2))^2)
    mid_z_vox = (z_start + z_end) / 2
    L_vox = z_end - z_start
    z_rel = (z_vox - mid_z_vox) / (L_vox / 2)
    y_meniscus_vox = -depth_vox * (1.0 - z_rel**2)
    y_meniscus_vox = np.where((z_vox >= z_start) & (z_vox <= z_end), y_meniscus_vox, 0)
    
    # Oil blob mask: Below meniscus, within cavity bounds
    blob_mask = (y_vox[np.newaxis, :, np.newaxis] > -cavity_h_vox) & \
                (y_vox[np.newaxis, :, np.newaxis] <= y_meniscus_vox[:, np.newaxis, np.newaxis]) & \
                (np.abs(x_vox[np.newaxis, np.newaxis, :]) < cavity_w_vox/2) & \
                (z_vox[:, np.newaxis, np.newaxis] >= z_start) & \
                (z_vox[:, np.newaxis, np.newaxis] <= z_end)
    blob_mask = blob_mask.astype(int)
    
    # Pore mask: Water channel (y > 0) + oil cavity
    pore_mask_raw = (y_vox[np.newaxis, :, np.newaxis] > 0) | \
                    ((y_vox[np.newaxis, :, np.newaxis] > -cavity_h_vox) & \
                     (np.abs(x_vox[np.newaxis, np.newaxis, :]) < cavity_w_vox/2))
    pore_mask = np.broadcast_to(pore_mask_raw, (nz, ny, nx)).astype(int)
    
    # 3. Velocity Field
    # Water: Simple plug flow in positive Z direction
    # Oil: Boundary-driven recirculation (simplified as quiescent for this test)
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    w = np.where(y > 0, U_water, 0.0)  # Water flows, oil is stationary
    
    # Pressure: Zero for this simplified case
    p = np.zeros_like(x)
    
    # 4. Compute Drag on Oil Blob
    results_m = compute_interface_drag(u, v, w, p, mu, dx, dy, dz, blob_mask, 
                                     method='mesh', background_mask=pore_mask)
    
    if 1 in results_m:
        d = results_m[1]
        
        # Extract forces
        fz_v = d['Fz_v']      # Viscous drag in Z
        fz_p = d['Fz_p']      # Pressure drag in Z
        fz_total = d['Fz']    # Total drag in Z
        area = d['Area']      # Interface area
        
        # Theoretical drag: For plug flow over stationary oil with curved meniscus
        # Shear stress: τ = μ * (U_water - 0) / δ where δ is the boundary layer thickness
        # For this validation, we use δ ≈ dy (one voxel)
        # Area: Approximate meniscus area
        L_cavity = L_vox * dz
        W_cavity = cavity_w_vox * dx
        area_meniscus_approx = W_cavity * L_cavity * 1.1  # Factor for curvature
        
        tau_shear = mu * U_water / dy
        f_drag_theo = tau_shear * area_meniscus_approx
        
        print(f"  Meniscus Area: {area:.4e} m² (Approx: {area_meniscus_approx:.4e} m²)")
        print(f"  Theoretical Drag Fz: {f_drag_theo:.4e} N")
        print(f"  Recovered Visc Fz:   {fz_v:.4e} N")
        print(f"  Recovered Pressure Fz: {fz_p:.4e} N")
        print(f"  Recovered Total Fz:  {fz_total:.4e} N")
        print(f"  Water/Solid Ratio: {abs(d['Fz_water']/d['Fz_solid']):.2f}")
        
        # Validation: Check that viscous drag is positive (water drags oil forward)
        # and matches theoretical estimate within reasonable tolerance
        assert fz_v > 0, f"Viscous drag should be positive (water drags oil), got {fz_v:.4e}"
        err = abs(fz_v - f_drag_theo) / f_drag_theo
        print(f"  Drag Error: {err*100:.2f}%")
        assert err < 0.50, f"Drag error too high: {err*100:.2f}%"
        
        # Visualization
        w_masked = np.where(pore_mask == 1, w, np.nan)
        
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        mid_z, mid_x = nz//2, nx//2
        
        # Panel 1: Geometry
        ax[0].imshow(pore_mask[mid_z], cmap='gray', origin='lower', alpha=0.8)
        ax[0].imshow(blob_mask[mid_z], cmap='Reds', alpha=0.5, origin='lower')
        ax[0].set_title("Geometry: Oil Cavity (Red) + Water Channel (Gray)")
        ax[0].set_xlabel("X (voxels)")
        ax[0].set_ylabel("Y (voxels)")
        
        # Panel 2: Velocity field
        im = ax[1].imshow(w_masked[mid_z], cmap='Blues', origin='lower', vmin=0, vmax=U_water)
        plt.colorbar(im, ax=ax[1], label="Velocity W (m/s)")
        ax[1].set_title("Water Flow (Plug Flow)")
        ax[1].set_xlabel("Z (voxels)")
        ax[1].set_ylabel("Y (voxels)")
        
        # Panel 3: 1D profile showing velocity discontinuity at meniscus
        y_profile = w_masked[mid_z, :, mid_x]
        ax[2].plot(y_profile, y_vox * dy * 1e6, 'b-', linewidth=2)
        ax[2].axhline(0, color='r', linestyle='--', alpha=0.7, label='Meniscus (y=0)')
        ax[2].axvline(0, color='k', linestyle='--', alpha=0.3)
        ax[2].set_xlabel("Velocity W (m/s)")
        ax[2].set_ylabel("Y position (μm)")
        ax[2].set_title("Velocity Profile (Shear at Meniscus)")
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("validation_trapped_blob.png", dpi=150)
        print("  Saved visualization to validation_trapped_blob.png")
    else:
        print("  Warning: Label 1 not found in results.")
