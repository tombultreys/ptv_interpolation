import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

class SliceViewer:

    def __init__(self, u, v, w, x, y, z, mask=None, input_df=None, fig=None):
        """
        u, v, w: 3D velocity components (nz, ny, nx). 
                 Can be tuples/lists (cleaned, initial) if both are provided.
        x, y, z: 1D coordinate arrays
        """
        if isinstance(u, (tuple, list)):
            self.u_cleaned, self.u_init = u
            self.v_cleaned, self.v_init = v
            self.w_cleaned, self.w_init = w
            self.has_dual_fields = True
            self.u, self.v, self.w = self.u_cleaned, self.v_cleaned, self.w_cleaned
            self.field_name = 'Cleaned'
        else:
            self.u, self.v, self.w = u, v, w
            self.has_dual_fields = False
            self.field_name = 'Velocity'
            
        self.coords = [z, y, x]
        self.dim_names = ['Z', 'Y', 'X']
        self.mask = mask
        self.input_df = input_df
        
        self.shape = self.u.shape # (nz, ny, nx)
        self.axis = 1 # Default: XZ plane (slice along Y)
        self.current_slice = self.shape[self.axis] // 2
        self.v_scale = 1.0 # Default vector scale
        
        # Calculate global max speed for color scaling
        all_speeds = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        valid_speeds = all_speeds[~np.isnan(all_speeds)]
        self.global_max_speed = np.max(valid_speeds) if valid_speeds.size > 0 else 1.0
        if self.global_max_speed > 1e10: self.global_max_speed = 100.0
        if self.global_max_speed <= 0: self.global_max_speed = 1.0
        
        self.vmin = 0.0
        self.vmax = self.global_max_speed
        if self.vmin >= self.vmax:
             self.vmax = self.vmin + 1e-4
        
        # Pre-calculate 3D speed for the background
        self.speed = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        self.color_type = '3D Speed'
        
        # Visibility flags
        self.show_grid_vectors = True
        self.show_input_vectors = True
        self.show_mask = True
        
        if fig is None:
            self.fig, self.ax = plt.subplots(figsize=(13, 8))
            self._is_custom_fig = False
        else:
            self.fig = fig
            self._is_custom_fig = True
            if len(fig.axes) > 0:
                self.ax = fig.axes[0]
            else:
                self.ax = fig.add_subplot(111)
                
        plt.subplots_adjust(bottom=0.2, left=0.25, right=0.9)
        
        # Colorbar axis
        self.cax = self.fig.add_axes([0.92, 0.25, 0.02, 0.6])
        self.cbar = None
        
        self.setup_widgets()
        self.update(self.current_slice)
        
    def setup_widgets(self):
        # Slice Index Slider
        ax_slider = plt.axes([0.35, 0.12, 0.55, 0.025], facecolor='lightgoldenrodyellow')
        self.slider = Slider(
            ax_slider, f'{self.dim_names[self.axis]} Index', 0, self.shape[self.axis] - 1, 
            valinit=self.current_slice, valstep=1
        )
        self.slider.on_changed(self.on_slider_change)
        
        # Vector Scale Slider
        ax_vscale = plt.axes([0.35, 0.08, 0.55, 0.025], facecolor='lightcyan')
        self.slider_vscale = Slider(
            ax_vscale, 'Vector Scale', 0.1, 15.0, valinit=self.v_scale
        )
        self.slider_vscale.on_changed(self.on_vscale_change)

        # Color Range Sliders
        vmax_init = min(4.0, self.global_max_speed)
        if self.vmin >= vmax_init:
            vmax_init = self.vmin + 0.1 * self.global_max_speed + 1e-3

        ax_vmin = plt.axes([0.35, 0.04, 0.22, 0.02], facecolor='whitesmoke')
        self.slider_vmin = Slider(ax_vmin, 'V min', -self.global_max_speed, self.global_max_speed, valinit=self.vmin)
        self.slider_vmin.on_changed(self.on_color_limit_change)

        ax_vmax = plt.axes([0.68, 0.04, 0.22, 0.02], facecolor='whitesmoke')
        self.slider_vmax = Slider(ax_vmax, 'V max', -self.global_max_speed, self.global_max_speed, valinit=vmax_init)
        self.slider_vmax.on_changed(self.on_color_limit_change)

        # Plane selection radio buttons
        ax_radio_plane = plt.axes([0.05, 0.55, 0.15, 0.18])
        self.radio_plane = RadioButtons(ax_radio_plane, ('XY', 'XZ', 'YZ'), active=1)
        self.radio_plane.on_clicked(self.on_plane_change)
        
        # Background color selection
        ax_radio_color = plt.axes([0.05, 0.35, 0.15, 0.16])
        self.radio_color = RadioButtons(ax_radio_color, ('3D Speed', 'U (vx)', 'V (vy)', 'W (vz)'), active=0)
        self.radio_color.on_clicked(self.on_color_type_change)

        # Field toggle (if dual)
        if self.has_dual_fields:
            ax_radio_field = plt.axes([0.05, 0.18, 0.15, 0.14])
            self.radio_field = RadioButtons(ax_radio_field, ('Cleaned', 'Original'), active=0)
            self.radio_field.on_clicked(self.on_field_change)
            
        # Visibility checkboxes - Stacking neatly at the bottom-left
        ax_check = plt.axes([0.05, 0.02, 0.15, 0.13])
        self.check = CheckButtons(ax_check, ('Grid Vectors', 'Input Data', 'Mask'), 
                                 (self.show_grid_vectors, self.show_input_vectors, self.show_mask))
        self.check.on_clicked(self.on_check_change)

    def on_slider_change(self, val):
        self.current_slice = int(val)
        self.update(self.current_slice)
        
    def on_vscale_change(self, val):
        self.v_scale = val
        self.update(self.current_slice)

    def on_color_limit_change(self, val):
        self.vmin = self.slider_vmin.val
        self.vmax = self.slider_vmax.val
        # Ensure vmin < vmax
        if self.vmin >= self.vmax:
            self.vmax = self.vmin + 1e-6
        self.update(self.current_slice)
        
    def on_plane_change(self, label):
        if label == 'XY': self.axis = 0
        elif label == 'XZ': self.axis = 1
        elif label == 'YZ': self.axis = 2
        
        # Reset slice to middle of new axis
        self.current_slice = self.shape[self.axis] // 2
        self.slider.valmax = self.shape[self.axis] - 1
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self.slider.label.set_text(f'{self.dim_names[self.axis]} Index')
        self.slider.set_val(self.current_slice)
        self.update(self.current_slice)

    def on_color_type_change(self, label):
        self.color_type = label
        self.update(self.current_slice)

    def on_field_change(self, label):
        print(f"Switching to field: {label}")
        self.field_name = label
        if label == 'Cleaned':
            self.u, self.v, self.w = self.u_cleaned, self.v_cleaned, self.w_cleaned
        else:
            self.u, self.v, self.w = self.u_init, self.v_init, self.w_init
        
        # Re-calc global max speed and color limits for the new field
        all_speeds = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        valid_speeds = all_speeds[~np.isnan(all_speeds)]
        self.global_max_speed = np.max(valid_speeds) if valid_speeds.size > 0 else 1.0
        if self.global_max_speed > 1e10: self.global_max_speed = 100.0
        
        self.vmin = 0.0
        self.vmax = self.global_max_speed
        
        # Update sliders to reflect new range
        self.slider_vmin.valmin = -self.global_max_speed
        self.slider_vmin.valmax = self.global_max_speed
        self.slider_vmax.valmin = -self.global_max_speed
        self.slider_vmax.valmax = self.global_max_speed
        self.slider_vmin.set_val(self.vmin)
        
        # Ensure vmin < vmax for safety
        if self.vmin >= self.vmax:
             self.vmax = self.vmin + 1e-6
        self.slider_vmax.set_val(self.vmax)

        # Re-calc speed for background
        self.speed = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        self.update(self.current_slice)

    def on_check_change(self, label):
        if label == 'Grid Vectors': self.show_grid_vectors = not self.show_grid_vectors
        elif label == 'Input Data': self.show_input_vectors = not self.show_input_vectors
        elif label == 'Mask': self.show_mask = not self.show_mask
        self.update(self.current_slice)
        
    def update(self, slice_idx):
        self.ax.clear()
        
        # Determine Plane and slice data
        if self.axis == 0: # XY plane
            h_axis, v_axis = 2, 1
            U_slice = self.u[slice_idx, :, :]
            V_slice = self.v[slice_idx, :, :]
            W_slice = self.w[slice_idx, :, :]
            plane_name = "XY"
        elif self.axis == 1: # XZ plane
            h_axis, v_axis = 2, 0
            U_slice = self.u[:, slice_idx, :]
            V_slice = self.v[:, slice_idx, :]
            W_slice = self.w[:, slice_idx, :]
            plane_name = "XZ"
        else: # YZ plane
            h_axis, v_axis = 1, 0
            U_slice = self.u[:, :, slice_idx]
            V_slice = self.v[:, :, slice_idx]
            W_slice = self.w[:, :, slice_idx]
            plane_name = "YZ"
            
        h_coords = self.coords[h_axis]
        v_coords = self.coords[v_axis]
        slice_val = self.coords[self.axis][slice_idx]
        extent = [h_coords.min(), h_coords.max(), v_coords.min(), v_coords.max()]
        base_scale = self.global_max_speed * 10.0 if self.global_max_speed > 0 else 1.0
        
        # Select background data
        if self.color_type == '3D Speed':
            bg_data = self.speed[slice_idx,:,:] if self.axis==0 else (self.speed[:,slice_idx,:] if self.axis==1 else self.speed[:,:,slice_idx])
        elif self.color_type == 'U (vx)': bg_data = self.u[slice_idx,:,:] if self.axis==0 else (self.u[:,slice_idx,:] if self.axis==1 else self.u[:,:,slice_idx])
        elif self.color_type == 'V (vy)': bg_data = self.v[slice_idx,:,:] if self.axis==0 else (self.v[:,slice_idx,:] if self.axis==1 else self.v[:,:,slice_idx])
        else: bg_data = self.w[slice_idx,:,:] if self.axis==0 else (self.w[:,slice_idx,:] if self.axis==1 else self.w[:,:,slice_idx])
        
        # Mask overlay
        if self.mask is not None and self.show_mask:
            if self.axis == 0: M_slice = self.mask[slice_idx, :, :]
            elif self.axis == 1: M_slice = self.mask[:, slice_idx, :]
            else: M_slice = self.mask[:, :, slice_idx]
            
            # Create a colored mask (grey for solid)
            overlay = np.zeros((M_slice.shape[0], M_slice.shape[1], 4))
            overlay[~M_slice] = [0.0, 0.0, 0.0, 0.5] # Grey with alpha
            self.ax.imshow(overlay, extent=extent, origin='lower', zorder=2)

        # Plot background speed
        im = self.ax.imshow(bg_data, extent=extent, origin='lower', cmap='viridis', alpha=0.9, vmin=self.vmin, vmax=self.vmax)
        
        if self.cbar is None:
            self.cbar = self.fig.colorbar(im, cax=self.cax)
        else:
            self.cbar.update_normal(im)
        
        # Plot grid vectors
        if self.show_grid_vectors:
            # Subsample for readability
            skip = max(1, len(h_coords) // 25)
            H_grid, V_grid = np.meshgrid(h_coords, v_coords)
            
            if self.axis == 0: u_plot, v_plot = U_slice, V_slice
            elif self.axis == 1: u_plot, v_plot = U_slice, W_slice
            else: u_plot, v_plot = V_slice, W_slice
            
            self.ax.quiver(H_grid[::skip, ::skip], V_grid[::skip, ::skip], 
                          u_plot[::skip, ::skip], v_plot[::skip, ::skip], 
                          color='white', pivot='mid', scale=base_scale / self.v_scale, width=0.003)
            
        # Plot raw input vectors (if within slice thickness)
        if self.input_df is not None and self.show_input_vectors:
            thick = (self.coords[self.axis][1] - self.coords[self.axis][0]) * 1.5
            mask_pts = np.abs(self.input_df[self.dim_names[self.axis].lower()] - slice_val) < thick
            pts = self.input_df[mask_pts]
            if len(pts) > 0:
                h_pts = pts[self.dim_names[h_axis].lower()]
                v_pts = pts[self.dim_names[v_axis].lower()]
                if self.axis == 0: u_pts, v_pts_vec = pts['u'], pts['v']
                elif self.axis == 1: u_pts, v_pts_vec = pts['u'], pts['w']
                else: u_pts, v_pts_vec = pts['v'], pts['w']
                
                self.ax.quiver(h_pts, v_pts, u_pts, v_pts_vec, color='red', scale=base_scale / self.v_scale, 
                                  pivot='tail', label='Input Data', zorder=3)

        self.ax.set_title(f"{self.field_name} Field: {plane_name} view at {self.dim_names[self.axis]} = {slice_val:.2f}")
        self.ax.set_xlabel(self.dim_names[h_axis])
        self.ax.set_ylabel(self.dim_names[v_axis])
        self.ax.legend(loc='upper right')
        self.ax.set_aspect('equal')

class ComparisonViewer(SliceViewer):
    def __init__(self, u1, v1, w1, u2, v2, w2, x, y, z, mask=None, labels=("Field 1", "Field 2"), fig=None):
        # Ref
        self.u2, self.v2, self.w2 = u2, v2, w2
        self.labels = labels
        self.speed2 = np.sqrt(u2**2 + v2**2 + w2**2)
        # Base handles PTV field and UI state
        super().__init__(u1, v1, w1, x, y, z, mask=mask, fig=fig)

    def setup_widgets(self):
        if not self._is_custom_fig:
            plt.close(self.fig)
            self.fig, self.axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        else:
            if len(self.fig.axes) >= 3:
                self.axs = self.fig.axes[:3]
            else:
                self.fig.clf()
                self.axs = self.fig.subplots(1, 3, sharex=True, sharey=True)
        
        plt.subplots_adjust(bottom=0.25, left=0.1, right=0.9, wspace=0.1)
        self.ax = self.axs[0] 
        self.cax1 = self.fig.add_axes([0.92, 0.3, 0.015, 0.5])
        self.cax2 = self.fig.add_axes([0.95, 0.3, 0.015, 0.5])
        self.cbar1, self.cbar2 = None, None
        super().setup_widgets()
        # Reposition central sliders
        for s in [self.slider, self.slider_vscale, self.slider_vmin, self.slider_vmax]:
            pos = s.ax.get_position()
            s.ax.set_position([0.33, pos.y0, 0.4, pos.height])
            
        # Field toggle (if dual) - move to avoid visibility overlap
        if self.has_dual_fields:
            self.radio_field.ax.set_position([0.05, 0.18, 0.1, 0.1])

    def update(self, slice_idx):
        if not hasattr(self, 'axs'): return
        for ax in self.axs: ax.clear()
        
        # Slicing PTV
        if self.axis == 0: # XY
            h_ax, v_ax = 2, 1
            U1, V1 = self.u[slice_idx,:,:], self.v[slice_idx,:,:]
            U2, V2 = self.u2[slice_idx,:,:], self.v2[slice_idx,:,:]
            M_sl = self.mask[slice_idx,:,:] if self.mask is not None else None
            plane = "XY"
        elif self.axis == 1: # XZ
            h_ax, v_ax = 2, 0
            U1, V1 = self.u[:,slice_idx,:], self.w[:,slice_idx,:]
            U2, V2 = self.u2[:,slice_idx,:], self.w2[:,slice_idx,:]
            M_sl = self.mask[:,slice_idx,:] if self.mask is not None else None
            plane = "XZ"
        else: # YZ
            h_ax, v_ax = 1, 0
            U1, V1 = self.v[:,:,slice_idx], self.w[:,:,slice_idx]
            U2, V2 = self.v2[:,:,slice_idx], self.w2[:,:,slice_idx]
            M_sl = self.mask[:,:,slice_idx] if self.mask is not None else None
            plane = "YZ"

        h_c, v_c = self.coords[h_ax], self.coords[v_ax]
        extent = [h_c.min(), h_c.max(), v_c.min(), v_c.max()]
        
        # Background
        if self.color_type == '3D Speed':
            b1, b2 = self.speed[slice_idx,:,:] if self.axis==0 else (self.speed[:,slice_idx,:] if self.axis==1 else self.speed[:,:,slice_idx]), \
                     self.speed2[slice_idx,:,:] if self.axis==0 else (self.speed2[:,slice_idx,:] if self.axis==1 else self.speed2[:,:,slice_idx])
        elif self.color_type == 'U (vx)':
            b1, b2 = self.u[slice_idx,:,:] if self.axis==0 else (self.u[:,slice_idx,:] if self.axis==1 else self.u[:,:,slice_idx]), \
                     self.u2[slice_idx,:,:] if self.axis==0 else (self.u2[:,slice_idx,:] if self.axis==1 else self.u2[:,:,slice_idx])
        elif self.color_type == 'V (vy)':
            b1, b2 = self.v[slice_idx,:,:] if self.axis==0 else (self.v[:,slice_idx,:] if self.axis==1 else self.v[:,:,slice_idx]), \
                     self.v2[slice_idx,:,:] if self.axis==0 else (self.v2[:,slice_idx,:] if self.axis==1 else self.v2[:,:,slice_idx])
        else:
            b1, b2 = self.w[slice_idx,:,:] if self.axis==0 else (self.w[:,slice_idx,:] if self.axis==1 else self.w[:,:,slice_idx]), \
                     self.w2[slice_idx,:,:] if self.axis==0 else (self.w2[:,slice_idx,:] if self.axis==1 else self.w2[:,:,slice_idx])

        diff = b1 - b2
        im1 = self.axs[0].imshow(b1, extent=extent, origin='lower', cmap='viridis', vmin=self.vmin, vmax=self.vmax, alpha=0.9)
        im2 = self.axs[1].imshow(b2, extent=extent, origin='lower', cmap='viridis', vmin=self.vmin, vmax=self.vmax, alpha=0.9)
        d_lim = max(abs(self.vmin), abs(self.vmax))
        im3 = self.axs[2].imshow(diff, extent=extent, origin='lower', cmap='RdBu_r', vmin=-d_lim/2, vmax=d_lim/2, alpha=0.9)
        
        if self.cbar1 is None:
            self.cbar1 = self.fig.colorbar(im1, cax=self.cax1, label="Velocity")
            self.cbar2 = self.fig.colorbar(im3, cax=self.cax2, label="Diff")
        else:
            self.cbar1.update_normal(im1)
            self.cbar2.update_normal(im3)

        if M_sl is not None and self.show_mask:
            ov = np.zeros(M_sl.shape + (4,))
            ov[~M_sl] = [0.2, 0.2, 0.2, 0.5]
            for ax in self.axs: ax.imshow(ov, extent=extent, origin='lower', zorder=2)

        if self.show_grid_vectors:
            skip = max(1, len(h_c)//20)
            H, V = np.meshgrid(h_c, v_c)
            b_sc = self.global_max_speed * 10
            q_p = dict(scale=b_sc/self.v_scale, color='white', width=0.004, pivot='mid')
            self.axs[0].quiver(H[::skip,::skip], V[::skip,::skip], U1[::skip,::skip], V1[::skip,::skip], **q_p)
            self.axs[1].quiver(H[::skip,::skip], V[::skip,::skip], U2[::skip,::skip], V2[::skip,::skip], **q_p)

        self.axs[0].set_title(self.labels[0])
        self.axs[1].set_title(self.labels[1])
        self.axs[2].set_title("Difference")
        for ax in self.axs: 
            ax.set_aspect('equal')
            ax.set_xlabel(self.dim_names[h_ax])
        self.axs[0].set_ylabel(self.dim_names[v_ax])
        self.fig.suptitle(f"{self.field_name} {plane} view at {self.dim_names[self.axis]}={self.coords[self.axis][slice_idx]:.2f}")

class SideBySideViewer(SliceViewer):
    def __init__(self, u1, v1, w1, u2, v2, w2, x, y, z, mask=None, labels=("Field 1", "Field 2"), fig=None):
        # Ref
        self.u2, self.v2, self.w2 = u2, v2, w2
        self.labels = labels
        self.speed2 = np.sqrt(u2**2 + v2**2 + w2**2)
        # Base handles PTV field and UI state
        super().__init__(u1, v1, w1, x, y, z, mask=mask, fig=fig)

    def setup_widgets(self):
        if not self._is_custom_fig:
            plt.close(self.fig)
            self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        else:
            if len(self.fig.axes) >= 2:
                self.axs = self.fig.axes[:2]
            else:
                self.fig.clf()
                self.axs = self.fig.subplots(1, 2, sharex=True, sharey=True)
                
        plt.subplots_adjust(bottom=0.25, left=0.1, right=0.9, wspace=0.1)
        self.ax = self.axs[0] 
        self.cax1 = self.fig.add_axes([0.92, 0.3, 0.015, 0.5])
        self.cbar1 = None
        super().setup_widgets()
        
        # Reposition central sliders to avoid overlap
        for s in [self.slider, self.slider_vscale]:
            pos = s.ax.get_position()
            s.ax.set_position([0.33, pos.y0, 0.4, pos.height])
            
        # Position vmin and vmax side-by-side instead of overlapping
        pos_min = self.slider_vmin.ax.get_position()
        self.slider_vmin.ax.set_position([0.33, pos_min.y0, 0.18, pos_min.height])
        pos_max = self.slider_vmax.ax.get_position()
        self.slider_vmax.ax.set_position([0.55, pos_max.y0, 0.18, pos_max.height])

    def update(self, slice_idx):
        if not hasattr(self, 'axs'): return
        for ax in self.axs: ax.clear()
        
        # Slicing
        if self.axis == 0: # XY
            h_ax, v_ax = 2, 1
            U1, V1 = self.u[slice_idx,:,:], self.v[slice_idx,:,:]
            U2, V2 = self.u2[slice_idx,:,:], self.v2[slice_idx,:,:]
            M_sl = self.mask[slice_idx,:,:] if self.mask is not None else None
            plane = "XY"
        elif self.axis == 1: # XZ
            h_ax, v_ax = 2, 0
            U1, V1 = self.u[:,slice_idx,:], self.w[:,slice_idx,:]
            U2, V2 = self.u2[:,slice_idx,:], self.w2[:,slice_idx,:]
            M_sl = self.mask[:,slice_idx,:] if self.mask is not None else None
            plane = "XZ"
        else: # YZ
            h_ax, v_ax = 1, 0
            U1, V1 = self.v[:,:,slice_idx], self.w[:,:,slice_idx]
            U2, V2 = self.v2[:,:,slice_idx], self.w2[:,:,slice_idx]
            M_sl = self.mask[:,:,slice_idx] if self.mask is not None else None
            plane = "YZ"

        h_c, v_c = self.coords[h_ax], self.coords[v_ax]
        extent = [h_c.min(), h_c.max(), v_c.min(), v_c.max()]
        
        # Background
        if self.color_type == '3D Speed':
            b1, b2 = self.speed[slice_idx,:,:] if self.axis==0 else (self.speed[:,slice_idx,:] if self.axis==1 else self.speed[:,:,slice_idx]), \
                     self.speed2[slice_idx,:,:] if self.axis==0 else (self.speed2[:,slice_idx,:] if self.axis==1 else self.speed2[:,:,slice_idx])
        elif self.color_type == 'U (vx)':
            b1, b2 = self.u[slice_idx,:,:] if self.axis==0 else (self.u[:,slice_idx,:] if self.axis==1 else self.u[:,:,slice_idx]), \
                     self.u2[slice_idx,:,:] if self.axis==0 else (self.u2[:,slice_idx,:] if self.axis==1 else self.u2[:,:,slice_idx])
        elif self.color_type == 'V (vy)':
            b1, b2 = self.v[slice_idx,:,:] if self.axis==0 else (self.v[:,slice_idx,:] if self.axis==1 else self.v[:,:,slice_idx]), \
                     self.v2[slice_idx,:,:] if self.axis==0 else (self.v2[:,slice_idx,:] if self.axis==1 else self.v2[:,:,slice_idx])
        else:
            b1, b2 = self.w[slice_idx,:,:] if self.axis==0 else (self.w[:,slice_idx,:] if self.axis==1 else self.w[:,:,slice_idx]), \
                     self.w2[slice_idx,:,:] if self.axis==0 else (self.w2[:,slice_idx,:] if self.axis==1 else self.w2[:,:,slice_idx])

        # Debug prints to check for valid data
        if slice_idx == self.shape[self.axis] // 2:
            print(f"Update: Slice {slice_idx}, plane {plane}")
            print(f"  B1 mean: {np.nanmean(b1):.4f}, max: {np.nanmax(b1):.4f}")
            print(f"  B2 mean: {np.nanmean(b2):.4f}, max: {np.nanmax(b2):.4f}")

        im1 = self.axs[0].imshow(b1, extent=extent, origin='lower', cmap='viridis', vmin=self.vmin, vmax=self.vmax, alpha=0.9)
        im2 = self.axs[1].imshow(b2, extent=extent, origin='lower', cmap='viridis', vmin=self.vmin, vmax=self.vmax, alpha=0.9)
        
        if self.cbar1 is None:
            self.cbar1 = self.fig.colorbar(im1, cax=self.cax1, label="Velocity")
        else:
            self.cbar1.update_normal(im1)

        if M_sl is not None and self.show_mask:
            ov = np.zeros(M_sl.shape + (4,))
            ov[~M_sl] = [0.2, 0.2, 0.2, 0.5]
            self.axs[0].imshow(ov, extent=extent, origin='lower', zorder=2)

        if self.show_grid_vectors:
            skip = max(1, len(h_c)//20)
            H, V = np.meshgrid(h_c, v_c)
            b_sc = self.global_max_speed * 10
            q_p = dict(scale=b_sc/self.v_scale, color='white', width=0.004, pivot='mid')
            self.axs[0].quiver(H[::skip,::skip], V[::skip,::skip], U1[::skip,::skip], V1[::skip,::skip], **q_p)
            self.axs[1].quiver(H[::skip,::skip], V[::skip,::skip], U2[::skip,::skip], V2[::skip,::skip], **q_p)

        self.axs[0].set_title(self.labels[0])
        self.axs[1].set_title(self.labels[1])
        for ax in self.axs: 
            ax.set_aspect('equal')
            ax.set_xlabel(self.dim_names[h_ax])
        self.axs[0].set_ylabel(self.dim_names[v_ax])
        self.fig.suptitle(f"{self.field_name} {plane} view at {self.dim_names[self.axis]}={self.coords[self.axis][slice_idx]:.2f}")

class ScalarSliceViewer(SliceViewer):
    def __init__(self, data, x, y, z, mask=None, title="Scalar Field", cmap="RdBu_r", fig=None):
        # We spoof u, v, w as zeros for the base class, or just handle it
        self.data_field = data
        self.cmap = cmap
        self.title_base = title
        # For base class to not crash
        super().__init__(np.zeros_like(data), np.zeros_like(data), np.zeros_like(data), x, y, z, mask=mask, fig=fig)
        
        # Override vmin/vmax for scalar field
        vabs = np.nanmax(np.abs(data))
        self.vmin, self.vmax = -vabs, vabs
        if self.vmin >= self.vmax:
            self.vmax = self.vmin + 1e-4
        self.slider_vmin.set_val(self.vmin)
        self.slider_vmax.set_val(self.vmax)

    def setup_widgets(self):
        super().setup_widgets()
        # Hide things we don't need, safely
        for attr in ['radio_color', 'radio_field', 'check']:
            if hasattr(self, attr):
                getattr(self, attr).ax.set_visible(False)

    def update(self, slice_idx):
        if not hasattr(self, 'ax'): return
        self.ax.clear()
        
        # Slicing
        if self.axis == 0: # XY
            d_sl = self.data_field[slice_idx,:,:]
            m_sl = self.mask[slice_idx,:,:] if self.mask is not None else None
            plane = "XY"
        elif self.axis == 1: # XZ
            d_sl = self.data_field[:,slice_idx,:]
            m_sl = self.mask[:,slice_idx,:] if self.mask is not None else None
            plane = "XZ"
        else: # YZ
            d_sl = self.data_field[:,:,slice_idx]
            m_sl = self.mask[:,:,slice_idx] if self.mask is not None else None
            plane = "YZ"

        h_ax, v_ax = (2, 1) if self.axis == 0 else ((2, 0) if self.axis == 1 else (1, 0))
        h_c, v_c = self.coords[h_ax], self.coords[v_ax]
        extent = [h_c.min(), h_c.max(), v_c.min(), v_c.max()]
        
        im = self.ax.imshow(d_sl, extent=extent, origin='lower', cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        
        if self.cbar is None:
            self.cbar = self.fig.colorbar(im, cax=self.cax, label="Value")
        else:
            self.cbar.update_normal(im)

        if m_sl is not None and self.show_mask:
            ov = np.zeros(m_sl.shape + (4,))
            ov[~m_sl] = [0.2, 0.2, 0.2, 0.5]
            self.ax.imshow(ov, extent=extent, origin='lower', zorder=2)

        self.ax.set_title(f"{self.title_base}: {plane} at {self.dim_names[self.axis]}={self.coords[self.axis][slice_idx]:.2f}")
        self.ax.set_xlabel(self.dim_names[h_ax])
        self.ax.set_ylabel(self.dim_names[v_ax])
        self.ax.set_aspect('equal')

class ScalarSideBySideViewer(ScalarSliceViewer):
    def __init__(self, data1, data2, x, y, z, mask=None, labels=("Field 1", "Field 2"), title="Scalar Comparison", cmap="RdBu_r", fig=None):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels
        self.title_base = title
        # For base class to not crash
        super().__init__(data1, x, y, z, mask=mask, title=title, cmap=cmap, fig=fig)
        
        # Override vmin/vmax for combined data
        vabs = max(np.nanmax(np.abs(data1)), np.nanmax(np.abs(data2)))
        self.vmin, self.vmax = -vabs, vabs
        if self.vmin >= self.vmax:
            self.vmax = self.vmin + 1e-4
        self.slider_vmin.set_val(self.vmin)
        self.slider_vmax.set_val(self.vmax)

    def setup_widgets(self):
        plt.close(self.fig)
        self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        plt.subplots_adjust(bottom=0.25, left=0.1, right=0.9, wspace=0.1)
        self.ax = self.axs[0] 
        self.cax1 = self.fig.add_axes([0.92, 0.3, 0.015, 0.5])
        self.cbar1 = None
        # Parent setup handles sliders
        super().setup_widgets()
        # Reposition sliders
        for s in [self.slider, self.slider_vscale]:
            pos = s.ax.get_position()
            s.ax.set_position([0.33, pos.y0, 0.4, pos.height])
        pos_min = self.slider_vmin.ax.get_position()
        self.slider_vmin.ax.set_position([0.33, pos_min.y0, 0.18, pos_min.height])
        pos_max = self.slider_vmax.ax.get_position()
        self.slider_vmax.ax.set_position([0.55, pos_max.y0, 0.18, pos_max.height])

    def update(self, slice_idx):
        if not hasattr(self, 'axs'): return
        for ax in self.axs: ax.clear()
        
        # Slicing
        if self.axis == 0: # XY
            d1, d2 = self.data1[slice_idx,:,:], self.data2[slice_idx,:,:]
            m_sl = self.mask[slice_idx,:,:] if self.mask is not None else None
            plane = "XY"
        elif self.axis == 1: # XZ
            d1, d2 = self.data1[:,slice_idx,:], self.data2[:,slice_idx,:]
            m_sl = self.mask[:,slice_idx,:] if self.mask is not None else None
            plane = "XZ"
        else: # YZ
            d1, d2 = self.data1[:,:,slice_idx], self.data2[:,:,slice_idx]
            m_sl = self.mask[:,:,slice_idx] if self.mask is not None else None
            plane = "YZ"

        h_ax, v_ax = (2, 1) if self.axis == 0 else ((2, 0) if self.axis == 1 else (1, 0))
        h_c, v_c = self.coords[h_ax], self.coords[v_ax]
        extent = [h_c.min(), h_c.max(), v_c.min(), v_c.max()]
        
        im1 = self.axs[0].imshow(d1, extent=extent, origin='lower', cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        im2 = self.axs[1].imshow(d2, extent=extent, origin='lower', cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        
        if self.cbar1 is None:
            self.cbar1 = self.fig.colorbar(im1, cax=self.cax1, label="Value")
        else:
            self.cbar1.update_normal(im1)

        if m_sl is not None and self.show_mask:
            ov = np.zeros(m_sl.shape + (4,))
            ov[~m_sl] = [0.2, 0.2, 0.2, 0.5]
            self.axs[0].imshow(ov, extent=extent, origin='lower', zorder=2)

        self.axs[0].set_title(self.labels[0])
        self.axs[1].set_title(self.labels[1])
        for ax in self.axs: 
            ax.set_aspect('equal')
            ax.set_xlabel(self.dim_names[h_ax])
        self.axs[0].set_ylabel(self.dim_names[v_ax])
        self.fig.suptitle(f"{self.title_base} {plane}: {self.dim_names[self.axis]}={self.coords[self.axis][slice_idx]:.2f}")

def show(u, v, w, x, y, z, mask=None, input_df=None, fig=None):
    viewer = SliceViewer(u, v, w, x, y, z, mask, input_df, fig=fig)
    if fig is None: plt.show()
    return viewer

def compare(u1, v1, w1, u2, v2, w2, x, y, z, mask=None, labels=("Field 1", "Field 2"), fig=None):
    viewer = ComparisonViewer(u1, v1, w1, u2, v2, w2, x, y, z, mask, labels, fig=fig)
    if fig is None: plt.show()
    return viewer

def side_by_side(u1, v1, w1, u2, v2, w2, x, y, z, mask=None, labels=("Field 1", "Field 2"), fig=None):
    viewer = SideBySideViewer(u1, v1, w1, u2, v2, w2, x, y, z, mask, labels, fig=fig)
    if fig is None: plt.show()
    return viewer

def show_scalar(data, x, y, z, mask=None, title="Scalar Field", cmap="RdBu_r", fig=None):
    viewer = ScalarSliceViewer(data, x, y, z, mask, title, cmap, fig=fig)
    if fig is None: plt.show()
    return viewer

def compare_scalars(data1, data2, x, y, z, mask=None, labels=("Field 1", "Field 2"), title="Scalar Comparison", cmap="RdBu_r", fig=None):
    viewer = ScalarSideBySideViewer(data1, data2, x, y, z, mask, labels, title, cmap, fig=fig)
    if fig is None: plt.show()
    return viewer
