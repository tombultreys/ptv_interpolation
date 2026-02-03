
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

class PreViewer:
    def __init__(self, df, mask, invert=False, bounds=None, initial_offset=None):
        self.df = df
        self.mask = mask
        self.bounds = bounds # ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        
        if invert:
            self.mask = ~self.mask
        
        # Mask shape is (Z, Y, X)
        self.shape = self.mask.shape
        self.dim_names = ['Z', 'Y', 'X']
        self.axis = 1 # Default: XZ plane (slide Y)
        
        if self.bounds is None:
            self.bounds = ((0, self.shape[2]), (0, self.shape[1]), (0, self.shape[0]))
            
        self.current_slice = self.shape[self.axis] // 2 
        
        # Initial offsets from CLI or auto-align
        if initial_offset is not None:
            self.ox, self.oy, self.oz = initial_offset
        else:
            self.ox, self.oy, self.oz = 0.0, 0.0, 0.0
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.35, left=0.2) # Make more room for sliders and radio
        
        self.setup_widgets()
        self.update(self.current_slice)
        
    def setup_widgets(self):
        # Slice slider
        ax_slider = plt.axes([0.3, 0.2, 0.6, 0.02])
        # Find bounds for current sliding axis
        # bounds are ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        # axis 0 (Z): zmin, zmax
        # axis 1 (Y): ymin, ymax
        # axis 2 (X): xmin, xmax
        b_idx_map = {0: 2, 1: 1, 2: 0} # map axis to bounds index
        b_min, b_max = self.bounds[b_idx_map[self.axis]]
        
        self.slider_s = Slider(ax_slider, f'{self.dim_names[self.axis]}-coord', b_min, b_max - 1, valinit=b_min + self.current_slice, valstep=1)
        self.slider_s.on_changed(self.on_slider_change)
        
        # Offset sliders (X, Y, Z)
        ax_ox = plt.axes([0.3, 0.14, 0.6, 0.02])
        self.slider_ox = Slider(ax_ox, 'Offs X', self.ox - 300, self.ox + 300, valinit=self.ox)
        
        ax_oy = plt.axes([0.3, 0.1, 0.6, 0.02])
        self.slider_oy = Slider(ax_oy, 'Offs Y', self.oy - 300, self.oy + 300, valinit=self.oy)
        
        ax_oz = plt.axes([0.3, 0.06, 0.6, 0.02])
        self.slider_oz = Slider(ax_oz, 'Offs Z', self.oz - 300, self.oz + 300, valinit=self.oz)
        
        self.slider_ox.on_changed(self.on_offset_change)
        self.slider_oy.on_changed(self.on_offset_change)
        self.slider_oz.on_changed(self.on_offset_change)
        
        # Radio Buttons
        ax_radio = plt.axes([0.02, 0.5, 0.12, 0.15], facecolor='#f0f0f0')
        self.radio = RadioButtons(ax_radio, ('XY (slide Z)', 'XZ (slide Y)', 'YZ (slide X)'), active=1)
        self.radio.on_clicked(self.on_axis_change)

    def on_axis_change(self, label):
        plane_map = {'XY (slide Z)': 0, 'XZ (slide Y)': 1, 'YZ (slide X)': 2}
        self.axis = plane_map[label]
        self.current_slice = self.shape[self.axis] // 2
        
        # Update slider label and range
        b_idx_map = {0: 2, 1: 1, 2: 0} 
        b_min, b_max = self.bounds[b_idx_map[self.axis]]
        
        self.slider_s.label.set_text(f'{self.dim_names[self.axis]}-coord')
        self.slider_s.valmin = b_min
        self.slider_s.valmax = b_max - 1
        self.slider_s.set_val(b_min + self.current_slice)
        
        self.update(self.current_slice)
        self.fig.canvas.draw_idle()

    def on_slider_change(self, val):
        b_idx_map = {0: 2, 1: 1, 2: 0} 
        b_min, _ = self.bounds[b_idx_map[self.axis]]
        self.current_slice = int(val) - int(b_min)
        self.update(self.current_slice)
        self.fig.canvas.draw_idle()

    def on_offset_change(self, val):
        self.ox = self.slider_ox.val
        self.oy = self.slider_oy.val
        self.oz = self.slider_oz.val
        self.update(self.current_slice)
        self.fig.canvas.draw_idle()
        
    def update(self, slice_idx):
        self.ax.clear()
        
        if self.axis == 0: # XY plane
            h_axis, v_axis = 2, 1
            m_slice = self.mask[slice_idx, :, :]
            plane_name = "XY"
        elif self.axis == 1: # XZ plane
            h_axis, v_axis = 2, 0
            m_slice = self.mask[:, slice_idx, :]
            plane_name = "XZ"
        else: # YZ plane
            h_axis, v_axis = 1, 0
            m_slice = self.mask[:, :, slice_idx]
            plane_name = "YZ"

        b_idx_list = [2, 1, 0] # map axis to bounds index
        h_min, h_max = self.bounds[b_idx_list[h_axis]]
        v_min, v_max = self.bounds[b_idx_list[v_axis]]
        s_min, _ = self.bounds[b_idx_list[self.axis]]
        slice_val = s_min + slice_idx
        
        # Adjust extent for imshow: pixels are centered at integers (0, 1, ..., N-1)
        # So the boundaries are [-0.5, N-0.5]
        self.ax.imshow(m_slice, cmap='gray', origin='lower', 
                       extent=[h_min - 0.5, h_max - 0.5, v_min - 0.5, v_max - 0.5])
        
        # Overlay points near this slice (Apply interactive offsets)
        dz = 2.0 
        points_x = self.df.x + self.ox
        points_y = self.df.y + self.oy
        points_z = self.df.z + self.oz
        
        points_list = [points_z, points_y, points_x]
        
        p_slice_coords = points_list[self.axis]
        p_h_coords = points_list[h_axis]
        p_v_coords = points_list[v_axis]
        
        mask_select = (p_slice_coords >= slice_val - dz) & (p_slice_coords <= slice_val + dz)
        subset_h = p_h_coords[mask_select]
        subset_v = p_v_coords[mask_select]
        
        if len(subset_h) > 0:
            self.ax.scatter(subset_h, subset_v, c='r', s=1, alpha=0.5, label='PTV Points')
            
        self.ax.set_title(f"Alignment: {plane_name} view at {self.dim_names[self.axis]}={slice_val:.1f}\nOffset: [{self.ox:.1f}, {self.oy:.1f}, {self.oz:.1f}]")
        self.ax.set_xlabel(self.dim_names[h_axis])
        self.ax.set_ylabel(self.dim_names[v_axis])
        self.ax.legend(loc='upper right')
        self.ax.set_aspect('equal')

def main():
    parser = argparse.ArgumentParser(description="Pre-visualize mask and PTV point alignment.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--mask", "-m", required=True, help="Input Mask TIFF")
    parser.add_argument("--invert-mask", action="store_true", help="Invert mask")
    parser.add_argument("--crop", type=int, nargs=6, help="Crop region: xmin xmax ymin ymax zmin zmax")
    parser.add_argument("--data-offset", type=int, nargs=3, help="Offset to align data to mask: x y z")
    parser.add_argument("--swap-xy", action="store_true", help="Swap X and Y coordinates")
    parser.add_argument("--mask-transpose", type=int, nargs=3, help="Transpose mask axes: e.g., 2 1 0")
    args = parser.parse_args()
    
    print("Loading data...")
    df = load_ptv_data(args.input)
    
    if args.swap_xy:
        print("Swapping X and Y coordinates and velocities...")
        df[['x', 'y']] = df[['y', 'x']]
        if 'u' in df.columns and 'v' in df.columns:
            df[['u', 'v']] = df[['v', 'u']]
    
    if args.data_offset:
        ox, oy, oz = args.data_offset
    else:
        ox, oy, oz = 0, 0, 0
        
    print("Loading mask...")
    mask = load_mask(args.mask)
    print(f"Loaded Mask Shape: {mask.shape}")
    
    if args.mask_transpose:
        print(f"Transposing mask with axes {args.mask_transpose}...")
        mask = np.transpose(mask, axes=args.mask_transpose)
        print(f"Mask Shape after transposition: {mask.shape}")
    
    print(f"Data bounds: X[{df.x.min():.1f}, {df.x.max():.1f}], Y[{df.y.min():.1f}, {df.y.max():.1f}], Z[{df.z.min():.1f}, {df.z.max():.1f}]")
    print(f"Mask shape: {mask.shape} (Z, Y, X)")
    
    bounds = None
    if args.crop:
        xs, xe, ys, ye, zs, ze = args.crop
        print(f"Cropping to X[{xs}:{xe}], Y[{ys}:{ye}], Z[{zs}:{ze}]...")
        mask = mask[zs:ze, ys:ye, xs:xe]
        df = df[(df.x >= xs) & (df.x < xe) & (df.y >= ys) & (df.y < ye) & (df.z >= zs) & (df.z < ze)]
        bounds = ((xs, xe), (ys, ye), (zs, ze))
        print(f"Mask shape after crop: {mask.shape}")
        print(f"Points after crop: {len(df)}")
    
    pv = PreViewer(df, mask, invert=args.invert_mask, bounds=bounds, initial_offset=(ox, oy, oz))
    plt.show()

if __name__ == "__main__":
    main()
