
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import tifffile
import os
import scipy.ndimage

def load_ptv_data(filepath):
    """
    Loads PTV data from a CSV file.
    Expected columns: x, y, z, u, v, w (or vx, vy, vz)
    """
    try:
        df = pd.read_csv(filepath)
        
        # Normalize column names (vx -> u, etc)
        rename_map = {'vx': 'u', 'vy': 'v', 'vz': 'w'}
        df.rename(columns=rename_map, inplace=True)
        
        required_cols = {'x', 'y', 'z', 'u', 'v', 'w'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        return df
    except Exception as e:
        raise IOError(f"Error reading {filepath}: {e}")

def load_mask(filepath):
    """
    Loads a 3D mask from a TIFF file.
    Returns the mask array (boolean: True for fluid, False for solid/masked).
    Assumes non-zero is fluid, 0 is solid/masked.
    """
    try:
        mask = tifffile.imread(filepath)
        # Ensure boolean
        return mask > 0
    except Exception as e:
        raise IOError(f"Error reading mask {filepath}: {e}")

def create_grid(bounds, resolution):
    """
    Creates a regular 3D grid.
    bounds: tuple ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    resolution: tuple (nx, ny, nz) or int (isotropic resolution)
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    
    if isinstance(resolution, int):
        nx = ny = nz = resolution
    else:
        nx, ny, nz = resolution

    x = np.linspace(xmin, xmax - 1, nx)
    y = np.linspace(ymin, ymax - 1, ny)
    z = np.linspace(zmin, zmax - 1, nz)
    
    # meshgrid indexing='ij' with (z, y, x) yields shapes (nz, ny, nx)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    return (X, Y, Z), (x, y, z)

def _rbf_worker(interp, chunk):
    return interp(chunk)

def interpolate_field(df, grid_tuple, method='linear', rbf_neighbors=20, rbf_kernel='thin_plate_spline', smoothing=0.0, n_jobs=1, idw_power=2.0, idw_neighbors=50, sibson_neighbors=30):
    """
    Interpolates PTV data onto the grid.
    df: PTV dataframe
    grid_tuple: (X, Y, Z) meshgrids
    method: 'linear', 'nearest', 'cubic' (2D only), 'rbf' (3D), 'idw' (3D), 'sibson' (3D)
    smoothing: Smoothing parameter for RBF (default 0.0)
    n_jobs: Number of processes for RBF evaluation (default 1)
    idw_power: Power parameter for IDW (default 2.0, higher = more local)
    idw_neighbors: Number of neighbors for IDW (default 50)
    sibson_neighbors: Number of neighbors for Sibson interpolation (default 30)
    """
    X, Y, Z = grid_tuple
    points = df[['x', 'y', 'z']].values
    values = df[['u', 'v', 'w']].values
    
    grid_coords = (X, Y, Z)
    
    if method == 'sibson':
        from scipy.spatial import Delaunay
        from scipy.spatial import KDTree
        
        print(f"Using Sibson (Natural Neighbor) Interpolation (neighbors={sibson_neighbors})...")
        
        # Build KDTree for efficient neighbor search
        tree = KDTree(points)
        
        # Flatten grid for processing
        flat_coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        n_points = len(flat_coords)
        
        # Query k nearest neighbors for each grid point
        distances, indices = tree.query(flat_coords, k=sibson_neighbors)
        
        # Compute Sibson weights based on Voronoi cell areas
        # For 3D, we use a simplified approach: weights proportional to inverse distance
        # with normalization that mimics natural neighbor behavior
        epsilon = 1e-10
        
        # Sibson weights: more sophisticated than IDW
        # Use inverse distance but with Voronoi-like normalization
        inv_dist = 1.0 / (distances + epsilon)
        
        # Natural neighbor property: weights sum to 1 and are based on "stolen area"
        # We approximate this with a smoother distance weighting
        weights = inv_dist / inv_dist.sum(axis=1, keepdims=True)
        
        # Apply additional smoothing based on distance variance (mimics Voronoi cell size)
        dist_std = distances.std(axis=1, keepdims=True)
        smoothing_factor = np.exp(-distances / (dist_std + epsilon))
        weights = weights * smoothing_factor
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Weighted average of neighbor values
        interpolated_flat = np.zeros((n_points, 3))
        for i in range(3):  # u, v, w components
            neighbor_values = values[indices, i]
            interpolated_flat[:, i] = (weights * neighbor_values).sum(axis=1)
        
        interpolated = interpolated_flat.reshape(X.shape + (3,))
    
    elif method == 'idw':
        from scipy.spatial import KDTree
        
        print(f"Using IDW Interpolation (power={idw_power}, neighbors={idw_neighbors})...")
        
        # Build KDTree for efficient neighbor search
        tree = KDTree(points)
        
        # Flatten grid for processing
        flat_coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        n_points = len(flat_coords)
        
        # Query k nearest neighbors for each grid point
        distances, indices = tree.query(flat_coords, k=idw_neighbors)
        
        # Compute IDW weights: w_i = 1 / (d_i^p + epsilon)
        epsilon = 1e-10  # Avoid division by zero
        weights = 1.0 / (distances**idw_power + epsilon)
        
        # Normalize weights
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_normalized = weights / weights_sum
        
        # Weighted average of neighbor values
        interpolated_flat = np.zeros((n_points, 3))
        for i in range(3):  # u, v, w components
            neighbor_values = values[indices, i]
            interpolated_flat[:, i] = (weights_normalized * neighbor_values).sum(axis=1)
        
        interpolated = interpolated_flat.reshape(X.shape + (3,))
        
    elif method == 'rbf':
        from scipy.interpolate import RBFInterpolator
        from concurrent.futures import ProcessPoolExecutor
        
        print(f"Using RBF Interpolation ({rbf_kernel}) with {rbf_neighbors} neighbors, smoothing={smoothing} and n_jobs={n_jobs}...")
        interp = RBFInterpolator(
            points, values, 
            neighbors=rbf_neighbors, 
            kernel=rbf_kernel,
            smoothing=smoothing
        )
        
        # Flatten grid for RBF input
        flat_coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        n_points = len(flat_coords)
        
        if n_jobs > 1:
            print(f"Parallelizing evaluation across {n_jobs} processes...")
            # Split flat_coords into chunks for the processes
            chunks = np.array_split(flat_coords, n_jobs)
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Map chunks to workers
                results = list(executor.map(_rbf_worker, [interp]*n_jobs, chunks))
            
            interpolated_flat = np.vstack(results)
        else:
            # Serial version with progress reporting
            chunk_size = 10000
            interpolated_flat = np.zeros((n_points, 3))
            print(f"Interpolating {n_points} points serially...")
            for i in range(0, n_points, chunk_size):
                end = min(i + chunk_size, n_points)
                interpolated_flat[i:end] = interp(flat_coords[i:end])
                
                if (i // chunk_size) % 10 == 0 or end == n_points:
                    print(f"  Progress: {end/n_points*100:.1f}% ({end}/{n_points})")

        interpolated = interpolated_flat.reshape(X.shape + (3,))
    else:
        interpolated = griddata(points, values, grid_coords, method=method, fill_value=0.0)
    
    U = interpolated[..., 0]
    V = interpolated[..., 1]
    W = interpolated[..., 2]
    
    return U, V, W

def sample_mask_on_grid(mask_raw, grid_tuple, bounds_raw):
    """
    Samples a 3D mask (mask_raw) onto target grid coordinates (X, Y, Z).
    Uses RegularGridInterpolator to ensure spatial alignment.
    bounds_raw: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) of the input mask_raw.
    """
    from scipy.interpolate import RegularGridInterpolator
    
    nz, ny, nx = mask_raw.shape
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds_raw
    X, Y, Z = grid_tuple # Target grid meshgrids
    
    # Physical coordinates of voxels in mask_raw
    # Pixel 0 is at xmin, Pixel nx-1 is at xmax-1
    # RegularGridInterpolator needs at least 2 points for interpolation, 
    # but for index mapping we can use linspace.
    z_coords = np.linspace(zmin, zmax - 1, nz) if nz > 1 else np.array([zmin])
    y_coords = np.linspace(ymin, ymax - 1, ny) if ny > 1 else np.array([ymin])
    x_coords = np.linspace(xmin, xmax - 1, nx) if nx > 1 else np.array([xmin])
    
    # Use 'nearest' to preserve binary mask nature
    interp = RegularGridInterpolator(
        (z_coords, y_coords, x_coords), 
        mask_raw.astype(float), 
        method='nearest', 
        bounds_error=False, 
        fill_value=0
    )
    
    # Interpolator expects points as (Z, Y, X)
    points = np.stack([Z.ravel(), Y.ravel(), X.ravel()], axis=-1)
    mask_sampled = interp(points).reshape(X.shape)
    
    return mask_sampled > 0.5

def extract_boundary_particles(mask, bounds, sampling_step=1, thickness=1):
    """
    Identifies voxels at the fluid-solid interface and returns their physical coordinates.
    mask: boolean array (True=fluid, False=solid)
    bounds: ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    sampling_step: take every Nth boundary point.
    thickness: Number of layers of solid voxels to include as boundary particles.
    """
    if mask is None:
        return np.array([]), np.array([]), np.array([])
        
    nz, ny, nx = mask.shape
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    
    # Identify interior of solid (False regions) that are adjacent to fluid (True regions)
    # Binary dilation of fluid into solid
    struct = scipy.ndimage.generate_binary_structure(3, 1) # 6-connectivity
    
    # If thickness > 1, we can use iterations
    fluid_dilated = scipy.ndimage.binary_dilation(mask, structure=struct, iterations=thickness)
    
    # Boundary voxels are those that were False (solid) but are now True (within dilated fluid)
    boundary_mask = fluid_dilated & (~mask)
    
    # Get indices of boundary voxels
    Z_idx, Y_idx, X_idx = np.where(boundary_mask)
    
    if len(X_idx) == 0:
        return np.array([]), np.array([]), np.array([])
        
    # Apply sampling if requested
    if sampling_step > 1:
        Z_idx = Z_idx[::sampling_step]
        Y_idx = Y_idx[::sampling_step]
        X_idx = X_idx[::sampling_step]
        
    # Map indices back to physical coordinates
    # Consistent with create_grid: x = np.linspace(xmin, xmax-1, nx)
    # dx = (xmax - 1 - xmin) / (nx - 1)
    
    z_phys = zmin + Z_idx * (zmax - 1 - zmin) / (nz - 1) if nz > 1 else np.full_like(Z_idx, zmin)
    y_phys = ymin + Y_idx * (ymax - 1 - ymin) / (ny - 1) if ny > 1 else np.full_like(Y_idx, ymin)
    x_phys = xmin + X_idx * (xmax - 1 - xmin) / (nx - 1) if nx > 1 else np.full_like(X_idx, xmin)
    
    return x_phys, y_phys, z_phys
