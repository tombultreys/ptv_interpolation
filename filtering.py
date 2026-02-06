import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def remove_outliers_knn(df, k=25, threshold=3.0):
    """
    Neighborhood-based statistical filter using Median and Median Absolute Deviation (MAD).
    df: PTV DataFrame with columns ['x', 'y', 'z', 'u', 'v', 'w']
    k: Number of neighbors to consider (neighborhood size)
    threshold: Number of MAD units away from median to trigger removal.
    """
    if len(df) <= k:
        print(f"  Warning: DataFrame too small ({len(df)}) for k-NN filter (k={k}). Skipping.")
        return df

    # 1. Compute velocity magnitude (speed)
    u, v, w = df['u'].values, df['v'].values, df['w'].values
    speed = np.sqrt(u**2 + v**2 + w**2)
    
    # 2. Build KDTree for spatial lookup
    points = df[['x', 'y', 'z']].values
    tree = KDTree(points)
    
    # 3. Query k nearest neighbors for everyone
    # k+1 because the point itself is its closest neighbor
    dist, idx = tree.query(points, k=k+1)
    
    # Remove the first index (the point itself)
    neighbor_indices = idx[:, 1:]
    neighbor_distances = dist[:, 1:]  # distances to k nearest neighbors
    
    # Report filtering radius: median of the furthest neighbor distance
    max_neighbor_dist = neighbor_distances[:, -1]  # distance to k-th (furthest) neighbor
    median_filter_radius = np.median(max_neighbor_dist)
    print(f"  Filtering radius: median voxel distance to {k}-th neighbor = {median_filter_radius:.4f}")
    
    # 4. Statistical analysis
    neighbor_speeds = speed[neighbor_indices]
    
    # Local Median and MAD
    local_medians = np.median(neighbor_speeds, axis=1)
    # MAD = median(|x - median(x)|)
    local_mads = np.median(np.abs(neighbor_speeds - local_medians[:, np.newaxis]), axis=1)
    
    # Add a small epsilon to local_mads to avoid division by zero in perfectly uniform regions
    epsilon = 1e-6
    z_scores = (np.abs(speed - local_medians)) / (local_mads + epsilon)
    
    # 5. Mask outliers
    keep_mask = z_scores <= threshold
    
    n_removed = np.sum(~keep_mask)
    if n_removed > 0:
        print(f"  Outlier Filter: Removed {n_removed} points ({n_removed/len(df)*100:.2f}%).")
        return df[keep_mask].reset_index(drop=True)
    else:
        print("  Outlier Filter: No outliers detected.")
        return df

def remove_outliers_threshold(df, max_speed=10.0):
    """
    Simple threshold filter to remove points with unphysical velocity magnitudes.
    """
    u, v, w = df['u'].values, df['v'].values, df['w'].values
    speed = np.sqrt(u**2 + v**2 + w**2)
    
    keep_mask = speed <= max_speed
    n_removed = np.sum(~keep_mask)
    
    if n_removed > 0:
        print(f"  Threshold Filter: Removed {n_removed} points with speed > {max_speed}.")
        return df[keep_mask].reset_index(drop=True)
    return df

def apply_filters(df, args):
    """
    Centralized entry point for all PTV data filtering.
    """
    if not args.filter_outliers:
        return df
        
    # Phase 1: Global speed threshold
    df = remove_outliers_threshold(df, max_speed=args.filter_max_speed)
    
    # Phase 2: Statistical k-NN (MAD) filter
    if len(df) > 0:
        df = remove_outliers_knn(df, k=args.filter_neighbors, threshold=args.filter_threshold)
        
    return df
