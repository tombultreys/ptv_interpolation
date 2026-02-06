
import os
import subprocess

# Paths provided by user
# Note: Using raw strings for Windows paths
INPUT_CSV = r"/media/tbultrey/data/PIV_20210622/XPTV_porousGlass/4_trackingOutput/sinteredGlass_unsmoothed_velocityPoints.csv"
INPUT_MASK = r"/media/tbultrey/data/PIV_20210622/XPTV_porousGlass/2_segmentedImage/000_6_HQ_sample2_poreMask_binary.tif"

# Output filenames
OUTPUT_TIF = "sinteredGlass_interpolatedFullRes_sibson20_var.tif"
OUTPUT_NPZ = "sinteredGlass_interpolatedFullRes_sibson20_var.npz"
CROP_COORDS = ["175", "497", "160", "497", "60", "546"] # cropping box [x1, x2, y1, y2, z1, z2]
OFFSET = ["175", "160", "60"] # offset to align velocity data with cropped box
METHOD = "sibson" # linear, rbf or sibson
DOWNSCALE = "1" # 1 for original resolution, 2 for half resolution, etc

def run():
    print("--- Setting up PTV Interpolation for Porous Glass Data ---")
    
    if not os.path.exists(INPUT_CSV):
        print(f"Warning: Input CSV not found at {INPUT_CSV}")
    if not os.path.exists(INPUT_MASK):
        print(f"Warning: Input Mask not found at {INPUT_MASK}")

    from interpolator import load_ptv_data
    print(f"Reading PTV data extent from {INPUT_CSV}...")
    df = load_ptv_data(INPUT_CSV)
    print(f"Data Spatial Extent:")
    print(f"  X: [{df.x.min():.2f}, {df.x.max():.2f}]")
    print(f"  Y: [{df.y.min():.2f}, {df.y.max():.2f}]")
    print(f"  Z: [{df.z.min():.2f}, {df.z.max():.2f}]")
    print(f"  Total points: {len(df)}")

    cmd = [
        "python", "main.py",

        ### Parameters for data handling
        "--input", INPUT_CSV,
        "--mask", INPUT_MASK,
        #"--resolution", "320", "340", "500", # Can now be e.g., "328" "328" "164"
        "--crop", *CROP_COORDS, 
        "--output-tif", OUTPUT_TIF,
        "--output-npz", OUTPUT_NPZ,
        "--data-offset", *OFFSET,
        # "--mask-transpose", "2", "1", "0",
        # "--swap-xy",
    
        ### Parameters for interpolation
        "--method", METHOD,
        "--downscale", DOWNSCALE,

        ###   Parameters for RBF interpolation
        "--rbf-neighbors", "20",
        "--smoothing", "5.0",

        ### Parameters for IDW interpolation
        "--idw-power", "2",
        "--idw-neighbors", "40",

        ### Parameters for Sibson interpolation
        "--sibson-neighbors", "20",

        # Parameters for divergence cleaning
        "--divergence-free",
        "--cleaning-method", "variational", #Cleaning method. "variational" or "iterative"
        "--cleaning-lambda", "200", #Strength for variational cleaning. higher is lower divergence and less data-controlled
        "--iter", "5", #Number of cleaning iterations for iterative cleaning

        #Handling screening by boundaries, using ghost particles
        "--boundary-particles",
        "--boundary-sampling", "50", # Take every xth boundary point
        "--boundary-thickness", "2", # Number of solid voxel layers to include as boundary particles.

        #Parameters for outlier filtering
        "--filter-outliers",
        "--filter-neighbors", "30", # number of nearest neighbors to consider for outlier removal
        "--filter-threshold", "4.0", # threshold for neighbourhood filtering: number of standard deviations from the local median to remove
        "--filter-max-speed", "5.0", # threshold large velocities

        #"--n-jobs", "4"
        # "--no-plot" # Uncomment to run headless
    ]
    
    print("\nCommand to run:")
    print(" ".join(cmd))
    
    print("\nExecuting...")
    subprocess.run(cmd) 

if __name__ == "__main__":
    run()
