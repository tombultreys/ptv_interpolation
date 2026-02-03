
import os
import subprocess

# Paths provided by user
# Note: Using raw strings for Windows paths
INPUT_CSV = r"C:\Users\tbultrey\Documents\XPTV_porousGlass\4_trackingOutput\sinteredGlass_unsmoothed_velocityPoints.csv"
INPUT_MASK = r"C:\Users\tbultrey\Documents\XPTV_porousGlass\2_segmentedImage\001_6_HQ_sample2_poreMaskEroded_binary.tif"

# Output filenames
OUTPUT_TIF = "sinteredGlass_interpolated.tif"
OUTPUT_NPZ = "sinteredGlass_interpolated.npz"

# Parameters
# Center of 657x657x656 is approx 328
# Crop 128 box -> 328 +/- 64 = [264, 392]
CROP_COORDS = ["175", "497", "160", "497", "60", "546"]
METHOD = "linear"

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
        "--input", INPUT_CSV,
        "--mask", INPUT_MASK,
        "--downscale", "2",
        #"--resolution", "320", "340", "500", # Can now be e.g., "328" "328" "164"
        "--crop", *CROP_COORDS, 
        "--divergence-free",
        "--output-tif", OUTPUT_TIF,
        "--output-npz", OUTPUT_NPZ,
        "--data-offset", "175", "160", "60",
        # "--mask-transpose", "2", "1", "0",
        # "--swap-xy",
        "--method", METHOD,
        "--n-jobs", "4"
        # "--no-plot" # Uncomment to run headless
    ]
    
    print("\nCommand to run:")
    print(" ".join(cmd))
    
    print("\nExecuting...")
    subprocess.run(cmd) 

if __name__ == "__main__":
    run()
