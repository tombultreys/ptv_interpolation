
import os
import numpy as np
import subprocess
from interpolator import load_ptv_data, load_mask
from auto_align import find_best_offset

# Paths
INPUT_CSV = r"C:\Users\tbultrey\Documents\XPTV_porousGlass\4_trackingOutput\sinteredGlass_unsmoothed_velocityPoints.csv"
INPUT_MASK = r"C:\Users\tbultrey\Documents\XPTV_porousGlass\2_segmentedImage\001_6_HQ_sample2_poreMaskEroded_binary.tif"

# Parameters
INVERT = False
SWAP_XY = False 
USE_AUTO_ALIGN = True
INITIAL_GUESS = (175, 160, 60)
MASK_TRANSPOSE = None # Not needed with aligned ZYX logic
SAMPLE_SIZE = 2000 

def main():
    print("--- Integrated PTV Alignment Tool ---")
    
    if not os.path.exists(INPUT_CSV) or not os.path.exists(INPUT_MASK):
        print("Error: Input files missing. Check paths.")
        return

    # 1. Run Auto-alignment or Use Initial Guess
    if USE_AUTO_ALIGN:
        print(f"\nPhase 1: Running Auto-Alignment (using {SAMPLE_SIZE} points)...")
        df_full = load_ptv_data(INPUT_CSV)
        
        if SWAP_XY:
            print("Swapping X and Y for alignment check...")
            df_full[['x', 'y']] = df_full[['y', 'x']]
            if 'u' in df_full.columns and 'v' in df_full.columns:
                df_full[['u', 'v']] = df_full[['v', 'u']]

        df_sample = df_full.sample(n=min(SAMPLE_SIZE, len(df_full)))
        mask = load_mask(INPUT_MASK)
        
        if MASK_TRANSPOSE:
            print(f"Transposing mask with axes {MASK_TRANSPOSE}...")
            mask = np.transpose(mask, axes=MASK_TRANSPOSE)

        best_offset, score = find_best_offset(df_sample, mask, initial_offset=INITIAL_GUESS, invert=INVERT)
        offset_final = np.round(best_offset).astype(int)
        print(f"\nAuto-Alignment Result: {offset_final}")
    else:
        print(f"\nAuto-Alignment disabled. Using initial guess: {INITIAL_GUESS}")
        offset_final = INITIAL_GUESS

    # 2. Launch Pre-viewer with these results
    print("\nPhase 2: Launching Interactive Pre-viewer for manual verification...")
    cmd = [
        "python", "pre_viewer.py",
        "--input", INPUT_CSV,
        "--mask", INPUT_MASK,
        "--data-offset", str(offset_final[0]), str(offset_final[1]), str(offset_final[2])
    ]
    if INVERT:
        cmd.append("--invert-mask")
    if SWAP_XY:
        cmd.append("--swap-xy")
    if MASK_TRANSPOSE:
        cmd.extend(["--mask-transpose"] + [str(a) for a in MASK_TRANSPOSE])
        
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
