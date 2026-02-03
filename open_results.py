import numpy as np
import argparse
from visualizer import show

def main():
    parser = argparse.ArgumentParser(description="Visualize PTV results from NPZ file.")
    parser.add_argument("file", nargs="?", default="sinteredGlass_interpolated.npz", help="Path to the .npz result file.")
    args = parser.parse_args()

    print(f"Loading data from {args.file}...")
    try:
        data = np.load(args.file)
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.")
        return

    # Extract coordinates
    x, y, z = data['x'], data['y'], data['z']
    mask = data['mask'] if 'mask' in data else None

    # Handle dual fields (Initial vs Cleaned)
    if 'u_init' in data:
        print("Found both initial and cleaned velocity fields.")
        u = (data['u'], data['u_init'])
        v = (data['v'], data['v_init'])
        w = (data['w'], data['w_init'])
    else:
        print("Found single velocity field.")
        u, v, w = data['u'], data['v'], data['w']

    print("Launching visualizer...")
    show(u, v, w, x, y, z, mask=mask)

if __name__ == "__main__":
    main()
