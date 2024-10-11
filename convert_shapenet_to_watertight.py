import os
import subprocess
from tqdm import tqdm

# Define the paths
SHAPENET_DIR = '/home/harsha/Documents/shapenet/02691156'
OUTPUT_DIR = '/home/harsha/Documents/watertight_shapenet/02691156'
MANIFOLD_TOOL = '/home/harsha/Manifold/build/manifold'

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_obj_file(input_file, output_file):
    """Run the manifold tool on the input file and save the output."""
    subprocess.run([MANIFOLD_TOOL, input_file, output_file], check=True)

def main():
    # Collect all obj files
    obj_files = []
    for root, _, files in os.walk(SHAPENET_DIR):
        if 'model_normalized.obj' in files:
            input_file = os.path.join(root, 'model_normalized.obj')
            obj_files.append(input_file)

    # Process each obj file with a progress bar
    for input_file in tqdm(obj_files, desc="Processing OBJ files"):
        relative_path = os.path.relpath(os.path.dirname(input_file), SHAPENET_DIR)
        output_subdir = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(output_subdir, exist_ok=True)
        output_file = os.path.join(output_subdir, 'model_watertight.obj')

        # Process the OBJ file
        process_obj_file(input_file, output_file)
        print(f"Processed {input_file} -> {output_file}")

if __name__ == '__main__':
    main()

