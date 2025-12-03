import os
import re
import pickle
import numpy as np
import shutil

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(PROJECT_DIR, "2_object_descriptions", "assets")

NUM_TO_KEEP = 5

def extract_scene_path(path):
    """
    Extracts 'scene_{}' and its subsequent content from the full path.
    """
    match = re.search(r'(scene_\d+.*)', path)
    if match:
        return match.group(1)
    else:
        return None

def calculate_mask_properties(mask):
    """
    Calculates the area and center point of a given mask.
    mask: numpy array, representing the binary mask.
    Returns: (area, center_x, center_y)
    """
    area = np.sum(mask)
    if area == 0:
        return 0, 0, 0

    # Get coordinates of all non-zero pixels
    coords = np.argwhere(mask > 0)
    # Calculate the center point
    center_y, center_x = np.mean(coords, axis=0)
    return area, center_x, center_y

def filter_assets():
    """
    Iterates through each asset folder under ASSETS_DIR, keeping only the NUM_TO_KEEP pkl files with the largest area and closest to the center for each asset.
    """
    for asset_name in os.listdir(ASSETS_DIR):
        asset_path = os.path.join(ASSETS_DIR, asset_name)
        if not os.path.isdir(asset_path):
            continue

        pkl_files = [f for f in os.listdir(asset_path) if f.endswith('.pkl')]
        if not pkl_files:
            print(f"Asset '{asset_name}' has no pkl files. Skipping.")
            continue

        file_data = []
        for pkl_file in pkl_files:
            file_path = os.path.join(asset_path, pkl_file)
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                # Assumes the pkl file contains a 'mask' key with a numpy array value
                if 'mask' in data and isinstance(data['mask'], np.ndarray):
                    area, center_x, center_y = calculate_mask_properties(data['mask'])
                    mask_height, mask_width = data['mask'].shape
                    target_center_x = mask_width / 2
                    target_center_y = mask_height / 2
                    distance_to_center = np.sqrt((center_x - target_center_x)**2 + (center_y - target_center_y)**2)
                    file_data.append({'path': file_path, 'area': area, 'distance': distance_to_center})
                else:
                    print(f"Warning: '{pkl_file}' in '{asset_name}' does not contain a 'mask' or 'mask' is not a numpy array. Skipping.")
            except Exception as e:
                print(f"Error processing {pkl_file} in {asset_name}: {e}")
                continue

        if not file_data:
            continue

        # Sort by area in descending order, then by distance to center in ascending order
        file_data.sort(key=lambda x: (x['area'], -x['distance']), reverse=True)

        # Keep the top NUM_TO_KEEP files
        files_to_keep_paths = {d['path'] for d in file_data[:NUM_TO_KEEP]}

        # import pdb;pdb.set_trace()
        # Process files: delete unnecessary ones, update retained ones
        for pkl_file in pkl_files:
            file_path = os.path.join(asset_path, pkl_file)
            if file_path not in files_to_keep_paths:
                # Delete unnecessary files
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
            else:
                # Update the 'image_path' field of the retained files
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    if 'image_path' in data:
                        original_path = data['image_path']
                        new_path = extract_scene_path(original_path)
                        if new_path:
                            data['image_path'] = new_path
                            with open(file_path, 'wb') as f:
                                pickle.dump(data, f)
                            print(f"Updated 'image_path' in {file_path} from '{original_path}' to '{new_path}'")
                        else:
                            print(f"Warning: 'scene_' not found in '{original_path}' for {file_path}. 'image_path' not updated.")
                    else:
                        print(f"Warning: 'image_path' not found in {file_path}. Skipping update.")
                except Exception as e:
                    print(f"Error processing {file_path} for update: {e}")




if __name__ == "__main__":
    filter_assets()