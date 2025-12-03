import os
import cv2
import argparse
from tqdm import tqdm

def create_video_from_images(image_folder, output_video_path, fps=3):
    state = output_video_path.split('/')[-1]
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    if not images:
        print(f"No images found in {image_folder}. Skipping video creation.")
        return

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Could not read first image {first_image_path}. Skipping video creation.")
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_name in tqdm(images, desc=f"Creating {state}", leave=False):    
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        if img is not None:
            video.write(img)
        else:
            print(f"Warning: Could not read image {image_path}. Skipping.")

    video.release()
    # print(f"Video created: {output_video_path}")

def process_scenes(base_input_path):
    scene_dirs = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d)) and d.startswith('scene_')]
    scene_dirs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for scene_name in scene_dirs:
        scene_path = os.path.join(base_input_path, scene_name)
        
        print(f"Processing scene: {scene_name}")

        state_dirs = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d)) and d.startswith('state_')]
        state_dirs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        for state_name in state_dirs:
            state_path = os.path.join(scene_path, state_name)
            
            image_folder = os.path.join(state_path, 'RGB')
            if not os.path.exists(image_folder):
                print(f"RGB folder not found in {state_path}. Skipping state.")
                continue

            output_video_name = f"{state_name}.mp4"
            output_video_path = os.path.join(scene_path, output_video_name)

            create_video_from_images(image_folder, output_video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create MP4 videos from RGB image sequences in scene/state directories.')
    parser.add_argument('--input_dir', type=str, help='Base directory containing scene_X/state_Y/rgb/ image structures.')
    args = parser.parse_args()

    process_scenes(args.input_dir)