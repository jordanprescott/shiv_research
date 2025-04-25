import os
import glob
import cv2
import torch
from DA2.depth_anything_v2.dpt import DepthAnythingV2

# Set device (forcing CPU as in your example)
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# DEVICE = 'cpu'

# Model configuration dictionary
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits'  # Choose among 'vits', 'vitb', 'vitl', 'vitg'

# Initialize and load the model
model = DepthAnythingV2(**model_configs[encoder])
model_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model = model.to(DEVICE).eval()

# Define input and output directories
input_dir = "/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_room/rgb"
output_dir = "/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_room/da2"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of image files (adjust the glob pattern if needed)
image_paths = glob.glob(os.path.join(input_dir, "*.*"))

for image_path in image_paths:
    # Load the image
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Infer the depth map (H x W numpy array)
    depth = model.infer_image(raw_img)

    # Construct the output file path (using the same filename)
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Save the depth map as an image
    cv2.imwrite(output_path, depth)
    
    print(f"Processed {image_path} and saved depth map to {output_path}")
