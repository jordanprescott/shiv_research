import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image
import DP.src.depth_pro as depth_pro


# Load model and preprocessing transform for Depth Pro.
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
model, transform = depth_pro.create_model_and_transforms(device=DEVICE)
model.eval()

# Define input and output directories.
input_dir = "/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_room/rgb"
output_dir = "/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_room/dp"  # dp for Depth Pro outputs
os.makedirs(output_dir, exist_ok=True)

# Get list of image files in the input directory.
image_paths = glob.glob(os.path.join(input_dir, "*.*"))

for image_path in image_paths:
    try:
        # Load image using depth_pro.load_rgb.
        image, _, f_px = depth_pro.load_rgb(image_path)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        continue

    # Apply the preprocessing transform to the image (expects a PIL Image).
    image_tensor = transform(image).to(DEVICE)

    # Run inference. The model returns a dictionary with depth and focal length.
    with torch.no_grad():
        prediction = model.infer(image_tensor, f_px=f_px)
    depth = prediction["depth"]  # Depth in meters.
    focallength_px = prediction["focallength_px"]

    # Convert the depth to a numpy array if it is a tensor.
    if torch.is_tensor(depth):
        depth = depth.cpu().detach().numpy().squeeze()
    else:
        depth = np.array(depth)

    # Normalize the depth map to the range [0, 255] for visualization.
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)

    # Construct the output file path (keeping the original filename).
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    
    # Save the depth map as an 8-bit grayscale image.
    cv2.imwrite(output_path, depth_uint8)
    
    print(f"Processed {image_path} and saved depth map to {output_path}")
