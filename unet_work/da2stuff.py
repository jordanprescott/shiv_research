import os
import glob
import cv2
import torch
# from DA2.depth_anything_v2.dpt import DepthAnythingV2
from DA2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
import gc
import numpy as np


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
max_depth = 20 # 20 for indoor model, 80 for outdoor model

# Initialize and load the model
# model = DepthAnythingV2(**model_configs[encoder])
model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})

model_path = f'checkpoints/depth_anything_v2_metric_hypersim_{encoder}.pth'
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model = model.to(DEVICE).eval()

base_dir = "/home/jordanprescott/shiv_research/tnt_data/"

for folder in os.listdir(base_dir):
    # Define input and output directories
    input_dir = f"/home/jordanprescott/shiv_research/tnt_data/{folder}/rgb"
    output_dir = f"/home/jordanprescott/shiv_research/tnt_data/{folder}/da2"

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



# # Define input and output directories.
# input_dir = "/home/jordanprescott/shiv_research/tnt_data/rgbd_scenenet/train/0"
# output_dir = "/home/jordanprescott/shiv_research/tnt_data/rgbd_scenenet/train/0"
# os.makedirs(output_dir, exist_ok=True)

# good_folders = [898, 657, 742, 809, 830, 604, 117, 797, 259, 47, 711, 4, 413, 450, 345, 792, 18, 555, 614, 70, 818, 3, 511, 275, 583, 182, 696, 459, 481, 980, 772, 966, 483, 641, 389, 330, 49, 437, 286]

# good_folders_subset = good_folders[:10]  # For testing, you can limit to the first 10 good folders
# # good_folders_subset = []

# for folder in os.listdir(input_dir):
#     os.makedirs(os.path.join(output_dir, folder, "da2"), exist_ok=True)

#     if int(folder) not in good_folders_subset:
#         print(f"Skipping folder {folder} as it is not in the list of good folders.")
#         continue
#     image_paths = glob.glob(os.path.join(input_dir, folder, "photo", "*.*"))

#     for image_path in image_paths:
#         print(f"Loading {image_path}...")
#         image = cv2.imread(image_path)

#         depth = model.infer_image(image)

#         # # display both depth and depth_og using matplotlib
#         # depth_path = image_path.replace("photo", "depth")
#         # depth_path = depth_path.replace(".jpg", ".png")
#         # print(depth_path)
#         # depth_og = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000
#         # plt.subplot(1, 2, 1)
#         # plt.imshow(depth, cmap='plasma')
#         # plt.title('Depth Map')
#         # plt.colorbar()
#         # plt.subplot(1, 2, 2)
#         # plt.imshow(depth_og, cmap='plasma')
#         # plt.title('Depth Map OG')
#         # plt.colorbar()
#         # plt.show()

#         # Construct the output file path (keeping the original filename).
#         filename = os.path.basename(image_path)
#         output_path = os.path.join(output_dir, folder, "da2", filename)
        
#         # Save the depth map as an 8-bit grayscale image.
#         cv2.imwrite(output_path, depth)
        
#         print(f"Processed {image_path} and saved depth map to {output_path}")


# # visualize the depth map and compare to ground truth for example image
# example_image_path = "/home/jordanprescott/shiv_research/tnt_data/rgbd_scenenet/train/0/898/photo/100.jpg"
# example_gt_path = "/home/jordanprescott/shiv_research/tnt_data/rgbd_scenenet/train/0/898/depth/100.png"
# example_depth_path = "/home/jordanprescott/shiv_research/tnt_data/rgbd_scenenet/train/0/898/da2/100.jpg"

# import matplotlib.pyplot as plt
# # Load the example image and depth map
# example_image = cv2.imread(example_image_path)
# example_depth = cv2.imread(example_depth_path, cv2.IMREAD_UNCHANGED)
# example_gt_depth = cv2.imread(example_gt_path, cv2.IMREAD_UNCHANGED) / 1000

# example_da2 = model.infer_image(example_image).astype(np.float32) 

# # Display the images
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))
# plt.title('Example Image')
# plt.axis('off')
# plt.subplot(1, 3, 2)
# plt.imshow(example_da2, cmap='plasma')
# plt.title('Depth Map (DA2)')
# plt.axis('off')
# plt.subplot(1, 3, 3)
# plt.imshow(example_gt_depth, cmap='plasma')
# plt.title('Ground Truth Depth Map')
# plt.axis('off')
# plt.tight_layout()
# plt.show()
