import os
from PIL import Image

# Input/output paths
input_dir = "/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_desk/depth"
output_gif = "/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_desk/depth/test.gif"

def make_gif(input_dir, output_path):
    images = []
    # Sort filenames numerically by stripping extension and converting to int
    for filename in sorted(
        os.listdir(input_dir),
        key=lambda x: int(os.path.splitext(x)[0])
    ):
        if filename.endswith(('.png', '.jpg')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert("RGBA")
            images.append(img)

    if not images:
        print("No images found in", input_dir)
        return

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=100
    )
    print(f"GIF saved to {output_path}")



# Run
# make_gif(input_dir, output_gif)
    
gt_path = '/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png'
dp_path = '/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_desk/dp/1305031453.323682.png'
# image_path = '/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_desk/rgb/1305031453.323682.png'
image_path = '/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_room/rgb/1305031952.736938.png'

# get depth pro prediction of img
import DP.src.depth_pro as depth_pro
import torch
import gc
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
model, transform = depth_pro.create_model_and_transforms(device=DEVICE)
model.eval()
torch.cuda.empty_cache()
gc.collect()

try:
    # Load image using depth_pro.load_rgb.
    print(f"Loading {image_path}...")
    image, _, f_px = depth_pro.load_rgb(image_path)
except Exception as e:
    print(f"Error loading {image_path}: {e}")

# Apply the preprocessing transform to the image (expects a PIL Image).
image_tensor = transform(image).to(DEVICE)

torch.cuda.empty_cache()
gc.collect()

# Run inference. The model returns a dictionary with depth and focal length.
with torch.no_grad():
    prediction = model.infer(image_tensor, f_px=f_px)
depth = prediction["depth"]  # Depth in meters.
focallength_px = prediction["focallength_px"]




# plot the two images side by side with individual colorbars
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
def plot_images_side_by_side(image1, image2, image3, image4, title1='Image 1', title2='Image 2', title3='Image 3', title4='Image 4'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Display the first image
    im1 = axes[0, 0].imshow(image1, cmap='plasma')
    axes[0, 0].set_title(title1)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Display the second image
    im2 = axes[0, 1].imshow(image2, cmap='plasma')
    axes[0, 1].set_title(title2)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Display the third image
    im3 = axes[1, 0].imshow(image3, cmap='plasma')
    axes[1, 0].set_title(title3)
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Display the fourth image
    im4 = axes[1, 1].imshow(image4, cmap='plasma')
    axes[1, 1].set_title(title4)
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.show()

# Load the images
gt_image = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
dp_image = cv2.imread(dp_path, cv2.IMREAD_UNCHANGED)
rgb_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
new_dp_image = depth.cpu().detach().numpy().squeeze()

# Convert to float32 for better visualization
gt_image = gt_image.astype(np.float32) / 5000
dp_image = dp_image.astype(np.float32)
rgb_image = rgb_image.astype(np.float32) / 255
new_dp_image = new_dp_image.astype(np.float32)

# plot the images
# plot_images_side_by_side(gt_image, dp_image, title1='Ground Truth Depth', title2='Predicted Depth')
plot_images_side_by_side(gt_image, dp_image, new_dp_image, rgb_image, title1='Ground Truth Depth', title2='Depth Pro Depth', title3='New Depth Pro Depth', title4='RGB Image')