import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- 1. Adjustable Spacing and File Loading -----------------------
spacing = 5  # For consecutive pairs, use 1; for pairs with one image skipped, use 2, etc.

# Define directories.
rgb_folder = '/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_desk/rgb'
depth_folder = '/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_desk/depth'
dp_dir = '/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_desk/dp'    # Saved Depth Pro frames
da2_dir = '/home/jordanprescott/shiv_research/tnt_data/rgbd_dataset_freiburg1_desk/da2'  # Saved Depth Anything frames

# Get sorted lists of files.
image_files = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
depth_files = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
dp_files = sorted(glob.glob(os.path.join(dp_dir, "*.png")))
da2_files = sorted(glob.glob(os.path.join(da2_dir, "*.png")))

print("Found {} RGB images".format(len(image_files)))
print("Found {} True depth files".format(len(depth_files)))
print("Found {} Depth Pro files".format(len(dp_files)))
print("Found {} Depth Anything files".format(len(da2_files)))

# ----------------------- 2. Camera Intrinsics and Baseline -----------------------
K = np.array([
    [525.0, 0.0, 319.5],
    [0.0, 525.0, 239.5],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
baseline = 0.075  # known baseline in meters

# ----------------------- 3. Initialize SIFT Detector -----------------------
sift = cv2.SIFT_create()
ratio_thresh = 0.5  # Adjust as needed

# ----------------------- 4. Process Image Pairs via Triangulation -----------------------
for i in range(len(image_files) - spacing):
    print("\nProcessing pair index {} and {}".format(i, i+spacing))
    
    # Load image pair: left is image_files[i] and right is image_files[i+spacing]
    img_left = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(image_files[i + spacing], cv2.IMREAD_GRAYSCALE)
    if img_left is None or img_right is None:
        print(f"Error loading RGB pair: {image_files[i]} or {image_files[i+spacing]}")
        continue

    # Optionally enhance contrast.
    img_left = cv2.equalizeHist(img_left)
    img_right = cv2.equalizeHist(img_right)

    # Detect SIFT features.
    kp_left, des_left = sift.detectAndCompute(img_left, None)
    kp_right, des_right = sift.detectAndCompute(img_right, None)

    # Match descriptors.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des_left, des_right, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    if len(good_matches) < 8:
        print(f"Not enough good matches in pair {i}-{i+spacing}, skipping...")
        continue

    # Extract matched keypoints.
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])

    # Compute the fundamental matrix.
    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)
    if F is None:
        print(f"Fundamental matrix estimation failed for pair {i}-{i+spacing}")
        continue
    pts_left_in = pts_left[mask.ravel() == 1]
    pts_right_in = pts_right[mask.ravel() == 1]

    # Compute the essential matrix.
    E, _ = cv2.findEssentialMat(pts_left_in, pts_right_in, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        print(f"Essential matrix estimation failed for pair {i}-{i+spacing}")
        continue

    # Recover relative pose.
    _, R, t, mask_pose = cv2.recoverPose(E, pts_left_in, pts_right_in, K)
    pts_left_final = pts_left_in[mask_pose.ravel() > 0]
    pts_right_final = pts_right_in[mask_pose.ravel() > 0]
    if pts_left_final.shape[0] < 8:
        print(f"Not enough inlier points after pose recovery for pair {i}-{i+spacing}, skipping...")
        continue

    # Compute scale factor based on known baseline.
    scale_factor = baseline / np.linalg.norm(t)
    
    # Define projection matrices.
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    # Triangulate points.
    pts4d = cv2.triangulatePoints(P1, P2, pts_left_final.T, pts_right_final.T)
    pts4d /= pts4d[3]  # Convert from homogeneous coordinates
    pts3d = pts4d[:3].T  # Shape (N, 3)

    # Scale the triangulated points to metric scale.
    pts3d_scaled = pts3d * scale_factor

    # The computed depth is the Z-coordinate of the scaled 3D points.
    computed_depths = pts3d_scaled[:, 2]

    # Reproject 3D points to the left image.
    proj_points = P1 @ np.vstack((pts3d_scaled.T, np.ones((1, pts3d_scaled.shape[0]))))
    proj_points /= proj_points[2, :]
    proj_points = proj_points[:2, :].T  # Shape (N, 2)

    # Filter out points outside the image boundaries.
    h, w = img_left.shape
    valid_idx = np.where((proj_points[:, 0] >= 0) & (proj_points[:, 0] < w) &
                           (proj_points[:, 1] >= 0) & (proj_points[:, 1] < h))[0]
    proj_points_valid = proj_points[valid_idx]
    computed_depths_valid = computed_depths[valid_idx]

    # ----------------------- Load Saved Depth Maps for the "skipped-to" Frame -----------------------
    # Use index i+spacing for the saved maps.
    if i+spacing >= len(dp_files) or i+spacing >= len(da2_files):
        print(f"Skipping pair {i} because saved depth frames for index {i+spacing} are not available.")
        continue
    print("Depth Pro file:", dp_files[i+spacing])
    print("Depth Anything file:", da2_files[i+spacing])
    depth_pro_img = cv2.imread(dp_files[i+spacing], cv2.IMREAD_GRAYSCALE)
    depth_anything_img = cv2.imread(da2_files[i+spacing], cv2.IMREAD_GRAYSCALE)

    # ----------------------- Load True Depth Map for the "skipped-to" Frame -----------------------
    if i+spacing >= len(depth_files):
        print(f"Skipping pair {i} because true depth file for index {i+spacing} is not available.")
        continue
    print("True Depth file:", depth_files[i+spacing])
    true_depth_raw = cv2.imread(depth_files[i+spacing], cv2.IMREAD_UNCHANGED)
    if true_depth_raw is None or depth_pro_img is None or depth_anything_img is None:
        print(f"Skipping pair {i} because not all required depth maps for index {i+spacing} are available.")
        continue
    true_depth = true_depth_raw.astype(np.float32) / 5000.0

    # ----------------------- Plot 2x2 Pairwise Comparison -----------------------
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Left image with computed sparse depth points (from the left frame).
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(img_left, cmap='gray')
    sc = ax1.scatter(proj_points_valid[:, 0], proj_points_valid[:, 1],
                     c=computed_depths_valid, cmap='plasma', s=10)
    ax1.text(10, 20, f"Left RGB Index: {i}", color='white', fontsize=12, backgroundcolor='black')
    plt.colorbar(sc, ax=ax1, label='Computed Depth (m)')
    ax1.set_title(f"Computed Depth (Pair {i}-{i+spacing})")
    ax1.set_xlabel("Pixel x")
    ax1.set_ylabel("Pixel y")
    
    # Subplot 2: True depth map for the skipped-to frame.
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(true_depth, cmap='plasma')
    ax2.text(10, 20, f"True Depth Index: {i+spacing}", color='white', fontsize=12, backgroundcolor='black')
    plt.colorbar(im2, ax=ax2, label='Depth (m)')
    ax2.set_title(f"True Depth Map (Image {i+spacing})")
    
    # Subplot 3: Depth Pro saved depth map (raw values) for the skipped-to frame.
    ax3 = plt.subplot(2, 2, 3)
    im3 = ax3.imshow(depth_pro_img, cmap='plasma')
    ax3.text(10, 20, f"Depth Pro Index: {i+spacing}", color='white', fontsize=12, backgroundcolor='black')
    plt.colorbar(im3, ax=ax3, label='Depth')
    ax3.set_title("Depth Pro Depth Map")
    
    # Subplot 4: Depth Anything saved depth map (raw values) for the skipped-to frame.
    ax4 = plt.subplot(2, 2, 4)
    im4 = ax4.imshow(depth_anything_img, cmap='plasma')
    ax4.text(10, 20, f"Depth Anything Index: {i+spacing}", color='white', fontsize=12, backgroundcolor='black')
    plt.colorbar(im4, ax=ax4, label='Depth')
    ax4.set_title("Depth Anything Depth Map")
    
    plt.tight_layout()
    plt.show()
