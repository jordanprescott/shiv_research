import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------- 1. Define File Lists -----------------------
image_files = [
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031910.765238.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031910.797230.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031910.835208.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031910.865025.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031910.897222.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031910.935211.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031910.965249.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031910.997325.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031911.035050.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031911.065269.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031911.097196.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031911.135664.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031911.165177.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031911.197213.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/rgb/1305031911.235651.png'
]

depth_files = [
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031910.771502.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031910.803249.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031910.835215.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031910.871167.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031910.903682.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031910.935221.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031910.971338.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031911.003202.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031911.035056.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031911.074509.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031911.103332.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031911.135683.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031911.171831.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031911.207568.png',
    '/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room/depth/1305031911.235660.png'
]

# ----------------------- 2. Camera Intrinsics and Baseline -----------------------
K = np.array([
    [525.0, 0.0, 319.5],
    [0.0, 525.0, 239.5],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
focal_length = K[0, 0]
baseline = 0.075  # approximate baseline in meters

# ----------------------- 3. Initialize SIFT Detector -----------------------
sift = cv2.SIFT_create()
ratio_thresh = 0.5  # Adjust as needed

# ----------------------- 4. Process Consecutive Image Pairs via Triangulation -----------------------
aggregated_proj_points = []
aggregated_depths = []

for i in range(len(image_files) - 1):
    # Load consecutive images.
    img_left = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(image_files[i+1], cv2.IMREAD_GRAYSCALE)
    if img_left is None or img_right is None:
        print(f"Error loading image pair {i} and {i+1}")
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
        print(f"Not enough good matches in pair {i}-{i+1}, skipping...")
        continue

    # Extract matched keypoints.
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])

    # Compute the fundamental matrix.
    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)
    if F is None:
        print(f"Fundamental matrix estimation failed for pair {i}-{i+1}")
        continue
    pts_left_in = pts_left[mask.ravel() == 1]
    pts_right_in = pts_right[mask.ravel() == 1]

    # Compute the essential matrix.
    E, _ = cv2.findEssentialMat(pts_left_in, pts_right_in, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        print(f"Essential matrix estimation failed for pair {i}-{i+1}")
        continue

    # Recover relative pose.
    _, R, t, mask_pose = cv2.recoverPose(E, pts_left_in, pts_right_in, K)
    pts_left_final = pts_left_in[mask_pose.ravel() > 0]
    pts_right_final = pts_right_in[mask_pose.ravel() > 0]
    if pts_left_final.shape[0] < 8:
        print(f"Not enough inlier points after pose recovery for pair {i}-{i+1}, skipping...")
        continue

    # Define projection matrices.
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    # Triangulate points.
    pts4d = cv2.triangulatePoints(P1, P2, pts_left_final.T, pts_right_final.T)
    pts4d /= pts4d[3]  # convert from homogeneous coordinates
    pts3d = pts4d[:3].T  # shape (N, 3)

    # The computed depth is the Z-coordinate of the triangulated 3D points.
    computed_depths = pts3d[:, 2]

    # Reproject 3D points to the left image.
    proj_points = P1 @ np.vstack((pts3d.T, np.ones((1, pts3d.shape[0]))))
    proj_points /= proj_points[2, :]
    proj_points = proj_points[:2, :].T  # shape (N, 2)

    # Filter out points outside the image boundaries.
    h, w = img_left.shape
    valid_idx = np.where((proj_points[:, 0] >= 0) & (proj_points[:, 0] < w) &
                           (proj_points[:, 1] >= 0) & (proj_points[:, 1] < h))[0]
    proj_points_valid = proj_points[valid_idx]
    computed_depths_valid = computed_depths[valid_idx]

    # Plot for this pair.
    plt.figure(figsize=(12, 6))
    # Left subplot: original left image with overlaid computed sparse depth points.
    plt.subplot(1, 2, 1)
    plt.imshow(img_left, cmap='gray')
    sc = plt.scatter(proj_points_valid[:, 0], proj_points_valid[:, 1],
                     c=computed_depths_valid, cmap='plasma', s=10)
    plt.colorbar(sc, label='Computed Depth (m)')
    plt.title(f"Original Left Image with Computed Depth (Pair {i}-{i+1})")
    plt.xlabel("Pixel x")
    plt.ylabel("Pixel y")
    
    # Right subplot: true depth map.
    true_depth_raw = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
    if true_depth_raw is None:
        print(f"Error loading true depth file for image {i}")
        continue
    true_depth = true_depth_raw.astype(np.float32) / 5000.0
    plt.subplot(1, 2, 2)
    plt.imshow(true_depth, cmap='plasma')
    plt.title(f"True Depth Map (Image {i})")
    plt.colorbar(label='Depth (m)')
    
    plt.tight_layout()
    plt.show()
    
    # Aggregate points and depths.
    aggregated_proj_points.append(proj_points_valid)
    aggregated_depths.append(computed_depths_valid)

# ----------------------- 5. Final Aggregated Plot -----------------------
# Concatenate all aggregated points and depths.
if aggregated_proj_points:
    all_points = np.vstack(aggregated_proj_points)
    all_depths = np.hstack(aggregated_depths)
    
    # Load a reference image (use the first image) and its true depth map.
    ref_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    true_depth_ref_raw = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
    if ref_img is None or true_depth_ref_raw is None:
        print("Error loading reference image or its true depth map.")
    else:
        true_depth_ref = true_depth_ref_raw.astype(np.float32) / 5000.0
        h_ref, w_ref = ref_img.shape
        
        # Filter aggregated points within reference image boundaries.
        valid_idx = np.where((all_points[:, 0] >= 0) & (all_points[:, 0] < w_ref) &
                               (all_points[:, 1] >= 0) & (all_points[:, 1] < h_ref))[0]
        all_points_valid = all_points[valid_idx]
        all_depths_valid = all_depths[valid_idx]
        
        # Final aggregated plot: side-by-side comparison.
        plt.figure(figsize=(14, 7))
        
        # Left: reference image with overlaid aggregated computed depth points.
        plt.subplot(1, 2, 1)
        plt.imshow(ref_img, cmap='gray')
        sc = plt.scatter(all_points_valid[:, 0], all_points_valid[:, 1],
                         c=all_depths_valid, cmap='plasma', s=10)
        plt.colorbar(sc, label='Computed Depth (m)')
        plt.title("Aggregated Computed Depth Points on Reference Image")
        plt.xlabel("Pixel x")
        plt.ylabel("Pixel y")
        
        # Right: true depth map of the reference image.
        plt.subplot(1, 2, 2)
        plt.imshow(true_depth_ref, cmap='plasma')
        plt.title("True Depth Map (Reference Image)")
        plt.colorbar(label='Depth (m)')
        
        plt.tight_layout()
        plt.show()
