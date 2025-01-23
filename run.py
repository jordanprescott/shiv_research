import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
from model import DepthPoseEstimationNN

# Define camera intrinsic matrix for TUM dataset
K = torch.tensor([
    [525.0, 0.0, 319.5],
    [0.0, 525.0, 239.5],
    [0.0, 0.0, 1.0]
], dtype=torch.float32)


# Quaternion to Rotation Matrix
def quaternion_to_rotation_matrix(q):
    """
    Converts a normalized quaternion to a rotation matrix.
    Args:
        q: [B, 4] tensor, where each quaternion is [qx, qy, qz, qw]
    Returns:
        R: [B, 3, 3] rotation matrices
    """
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros((q.shape[0], 3, 3), device=q.device)

    R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)

    return R



def compute_relative_pose(pose1, pose2):
    """
    Compute relative pose (translation and rotation) between two poses.

    Args:
        pose1: [tx, ty, tz, qx, qy, qz, qw] of first pose
        pose2: [tx, ty, tz, qx, qy, qz, qw] of second pose

    Returns:
        relative_pose: [relative_tx, relative_ty, relative_tz, relative_qx, relative_qy, relative_qz, relative_qw]
    """
    # Extract translation and quaternion components
    trans1, quat1 = pose1[:3], pose1[3:]
    trans2, quat2 = pose2[:3], pose2[3:]

    # Compute relative translation
    relative_translation = trans2 - trans1

    # Compute relative rotation (q1^-1 * q2)
    quat1_inv = np.array([-quat1[0], -quat1[1], -quat1[2], quat1[3]])  # Inverse of quaternion
    relative_rotation = quaternion_multiply(quat1_inv, quat2)

    return np.hstack([relative_translation, relative_rotation])


def quaternion_multiply(q1, q2):
    """
    Perform quaternion multiplication.
    
    Args:
        q1, q2: Quaternions [qx, qy, qz, qw]

    Returns:
        result: Result of q1 * q2 as [qx, qy, qz, qw]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])



# Preprocess the TUM sequence
def preprocess_tum_sequence(sequence_path, frame_spacing, target_size=(128, 128)):
    depth_dir = os.path.join(sequence_path, "depth")
    groundtruth_file = os.path.join(sequence_path, "groundtruth.txt")

    # Load ground-truth poses
    groundtruth = pd.read_csv(
        groundtruth_file, sep=" ", header=None, comment="#",
        names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    )

    depth_files = sorted(os.listdir(depth_dir))
    depth_files = [os.path.join(depth_dir, f) for f in depth_files if f.endswith(".png")]

    depth_data = []
    pose_data = []

    for i in range(len(depth_files) - frame_spacing):
        depth1 = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth2 = cv2.imread(depth_files[i + frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth1 = cv2.resize(depth1, target_size) / 5000.0  # Normalize depth
        depth2 = cv2.resize(depth2, target_size) / 5000.0

        pose1 = groundtruth.iloc[i][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
        pose2 = groundtruth.iloc[i + frame_spacing][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values

        # Compute relative pose (translation and rotation)
        relative_translation = pose2[:3] - pose1[:3]
        relative_rotation = pose2[3:]
        
        relative_pose = compute_relative_pose(pose1, pose2)
        
        

        depth_data.append((depth1, depth2))
        # pose_data.append(np.hstack([relative_translation, relative_rotation]))
        
        pose_data.append(relative_pose)
        

    return depth_data, pose_data


# Dataset Class
class DepthPoseDataset(Dataset):
    def __init__(self, depth_data, pose_data):
        self.depth_data = depth_data
        self.pose_data = pose_data

    def __len__(self):
        return len(self.depth_data)

    def __getitem__(self, idx):
        d1, d2 = self.depth_data[idx]
        pose = self.pose_data[idx]
        d1 = torch.tensor(d1, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        d2 = torch.tensor(d2, dtype=torch.float32).unsqueeze(0)
        pose = torch.tensor(pose, dtype=torch.float32)
        return d1, d2, pose


# Reprojection Loss
def reprojection_loss(pred_pose, true_pose, depth1, depth2, K, lambda_geo=0.1, lambda_trans=1.0, lambda_rot=1.0):
    pred_trans = pred_pose[:, :3]
    true_trans = true_pose[:, :3]
    pred_rot = pred_pose[:, 3:]
    true_rot = true_pose[:, 3:]

    # Translation loss
    trans_loss = torch.nn.functional.mse_loss(pred_trans, true_trans)

    # Normalize quaternions
    pred_rot = pred_rot / torch.norm(pred_rot, dim=1, keepdim=True)
    true_rot = true_rot / torch.norm(true_rot, dim=1, keepdim=True)

    # Rotation loss (geodesic distance)
    rot_loss = 1 - torch.abs(torch.sum(pred_rot * true_rot, dim=1)).mean()

    # Compute rotation matrix from quaternion
    pred_R = quaternion_to_rotation_matrix(pred_rot)

    # Ensure depth shape is [B, H, W]
    if len(depth1.shape) == 4:  # If [B, C, H, W], remove channel dimension
        depth1 = depth1.squeeze(1)
        depth2 = depth2.squeeze(1)

    # Back-project depth map to 3D points
    B, H, W = depth1.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device=depth1.device), torch.arange(H, device=depth1.device), indexing="ij")
    grid_x, grid_y = grid_x.float(), grid_y.float()
    pixel_coords = torch.stack((grid_x.flatten(), grid_y.flatten(), torch.ones_like(grid_x.flatten())), dim=1)
    pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1)  # Repeat for batch size

    # Adjust K for batch size
    K_batch = K.unsqueeze(0).repeat(B, 1, 1)  # [B, 3, 3]

    depth1_flat = depth1.view(B, -1)
    cam_coords = torch.bmm(torch.inverse(K_batch), pixel_coords.permute(0, 2, 1))  # Adjust for batched K
    cam_coords = cam_coords * depth1_flat.unsqueeze(1)

    # Transform 3D points to the second camera frame
    cam_coords_2 = torch.bmm(pred_R, cam_coords) + pred_trans.unsqueeze(-1)

    # Project 3D points back to the 2D plane
    proj_coords = torch.bmm(K_batch, cam_coords_2)
    proj_coords = proj_coords / proj_coords[:, 2:3, :]  # Normalize all (x, y, z) by z

    # Reshape projected depth to [B, H, W]
    proj_depth = proj_coords[:, 2, :].view(B, H, W)

    # Compute reprojection error
    geo_loss = torch.nn.functional.mse_loss(proj_depth, depth2)  # Compare directly with depth2

    # Combined loss
    return lambda_trans * trans_loss + lambda_rot * rot_loss + lambda_geo * geo_loss




def smooth_predictions_ema(predictions, alpha=0.2):
    """
    Applies exponential moving average (EMA) smoothing to the predictions.
    
    Args:
        predictions (np.array): Array of predictions to smooth.
        alpha (float): Smoothing factor (0 < alpha <= 1).
    
    Returns:
        np.array: Smoothed predictions.
    """
    smoothed = np.zeros_like(predictions)
    smoothed[0] = predictions[0]  # Initialize with the first value
    for t in range(1, len(predictions)):
        smoothed[t] = alpha * predictions[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed



def smooth_predictions(predictions, window_size=5):
    """
    Applies simple moving average (SMA) smoothing to the predictions.
    
    Args:
        predictions (np.array): Array of predictions to smooth.
        window_size (int): Size of the moving window.
    
    Returns:
        np.array: Smoothed predictions.
    """
    smoothed = np.convolve(predictions, np.ones(window_size) / window_size, mode='valid')
    # To keep the same length, pad the start with the first value
    pad_size = (len(predictions) - len(smoothed)) // 2
    smoothed = np.pad(smoothed, (pad_size, pad_size), mode='edge')
    return smoothed




# Load data and split into train, validation, and test sets
print("Loading data...")
data_dir = os.path.dirname(__file__)
depth_data_path = os.path.join(data_dir, "depth_data.npy")
pose_data_path = os.path.join(data_dir, "pose_data.npy")

if os.path.exists(depth_data_path) and os.path.exists(pose_data_path):
    depth_data = np.load(depth_data_path, allow_pickle=True)
    pose_data = np.load(pose_data_path, allow_pickle=True)
else:
    sequence_path = os.path.join(data_dir, "rgbd_dataset_freiburg1_room")
    depth_data, pose_data = preprocess_tum_sequence(sequence_path, 3)
    np.save(depth_data_path, depth_data)
    np.save(pose_data_path, pose_data)

train_depth, val_depth, train_pose, val_pose = train_test_split(depth_data, pose_data, test_size=0.2, random_state=42)
val_depth, test_depth, val_pose, test_pose = train_test_split(val_depth, val_pose, test_size=0.5, random_state=42)

train_dataset = DepthPoseDataset(train_depth, train_pose)
val_dataset = DepthPoseDataset(val_depth, val_pose)
test_dataset = DepthPoseDataset(test_depth, test_pose)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Initialize model, optimizer, and training loop
model = DepthPoseEstimationNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

if os.path.exists("depth_pose_model.pth"):
    print("Loading pretrained model...")
    model.load_state_dict(torch.load("depth_pose_model.pth"))
else:
    print("Training model...")
    for epoch in range(20):
        model.train()
        train_loss = 0.0
        for d1, d2, pose in train_loader:
            optimizer.zero_grad()
            pred_pose = model(d1, d2, pose)
            # loss = reprojection_loss(pred_pose, pose, depth1=d1, depth2=d2, K=K)
            loss = loss_fn(pred_pose, pose)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}")

    torch.save(model.state_dict(), "depth_pose_model.pth")

# Evaluate model
test_loss = 0.0

true_vals, pred_vals = [], []
model.eval()
with torch.no_grad():
    for d1, d2, pose in test_loader:
        pred_pose = model(d1, d2, pose)
        # loss = reprojection_loss(pred_pose, pose, depth1=d1, depth2=d2, K=K)
        loss = loss_fn(pred_pose, pose)
        test_loss += loss.item()
        true_vals.append(pose.numpy())
        pred_vals.append(pred_pose.numpy())

true_vals = np.vstack(true_vals)
pred_vals = np.vstack(pred_vals)

print(f"Test Loss: {test_loss / len(test_loader)}")

# Plot results
plt.figure()
plt.plot(true_vals[:60, 3], label='True qx')
plt.plot(pred_vals[:60, 3], label='Predicted qx')
plt.legend()
plt.title("Rotation X Prediction")
plt.show()





# Apply smoothing
smoothed_predictions = smooth_predictions(pred_vals[:, 0], window_size=3)  # Simple Moving Average
smoothed_predictions_ema = smooth_predictions_ema(pred_vals[:, 0], alpha=0.5)  # Exponential Moving Average

smoothed_loss = np.mean((true_vals[:, 0] - smoothed_predictions) ** 2)
print(f"Smoothed Loss (SMA): {smoothed_loss}")

smoothed_ema_loss = np.mean((true_vals[:, 0] - smoothed_predictions_ema) ** 2)
print(f"Smoothed Loss (EMA): {smoothed_ema_loss}")


# Number of components (translation: 3, rotation: 4)
num_components = true_vals.shape[1]  # Assuming true_vals and pred_vals are numpy arrays

# Create subplots
fig, axes = plt.subplots(num_components, 1, figsize=(10, 2 * num_components), sharex=True)

# Titles for components
titles = ["Translation X (tx)", "Translation Y (ty)", "Translation Z (tz)",
          "Rotation X (qx)", "Rotation Y (qy)", "Rotation Z (qz)", "Rotation W (qw)"]

for i in range(num_components):
    axes[i].plot(true_vals[:, i], label='Ground Truth', marker='o', markersize=3, linestyle='-')
    axes[i].plot(pred_vals[:, i], label='Predicted', marker='x', markersize=3, linestyle='--')
    axes[i].set_ylabel(titles[i])
    axes[i].legend()
    axes[i].grid(True)

# Set shared x-label
axes[-1].set_xlabel("Sample Index")

# Add overall title
fig.suptitle("Ground Truth vs Predicted for All Components", fontsize=16)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the suptitle

plt.show()
