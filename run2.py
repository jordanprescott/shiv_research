import os
import numpy as np
import cv2
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from model2 import DepthPoseEstimationNN


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





# Preprocessing TUM dataset
def preprocess_tum_sequence(sequence_path, frame_spacing, target_size=(128, 128)):
    depth_dir = os.path.join(sequence_path, "depth")
    groundtruth_file = os.path.join(sequence_path, "groundtruth.txt")

    # Load ground-truth poses and skip comment lines
    groundtruth = pd.read_csv(
        groundtruth_file,
        delim_whitespace=True,
        header=None,
        comment="#",
        names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    )

    # Sort depth images by timestamp
    depth_files = sorted(os.listdir(depth_dir))
    depth_files = [os.path.join(depth_dir, f) for f in depth_files if f.endswith(".png")]

    depth_data = []
    pose_data = []

    for i in range(len(depth_files) - frame_spacing):
        # Load and resize depth maps
        depth1 = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth2 = cv2.imread(depth_files[i + frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth1 = cv2.resize(depth1, target_size) / 5000.0  # Normalize (assume max depth ~5m)
        depth2 = cv2.resize(depth2, target_size) / 5000.0

        # Get corresponding ground-truth poses
        pose1 = groundtruth.iloc[i][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
        pose2 = groundtruth.iloc[i + frame_spacing][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values

        # Compute relative pose        
        relative_pose = compute_relative_pose(pose1, pose2)

        # Append processed data
        depth_data.append((depth1, depth2))
        pose_data.append(relative_pose)
        

    return depth_data, pose_data


# Dataset and Dataloader
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


# Preprocess the dataset
sequence_path = "/home/jordanprescott/shiv_research/rgbd_dataset_freiburg1_room"
depth_data, pose_data = preprocess_tum_sequence(sequence_path, 5)

# Train-validation-test split
train_depth, val_depth, train_pose, val_pose = train_test_split(depth_data, pose_data, test_size=0.2, random_state=42)
val_depth, test_depth, val_pose, test_pose = train_test_split(val_depth, val_pose, test_size=0.5, random_state=42)

# Create datasets and dataloaders
train_dataset = DepthPoseDataset(train_depth, train_pose)
val_dataset = DepthPoseDataset(val_depth, val_pose)
test_dataset = DepthPoseDataset(test_depth, test_pose)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



# Initialize model, optimizer, loss, and scheduler
model = DepthPoseEstimationNN()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = nn.MSELoss()

if os.path.exists("depth_pose_model2.pth"):
    print("Loading model...")
    model.load_state_dict(torch.load("depth_pose_model2.pth"))
else:
    # Training loop
    print("Training model...")
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        for d1, d2, pose in train_loader:
            optimizer.zero_grad()
            pred_pose = model(d1, d2, pose)
            loss = loss_fn(pred_pose, pose)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for d1, d2, pose in val_loader:
                pred_pose = model(d1, d2, pose)
                loss = loss_fn(pred_pose, pose)
                val_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")

    torch.save(model.state_dict(), "depth_pose_model2.pth")

# Test loop
model.eval()
test_loss = 0.0
true_vals, pred_vals = [], []

with torch.no_grad():
    for d1, d2, pose in test_loader:
        pred_pose = model(d1, d2, pose)
        loss = loss_fn(pred_pose, pose)
        test_loss += loss.item()

        true_vals.append(pose.numpy())  # Collect ground truth
        pred_vals.append(pred_pose.numpy())  # Collect predictions

# Combine into single arrays
true_vals = np.vstack(true_vals)  # Shape: [num_samples, 3]
pred_vals = np.vstack(pred_vals)  # Shape: [num_samples, 3]

print(f"Test Loss: {test_loss / len(test_loader)}")

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
