import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn

# Define camera intrinsic matrix for TUM dataset
K = torch.tensor([
    [525.0, 0.0, 319.5],
    [0.0, 525.0, 239.5],
    [0.0, 0.0, 1.0]
], dtype=torch.float32)

# Quaternion to Rotation Matrix
def quaternion_to_rotation_matrix(q):
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

# Compute Relative Pose
def compute_relative_pose(pose1, pose2):
    trans1, quat1 = pose1[:3], pose1[3:]
    trans2, quat2 = pose2[:3], pose2[3:]
    relative_translation = trans2 - trans1
    quat1_inv = np.array([-quat1[0], -quat1[1], -quat1[2], quat1[3]])
    relative_rotation = quaternion_multiply(quat1_inv, quat2)
    return np.hstack([relative_translation, relative_rotation])

# Quaternion Multiplication
def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])

# Preprocess TUM Sequence
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

    for i in range(len(depth_files) - 2 * frame_spacing):
        depth1 = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth2 = cv2.imread(depth_files[i + frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth3 = cv2.imread(depth_files[i + 2 * frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)

        depth1 = cv2.resize(depth1, target_size) / 5000.0  # Normalize depth
        depth2 = cv2.resize(depth2, target_size) / 5000.0
        depth3 = cv2.resize(depth3, target_size) / 5000.0

        pose1 = groundtruth.iloc[i][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
        pose2 = groundtruth.iloc[i + 2 * frame_spacing][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values

        relative_pose = compute_relative_pose(pose1, pose2)

        depth_data.append((depth1, depth2, depth3))  # Add three depth maps
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
        d1, d2, d3 = self.depth_data[idx]  # Expect three depth maps
        pose = self.pose_data[idx]
        d1 = torch.tensor(d1, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        d2 = torch.tensor(d2, dtype=torch.float32).unsqueeze(0)
        d3 = torch.tensor(d3, dtype=torch.float32).unsqueeze(0)
        pose = torch.tensor(pose, dtype=torch.float32)
        return d1, d2, d3, pose

# Load and Split Data
print("Loading data...")
data_dir = os.path.dirname(__file__)
depth_data_path = os.path.join(data_dir, "depth_data.npy")
pose_data_path = os.path.join(data_dir, "pose_data.npy")

if os.path.exists(depth_data_path) and os.path.exists(pose_data_path):
    depth_data = np.load(depth_data_path, allow_pickle=True)
    pose_data = np.load(pose_data_path, allow_pickle=True)
else:
    sequence_path = os.path.join(data_dir, "rgbd_dataset_freiburg1_room")
    depth_data, pose_data = preprocess_tum_sequence(sequence_path, frame_spacing=3)
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

# Initialize Model
class DepthPoseEstimationNN(nn.Module):
    def __init__(self):
        super(DepthPoseEstimationNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )

    def forward(self, d1, d2, d3, pose):
        depth_input = torch.cat((d1, d2, d3), dim=1)
        conv_out = self.conv_block(depth_input)
        conv_out = conv_out.view(conv_out.size(0), -1)
        pred_pose = self.fc(conv_out)
        return pred_pose

# Training Loop
model = DepthPoseEstimationNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

if os.path.exists("depth_pose_model.pth"):
    print("Loading pretrained model...")
    model.load_state_dict(torch.load("depth_pose_model.pth"))
else:
    print("Training model...")
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        for d1, d2, d3, pose in train_loader:
            optimizer.zero_grad()
            pred_pose = model(d1, d2, d3, pose)
            loss = loss_fn(pred_pose, pose)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}")
    torch.save(model.state_dict(), "depth_pose_model.pth")

# Evaluation
test_loss = 0.0
true_vals, pred_vals = [], []
model.eval()
with torch.no_grad():
    for d1, d2, d3, pose in test_loader:
        pred_pose = model(d1, d2, d3, pose)
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





