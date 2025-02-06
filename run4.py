# no randomization or jolts, clean prediction
import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch.nn as nn

# ---------------------------
# Camera Intrinsic Matrix (TUM)
# ---------------------------
K = torch.tensor([
    [525.0, 0.0, 319.5],
    [0.0, 525.0, 239.5],
    [0.0, 0.0, 1.0]
], dtype=torch.float32)

# ---------------------------
# Quaternion Utilities
# ---------------------------
def quaternion_multiply(q1, q2):
    """
    Quaternion multiplication: q1 * q2
    Each quaternion is [qx, qy, qz, qw].
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
    Compute relative pose from pose1 to pose2.
    pose: [tx, ty, tz, qx, qy, qz, qw]
    """
    trans1, quat1 = pose1[:3], pose1[3:]
    trans2, quat2 = pose2[:3], pose2[3:]

    # Relative translation
    relative_translation = trans2 - trans1

    # Relative rotation = inv(quat1) * quat2
    quat1_inv = np.array([-quat1[0], -quat1[1], -quat1[2], quat1[3]])
    relative_rotation = quaternion_multiply(quat1_inv, quat2)

    return np.hstack([relative_translation, relative_rotation])

def quaternion_to_rotation_matrix(q):
    """
    Converts batched quaternions to rotation matrices.
    q: [B, 4] => [B, 3, 3]
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

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_tum_sequence(parent_sequence_path, frame_spacing, target_size=(128, 128)):
    """
    Creates triplets of frames spaced by 'frame_spacing'.
    Example: i, i+frame_spacing, i+2*frame_spacing
    Useful for normal sequential data.
    """
    for sequence_folder in os.listdir(parent_sequence_path):
        sequence_path = os.path.join(parent_sequence_path, sequence_folder)

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

        # We need up to i + 2*frame_spacing
        max_idx = len(depth_files) - 2 * frame_spacing
        for i in range(max_idx):
            depth1 = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth2 = cv2.imread(depth_files[i + frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth3 = cv2.imread(depth_files[i + 2 * frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)

            # Resize and normalize
            depth1 = cv2.resize(depth1, target_size) / 5000.0
            depth2 = cv2.resize(depth2, target_size) / 5000.0
            depth3 = cv2.resize(depth3, target_size) / 5000.0

            pose1 = groundtruth.iloc[i][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
            pose3 = groundtruth.iloc[i + 2 * frame_spacing][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values

            # Relative pose from frame i to frame i + 2*frame_spacing
            relative_pose = compute_relative_pose(pose1, pose3)

            depth_data.append((depth1, depth2, depth3))
            pose_data.append(relative_pose)

    return depth_data, pose_data


# ---------------------------
# Dataset Class
# ---------------------------
class DepthPoseDataset(Dataset):
    def __init__(self, depth_data, pose_data):
        self.depth_data = depth_data
        self.pose_data = pose_data

    def __len__(self):
        return len(self.depth_data)

    def __getitem__(self, idx):
        d1, d2, d3 = self.depth_data[idx]
        pose = self.pose_data[idx]
        d1 = torch.tensor(d1, dtype=torch.float32).unsqueeze(0)
        d2 = torch.tensor(d2, dtype=torch.float32).unsqueeze(0)
        d3 = torch.tensor(d3, dtype=torch.float32).unsqueeze(0)
        pose = torch.tensor(pose, dtype=torch.float32)
        return d1, d2, d3, pose

# ---------------------------
# Model Definition
# ---------------------------
class DepthPoseEstimationNN(nn.Module):
    def __init__(self):
        super(DepthPoseEstimationNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # [tx, ty, tz, qx, qy, qz, qw]
        )

    def forward(self, d1, d2, d3):
        # Concatenate along channel dimension => [B, 3, 128, 128]
        depth_input = torch.cat((d1, d2, d3), dim=1)
        conv_out = self.conv_block(depth_input)
        conv_out = conv_out.view(conv_out.size(0), -1)
        pred_pose = self.fc(conv_out)

        return pred_pose



# ---------------------------
# Main Script with CUDA Support
# ---------------------------


if __name__ == "__main__":
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directory of this file
    data_dir = os.path.dirname(__file__)
    save_folder = os.path.join(data_dir, "saved_data_4")
    os.makedirs(save_folder, exist_ok=True)
    # sequence_path = os.path.join(data_dir, "rgbd_dataset_freiburg1_room")
    parent_sequence_path = os.path.join(data_dir, "tnt_data")



    # Numpy files for caching
    depth_data_path = os.path.join(save_folder, "depth_data_seq.npy")
    pose_data_path = os.path.join(save_folder, "pose_data_seq.npy")

    # ---------------------------
    # 1) Prepare data
    # ---------------------------
    if os.path.exists(depth_data_path) and os.path.exists(pose_data_path):
        print("Loading cached data...")
        depth_data_seq = np.load(depth_data_path, allow_pickle=True)
        pose_data_seq = np.load(pose_data_path, allow_pickle=True)
    else:
        print("Preprocessing TUM sequence...")
        depth_data_seq, pose_data_seq = preprocess_tum_sequence(
            parent_sequence_path, frame_spacing=7
        )
        np.save(depth_data_path, depth_data_seq)
        np.save(pose_data_path, pose_data_seq)

    # Split into train/val/test
    train_depth, val_depth, train_pose, val_pose = train_test_split(
        depth_data_seq, pose_data_seq, test_size=0.2, random_state=42
    )
    val_depth, test_depth, val_pose, test_pose = train_test_split(
        val_depth, val_pose, test_size=0.5, random_state=42
    )

    train_dataset = DepthPoseDataset(train_depth, train_pose)
    val_dataset = DepthPoseDataset(val_depth, val_pose)
    test_dataset = DepthPoseDataset(test_depth, test_pose)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # ---------------------------
    # 3) Initialize & Train Model
    # ---------------------------
    model = DepthPoseEstimationNN().to(device)  # Move model to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    pretrained_model_path = os.path.join(save_folder, "depth_pose_model.pth")
    if os.path.exists(pretrained_model_path):
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    else:
        print("Training model...")
        num_epochs = 200
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for d1, d2, d3, pose in train_loader:
                # Move data to GPU
                d1, d2, d3, pose = d1.to(device), d2.to(device), d3.to(device), pose.to(device)

                optimizer.zero_grad()
                pred_pose = model(d1, d2, d3)
                loss = loss_fn(pred_pose, pose)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.8f}")

        # Save trained model
        torch.save(model.state_dict(), os.path.join(save_folder, "depth_pose_model.pth"))

    # ---------------------------
    # 4) Evaluate
    # ---------------------------
    print("\nEvaluating on test set...")
    test_loss = 0.0
    true_vals, pred_vals = [], []
    model.eval()
    with torch.no_grad():
        for d1, d2, d3, pose in test_loader:
            # Move data to GPU
            d1, d2, d3, pose = d1.to(device), d2.to(device), d3.to(device), pose.to(device)

            pred_pose = model(d1, d2, d3)
            loss = loss_fn(pred_pose, pose)
            test_loss += loss.item()
            true_vals.append(pose.cpu().numpy())
            pred_vals.append(pred_pose.cpu().numpy())

    true_vals = np.vstack(true_vals)
    pred_vals = np.vstack(pred_vals)
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")


    import matplotlib.pyplot as plt

    # Number of components (translation: 3, rotation: 4)
    num_components = true_vals.shape[1]  # Assuming true_vals and pred_vals are numpy arrays

    # Create subplots
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111, projection='3d')

    # Plot actual path vs predicted path in 3D
    ax1.plot(true_vals[100:110, 0], true_vals[100:110, 1], true_vals[100:110, 2], label='Ground Truth', marker='o', markersize=3, linestyle='-')
    ax1.plot(pred_vals[100:110, 0], pred_vals[100:110, 1], pred_vals[100:110, 2], label='Predicted', marker='x', markersize=3, linestyle='--')
    ax1.set_xlabel("Translation X (tx)")
    ax1.set_ylabel("Translation Y (ty)")
    ax1.set_zlabel("Translation Z (tz)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("Actual Path vs Predicted Path (3D)")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for the suptitle

    plt.show()

    # Convert to numpy arrays for analysis
    true_vals = np.vstack(true_vals)
    pred_vals = np.vstack(pred_vals)

    # ---------------------------
    # 4) Visualizations
    # ---------------------------
    # 4a) Pose Prediction Plot
    def plot_pose_predictions(true_vals, pred_vals, sample_range=60):
        titles = [
            "Translation X (tx)", "Translation Y (ty)", "Translation Z (tz)",
            "Rotation X (qx)", "Rotation Y (qy)", "Rotation Z (qz)", "Rotation W (qw)"
        ]
        num_components = true_vals.shape[1]
        fig, axes = plt.subplots(num_components, 1, figsize=(10, 2 * num_components), sharex=True)

        for i in range(num_components):
            axes[i].plot(true_vals[:sample_range, i], label='Ground Truth',
                         marker='o', markersize=3, linestyle='-')
            axes[i].plot(pred_vals[:sample_range, i], label='Predicted',
                         marker='x', markersize=3, linestyle='--')
            axes[i].set_ylabel(titles[i])
            axes[i].legend()
            axes[i].grid(True)

        axes[-1].set_xlabel("Sample Index")
        fig.suptitle("Ground Truth vs Predicted Pose Components", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    plot_pose_predictions(true_vals, pred_vals, sample_range=60)
