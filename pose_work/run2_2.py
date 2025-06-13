import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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
# Preprocessing (Sequential)
# ---------------------------
def preprocess_tum_sequence(sequence_path, frame_spacing, target_size=(128, 128)):
    """
    Creates triplets of frames spaced by 'frame_spacing'.
    Example: i, i+frame_spacing, i+2*frame_spacing
    Useful for normal sequential data.
    """
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
# Preprocessing (Non-Sequential)
# ---------------------------
def preprocess_tum_sequence_nonsequential(sequence_path, 
                                          num_samples=200, 
                                          min_gap=30, 
                                          target_size=(128, 128)):
    """
    Creates triplets of frames that are far apart (at least 'min_gap' frames apart).
    This is intended to produce data where there's little to no overlap.
    """
    depth_dir = os.path.join(sequence_path, "depth")
    groundtruth_file = os.path.join(sequence_path, "groundtruth.txt")

    # Load ground-truth poses
    groundtruth = pd.read_csv(
        groundtruth_file, sep=" ", header=None, comment="#",
        names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    )

    depth_files = sorted(os.listdir(depth_dir))
    depth_files = [os.path.join(depth_dir, f) for f in depth_files if f.endswith(".png")]
    total_frames = len(depth_files)

    depth_data = []
    pose_data = []

    # Randomly pick triplets with a minimum gap
    # Example approach:
    #   1) Pick i randomly in [0, total_frames - 2*min_gap)
    #   2) Pick j in [i+min_gap, total_frames - min_gap)
    #   3) Pick k in [j+min_gap, total_frames)
    # Do this 'num_samples' times or until we run out of valid picks.
    valid_range = total_frames - 2 * min_gap
    if valid_range < 0:
        print("Not enough frames to create non-sequential triplets with the desired min_gap.")
        return [], []

    for _ in range(num_samples):
        i = np.random.randint(0, valid_range)
        j = np.random.randint(i + min_gap, total_frames - min_gap)
        k = np.random.randint(j + min_gap, total_frames)

        depth1 = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth2 = cv2.imread(depth_files[j], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth3 = cv2.imread(depth_files[k], cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Resize and normalize
        depth1 = cv2.resize(depth1, target_size) / 5000.0
        depth2 = cv2.resize(depth2, target_size) / 5000.0
        depth3 = cv2.resize(depth3, target_size) / 5000.0

        pose1 = groundtruth.iloc[i][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
        pose3 = groundtruth.iloc[k][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values

        # Relative pose from frame i to frame k
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

    def forward(self, d1, d2, d3, pose):
        # Concatenate along channel dimension => [B, 3, 128, 128]
        depth_input = torch.cat((d1, d2, d3), dim=1)
        conv_out = self.conv_block(depth_input)
        conv_out = conv_out.view(conv_out.size(0), -1)
        pred_pose = self.fc(conv_out)
        return pred_pose



def plot_pose_results(true_vals_seq, pred_vals_seq,
                      true_vals_nonseq, pred_vals_nonseq):
    """
    Plots a comparison of ground truth vs. predicted poses
    for both Sequential and Non-Sequential test sets.
    """
    # Titles for each component in [tx, ty, tz, qx, qy, qz, qw]
    titles = ["Translation X (tx)", "Translation Y (ty)", "Translation Z (tz)",
              "Rotation X (qx)", "Rotation Y (qy)", "Rotation Z (qz)", "Rotation W (qw)"]
    
    # Create subplots: 2 rows (Sequential, Non-Sequential) x 7 columns (each pose component)
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(20, 6), sharex=False)
    
    # --- Row 1: Sequential data ---
    for i in range(7):
        axes[0, i].plot(
            true_vals_seq[:, i], label='True (seq)', marker='o', markersize=3, linestyle='-'
        )
        axes[0, i].plot(
            pred_vals_seq[:, i], label='Pred (seq)', marker='x', markersize=3, linestyle='--'
        )
        axes[0, i].set_title(titles[i] + "\n(Sequential)")
        axes[0, i].legend()
        axes[0, i].grid(True)
    
    # --- Row 2: Non-Sequential data ---
    for i in range(7):
        axes[1, i].plot(
            true_vals_nonseq[:, i], label='True (nonseq)', marker='o', markersize=3, linestyle='-'
        )
        axes[1, i].plot(
            pred_vals_nonseq[:, i], label='Pred (nonseq)', marker='x', markersize=3, linestyle='--'
        )
        axes[1, i].set_title(titles[i] + "\n(Non-Sequential)")
        axes[1, i].legend()
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.show()


# ---------------------------
# Main Script with CUDA Support
# ---------------------------
if __name__ == "__main__":
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directory of this file
    data_dir = os.path.dirname(__file__)
    sequence_path = os.path.join(data_dir, "rgbd_dataset_freiburg1_room")

    # Numpy files for caching
    depth_data_path = os.path.join(data_dir, "depth_data_seq.npy")
    pose_data_path = os.path.join(data_dir, "pose_data_seq.npy")

    # ---------------------------
    # 1) Prepare SEQUENTIAL data
    # ---------------------------
    if os.path.exists(depth_data_path) and os.path.exists(pose_data_path):
        depth_data_seq = np.load(depth_data_path, allow_pickle=True)
        pose_data_seq = np.load(pose_data_path, allow_pickle=True)
    else:
        depth_data_seq, pose_data_seq = preprocess_tum_sequence(
            sequence_path, frame_spacing=3
        )
        np.save(depth_data_path, depth_data_seq)
        np.save(pose_data_path, pose_data_seq)

    # Split into train/val/test for SEQUENTIAL data
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
    # 2) Prepare NON-SEQUENTIAL data
    # ---------------------------
    depth_data_nonseq, pose_data_nonseq = preprocess_tum_sequence_nonsequential(
        sequence_path, 
        num_samples=100,   # How many random triplets you want
        min_gap=30,        # Minimum gap between frames
        target_size=(128, 128)
    )

    if len(depth_data_nonseq) > 0:
        nonseq_dataset = DepthPoseDataset(depth_data_nonseq, pose_data_nonseq)
        nonseq_loader = DataLoader(nonseq_dataset, batch_size=16, shuffle=False)
    else:
        nonseq_dataset, nonseq_loader = None, None

    # ---------------------------
    # 3) Initialize & Train Model
    # ---------------------------
    model = DepthPoseEstimationNN().to(device)  # Move model to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    if os.path.exists("depth_pose_model.pth"):
        print("Loading pretrained model...")
        model.load_state_dict(torch.load("depth_pose_model.pth", map_location=device))
    else:
        print("Training model on SEQUENTIAL data...")
        num_epochs = 30
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for d1, d2, d3, pose in train_loader:
                # Move data to GPU
                d1, d2, d3, pose = d1.to(device), d2.to(device), d3.to(device), pose.to(device)

                optimizer.zero_grad()
                pred_pose = model(d1, d2, d3, pose)
                loss = loss_fn(pred_pose, pose)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

        # Save trained model
        torch.save(model.state_dict(), "depth_pose_model.pth")

    # ---------------------------
    # 4) Evaluate on SEQUENTIAL TEST
    # ---------------------------
    print("\nEvaluating on SEQUENTIAL test set...")
    test_loss = 0.0
    true_vals, pred_vals = [], []
    model.eval()
    with torch.no_grad():
        for d1, d2, d3, pose in test_loader:
            # Move data to GPU
            d1, d2, d3, pose = d1.to(device), d2.to(device), d3.to(device), pose.to(device)

            pred_pose = model(d1, d2, d3, pose)
            loss = loss_fn(pred_pose, pose)
            test_loss += loss.item()
            true_vals.append(pose.cpu().numpy())
            pred_vals.append(pred_pose.cpu().numpy())

    true_vals = np.vstack(true_vals)
    pred_vals = np.vstack(pred_vals)
    print(f"SEQUENTIAL Test Loss: {test_loss / len(test_loader):.4f}")

    # ---------------------------
    # 5) Evaluate on NON-SEQUENTIAL TEST
    # ---------------------------
    if nonseq_loader is not None:
        print("\nEvaluating on NON-SEQUENTIAL test set...")
        nonseq_loss = 0.0
        nonseq_true_vals, nonseq_pred_vals = [], []
        model.eval()
        with torch.no_grad():
            for d1, d2, d3, pose in nonseq_loader:
                # Move data to GPU
                d1, d2, d3, pose = d1.to(device), d2.to(device), d3.to(device), pose.to(device)

                pred_pose = model(d1, d2, d3, pose)
                loss = loss_fn(pred_pose, pose)
                nonseq_loss += loss.item()
                nonseq_true_vals.append(pose.cpu().numpy())
                nonseq_pred_vals.append(pred_pose.cpu().numpy())

        nonseq_true_vals = np.vstack(nonseq_true_vals)
        nonseq_pred_vals = np.vstack(nonseq_pred_vals)
        print(f"NON-SEQUENTIAL Test Loss: {nonseq_loss / len(nonseq_loader):.4f}")
    else:
        print("\nNo non-sequential data created (possibly not enough frames).")

    # Plot results
    plot_pose_results(true_vals, pred_vals, nonseq_true_vals, nonseq_pred_vals)

    # Plot specific component for non-sequential data
    plt.figure()
    plt.plot(nonseq_true_vals[:60, 3], label='True qx')
    plt.plot(nonseq_pred_vals[:60, 3], label='Predicted qx')
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
