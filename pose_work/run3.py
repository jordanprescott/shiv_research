import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from model3 import DepthPoseEstimationNN

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

def preprocess_tum_sequence(sequence_path, frame_spacing, target_size=(128, 128), skip_probability=0.2, min_skip_frames=10, max_skip_frames=30):
    """
    Preprocess the TUM sequence to generate depth and pose data with optional skips for non-sequential labeling.

    Args:
        sequence_path (str): Path to the dataset.
        frame_spacing (int): Frame spacing between depth maps.
        target_size (tuple): Size to resize depth maps (default: (128, 128)).
        skip_probability (float): Probability of introducing a skip to non-sequential data (default: 0.3).
        max_skip_frames (int): Maximum number of frames to skip when creating non-sequential data (default: 10).

    Returns:
        depth_data: List of depth map triplets (sequential and non-sequential).
        pose_data: List of relative poses.
        sequential_labels: List of labels (1 for sequential, 0 for non-sequential).
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
    sequential_labels = []

    for i in range(len(depth_files) - 2 * frame_spacing):
        # Load three sequential depth maps
        depth1 = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth2 = cv2.imread(depth_files[i + frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth3 = cv2.imread(depth_files[i + 2 * frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)

        depth1 = cv2.resize(depth1, target_size) / np.max(depth1)  # Normalize depth
        depth2 = cv2.resize(depth2, target_size) / np.max(depth2)
        depth3 = cv2.resize(depth3, target_size) / np.max(depth3)

        # Sequential data
        pose1 = groundtruth.iloc[i][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
        pose2 = groundtruth.iloc[i + 2 * frame_spacing][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
        relative_pose = compute_relative_pose(pose1, pose2)

        # Add sequential example
        depth_data.append((depth1, depth2, depth3))
        pose_data.append(relative_pose)
        sequential_labels.append(1)  # Label as sequential

        # Add non-sequential data with skip
        if np.random.rand() < skip_probability:  # Decide whether to add a skip
            skip_frames = np.random.randint(min_skip_frames, max_skip_frames + 1)  # Random skip between 1 and max_skip_frames
            if i + 2 * frame_spacing + skip_frames < len(depth_files):
                non_seq_depth3 = cv2.imread(depth_files[i + 2 * frame_spacing + skip_frames], cv2.IMREAD_UNCHANGED).astype(np.float32)
                non_seq_depth3 = cv2.resize(non_seq_depth3, target_size)
                non_seq_depth3 = non_seq_depth3 / np.max(non_seq_depth3)  # Normalize depth
                depth_data.append((depth1, depth2, non_seq_depth3))
                pose_data.append(relative_pose)  # Pose remains valid but depth maps are non-sequential
                sequential_labels.append(0)  # Label as non-sequential

    return depth_data, pose_data, sequential_labels

# Dataset Class
class DepthPoseDataset(Dataset):
    def __init__(self, depth_data, pose_data, sequential_labels):
        self.depth_data = depth_data
        self.pose_data = pose_data
        self.sequential_labels = sequential_labels

    def __len__(self):
        return len(self.depth_data)

    def __getitem__(self, idx):
        d1, d2, d3 = self.depth_data[idx]
        pose = self.pose_data[idx]
        sequential_label = self.sequential_labels[idx]
        d1 = torch.tensor(d1, dtype=torch.float32).unsqueeze(0)
        d2 = torch.tensor(d2, dtype=torch.float32).unsqueeze(0)
        d3 = torch.tensor(d3, dtype=torch.float32).unsqueeze(0)
        pose = torch.tensor(pose, dtype=torch.float32)
        sequential_label = torch.tensor(sequential_label, dtype=torch.float32).unsqueeze(0)
        return d1, d2, d3, pose, sequential_label

# Ensure CUDA is enabled if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and Split Data
print("Loading data...")
data_dir = os.path.dirname(__file__)
save_folder = os.path.join(data_dir, "saved_data_3")
os.makedirs(save_folder, exist_ok=True)
depth_data_path = os.path.join(save_folder, "depth_data.npy")
pose_data_path = os.path.join(save_folder, "pose_data.npy")

sequential_labels_path = os.path.join(data_dir, "sequential_labels.npy")

if os.path.exists(depth_data_path) and os.path.exists(pose_data_path) and os.path.exists(sequential_labels_path):
    sequential_labels_path = os.path.join(save_folder, "sequential_labels_seq.npy")
else:
    sequence_path = os.path.join(data_dir, "rgbd_dataset_freiburg1_room")
    depth_data, pose_data, sequential_labels = preprocess_tum_sequence(sequence_path, frame_spacing=3)
    np.save(depth_data_path, depth_data)
    np.save(pose_data_path, pose_data)
    np.save(sequential_labels_path, sequential_labels)

train_depth, val_depth, train_pose, val_pose, train_labels, val_labels = train_test_split(
    depth_data, pose_data, sequential_labels, test_size=0.2, random_state=42
)
val_depth, test_depth, val_pose, test_pose, val_labels, test_labels = train_test_split(
    val_depth, val_pose, val_labels, test_size=0.5, random_state=42
)

train_dataset = DepthPoseDataset(train_depth, train_pose, train_labels)
val_dataset = DepthPoseDataset(val_depth, val_pose, val_labels)
test_dataset = DepthPoseDataset(test_depth, test_pose, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Training Loop
model = DepthPoseEstimationNN().to(device)  # Move model to GPU
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
pose_loss_fn = nn.MSELoss()
seq_loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss for sequential prediction

if os.path.exists("depth_pose_model.pth"):
    print("Loading pretrained model...")
    model.load_state_dict(torch.load("depth_pose_model.pth", map_location=device))
else:
    print("Training model...")
    for epoch in range(60):
        model.train()
        train_loss = 0.0
        for d1, d2, d3, pose, seq_label in train_loader:
            # Move data to GPU
            d1, d2, d3 = d1.to(device), d2.to(device), d3.to(device)
            pose = pose.to(device)
            seq_label = seq_label.to(device)

            optimizer.zero_grad()
            pred_pose, pred_seq = model(d1, d2, d3)
            pose_loss = pose_loss_fn(pred_pose, pose)
            seq_loss = seq_loss_fn(pred_seq, seq_label)
            loss = pose_loss + seq_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}")
    torch.save(model.state_dict(), os.path.join(save_folder, "depth_pose_model.pth"))


# Evaluation
test_loss = 0.0
true_vals, pred_vals = [], []
true_seq, pred_seq = [], []
model.eval()
with torch.no_grad():
    for d1, d2, d3, pose, seq_label in test_loader:
        # Move data to GPU
        d1, d2, d3 = d1.to(device), d2.to(device), d3.to(device)
        pose = pose.to(device)
        seq_label = seq_label.to(device)

        pred_pose, pred_seq_logits = model(d1, d2, d3)
        pose_loss = pose_loss_fn(pred_pose, pose)
        seq_loss = seq_loss_fn(pred_seq_logits, seq_label)
        loss = pose_loss + seq_loss
        test_loss += loss.item()
        true_vals.append(pose.cpu().numpy())  # Move data back to CPU for plotting
        pred_vals.append(pred_pose.cpu().numpy())
        true_seq.append(seq_label.cpu().numpy())
        pred_seq.append(pred_seq_logits.cpu().numpy())

true_vals = np.vstack(true_vals)
pred_vals = np.vstack(pred_vals)
true_seq = np.vstack(true_seq)
pred_seq = np.vstack(pred_seq)

print(f"Test Loss: {test_loss / len(test_loader)}")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

# Plot Pose Prediction Results
def plot_pose_predictions(true_vals, pred_vals, sample_range=60):
    """
    Plots the ground truth and predicted values for the pose prediction.

    Args:
        true_vals: Ground truth poses (numpy array of shape [N, 7]).
        pred_vals: Predicted poses (numpy array of shape [N, 7]).
        sample_range: Number of samples to plot.
    """
    titles = ["Translation X (tx)", "Translation Y (ty)", "Translation Z (tz)",
              "Rotation X (qx)", "Rotation Y (qy)", "Rotation Z (qz)", "Rotation W (qw)"]

    # Create subplots
    num_components = true_vals.shape[1]
    fig, axes = plt.subplots(num_components, 1, figsize=(10, 2 * num_components), sharex=True)

    for i in range(num_components):
        axes[i].plot(true_vals[:sample_range, i], label="Ground Truth", marker='o', markersize=3, linestyle='-')
        axes[i].plot(pred_vals[:sample_range, i], label="Predicted", marker='x', markersize=3, linestyle='--')
        axes[i].set_ylabel(titles[i])
        axes[i].legend()
        axes[i].grid(True)

    # Set shared x-label
    axes[-1].set_xlabel("Sample Index")

    # Add overall title
    fig.suptitle("Ground Truth vs Predicted Pose Components", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# Plot Sequential Classification Results
def plot_sequential_classification(true_seq, pred_seq):
    """
    Plots the confusion matrix for the sequential classification task.

    Args:
        true_seq: Ground truth sequential labels (numpy array of shape [N, 1]).
        pred_seq: Predicted sequential probabilities (numpy array of shape [N, 1]).
    """
    # Binarize predictions based on a threshold of 0.5
    pred_seq_binary = (pred_seq >= 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(true_seq, pred_seq_binary)
    print(f"Sequential Classification Accuracy: {accuracy:.4f}")

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(true_seq, pred_seq_binary, target_names=["Non-Sequential", "Sequential"]))

    # Confusion matrix
    cm = confusion_matrix(true_seq, pred_seq_binary)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Sequential", "Sequential"],
                yticklabels=["Non-Sequential", "Sequential"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# Call plotting functions
plot_pose_predictions(true_vals, pred_vals, sample_range=60)
plot_sequential_classification(true_seq, pred_seq)
