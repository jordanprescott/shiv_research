# gpu support version

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import cv2
from model import DepthPoseEstimationNN
import pandas as pd

# ---------------------------
# GPU/Device Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Preprocessing Functions
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
        relative_pose = pose3 - pose1

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
# Output Directory
# ---------------------------
output_dir = os.path.join(os.getcwd(), "output_results")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":
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
    # 2) Initialize & Train Model
    # ---------------------------
    model = DepthPoseEstimationNN().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = MSELoss()

    checkpoint_path = os.path.join(output_dir, "depth_pose_model.pth")

    if os.path.exists(checkpoint_path):
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Training model on SEQUENTIAL data...")
        num_epochs = 30
        train_losses = []

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for d1, d2, d3, pose in train_loader:
                d1, d2, d3, pose = d1.to(device), d2.to(device), d3.to(device), pose.to(device)
                optimizer.zero_grad()
                pred_pose = model(d1, d2, d3)
                loss = loss_fn(pred_pose, pose)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Save trained model
        torch.save(model.state_dict(), checkpoint_path)

        # Save training loss
        np.save(os.path.join(output_dir, "train_losses.npy"), train_losses)

    # ---------------------------
    # 3) Evaluate on SEQUENTIAL TEST
    # ---------------------------
    print("\nEvaluating on SEQUENTIAL test set...")
    test_loss = 0.0
    true_vals, pred_vals = [], []
    model.eval()

    with torch.no_grad():
        for d1, d2, d3, pose in test_loader:
            d1, d2, d3, pose = d1.to(device), d2.to(device), d3.to(device), pose.to(device)
            pred_pose = model(d1, d2, d3)
            loss = loss_fn(pred_pose, pose)
            test_loss += loss.item()
            true_vals.append(pose.cpu().numpy())
            pred_vals.append(pred_pose.cpu().numpy())

    true_vals = np.vstack(true_vals)
    pred_vals = np.vstack(pred_vals)
    avg_test_loss = test_loss / len(test_loader)

    # Save evaluation results
    np.save(os.path.join(output_dir, "true_vals_seq.npy"), true_vals)
    np.save(os.path.join(output_dir, "pred_vals_seq.npy"), pred_vals)
    with open(os.path.join(output_dir, "test_loss.txt"), "w") as f:
        f.write(f"SEQUENTIAL Test Loss: {avg_test_loss:.4f}\n")

    print(f"SEQUENTIAL Test Loss: {avg_test_loss:.4f}")

    # ---------------------------
    # Plot Results
    # ---------------------------
    plt.figure(figsize=(12, 18))

    pose_labels = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    for i, label in enumerate(pose_labels):
        plt.subplot(2, 4, i+1)  # Reordered subplot positions
        plt.plot(true_vals[:, i], label=f"True {label}", linestyle="-")
        plt.plot(pred_vals[:, i], label=f"Predicted {label}", linestyle="--")
        plt.legend()
        plt.title(f"True vs. Predicted {label} (Sequential Data)")
        plt.xlabel("Sample Index")
        plt.ylabel("Pose Component Value")
        plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, "pose_comparison_plot.png")
    plt.savefig(plot_path)
    print(f"Pose comparison plot saved to {plot_path}")
    plt.show()
