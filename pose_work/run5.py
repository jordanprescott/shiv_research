import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

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

    # Inverse of quat1
    quat1_inv = np.array([-quat1[0], -quat1[1], -quat1[2], quat1[3]])
    # Relative rotation
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
# Preprocessing with Skips
# ---------------------------
def preprocess_tum_sequence(
        parent_sequence_path,
        frame_spacing,
        target_size=(128, 128),
        skip_probability=0.2,
        min_skip_frames=10,
        max_skip_frames=30
    ):
    """
    Creates triplets of frames spaced by 'frame_spacing', plus optional non-sequential data.
    Example of sequential: i, i+frame_spacing, i+2*frame_spacing
    For non-sequential, we skip extra frames for the 3rd image.
    Returns:
        depth_data: list of (depth1, depth2, depth3)
        pose_data: list of [tx, ty, tz, qx, qy, qz, qw] (relative)
        sequential_labels: list of 1 (sequential) or 0 (non-sequential)
    """
    for sequence_folder in os.listdir(parent_sequence_path):
        sequence_path = os.path.join(parent_sequence_path, sequence_folder)

        depth_dir = os.path.join(sequence_path, "da2")
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

        # We need up to i + 2*frame_spacing
        max_idx = len(depth_files) - 2 * frame_spacing
        for i in range(max_idx):
            depth1 = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth2 = cv2.imread(depth_files[i + frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth3 = cv2.imread(depth_files[i + 2 * frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)

            # Guard against zero max, just in case
            max1, max2, max3 = np.max(depth1), np.max(depth2), np.max(depth3)
            # Avoid division by zero
            max1 = max1 if max1 > 0 else 1.0
            max2 = max2 if max2 > 0 else 1.0
            max3 = max3 if max3 > 0 else 1.0

            depth1 = cv2.resize(depth1, target_size) / max1
            depth2 = cv2.resize(depth2, target_size) / max2
            depth3 = cv2.resize(depth3, target_size) / max3

            # Relative pose from frame i to frame i + 2*frame_spacing
            pose1 = groundtruth.iloc[i][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
            pose3 = groundtruth.iloc[i + 2 * frame_spacing][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
            relative_pose = compute_relative_pose(pose1, pose3)

            # This is the standard "sequential" sample
            depth_data.append((depth1, depth2, depth3))
            pose_data.append(relative_pose)
            sequential_labels.append(1)  # 1 => sequential

            # Possibly add a "non-sequential" sample
            if np.random.rand() < skip_probability:
                skip_frames = np.random.randint(min_skip_frames, max_skip_frames + 1)
                # Make sure we don't exceed the dataset length
                non_seq_index = i + 2 * frame_spacing + skip_frames
                if non_seq_index < len(depth_files):
                    # third image is now further away
                    non_seq_depth3 = cv2.imread(depth_files[non_seq_index], cv2.IMREAD_UNCHANGED).astype(np.float32)
                    m3 = np.max(non_seq_depth3)
                    m3 = m3 if m3 > 0 else 1.0
                    non_seq_depth3 = cv2.resize(non_seq_depth3, target_size) / m3

                    # We'll reuse the same relative_pose for demonstration, 
                    # but note that physically this might not match the frames.
                    depth_data.append((depth1, depth2, non_seq_depth3))
                    pose_data.append(relative_pose)
                    sequential_labels.append(0)  # 0 => non-sequential

    return depth_data, pose_data, sequential_labels

# ---------------------------
# Dataset Class
# ---------------------------
class DepthPoseDataset(Dataset):
    def __init__(self, depth_data, pose_data, seq_labels):
        self.depth_data = depth_data
        self.pose_data = pose_data
        self.seq_labels = seq_labels  # 1 => sequential, 0 => non-sequential

    def __len__(self):
        return len(self.depth_data)

    def __getitem__(self, idx):
        d1, d2, d3 = self.depth_data[idx]
        pose = self.pose_data[idx]
        seq_label = self.seq_labels[idx]

        # Convert to torch tensors
        d1 = torch.tensor(d1, dtype=torch.float32).unsqueeze(0)
        d2 = torch.tensor(d2, dtype=torch.float32).unsqueeze(0)
        d3 = torch.tensor(d3, dtype=torch.float32).unsqueeze(0)
        pose = torch.tensor(pose, dtype=torch.float32)
        seq_label = torch.tensor(seq_label, dtype=torch.float32).unsqueeze(0)

        return d1, d2, d3, pose, seq_label

# ---------------------------
# Model Definition
# ---------------------------
class DepthPoseEstimationNN(nn.Module):
    """
    Now returns both:
        pred_pose: [tx, ty, tz, qx, qy, qz, qw]
        pred_seq: Probability of the triplet being sequential (range ~ [0,1])
    """
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

        # Shared fully-connected part
        self.fc_pose1 = nn.Linear(128 * 8 * 8, 256)
        self.fc_pose2 = nn.Linear(256, 128)
        self.fc_pose_out = nn.Linear(128, 7)  # [tx, ty, tz, qx, qy, qz, qw]

        # Classification head for sequential vs. non-sequential
        self.fc_seq1 = nn.Linear(128 * 8 * 8, 128)
        self.fc_seq2 = nn.Linear(128, 1)  # single probability

    def forward(self, d1, d2, d3):
        # Concatenate along channel dimension => [B, 3, 128, 128]
        depth_input = torch.cat((d1, d2, d3), dim=1)
        conv_out = self.conv_block(depth_input)
        conv_out = conv_out.view(conv_out.size(0), -1)

        # Pose branch
        x_pose = F.relu(self.fc_pose1(conv_out))
        x_pose = F.relu(self.fc_pose2(x_pose))
        pred_pose = self.fc_pose_out(x_pose)

        # Seq classification branch
        x_seq = F.relu(self.fc_seq1(conv_out))
        pred_seq = torch.sigmoid(self.fc_seq2(x_seq))  # Probability in [0,1]

        return pred_pose, pred_seq


# ---------------------------
# Main Script with CUDA Support
# ---------------------------

if __name__ == "__main__":
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directory of this file
    data_dir = os.path.dirname(__file__)
    save_folder = os.path.join(data_dir, "saved_data_5")
    os.makedirs(save_folder, exist_ok=True)
    parent_sequence_path = os.path.join(data_dir, "tnt_data")

    # Numpy files for caching
    depth_data_path = os.path.join(save_folder, "depth_data_seq.npy")
    pose_data_path = os.path.join(save_folder, "pose_data_seq.npy")
    seq_labels_path = os.path.join(save_folder, "seq_labels_seq.npy")

    # ---------------------------
    # 1) Prepare data (with skipping)
    # ---------------------------
    if (
        os.path.exists(depth_data_path)
        and os.path.exists(pose_data_path)
        and os.path.exists(seq_labels_path)
    ):
        print("Loading cached data...")
        depth_data_seq = np.load(depth_data_path, allow_pickle=True)
        pose_data_seq = np.load(pose_data_path, allow_pickle=True)
        seq_labels_seq = np.load(seq_labels_path, allow_pickle=True)
    else:
        print("Preprocessing TUM sequence with skipping...")
        depth_data_seq, pose_data_seq, seq_labels_seq = preprocess_tum_sequence(
            parent_sequence_path,
            frame_spacing=10,
            skip_probability=0.20,
            min_skip_frames=30,
            max_skip_frames=50
        )
        np.save(depth_data_path, depth_data_seq)
        np.save(pose_data_path, pose_data_seq)
        np.save(seq_labels_path, seq_labels_seq)

    # Split into train/val/test
    train_depth, val_depth, train_pose, val_pose, train_labels, val_labels = train_test_split(
        depth_data_seq, pose_data_seq, seq_labels_seq, test_size=0.2, random_state=42
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

    # ---------------------------
    # 2) Initialize & Train Model
    # ---------------------------
    model = DepthPoseEstimationNN().to(device)  # Move model to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=0.25e-3)
    pose_loss_fn = nn.MSELoss()
    seq_loss_fn = nn.BCELoss()

    lamda = 0.7

    pretrained_model_path = os.path.join(save_folder, "depth_pose_model.pth")
    if os.path.exists(pretrained_model_path):
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    else:
        print("Training model...")
        num_epochs = 100
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for d1, d2, d3, pose, seq_label in train_loader:
                d1, d2, d3 = d1.to(device), d2.to(device), d3.to(device)
                pose = pose.to(device)
                seq_label = seq_label.to(device)

                optimizer.zero_grad()
                pred_pose, pred_seq = model(d1, d2, d3)
                loss_pose = pose_loss_fn(pred_pose, pose)
                loss_seq = seq_loss_fn(pred_seq, seq_label)
                loss = lamda * loss_pose + (1 - lamda) * loss_seq
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase: do NOT backpropagate or update model parameters
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for d1, d2, d3, pose, seq_label in val_loader:
                    d1, d2, d3 = d1.to(device), d2.to(device), d3.to(device)
                    pose = pose.to(device)
                    seq_label = seq_label.to(device)

                    pred_pose, pred_seq = model(d1, d2, d3)
                    loss_pose = pose_loss_fn(pred_pose, pose)
                    loss_seq = seq_loss_fn(pred_seq, seq_label)
                    loss = lamda * loss_pose + (1 - lamda) * loss_seq
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_folder, "depth_pose_model.pth"))
                print(f"Saved best model at epoch {epoch + 1} with validation loss: {val_loss:.8f}")

            
        # torch.save(model.state_dict(), os.path.join(save_folder, "depth_pose_model.pth"))

    # ---------------------------
    # 3) Evaluate on Test Set
    # ---------------------------
    print("\nEvaluating on test set...")
    test_loss = 0.0
    true_vals, pred_vals = [], []
    true_seq, pred_seq = [], []

    model.eval()
    with torch.no_grad():
        for d1, d2, d3, pose, seq_label in test_loader:
            d1, d2, d3 = d1.to(device), d2.to(device), d3.to(device)
            pose = pose.to(device)
            seq_label = seq_label.to(device)

            pred_pose, pred_seq_logits = model(d1, d2, d3)
            loss_pose = pose_loss_fn(pred_pose, pose)
            loss_seq = seq_loss_fn(pred_seq_logits, seq_label)
            loss = loss_pose + loss_seq
            test_loss += loss.item()

            true_vals.append(pose.cpu().numpy())
            pred_vals.append(pred_pose.cpu().numpy())

            true_seq.append(seq_label.cpu().numpy())
            pred_seq.append(pred_seq_logits.cpu().numpy())

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    # Convert to numpy arrays for analysis
    true_vals = np.vstack(true_vals)
    pred_vals = np.vstack(pred_vals)
    true_seq = np.vstack(true_seq)
    pred_seq = np.vstack(pred_seq)

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

    # 4b) Sequential vs Non-sequential Classification
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    import seaborn as sns

    def plot_sequential_classification(true_seq, pred_seq):
        # Binarize predictions with threshold=0.5
        pred_seq_binary = (pred_seq >= 0.5).astype(int)

        # Accuracy
        accuracy = accuracy_score(true_seq, pred_seq_binary)
        print(f"Sequential Classification Accuracy: {accuracy:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_seq, pred_seq_binary,
                                    target_names=["Non-Sequential (0)", "Sequential (1)"]))

        # Confusion matrix
        cm = confusion_matrix(true_seq, pred_seq_binary)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Seq (0)", "Seq (1)"],
                    yticklabels=["Non-Seq (0)", "Seq (1)"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    # Plot results
    plot_pose_predictions(true_vals, pred_vals, sample_range=60)
    plot_sequential_classification(true_seq, pred_seq)
