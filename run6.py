import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

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
    Each pose is [tx, ty, tz, qx, qy, qz, qw].
    """
    trans1, quat1 = pose1[:3], pose1[3:]
    trans2, quat2 = pose2[:3], pose2[3:]
    relative_translation = trans2 - trans1
    quat1_inv = np.array([-quat1[0], -quat1[1], -quat1[2], quat1[3]])
    relative_rotation = quaternion_multiply(quat1_inv, quat2)
    return np.hstack([relative_translation, relative_rotation])


# ---------------------------
# Preprocessing Function for Pairs (Depth + RGB)
# ---------------------------
def preprocess_tum_sequence_pairs(
        parent_sequence_path,
        frame_spacing,
        target_size=(128, 128),
        skip_probability=0.2,
        min_skip_frames=10,
        max_skip_frames=30
    ):
    """
    Processes multiple TUM sequences from a parent folder.
    
    The parent folder (e.g., "tnt_data") is expected to contain one or more subfolders,
    each corresponding to a TUM sequence (e.g., "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_room"). For each sequence, this function loads:
      - Two depth maps from the "depth" folder.
      - Two corresponding RGB images from the "rgb" folder.
      - The ground truth poses from "groundtruth.txt".
      - Computes the relative pose from frame i to frame i+frame_spacing.
      - Optionally, a non-sequential sample is generated by skipping extra frames.
    
    Returns:
        all_depth_data (list): List of tuples (depth1, depth2)
        all_image_data (list): List of tuples (img1, img2)
        all_pose_data (list): List of relative poses [tx, ty, tz, qx, qy, qz, qw]
        all_seq_labels (list): List of labels (1 for sequential, 0 for non-sequential)
    """
    import os
    import cv2
    import numpy as np
    import pandas as pd

    all_depth_data = []
    all_image_data = []
    all_pose_data = []
    all_seq_labels = []
    
    # Iterate over each subfolder in the parent directory
    for seq_folder in sorted(os.listdir(parent_sequence_path)):
        seq_path = os.path.join(parent_sequence_path, seq_folder)
        if not os.path.isdir(seq_path):
            continue
        
        # Define required paths for the current sequence folder
        depth_dir = os.path.join(seq_path, "depth")
        rgb_dir = os.path.join(seq_path, "rgb")
        groundtruth_file = os.path.join(seq_path, "groundtruth.txt")
        
        # Skip the sequence if any required file/folder is missing
        if not (os.path.exists(depth_dir) and os.path.exists(rgb_dir) and os.path.exists(groundtruth_file)):
            print(f"Skipping {seq_path} as required files are missing.")
            continue
        
        # Load ground truth for the current sequence
        gt = pd.read_csv(groundtruth_file, sep=" ", header=None, comment="#",
                         names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        
        depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(".png")])
        rgb_files   = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(".png")])
        
        max_idx = len(depth_files) - frame_spacing
        for i in range(max_idx):
            # Sequential sample: load frame i and frame i+frame_spacing
            depth1 = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth2 = cv2.imread(depth_files[i+frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)
            max1 = np.max(depth1) if np.max(depth1) > 0 else 1.0
            max2 = np.max(depth2) if np.max(depth2) > 0 else 1.0
            depth1 = cv2.resize(depth1, target_size) / max1
            depth2 = cv2.resize(depth2, target_size) / max2
            
            img1 = cv2.imread(rgb_files[i])
            img2 = cv2.imread(rgb_files[i+frame_spacing])
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img1, target_size).astype(np.float32) / 255.0
            img2 = cv2.resize(img2, target_size).astype(np.float32) / 255.0
            
            pose1 = gt.iloc[i][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
            pose2 = gt.iloc[i+frame_spacing][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
            relative_pose = compute_relative_pose(pose1, pose2)
            
            all_depth_data.append((depth1, depth2))
            all_image_data.append((img1, img2))
            all_pose_data.append(relative_pose)
            all_seq_labels.append(1)
            
            # Optionally add a non-sequential sample by skipping extra frames
            if np.random.rand() < skip_probability:
                skip_frames = np.random.randint(min_skip_frames, max_skip_frames + 1)
                non_seq_index = i + frame_spacing + skip_frames
                if non_seq_index < len(depth_files):
                    depth2_non = cv2.imread(depth_files[non_seq_index], cv2.IMREAD_UNCHANGED).astype(np.float32)
                    max2_non = np.max(depth2_non) if np.max(depth2_non) > 0 else 1.0
                    depth2_non = cv2.resize(depth2_non, target_size) / max2_non
                    
                    img2_non = cv2.imread(rgb_files[non_seq_index])
                    img2_non = cv2.cvtColor(img2_non, cv2.COLOR_BGR2RGB)
                    img2_non = cv2.resize(img2_non, target_size).astype(np.float32) / 255.0
                    
                    pose2_non = gt.iloc[non_seq_index][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].values
                    relative_pose_non = compute_relative_pose(pose1, pose2_non)
                    
                    all_depth_data.append((depth1, depth2_non))
                    all_image_data.append((img1, img2_non))
                    all_pose_data.append(relative_pose_non)
                    all_seq_labels.append(0)
                    
    return all_depth_data, all_image_data, all_pose_data, all_seq_labels


# ---------------------------
# Updated Dataset Class
# ---------------------------
class DepthPoseImageDataset(Dataset):
    def __init__(self, depth_data, image_data, pose_data, seq_labels):
        self.depth_data = depth_data
        self.image_data = image_data
        self.pose_data = pose_data
        self.seq_labels = seq_labels

    def __len__(self):
        return len(self.depth_data)

    def __getitem__(self, idx):
        d1, d2 = self.depth_data[idx]
        img1, img2 = self.image_data[idx]
        pose = self.pose_data[idx]
        seq_label = self.seq_labels[idx]

        # Convert depth maps to shape [1, H, W]
        d1 = torch.tensor(d1, dtype=torch.float32).unsqueeze(0)
        d2 = torch.tensor(d2, dtype=torch.float32).unsqueeze(0)
        # Convert RGB images from [H, W, 3] to [3, H, W]
        img1 = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1)
        img2 = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1)
        pose = torch.tensor(pose, dtype=torch.float32)
        seq_label = torch.tensor(seq_label, dtype=torch.float32).unsqueeze(0)

        return d1, d2, img1, img2, pose, seq_label

# ---------------------------
# Updated Model Definition
# ---------------------------
class DepthPoseEstimationNN(nn.Module):
    """
    The model takes as input two depth maps and two associated RGB images.
    Inputs are concatenated along the channel dimension (total channels = 1+1+3+3 = 8).
    Outputs:
      - pred_pose: 7D pose [tx, ty, tz, qx, qy, qz, qw]
      - pred_seq: predicted probability that the sample is sequential.
    """
    def __init__(self):
        super(DepthPoseEstimationNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
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
        
        # Pose estimation branch
        self.fc_pose1 = nn.Linear(128 * 8 * 8, 256)
        self.fc_pose2 = nn.Linear(256, 128)
        self.fc_pose_out = nn.Linear(128, 7)
        
        # Sequential classification branch
        self.fc_seq1 = nn.Linear(128 * 8 * 8, 128)
        self.fc_seq2 = nn.Linear(128, 1)

    def forward(self, d1, d2, img1, img2):
        # Concatenate inputs along channel dimension: shape [B, 8, H, W]
        x = torch.cat((d1, d2, img1, img2), dim=1)
        conv_out = self.conv_block(x)
        conv_out = conv_out.view(conv_out.size(0), -1)

        # Pose estimation branch
        x_pose = F.relu(self.fc_pose1(conv_out))
        x_pose = F.relu(self.fc_pose2(x_pose))
        pred_pose = self.fc_pose_out(x_pose)

        # Sequential classification branch
        x_seq = F.relu(self.fc_seq1(conv_out))
        pred_seq = torch.sigmoid(self.fc_seq2(x_seq))

        return pred_pose, pred_seq

# ---------------------------
# Plotting Function for Trajectories
# ---------------------------
from mpl_toolkits.mplot3d import Axes3D
def plot_test_path(true_poses, pred_poses, save_path_2d=None, save_path_3d=None):
    """
    Plots the trajectory of poses in both 2D (X vs. Z) and 3D.
    
    Args:
        true_poses (np.ndarray): Array of ground truth poses (N, 7), where [:, :3] are translations.
        pred_poses (np.ndarray): Array of predicted poses (N, 7).
        save_path_2d (str, optional): File path to save the 2D plot.
        save_path_3d (str, optional): File path to save the 3D plot.
    """
    true_trans = true_poses[:, :3]
    pred_trans = pred_poses[:, :3]
    
    # 2D Plot (X vs. Z)
    plt.figure(figsize=(10, 8))
    plt.plot(true_trans[:, 0], true_trans[:, 2], label='Ground Truth', marker='o', markersize=2, linestyle='-')
    plt.plot(pred_trans[:, 0], pred_trans[:, 2], label='Predicted', marker='x', markersize=2, linestyle='--')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Trajectory (X vs Z)')
    plt.legend()
    plt.grid(True)
    if save_path_2d is not None:
        plt.savefig(save_path_2d)
    plt.show()

    # 3D Plot (X, Y, Z)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(true_trans[:, 0], true_trans[:, 1], true_trans[:, 2], label='Ground Truth', marker='o', markersize=2, linestyle='-')
    ax.plot(pred_trans[:, 0], pred_trans[:, 1], pred_trans[:, 2], label='Predicted', marker='x', markersize=2, linestyle='--')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')
    ax.legend()
    if save_path_3d is not None:
        plt.savefig(save_path_3d)
    plt.show()

# ---------------------------
# Main Training, Testing, and Extra Path Tracking
# ---------------------------
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up paths for saving data and caching
    saved_data_dir = os.path.dirname(__file__)
    sequence_path = os.path.join(saved_data_dir, "tnt_data")
    save_folder = os.path.join(saved_data_dir, "saved_data_6")
    os.makedirs(save_folder, exist_ok=True)
    
    # Cache file paths for paired data (with possible skips)
    depth_data_path = os.path.join(save_folder, "depth_data_pairs.npy")
    image_data_path = os.path.join(save_folder, "image_data_pairs.npy")
    pose_data_path = os.path.join(save_folder, "pose_data_pairs.npy")
    seq_labels_path = os.path.join(save_folder, "seq_labels_pairs.npy")

    # Load or preprocess paired data (with skips)
    if (os.path.exists(depth_data_path) and os.path.exists(image_data_path) and
        os.path.exists(pose_data_path) and os.path.exists(seq_labels_path)):
        print("Loading cached paired data...")
        depth_data = np.load(depth_data_path, allow_pickle=True)
        image_data = np.load(image_data_path, allow_pickle=True)
        pose_data = np.load(pose_data_path, allow_pickle=True)
        seq_labels = np.load(seq_labels_path, allow_pickle=True)
    else:
        print("Preprocessing TUM sequence (pairs with skips)...")
        depth_data, image_data, pose_data, seq_labels = preprocess_tum_sequence_pairs(
            sequence_path,
            frame_spacing=3,
            target_size=(128, 128),
            skip_probability=0.2,
            min_skip_frames=10,
            max_skip_frames=30
        )
        np.save(depth_data_path, depth_data)
        np.save(image_data_path, image_data)
        np.save(pose_data_path, pose_data)
        np.save(seq_labels_path, seq_labels)

    # Split the data into train/validation/test sets
    train_depth, val_depth, train_pose, val_pose, train_labels, val_labels, train_images, val_images = train_test_split(
        depth_data, pose_data, seq_labels, image_data, test_size=0.2, random_state=42
    )
    val_depth, test_depth, val_pose, test_pose, val_labels, test_labels, val_images, test_images = train_test_split(
        val_depth, val_pose, val_labels, val_images, test_size=0.5, random_state=42
    )

    # Create dataset and dataloaders
    train_dataset = DepthPoseImageDataset(train_depth, train_images, train_pose, train_labels)
    val_dataset   = DepthPoseImageDataset(val_depth, val_images, val_pose, val_labels)
    test_dataset  = DepthPoseImageDataset(test_depth, test_images, test_pose, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model, optimizer, and loss functions
    model = DepthPoseEstimationNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pose_loss_fn = nn.MSELoss()
    seq_loss_fn = nn.BCELoss()
    lamda = 0.7  # Weighting factor between pose and sequence losses


    pretrained_model_path = os.path.join(save_folder, "depth_pose_model.pth")
    if os.path.exists(pretrained_model_path):
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    else:
        print("Training model from scratch...")

        num_epochs = 100
        best_val_loss = float('inf')

        # Training Loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for d1, d2, img1, img2, pose, seq_label in train_loader:
                d1, d2 = d1.to(device), d2.to(device)
                img1, img2 = img1.to(device), img2.to(device)
                pose, seq_label = pose.to(device), seq_label.to(device)

                optimizer.zero_grad()
                pred_pose, pred_seq = model(d1, d2, img1, img2)
                loss_pose = pose_loss_fn(pred_pose, pose)
                loss_seq = seq_loss_fn(pred_seq, seq_label)
                loss = lamda * loss_pose + (1 - lamda) * loss_seq
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * d1.size(0)

            epoch_loss = running_loss / len(train_dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for d1, d2, img1, img2, pose, seq_label in val_loader:
                    d1, d2 = d1.to(device), d2.to(device)
                    img1, img2 = img1.to(device), img2.to(device)
                    pose, seq_label = pose.to(device), seq_label.to(device)

                    pred_pose, pred_seq = model(d1, d2, img1, img2)
                    loss_pose = pose_loss_fn(pred_pose, pose)
                    loss_seq = seq_loss_fn(pred_seq, seq_label)
                    loss = lamda * loss_pose + (1 - lamda) * loss_seq
                    val_loss += loss.item() * d1.size(0)

            val_loss /= len(val_dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_folder, "best_depth_pose_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at epoch {epoch+1} with Validation Loss: {val_loss:.4f}")
            
        # Save final model
        # torch.save(model.state_dict(), pretrained_model_path)

    # Test evaluation on the main test set
    model.eval()
    test_loss = 0.0
    all_true_pose = []
    all_pred_pose = []
    all_true_seq = []
    all_pred_seq = []
    with torch.no_grad():
        for d1, d2, img1, img2, pose, seq_label in test_loader:
            d1, d2 = d1.to(device), d2.to(device)
            img1, img2 = img1.to(device), img2.to(device)
            pose, seq_label = pose.to(device), seq_label.to(device)
            pred_pose, pred_seq = model(d1, d2, img1, img2)
            loss_pose = pose_loss_fn(pred_pose, pose)
            loss_seq = seq_loss_fn(pred_seq, seq_label)
            loss = lamda * loss_pose + (1 - lamda) * loss_seq
            test_loss += loss.item() * d1.size(0)
            all_true_pose.append(pose.cpu().numpy())
            all_pred_pose.append(pred_pose.cpu().numpy())
            all_true_seq.append(seq_label.cpu().numpy())
            all_pred_seq.append(pred_seq.cpu().numpy())

    test_loss /= len(test_dataset)
    print(f"Test Loss: {test_loss:.4f}")

    all_true_pose = np.vstack(all_true_pose)
    all_pred_pose = np.vstack(all_pred_pose)
    all_true_seq = np.vstack(all_true_seq)
    all_pred_seq = np.vstack(all_pred_seq)

    # Plot Pose Predictions for Test Set
    pose_titles = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    fig, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    for i in range(7):
        axes[i].plot(all_true_pose[:100, i], label='Ground Truth', marker='o', linestyle='-')
        axes[i].plot(all_pred_pose[:100, i], label='Predicted', marker='x', linestyle='--')
        axes[i].set_title(pose_titles[i])
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    plt.show()

    # Plot Sequential Classification Results
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import seaborn as sns

    true_seq_bin = (all_true_seq >= 0.5).astype(int)
    pred_seq_bin = (all_pred_seq >= 0.5).astype(int)
    acc = accuracy_score(true_seq_bin, pred_seq_bin)
    print("Sequential Classification Accuracy:", acc)
    print("Classification Report:")
    print(classification_report(true_seq_bin, pred_seq_bin, target_names=["Non-Sequential", "Sequential"]))
    cm = confusion_matrix(true_seq_bin, pred_seq_bin)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Sequential", "Sequential"],
                yticklabels=["Non-Sequential", "Sequential"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ---------------------------
    # Extra Test Sequence: No Skips (for path tracking evaluation)
    # ---------------------------
    print("Processing extra test sequence (no skips) for path tracking evaluation...")
    extra_depth_data, extra_image_data, extra_pose_data, extra_seq_labels = preprocess_tum_sequence_pairs(
        sequence_path,
        frame_spacing=3,
        target_size=(128, 128),
        skip_probability=0.0  # No skips
    )
    # Only use the first 100 points for path tracking evaluation
    extra_depth_data = extra_depth_data[:100]
    extra_image_data = extra_image_data[:100]
    extra_pose_data  = extra_pose_data[:100]
    extra_seq_labels = extra_seq_labels[:100]

    extra_dataset = DepthPoseImageDataset(extra_depth_data, extra_image_data, extra_pose_data, extra_seq_labels)
    extra_loader = DataLoader(extra_dataset, batch_size=1, shuffle=False)

    extra_true_poses = []
    extra_pred_poses = []
    model.eval()
    with torch.no_grad():
        for d1, d2, img1, img2, pose, seq_label in extra_loader:
            d1, d2 = d1.to(device), d2.to(device)
            img1, img2 = img1.to(device), img2.to(device)
            pose = pose.to(device)
            pred_pose, _ = model(d1, d2, img1, img2)
            extra_true_poses.append(pose.cpu().numpy())
            extra_pred_poses.append(pred_pose.cpu().numpy())

    extra_true_poses = np.vstack(extra_true_poses)
    extra_pred_poses = np.vstack(extra_pred_poses)
    plot_test_path(extra_true_poses, extra_pred_poses)

