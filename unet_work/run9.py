import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import imageio

# ---------------------------
# 1) Quaternion Utilities
# ---------------------------
def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])

def compute_relative_pose(p1, p2):
    t1, q1 = p1[:3], p1[3:]
    t2, q2 = p2[:3], p2[3:]
    rel_t = t2 - t1
    inv_q1 = np.array([-q1[0], -q1[1], -q1[2], q1[3]])
    rel_q = quaternion_multiply(inv_q1, q2)
    # Normalize quaternion so that it is unit length
    norm_q = np.linalg.norm(rel_q) + 1e-12
    rel_q = rel_q / norm_q
    return np.hstack([rel_t, rel_q])

# ---------------------------
# 2) Unified Preprocessing
# ---------------------------
def preprocess_multitask_sequence(
    parent_sequence_path,
    frame_spacing=3,
    target_size=(128,128),
    skip_probability=0.2,
    min_skip_frames=10,
    max_skip_frames=30
):
    inputs, depth_gts, poses, seq_labels = [], [], [], []

    for seq in os.listdir(parent_sequence_path):
        seq_path = os.path.join(parent_sequence_path, seq)
        dp_folder = os.path.join(seq_path, "dp")
        depth_folder = os.path.join(seq_path, "depth")
        gt_file = os.path.join(seq_path, "groundtruth.txt")

        if not (os.path.isdir(dp_folder) and os.path.isdir(depth_folder) and os.path.isfile(gt_file)):
            continue

        gt_data = np.loadtxt(gt_file, comments='#')
        trans_quat = gt_data[:,1:8]  # [tx,ty,tz,qx,qy,qz,qw]

        dp_files    = sorted(f for f in os.listdir(dp_folder)   if f.endswith(".png"))
        depth_files = sorted(f for f in os.listdir(depth_folder) if f.endswith(".png"))
        max_idx = min(len(dp_files) - 2*frame_spacing, len(depth_files) - frame_spacing)

        for i in range(max_idx):
            # Load and preprocess dp inputs (raw depth)
            raw1 = cv2.imread(os.path.join(dp_folder,    dp_files[i]),            cv2.IMREAD_UNCHANGED).astype(np.float32)
            raw2 = cv2.imread(os.path.join(dp_folder,    dp_files[i+frame_spacing]), cv2.IMREAD_UNCHANGED).astype(np.float32)
            raw3 = cv2.imread(os.path.join(dp_folder,    dp_files[i+2*frame_spacing]), cv2.IMREAD_UNCHANGED).astype(np.float32)
            d1 = cv2.resize(raw1, target_size) / 1000.0
            d2 = cv2.resize(raw2, target_size) / 1000.0
            d3 = cv2.resize(raw3, target_size) / 1000.0

            # Ground-truth depth from /depth (use middle frame)
            gt_img = cv2.imread(os.path.join(depth_folder, depth_files[i+frame_spacing]),
                                cv2.IMREAD_UNCHANGED).astype(np.float32)
            gt_img = cv2.resize(gt_img, target_size) / 5000.0

            # Relative pose between frame i → i+2*spacing (normalized inside)
            rel_pose = compute_relative_pose(trans_quat[i], trans_quat[i+2*frame_spacing])

            # --- sequential sample ---
            inputs.append((d1, d2, d3))
            depth_gts.append(gt_img)
            poses.append(rel_pose)
            seq_labels.append(1.0)

            # --- optional non-sequential sample ---
            if np.random.rand() < skip_probability:
                skip = np.random.randint(min_skip_frames, max_skip_frames+1)
                idx3 = i + 2*frame_spacing + skip
                if idx3 < len(dp_files):
                    raw3ns = cv2.imread(os.path.join(dp_folder, dp_files[idx3]),
                                        cv2.IMREAD_UNCHANGED).astype(np.float32)
                    d3ns = cv2.resize(raw3ns, target_size) / 5000.0

                    inputs.append((d1, d2, d3ns))
                    depth_gts.append(gt_img)
                    # For a non-sequential pose, compute actual relative pose i→idx3 and normalize
                    rel_pose_ns = compute_relative_pose(trans_quat[i], trans_quat[idx3])
                    poses.append(rel_pose_ns)
                    seq_labels.append(0.0)

    return inputs, depth_gts, poses, seq_labels

# ---------------------------
# 3) Dataset
# ---------------------------
class MultiTaskDepthPoseDataset(Dataset):
    def __init__(self, inputs, depth_gts, poses, seq_labels):
        self.inputs    = inputs
        self.depth_gts = depth_gts
        self.poses     = poses
        self.seq_labels= seq_labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        d1, d2, d3 = self.inputs[idx]
        x = np.stack([d1, d2, d3], axis=0).astype(np.float32)       # [3,H,W]
        y_depth = self.depth_gts[idx][None,...].astype(np.float32) # [1,H,W]
        y_pose = self.poses[idx].astype(np.float32)                # [7]
        y_skip = np.array([self.seq_labels[idx]], dtype=np.float32)# [1]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y_depth),
            torch.from_numpy(y_pose),
            torch.from_numpy(y_skip),
        )

# ---------------------------
# 4) Rotation-Aware Loss
# ---------------------------
def quaternion_angle_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean geodesic/angle loss between two unit quaternions.
    q_pred, q_gt: each [B,4], both should already be normalized.
    Returns: scalar = mean(1 - |dot(q_pred, q_gt)|).
    """
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=1))    # [B]
    dot = torch.clamp(dot, -1.0, 1.0)
    return torch.mean(1.0 - dot)

# ---------------------------
# 5) Multitask Model Definition
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    A simple “Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU” block,
    with no residual connections anywhere.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskUNet(nn.Module):
    """
    A U-Net backbone with three “heads”:
      1) Depth reconstruction (full UNet decoder → 1×1 conv)
      2) Pose regression (conv stem + encoder features → MLP → 7D output)
      3) Skip (sequential vs non-seq) classification (bottleneck → global-avg → MLP → sigmoid)

    All convolutional blocks are plain Conv→BN→ReLU (no residuals).
    """

    def __init__(
        self,
        in_channels: int = 3,       # we feed [d1, d2, d3] as 3 channels
        depth_out_ch: int = 1,      # single-channel depth output
        features: list[int] = [64, 128, 256, 512]
    ):
        super().__init__()

        # ---------------------------
        # 1) Encoder (“down” path)
        # ---------------------------
        self.downs = nn.ModuleList()
        prev_ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(prev_ch, feat))
            prev_ch = feat

        # ---------------------------
        # 2) Bottleneck (no residual)
        # ---------------------------
        # Takes “features[-1] → features[-1] * 2”
        self.bottleneck = DoubleConv(prev_ch, prev_ch * 2)
        bottleneck_ch = prev_ch * 2  # e.g. if prev_ch=512, then bottleneck_ch=1024

        # ---------------------------
        # 3) Decoder (“up” path)
        # ---------------------------
        self.ups = nn.ModuleList()
        curr_ch = bottleneck_ch
        for feat in reversed(features):
            # (a) upsample from curr_ch → feat
            self.ups.append(nn.ConvTranspose2d(curr_ch, feat, kernel_size=2, stride=2))
            # (b) after concatenating skip+upsampled, channels = feat*2 → feat
            self.ups.append(DoubleConv(feat * 2, feat))
            curr_ch = feat

        # Final 1×1 → depth_out_ch
        self.final_depth = nn.Conv2d(features[0], depth_out_ch, kernel_size=1)

        # ---------------------------
        # 4) Pose regression stem (no residual)
        #    A small ConvNet “stem” that eventually Global-AvgPools to a 256-dim vector.
        # ---------------------------
        # Input to this stem: concatenated depth triplet [B,3,128,128]
        self.pose_conv1 = nn.Sequential(
            nn.Conv2d(3,   32, kernel_size=3, stride=2, padding=1),  # → [B,32,64,64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pose_conv2 = nn.Sequential(
            nn.Conv2d(32,  64, kernel_size=3, stride=2, padding=1),  # → [B,64,32,32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Dilated conv to increase receptive field (keeps 32×32)
        self.pose_dilated = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2, stride=1),  # → [B,128,32,32]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pose_conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # → [B,128,16,16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pose_conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # → [B,256,8,8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pose_gap   = nn.AdaptiveAvgPool2d((1, 1))                 # → [B,256,1,1]

        # We will concatenate this 256-dim vector with the UNet’s last encoder feature (pooled to a vector).
        # If features[-1] = 512, then MLP input = 256 + 512 = 768.
        concat_ch = 256 + features[-1]
        self.pose_mlp = nn.Sequential(
            nn.Linear(concat_ch, 256),    # 768→256
            nn.ReLU(inplace=True),
            nn.Linear(256,     128),
            nn.ReLU(inplace=True),
            nn.Linear(128,     64),
            nn.ReLU(inplace=True),
            nn.Linear(64,      7),        # 7D output (tx, ty, tz, qx, qy, qz, qw)
        )

        # ---------------------------
        # 5) Skip (sequential vs non-seq) head
        #    We take the Bottleneck feature (B, bottleneck_ch, 8,8), GAP → MLP → sigmoid
        # ---------------------------
        self.skip_gap = nn.AdaptiveAvgPool2d((1, 1))   # → [B,bottleneck_ch,1,1]
        self.skip_mlp = nn.Sequential(
            nn.Linear(bottleneck_ch, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, d1: torch.Tensor, d2: torch.Tensor, d3: torch.Tensor):
        B = d1.size(0)

        # ------------------------------------------------------
        # 1) Pose stem (small ConvNet → GAP → 256-D vector)
        # ------------------------------------------------------
        x_pose = torch.cat((d1, d2, d3), dim=1)       # [B,3,H,W]
        x_pose = self.pose_conv1(x_pose)              # [B,32,64,64]
        x_pose = self.pose_conv2(x_pose)              # [B,64,32,32]
        x_pose = self.pose_dilated(x_pose)            # [B,128,32,32]
        x_pose = self.pose_conv3(x_pose)              # [B,128,16,16]
        x_pose = self.pose_conv4(x_pose)              # [B,256, 8, 8]
        x_pose_flat = self.pose_gap(x_pose).view(B, 256)  # [B,256]

        # ------------------------------------------------------
        # 2) UNet encoder (stack depth triplet → “downs”)
        # ------------------------------------------------------
        x = torch.cat((d1, d2, d3), dim=1)  # [B,3,128,128]
        skip_feats = []
        for down in self.downs:
            x = down(x)
            skip_feats.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Now x is fed to the bottleneck

        # ------------------------------------------------------
        # 3) UNet bottleneck
        # ------------------------------------------------------
        btl = self.bottleneck(x)  # [B, bottleneck_ch, 8, 8]

        # ------------------------------------------------------
        # 4) Skip head (bottleneck → GAP → MLP → sigmoid)
        # ------------------------------------------------------
        s = self.skip_gap(btl).view(B, -1)  # [B, bottleneck_ch]
        pred_skip = self.skip_mlp(s)        # [B,1]

        # ------------------------------------------------------
        # 5) UNet decoder (upsample + concat skip_feats → “ups”)
        # ------------------------------------------------------
        x_dec = btl
        for idx in range(0, len(self.ups), 2):
            upsample_layer = self.ups[idx]
            double_layer   = self.ups[idx + 1]

            x_dec = upsample_layer(x_dec)
            skip_feat = skip_feats[-(idx//2 + 1)]
            dy, dx = skip_feat.size(2) - x_dec.size(2), skip_feat.size(3) - x_dec.size(3)
            x_dec = F.pad(x_dec, [dx//2, dx-dx//2, dy//2, dy-dy//2])
            x_dec = torch.cat([skip_feat, x_dec], dim=1)
            x_dec = double_layer(x_dec)

        pred_depth = self.final_depth(x_dec)  # [B,1,128,128]

        # ------------------------------------------------------
        # 6) Pose head (concatenate “pose stem” + last encoder feature)
        # ------------------------------------------------------
        # Last encoder feature = skip_feats[-1]  →  [B, features[-1], 16,16]
        last_enc = skip_feats[-1]
        pooled_enc = F.adaptive_avg_pool2d(last_enc, (1, 1)).view(B, -1)  # [B, features[-1]]

        pose_input = torch.cat([x_pose_flat, pooled_enc], dim=1)  # [B, 256 + features[-1]]
        raw_pose   = self.pose_mlp(pose_input)                    # [B,7]

        # Enforce unit quaternion (q_w ≥ 0)
        t_raw = raw_pose[:, :3]
        q_raw = raw_pose[:, 3:]
        q_norm = F.normalize(q_raw, p=2, dim=1).clamp(min=1e-6)   # [B,4]
        # If q_w < 0, flip sign so q_w ≥ 0
        sign_mask = (q_norm[:, 3] < 0).view(B, 1).float()         # [B,1]
        q_norm = q_norm * (1 - 2 * sign_mask)                     # flip entire quaternion if needed
        pred_pose = torch.cat([t_raw, q_norm], dim=1)             # [B,7]

        return pred_depth, pred_pose, pred_skip

# ---------------------------
# 6) Loss Functions
# ---------------------------
def indexed_masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, crop: float = 0.0):
    mask = target != 0.0
    if crop > 0:
        b, h, w = mask.shape[0], mask.shape[2], mask.shape[3]
        h1,h2 = int(h * crop), int(h * (1 - crop))
        w1,w2 = int(w * crop), int(w * (1 - crop))
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, h1:h2, w1:w2] = 1
        mask = mask & crop_mask

    p = pred[mask]
    t = target[mask]
    if p.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(p, t)

# ---------------------------
# 7) Visualization Utilities (unchanged)
# ---------------------------
def plot_pose_predictions(true_vals: np.ndarray, pred_vals: np.ndarray, sample_range: int = 60):
    titles = ["tx","ty","tz","qx","qy","qz","qw"]
    num = true_vals.shape[1]
    fig, axes = plt.subplots(num, 1, figsize=(8, 2 * num), sharex=True)
    for i in range(num):
        axes[i].plot(true_vals[:sample_range, i], 'o-', label="GT")
        axes[i].plot(pred_vals[:sample_range, i], 'x--', label="Pred")
        axes[i].set_ylabel(titles[i])
        axes[i].legend(); axes[i].grid(True)
    axes[-1].set_xlabel("Sample")
    fig.suptitle("Pose Predictions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_sequential_classification(true_seq: list, pred_seq: list):
    cm = confusion_matrix(true_seq, pred_seq)
    acc = accuracy_score(true_seq, pred_seq)
    print(f"Skip Classification Accuracy: {acc:.4f}")
    print(classification_report(true_seq, pred_seq, target_names=["Non-seq (0)", "Seq (1)"]))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non", "Seq"], yticklabels=["Non", "Seq"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def visualize_prediction(input_seq: torch.Tensor,
                         gt: torch.Tensor,
                         pred: torch.Tensor,
                         crop: float = 0.0,
                         show_disp: bool = True):
    def compute_mae(pred_np: np.ndarray, gt_np: np.ndarray, mask_value=0.0):
        valid = gt_np != mask_value
        if not np.any(valid):
            return np.nan
        return np.mean(np.abs(pred_np[valid] - gt_np[valid]))

    seq_np = input_seq.cpu().numpy()         # [3, H, W]
    gt_np  = gt.detach().cpu().numpy().squeeze()   # [H, W]
    pred_np= pred.detach().cpu().numpy().squeeze() # [H, W]

    if crop > 0:
        h, w = gt_np.shape
        h1, h2 = int(h * crop), int(h * (1 - crop))
        w1, w2 = int(w * crop), int(w * (1 - crop))
        seq_np = seq_np[:, h1:h2, w1:w2]
        gt_np  = gt_np[h1:h2, w1:w2]
        pred_np= pred_np[h1:h2, w1:w2]

    disparity_input = gt_np - seq_np[1]
    disparity_pred  = gt_np - pred_np

    mae_input = compute_mae(seq_np[1], gt_np)
    mae_pred  = compute_mae(pred_np, gt_np)
    mse_input = np.mean((seq_np[1] - gt_np) ** 2)
    mse_pred  = np.mean((pred_np - gt_np) ** 2)

    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    images = [seq_np[0], seq_np[1], seq_np[2], gt_np, pred_np]
    titles = [
        "Input t-1",
        f"Input t0\n(MAE={mae_input:.3f}, MSE={mse_input:.3f})",
        "Input t+1",
        "Ground Truth",
        f"Prediction\n(MAE={mae_pred:.3f}, MSE={mse_pred:.3f})"
    ]
    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, cmap='plasma')
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    if show_disp:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        for ax, disp, title in zip(axes2,
                                   [disparity_input, disparity_pred],
                                   ["Disparity: GT - t0", "Disparity: GT - Prediction"]):
            im = ax.imshow(disp, cmap='bwr')
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    print(f"MAE (GT - t0):   {mae_input:.4f}")
    print(f"MSE (GT - t0):   {mse_input:.4f}")
    print(f"MAE (GT - pred): {mae_pred:.4f}")
    print(f"MSE (GT - pred): {mse_pred:.4f}")

def create_sequence_gif(model, dataloader, device, gif_path='preds.gif', fps=4, crop=0.1):
    model.eval()
    frames = []
    all_vals = []

    # 1) First pass to collect global vmin/vmax
    with torch.no_grad():
        for x, y_d, _, _ in dataloader:
            x, y_d = x.to(device), y_d.to(device)
            d1, d2, d3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            pd, _, _ = model(d1, d2, d3)

            seqs = x.cpu().numpy()      # [B,3,H,W]
            gts  = y_d.cpu().numpy()    # [B,1,H,W]
            preds= pd.cpu().numpy()     # [B,1,H,W]

            for j in range(3):
                all_vals.append((seqs[:, j] / 1000.0).flatten())
            all_vals.append(gts[:, 0].flatten())
            all_vals.append(preds[:, 0].flatten())

    all_vals = np.concatenate(all_vals)
    vmin, vmax = all_vals.min(), all_vals.max()

    # 2) Second pass to build frames
    with torch.no_grad():
        for x, y_d, _, _ in dataloader:
            x, y_d = x.to(device), y_d.to(device)
            d1, d2, d3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            pd, _, _ = model(d1, d2, d3)

            seqs = x.cpu().numpy() / 1000.0  # [B,3,H,W] in meters
            gts  = y_d.cpu().numpy()[0, 0]   # [H,W]
            preds= pd.cpu().numpy()[0, 0]    # [H,W]

            fig, axs = plt.subplots(1, 5, figsize=(15, 3))
            for j in range(3):
                axs[j].imshow(seqs[0, j], cmap='magma', vmin=vmin, vmax=vmax)
                axs[j].set_title(f"t{['-1','0','+1'][j]}")
                axs[j].axis('off')
            axs[3].imshow(gts, cmap='magma', vmin=vmin, vmax=vmax)
            axs[3].set_title("GT"); axs[3].axis('off')
            axs[4].imshow(preds, cmap='magma', vmin=vmin, vmax=vmax)
            axs[4].set_title("Pred"); axs[4].axis('off')

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            h, w = fig.canvas.get_width_height()
            frames.append(image.reshape((h, w, 3)))
            plt.close(fig)

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"Saved GIF to {gif_path}")

def evaluate_sequence(input_seqs: np.ndarray,
                      gt_seqs: np.ndarray,
                      pred_seqs: np.ndarray,
                      crop: float = 0.0,
                      visualize_indices: list = None,
                      show_disp: bool = True):
    totals = {"mae_pred":0.0, "mse_pred":0.0, "mae_in":0.0, "mse_in":0.0}
    count = 0

    for i in range(input_seqs.shape[0]):
        seq = input_seqs[i]            # [3,H,W]
        gt  = gt_seqs[i, 0]            # [H,W]
        pd  = pred_seqs[i, 0]          # [H,W]
        input_t0 = seq[1]              # [H,W]

        if crop > 0:
            h, w = gt.shape
            h1, h2 = int(h * crop), int(h * (1 - crop))
            w1, w2 = int(w * crop), int(w * (1 - crop))
            seq = seq[:, h1:h2, w1:w2]
            gt  = gt[h1:h2, w1:w2]
            pd  = pd[h1:h2, w1:w2]
            input_t0 = input_t0[h1:h2, w1:w2]

        valid_mask = gt != 0.0
        if not np.any(valid_mask):
            continue

        err_in  = input_t0[valid_mask] - gt[valid_mask]
        mae_in = np.mean(np.abs(err_in))
        mse_in = np.mean(err_in ** 2)

        err_pd  = pd[valid_mask] - gt[valid_mask]
        mae_pd = np.mean(np.abs(err_pd))
        mse_pd = np.mean(err_pd ** 2)

        totals["mae_pred"] += mae_pd
        totals["mse_pred"] += mse_pd
        totals["mae_in"]   += mae_in
        totals["mse_in"]   += mse_in
        count += 1

        if visualize_indices and i in visualize_indices:
            visualize_prediction(torch.tensor(seq),
                                 torch.tensor(gt)[None, ...],
                                 torch.tensor(pd)[None, ...],
                                 crop=0.0,
                                 show_disp=show_disp)

    for k in totals:
        totals[k] /= count
    print(f"\n[Sequence Eval] MAE pred: {totals['mae_pred']:.4f}, MSE pred: {totals['mse_pred']:.4f}")
    print(f"[Sequence Eval] MAE in:   {totals['mae_in']:.4f}, MSE in:   {totals['mse_in']:.4f}")
    return totals

# ---------------------------
# 7) Training & Evaluation
# ---------------------------
if __name__ == "__main__":
    BASE = os.path.dirname(__file__)
    DATA = os.path.join(BASE, "tnt_data")
    CACHE= os.path.join(BASE, "saved_data_9")
    os.makedirs(CACHE, exist_ok=True)

    FS, TS = 3, (128,128)
    BATCH = 16
    LR = 1e-3
    EPOCHS = 100
    # Loss weights: (depth, pose, skip)
    # Rebalancing: give pose a higher weight so it isn't drowned by pixel‐MSE
    Wd, Wp, Ws = 1.0, 10.0, 0.75

    # 7.1) Preprocess or load cached arrays
    paths = [os.path.join(CACHE,n) for n in ("inp.npy","dgt.npy","pose.npy","skip.npy")]
    if all(os.path.exists(p) for p in paths):
        inputs    = np.load(paths[0], allow_pickle=True)
        depth_gts = np.load(paths[1], allow_pickle=True)
        poses     = np.load(paths[2], allow_pickle=True)
        seq_labels= np.load(paths[3], allow_pickle=True)
    else:
        print("Preprocessing…")
        inputs, depth_gts, poses, seq_labels = preprocess_multitask_sequence(DATA, frame_spacing=FS, target_size=TS)
        np.save(paths[0], inputs,    allow_pickle=True)
        np.save(paths[1], depth_gts, allow_pickle=True)
        np.save(paths[2], poses,     allow_pickle=True)
        np.save(paths[3], seq_labels,allow_pickle=True)

    # 7.2) Splits & loaders
    ds = MultiTaskDepthPoseDataset(inputs, depth_gts, poses, seq_labels)
    n = len(ds)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test],
                                             generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)
    viz_loader   = DataLoader(test_ds,  batch_size=1,   shuffle=False)

    # 7.3) Model, optimizer, loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskUNet(in_channels=3, depth_out_ch=1).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)

    skip_fn = nn.BCELoss()

    # ---------------------------
    # 7.4) Training loop (with Early Stopping)
    # ---------------------------
    if os.path.exists(os.path.join(CACHE, "model.pth")):
        print("Loading cached model…")
        model.load_state_dict(torch.load(os.path.join(CACHE, "model.pth")))
    else:
        best_val_loss = float('inf')
        no_improve_cnt = 0
        patience = 15   # stop if no improvement for this many epochs

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_train_loss = 0.0

            for x, y_d, y_p, y_s in train_loader:
                x, y_d, y_p, y_s = x.to(device), y_d.to(device), y_p.to(device), y_s.to(device)
                d1 = x[:, 0:1]   # [B,1,H,W]
                d2 = x[:, 1:2]
                d3 = x[:, 2:3]

                opt.zero_grad()
                pd, pp, ps = model(d1, d2, d3)

                # 1) Depth loss
                loss_d = indexed_masked_mse_loss(pd, y_d)

                # 2) Pose loss: translation MSE + quaternion angle loss
                t_pred, q_pred = pp[:, :3], pp[:, 3:]
                t_gt,   q_gt   = y_p[:, :3],  y_p[:, 3:]
                loss_trans = F.mse_loss(t_pred, t_gt)
                loss_rot   = quaternion_angle_loss(q_pred, q_gt)
                lambda_rot = 0.25
                loss_p = loss_trans + lambda_rot * loss_rot

                # 3) Skip loss
                loss_s = skip_fn(ps, y_s)

                # 4) Weighted total loss
                loss = Wd * loss_d + Wp * loss_p + Ws * loss_s
                loss.backward()
                opt.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation pass
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for x, y_d, y_p, y_s in val_loader:
                    x, y_d, y_p, y_s = x.to(device), y_d.to(device), y_p.to(device), y_s.to(device)
                    d1 = x[:, 0:1]
                    d2 = x[:, 1:2]
                    d3 = x[:, 2:3]

                    pd, pp, ps = model(d1, d2, d3)

                    loss_d = indexed_masked_mse_loss(pd, y_d)
                    t_pred, q_pred = pp[:, :3], pp[:, 3:]
                    t_gt,   q_gt   = y_p[:, :3],  y_p[:, 3:]
                    loss_trans = F.mse_loss(t_pred, t_gt)
                    loss_rot   = quaternion_angle_loss(q_pred, q_gt)
                    loss_p = loss_trans + lambda_rot * loss_rot
                    loss_s = skip_fn(ps, y_s)

                    total_val_loss += (Wd * loss_d + Wp * loss_p + Ws * loss_s).item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch:02d}  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_cnt = 0
                torch.save(model.state_dict(), os.path.join(CACHE, "model.pth"))
                print(f"  ↳ New best model (Val Loss: {best_val_loss:.4f}) saved.")
            else:
                no_improve_cnt += 1
                print(f"  ↳ No improvement for {no_improve_cnt}/{patience} epochs.")

            if no_improve_cnt >= patience:
                print(f"Early stopping triggered (no improvement in {patience} epochs).")
                break

        if best_val_loss == float('inf'):
            torch.save(model.state_dict(), os.path.join(CACHE, "model.pth"))
            print("Saved final model checkpoint (no validation improvement observed).")

    # 7.5) Testing & collecting predictions
    model.eval()
    true_poses, pred_poses = [], []
    true_skips, pred_skips = [], []
    all_seqs, all_gts, all_preds = [], [], []

    with torch.no_grad():
        for x, y_d, y_p, y_s in test_loader:
            x, y_d, y_p, y_s = x.to(device), y_d.to(device), y_p.to(device), y_s.to(device)
            d1 = x[:, 0:1]
            d2 = x[:, 1:2]
            d3 = x[:, 2:3]

            pd, pp, ps = model(d1, d2, d3)

            # Collect poses
            true_poses.append(y_p.cpu().numpy())
            pred_poses.append(pp.cpu().numpy())

            # Collect skip flags
            ts = y_s.cpu().numpy().ravel()
            pb = (ps.cpu().numpy().ravel() >= 0.5).astype(int)
            true_skips.extend(ts.tolist())
            pred_skips.extend(pb.tolist())

            # Collect for depth evaluation
            all_seqs.append(x.cpu().numpy()[0])     # [3,H,W]
            all_gts.append(y_d.cpu().numpy()[0])    # [1,H,W]
            all_preds.append(pd.cpu().numpy()[0])   # [1,H,W]

    true_poses = np.vstack(true_poses)
    pred_poses = np.vstack(pred_poses)
    all_seqs   = np.stack(all_seqs)   # [N,3,H,W]
    all_gts    = np.stack(all_gts)    # [N,1,H,W]
    all_preds  = np.stack(all_preds)  # [N,1,H,W]

    # 7.6) Print test metrics
    depth_mses = [
        indexed_masked_mse_loss(torch.tensor(pd[None], dtype=torch.float32),
                                torch.tensor(gt[None], dtype=torch.float32)).item()
        for pd, gt in zip(all_preds, all_gts)
    ]
    # For rotation/translation error, compute MSE on translation and angle on quaternion
    trans_errors = np.linalg.norm(pred_poses[:, :3] - true_poses[:, :3], axis=1)
    q_pred = pred_poses[:, 3:]
    q_gt   = true_poses[:, 3:]
    cos_half = np.abs(np.sum(q_pred * q_gt, axis=1))
    cos_half = np.clip(cos_half, -1.0, 1.0)
    rot_angles = 2 * np.arccos(cos_half) * (180.0 / np.pi)  # degrees

    print("\n=== Test Results ===")
    print(f"Depth MSE (averaged over samples): {np.mean(depth_mses):.4f}")
    print(f"Translation Error (mean L2 in meters): {np.mean(trans_errors):.4f}")
    print(f"Rotation Error (mean deg):             {np.mean(rot_angles):.4f}")
    print(f"Skip Classification Accuracy:         {accuracy_score(true_skips, pred_skips):.4f}")

    # 7.7) Visualizations
    plot_pose_predictions(true_poses, pred_poses, sample_range=30)
    plot_sequential_classification(true_skips, pred_skips)

    # 7.8) Depth visualization examples (optional)
    # for idx in [0, 1, 2]:
    #     visualize_prediction(
    #         torch.tensor(all_seqs[idx]),
    #         torch.tensor(all_gts[idx:idx+1]),
    #         torch.tensor(all_preds[idx:idx+1]),
    #         crop=0.1,
    #         show_disp=True
    #     )

    # 7.9) Sequence-level evaluation
    evaluate_sequence(all_seqs, all_gts, all_preds, crop=0.1, visualize_indices=[0, 1], show_disp=False)
