import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
from torch.utils.data import Subset
from partialconv.models.partialconv2d import PartialConv2d


# ---------------------------
# 1. Preprocessing Functions
# ---------------------------
def preprocess_depth_sequence(parent_sequence_path, frame_spacing=3, target_size=(128,128)):
    """
    For each sequence:
      - "dp" holds raw depth frames
      - "depth" holds ground‑truth depth maps
    Builds triplets (i, i+frame_spacing, i+2*frame_spacing) from dp,
    uses the middle frame in "depth" as GT, resizes, normalizes by 5000.0,
    and returns two lists: inputs and targets.
    """
    all_inputs, all_gts = [], []
    for seq in os.listdir(parent_sequence_path):
        if seq == "rgbd_scenenet":
            print(f"Skipping {seq}...")
            continue
        print(f"Processing {seq}...")
        dp_folder    = os.path.join(parent_sequence_path, seq, "dp")
        gt_folder    = os.path.join(parent_sequence_path, seq, "depth")
        if not (os.path.isdir(dp_folder) and os.path.isdir(gt_folder)):
            continue

        dp_files  = sorted(f for f in os.listdir(dp_folder)  if f.endswith(".png"))
        gt_files  = sorted(f for f in os.listdir(gt_folder) if f.endswith(".png"))
        max_idx = min(len(dp_files) - 2*frame_spacing, len(gt_files) - frame_spacing)

        for i in range(max_idx):
            # Load three input depth maps
            imgs = []
            for offset in (0, frame_spacing, 2*frame_spacing):
                path = os.path.join(dp_folder, dp_files[i + offset])
                img  = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # OpenCV imread :contentReference[oaicite:2]{index=2}
                img  = cv2.resize(img, target_size)
                imgs.append(img)
            all_inputs.append(tuple(imgs))

            # Load GT for middle frame
            gt_path = os.path.join(gt_folder, gt_files[i + frame_spacing])
            gt_img   = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            gt_img   = cv2.resize(gt_img, target_size) / 5000.0  # Normalize by 5000.0 to convert ground truth to meters
            all_gts.append(gt_img)
    return all_inputs, all_gts

def preprocess_depth_sequence_scenenet(parent_sequence_path, frame_spacing=3, target_size=(128,128)):
    """
    For each sequence:
      - "dp" holds raw depth frames
      - "depth" holds ground‑truth depth maps
    Builds triplets (i, i+frame_spacing, i+2*frame_spacing) from dp,
    uses the middle frame in "depth" as GT, resizes, normalizes by 5000.0,
    and returns two lists: inputs and targets.
    """
    all_inputs, all_gts = [], []
    # seq = os.listdir(parent_sequence_path)
    # seq = parent_sequence_path

    for seq in os.listdir(parent_sequence_path):
        if seq not in ["0", "1", "2"]:
            continue
        dp_folder    = os.path.join(parent_sequence_path, seq, "dp")
        gt_folder    = os.path.join(parent_sequence_path, seq, "depth")

        dp_files = sorted(
            (f for f in os.listdir(dp_folder) if f.endswith(".jpg")),
            key=lambda x: int(os.path.splitext(x)[0])
        )

        gt_files = sorted(
            (f for f in os.listdir(gt_folder) if f.endswith(".png")),
            key=lambda x: int(os.path.splitext(x)[0])
        )
        max_idx = min(len(dp_files) - 2*frame_spacing, len(gt_files) - frame_spacing)

        for i in range(max_idx):
            # Load three input depth maps
            imgs = []
            for offset in (0, frame_spacing, 2*frame_spacing):
                path = os.path.join(dp_folder, dp_files[i + offset])
                img  = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # OpenCV imread :contentReference[oaicite:2]{index=2}
                img  = cv2.resize(img, target_size)
                imgs.append(img)
            all_inputs.append(tuple(imgs))

            # Load GT for middle frame
            gt_path = os.path.join(gt_folder, gt_files[i + frame_spacing])
            gt_img   = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            gt_img   = cv2.resize(gt_img, target_size) / 1000.0  # Normalize by 1000.0 to convert ground truth to meters
            all_gts.append(gt_img)
    return all_inputs, all_gts

# ---------------------------
# 2. Dataset Class
# ---------------------------
class CachedDepthDataset(Dataset):
    """
    Stores preprocessed depth triplets (d1, d2, d3) and corresponding GT.
    Returns:
      input_tensor: shape (3, H, W)
      gt_tensor:    shape (1, H, W)
    """
    def __init__(self, inputs, gts):
        self.inputs = inputs
        self.gts    = gts

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        d1, d2, d3 = self.inputs[idx]
        inp = np.stack([d1, d2, d3], axis=0).astype(np.float32)
        gt  = self.gts[idx].astype(np.float32)[None, ...]
        return torch.from_numpy(inp), torch.from_numpy(gt)

# ---------------------------
# 3. Model Definition
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet3In(nn.Module):
    """
    U‑Net that takes 3 input channels (e.g., three sequential depth maps)
    and produces 1 output channel (the middle depth map prediction).
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list[int] = [64, 128, 256, 512]
    ):
        super().__init__()

        # Encoder path
        self.downs = nn.ModuleList()
        prev_ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(prev_ch, feat))
            prev_ch = feat

        # Bottleneck
        self.bottleneck = DoubleConv(prev_ch, prev_ch * 2)
        prev_ch = prev_ch * 2  # now 512*2 = 1024 for features=[64,128,256,512]

        # Decoder path: for each feature size in reverse, upsample then double‑conv
        self.ups = nn.ModuleList()
        for feat in reversed(features):
            # 1) transpose conv to upsample from prev_ch → feat channels
            self.ups.append(nn.ConvTranspose2d(prev_ch, feat, kernel_size=2, stride=2))
            # 2) double conv on concatenated skip (feat) + upsampled (feat) = feat*2 → feat
            self.ups.append(DoubleConv(feat * 2, feat))
            prev_ch = feat  # update for next stage

        # Final 1×1 conv to map to the desired output channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_feats: list[torch.Tensor] = []

        # Encoder: store features then downsample
        for down in self.downs:
            x = down(x)
            enc_feats.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: upsample, pad if needed, concat skip, then double‑conv
        for idx in range(0, len(self.ups), 2):
            upsample = self.ups[idx]
            double_conv = self.ups[idx + 1]

            x = upsample(x)
            skip = enc_feats[-(idx // 2 + 1)]

            # Pad to handle odd sizes
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

            # Concatenate along channel dimension
            x = torch.cat([skip, x], dim=1)
            x = double_conv(x)

        # Final conv + ReLU clamp
        x = self.final_conv(x)
        # x = F.relu(x)  # ⟵ ensure outputs are ≥ 0
        return x

# ------ partialconv ------
class PartialDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            # First partial convolution + normalization + activation
            PartialConv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Second partial convolution
            PartialConv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet3InPartial(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        # Encoder
        self.downs = nn.ModuleList()
        prev_ch = in_channels
        for feat in features:
            self.downs.append(PartialDoubleConv(prev_ch, feat))
            prev_ch = feat

        # Bottleneck
        self.bottleneck = PartialDoubleConv(prev_ch, prev_ch * 2)
        prev_ch *= 2

        # Decoder
        self.ups = nn.ModuleList()
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(prev_ch, feat, kernel_size=2, stride=2))
            self.ups.append(PartialDoubleConv(feat * 2, feat))
            prev_ch = feat

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[-(i//2 + 1)]
            # Pad if needed (should rarely artifact thanks to partial convs)
            dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
            x = nn.functional.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x)

        return self.final(x)

# ---------------------------
# 4. Loss Function
# ---------------------------
def indexed_masked_mse_loss(pred, target):
    mask = target != 0.0
    pred_vals   = pred[mask]
    target_vals = target[mask]
    if pred_vals.numel() == 0:
        return torch.tensor(0., device=pred.device, requires_grad=True)
    return F.mse_loss(pred_vals, target_vals)

# ---------------------------
# 5. Data Visualization
# ---------------------------
def visualize_prediction(input_seq: torch.Tensor,
                                  gt: torch.Tensor,
                                  pred: torch.Tensor):
    """
    Displays three input frames, the ground truth, and the prediction
    in two rows:
      - Row 1: shared scale (vmin=0, vmax=1)
      - Row 2: individual auto scales
    """
    # Convert to NumPy
    seq_np  = input_seq.cpu().numpy()
    gt_np   = gt.detach().cpu().numpy().squeeze(0)
    pred_np = pred.detach().cpu().numpy().squeeze(0)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))  # 2 rows, 5 cols :contentReference[oaicite:0]{index=0}

    # --- Row 1: fixed scale ---
    for i in range(3):
        im = axes[0, i].imshow(seq_np[i], cmap='plasma', vmin=0, vmax=1)
        axes[0, i].set_title(f'Input t{["-1","0","+1"][i]} (fixed)')
        axes[0, i].axis('off')
        fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)  # per‑panel colorbar :contentReference[oaicite:1]{index=1}

    im_gt = axes[0, 3].imshow(gt_np,   cmap='plasma', vmin=0, vmax=1)
    axes[0, 3].set_title('Ground Truth (fixed)')
    axes[0, 3].axis('off')
    fig.colorbar(im_gt, ax=axes[0, 3], fraction=0.046, pad=0.04)

    im_pr = axes[0, 4].imshow(pred_np, cmap='plasma', vmin=0, vmax=1)
    axes[0, 4].set_title('Prediction (fixed)')
    axes[0, 4].axis('off')
    fig.colorbar(im_pr, ax=axes[0, 4], fraction=0.046, pad=0.04)

    # --- Row 2: auto scale per image ---
    items = list(seq_np) + [gt_np, pred_np]
    titles = [f'Input t{t} (auto)' for t in ["-1","0","+1"]] + ['Ground Truth (auto)', 'Prediction (auto)']
    for j, (img, title) in enumerate(zip(items, titles)):
        ax = axes[1, j]
        im = ax.imshow(img, cmap='plasma')  # no vmin/vmax → auto-scale :contentReference[oaicite:2]{index=2}
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    print(f"Shapes of all images: {[img.shape for img in items]}")

    plt.tight_layout()
    plt.show()

def create_sequence_gif(model, dataloader, device, gif_path='predictions.gif', fps=2):
    """
    Runs the model on each batch of depth‑sequence samples and
    saves an animated GIF of the inputs, ground truth, and predictions.

    Args:
      model       : your UNet3In instance (in eval mode)
      dataloader  : torch DataLoader yielding (seqs, gts)
      device      : 'cuda' or 'cpu'
      gif_path    : output file path for the GIF
      fps         : frames per second for the GIF
    """
    model.eval()
    frames = []

    with torch.no_grad():
        for seqs, gts in dataloader:
            seqs, gts = seqs.to(device), gts.to(device)            # send to device
            preds = model(seqs)                                    # get predictions

            # Loop over batch
            for i in range(seqs.size(0)):
                fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                # Plot inputs t-1, t0, t+1
                for j in range(3):
                    axes[j].imshow(seqs[i, j].cpu().numpy(), cmap='magma')
                    axes[j].set_title(f'Input t{["-1","0","+1"][j]}')
                    axes[j].axis('off')
                # Ground truth
                axes[3].imshow(gts[i,0].cpu().numpy(), cmap='magma')
                axes[3].set_title('Ground Truth')
                axes[3].axis('off')
                # Prediction
                axes[4].imshow(preds[i,0].cpu().numpy(), cmap='magma')
                axes[4].set_title('Prediction')
                axes[4].axis('off')

                # Draw the canvas, extract as RGBA buffer
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                w, h = fig.canvas.get_width_height()
                image = image.reshape((h, w, 3))

                frames.append(image)                                   # collect frame
                plt.close(fig)

    # Write out GIF
    imageio.mimsave(gif_path, frames, fps=fps)                       # write GIF :contentReference[oaicite:1]{index=1}
    print(f"Saved GIF to {gif_path}")

# ---------------------------
# 6. Main: Data & Training
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "saved_data_8")
    os.makedirs(cache_dir, exist_ok=True)

    inp_cache = os.path.join(cache_dir, "depth_input_seq.npy")
    gt_cache  = os.path.join(cache_dir, "depth_gt_seq.npy")

    if os.path.exists(inp_cache) and os.path.exists(gt_cache):
        print("Loading cached data...")
        inputs = np.load(inp_cache, allow_pickle=True)
        gts    = np.load(gt_cache, allow_pickle=True)
    else:
        print("Preprocessing data...")
        inputs, gts = preprocess_depth_sequence(
            os.path.join(base_dir, "tnt_data/"),
            frame_spacing=10, target_size=(128,128)
        )
        # inputs, gts = preprocess_depth_sequence_scenenet(
        #     os.path.join(base_dir, "tnt_data/rgbd_scenenet/train/0"),
        #     frame_spacing=3, target_size=(128,128)
        # )
        np.save(inp_cache, np.array(inputs, dtype=object))
        np.save(gt_cache,  np.array(gts,    dtype=object))

    dataset  = CachedDepthDataset(inputs, gts)
    total    = len(dataset)
    train_n  = int(0.7 * total)
    val_n    = int(0.15 * total)
    test_n   = total - train_n - val_n

    # train_ds, val_ds, test_ds = random_split(
    #     dataset, [train_n, val_n, test_n], generator=torch.Generator().manual_seed(42)
    # )
    # train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # PyTorch DataLoader :contentReference[oaicite:3]{index=3}
    # val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)
    # test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)

    # 1. Split sequentially
    train_indices = range(0,   train_n)
    val_indices   = range(train_n,   train_n + val_n)
    test_indices  = range(train_n + val_n, total)

    train_ds = Subset(dataset, train_indices)
    val_ds   = Subset(dataset, val_indices)
    test_ds  = Subset(dataset, test_indices)

    # 2. Create loaders
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=True)

    # 3. For GIF visualization (one frame at a time, in order)
    viz_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model     = UNet3In(in_channels=3, out_channels=1).to(device)
    # model     = UNet3InPartial(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if os.path.exists(os.path.join(cache_dir, "unet3in_depth.pth")):
        print("Loading cached model...")
        model.load_state_dict(torch.load(os.path.join(cache_dir, "unet3in_depth.pth")))
    else:
        print("Training model...")
        for epoch in range(1, 1 + 10):
            model.train()
            epoch_loss = 0.0
            for seqs, targets in train_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                optimizer.zero_grad()
                preds = model(seqs)
                loss  = indexed_masked_mse_loss(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch:02d} — Train Loss: {epoch_loss/len(train_loader):.4f}")
        model_save_path = os.path.join(cache_dir, "unet3in_depth.pth")
        torch.save(model.state_dict(), model_save_path)


    # Get one batch
    seqs, gts = next(iter(test_loader))     # get first batch from your DataLoader
    # Move data & model to device
    seqs = seqs.to(device)
    gts  = gts.to(device)
    # Compute predictions
    with torch.no_grad():
        preds = model(seqs)
    # Visualize the first sample in the batch:
    visualize_prediction(
        input_seq=seqs[0],
        gt=       gts[0],
        pred=     preds[0]
    )

    create_sequence_gif(model, viz_loader, device, gif_path='saved_data_8/vis.gif', fps=8)
