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


# ---------------------------
# 1. Preprocessing Functions
# ---------------------------
def preprocess_depth_sequence(parent_sequence_path, frame_spacing=3, target_size=(128,128)):
    """
    For each sequence:
      - "dp" holds raw depth frames
      - "depth" holds ground truth depth maps
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
                img  = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                img  = cv2.resize(img, target_size)
                imgs.append(img)
            all_inputs.append(tuple(imgs))

            # Load GT for middle frame
            gt_path = os.path.join(gt_folder, gt_files[i + frame_spacing])
            gt_img   = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            gt_img   = cv2.resize(gt_img, target_size) / 5000.0  # Normalize by 5000.0 to convert ground truth to meters for Freiburg dataset
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
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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

        # Final conv + clamp
        x = self.final_conv(x)
        # x = F.sigmoid(x)
        # x = F.relu(x)
        return x


# ---------------------------
# 4. Loss Function
# ---------------------------
def indexed_masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, crop: float=0.0):
    mask = target != 0.0
    if crop > 0:
        crop_mask = torch.zeros_like(mask)
        h, w = mask.shape[2:]
        h1, h2 = int(h * crop), int(h * (1 - crop))
        w1, w2 = int(w * crop), int(w * (1 - crop))
        crop_mask[:, :, h1:h2, w1:w2] = 1
        mask = mask * crop_mask
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
                         pred: torch.Tensor,
                         crop: float = 0.0,
                         show_disparities: bool = True):
    """
    Displays prediction results:
      - Main Figure: Input t-1, t0, t+1, Ground Truth, Prediction (shared color scale)
      - Optional Second Figure: Disparity (GT - t0), Disparity (GT - Prediction)
    Includes MAE and MSE in titles.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def compute_mae(pred_np: np.ndarray, gt_np: np.ndarray, mask_value=0.0):
        valid = gt_np != mask_value
        if not np.any(valid):
            return np.nan
        return np.mean(np.abs(pred_np[valid] - gt_np[valid]))

    # Convert tensors to NumPy
    seq_np = input_seq.cpu().numpy() / 1000  # Convert mm → meters
    gt_np = gt.detach().cpu().numpy().squeeze(0)
    pred_np = pred.detach().cpu().numpy().squeeze(0)

    # Optional crop
    if crop > 0:
        h, w = gt_np.shape
        h1, h2 = int(h * crop), int(h * (1 - crop))
        w1, w2 = int(w * crop), int(w * (1 - crop))
        gt_np = gt_np[h1:h2, w1:w2]
        pred_np = pred_np[h1:h2, w1:w2]
        seq_np = seq_np[:, h1:h2, w1:w2]

    # Compute disparities
    disparity_input = gt_np - seq_np[1]
    disparity_pred = gt_np - pred_np

    # MAE & MSE
    mae_input = compute_mae(seq_np[1], gt_np)
    mae_pred = compute_mae(pred_np, gt_np)
    mse_input = np.mean((seq_np[1] - gt_np) ** 2)
    mse_pred = np.mean((pred_np - gt_np) ** 2)

    # Shared color scale for the main maps
    core_maps = [seq_np[0], seq_np[1], seq_np[2], gt_np, pred_np]
    vmin_core = min(map(np.min, core_maps))
    vmax_core = max(map(np.max, core_maps))

    # Main figure
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    images = core_maps
    titles = [
        "Input t-1",
        f"Input t0\n(MAE={mae_input:.3f}, MSE={mse_input:.3f})",
        "Input t+1",
        "Ground Truth",
        f"Prediction\n(MAE={mae_pred:.3f}, MSE={mse_pred:.3f})"
    ]
    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, cmap='plasma', vmin=vmin_core, vmax=vmax_core)
        ax.set_title(title)
        ax.axis('off')
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Add colorbar to the last axis
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # Optional disparity figure
    if show_disparities:
        vmin_disp = min(disparity_input.min(), disparity_pred.min())
        vmax_disp = max(disparity_input.max(), disparity_pred.max())

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        disparities = [disparity_input, disparity_pred]
        disp_titles = ["Disparity: GT - t0", "Disparity: GT - Prediction"]
        for ax, img, title in zip(axes2, disparities, disp_titles):
            im = ax.imshow(img, cmap='bwr', vmin=vmin_disp, vmax=vmax_disp)
            ax.set_title(title)
            ax.axis('off')
            fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    print(f"MAE (GT - t0):   {mae_input:.4f}")
    print(f"MSE (GT - t0):   {mse_input:.4f}")
    print(f"MAE (GT - pred): {mae_pred:.4f}")
    print(f"MSE (GT - pred): {mse_pred:.4f}")


def create_sequence_gif(model, dataloader, device, gif_path='predictions.gif', fps=2, crop=0.1):
    import matplotlib.pyplot as plt
    import imageio
    import numpy as np

    model.eval()
    frames = []
    all_values = []

    # First pass: collect vmin/vmax using inputs (converted), GT, and predictions
    with torch.no_grad():
        for seqs, gts in dataloader:
            seqs, gts = seqs.to(device), gts.to(device)
            preds = model(seqs)

            if crop > 0:
                h, w = gts.shape[2:]
                h1, h2 = int(h * crop), int(h * (1 - crop))
                w1, w2 = int(w * crop), int(w * (1 - crop))
                seqs = seqs[:, :, h1:h2, w1:w2]
                gts = gts[:, :, h1:h2, w1:w2]
                preds = preds[:, :, h1:h2, w1:w2]

            all_values.extend([(seqs[:, j] / 1000).cpu().numpy().flatten() for j in range(3)])
            all_values.append(gts[:, 0].cpu().numpy().flatten())
            all_values.append(preds[:, 0].cpu().numpy().flatten())

    all_values = np.concatenate(all_values)
    vmin = np.min(all_values)
    vmax = np.max(all_values)

    # Second pass: generate frames
    with torch.no_grad():
        for seqs, gts in dataloader:
            seqs, gts = seqs.to(device), gts.to(device)
            preds = model(seqs)

            if crop > 0:
                h, w = gts.shape[2:]
                h1, h2 = int(h * crop), int(h * (1 - crop))
                w1, w2 = int(w * crop), int(w * (1 - crop))
                seqs = seqs[:, :, h1:h2, w1:w2]
                gts = gts[:, :, h1:h2, w1:w2]
                preds = preds[:, :, h1:h2, w1:w2]

            for i in range(seqs.size(0)):
                fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                for j in range(3):
                    axes[j].imshow((seqs[i, j].cpu().numpy() / 1000), cmap='magma', vmin=vmin, vmax=vmax)
                    axes[j].set_title(f'Input t{["-1", "0", "+1"][j]}')
                    axes[j].axis('off')
                axes[3].imshow(gts[i, 0].cpu().numpy(), cmap='magma', vmin=vmin, vmax=vmax)
                axes[3].set_title('Ground Truth')
                axes[3].axis('off')
                axes[4].imshow(preds[i, 0].cpu().numpy(), cmap='magma', vmin=vmin, vmax=vmax)
                axes[4].set_title('Prediction')
                axes[4].axis('off')

                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                width, height = fig.canvas.get_width_height()
                image = image.reshape((height, width, 3))
                frames.append(image)
                plt.close(fig)

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"Saved GIF to {gif_path}")

def evaluate_sequence(input_seqs: torch.Tensor, gt_seqs: torch.Tensor, pred_seqs: torch.Tensor, crop: float = 0.0, mask_value: float = 0.0, visualize_indices: list = None):
    """
    Evaluate the full sequence using MAE and MSE. Optionally visualize a few frames.

    Args:
        input_seqs: Tensor [B, 3, H, W]
        gt_seqs:    Tensor [B, 1, H, W]
        pred_seqs:  Tensor [B, 1, H, W]
        crop: Fractional crop applied on height/width (e.g. 0.1)
        mask_value: Invalid depth value to ignore (e.g. 0.0)
        visualize_indices: Optional list of indices to visualize
    """

    def compute_mae_mse(pred_np, gt_np, mask_value):
        valid = gt_np != mask_value
        if not np.any(valid):
            return np.nan, np.nan
        error = pred_np[valid] - gt_np[valid]
        mae = np.mean(np.abs(error))
        mse = np.mean(error ** 2)
        return mae, mse

    total_mae_pred = total_mse_pred = 0.0
    total_mae_input = total_mse_input = 0.0
    count = 0

    input_seqs = input_seqs.cpu()
    gt_seqs = gt_seqs.cpu()
    pred_seqs = pred_seqs.detach().cpu()

    B = input_seqs.shape[0]

    for i in range(B):
        seq_np = input_seqs[i].numpy()
        gt_np = gt_seqs[i].numpy().squeeze(0)
        pred_np = pred_seqs[i].numpy().squeeze(0)
        input_t0_np = seq_np[1]

        # Apply crop
        if crop > 0:
            h, w = gt_np.shape
            h1, h2 = int(h * crop), int(h * (1 - crop))
            w1, w2 = int(w * crop), int(w * (1 - crop))
            gt_np = gt_np[h1:h2, w1:w2]
            pred_np = pred_np[h1:h2, w1:w2]
            input_t0_np = input_t0_np[h1:h2, w1:w2]
            seq_np = seq_np[:, h1:h2, w1:w2]  # crop all 3 input frames

        mae_pred, mse_pred = compute_mae_mse(pred_np, gt_np, mask_value)
        mae_input, mse_input = compute_mae_mse(input_t0_np/1000, gt_np, mask_value)

        total_mae_pred += mae_pred
        total_mse_pred += mse_pred
        total_mae_input += mae_input
        total_mse_input += mse_input
        count += 1

        if visualize_indices and i in visualize_indices:
            visualize_prediction(torch.tensor(seq_np),
                                 torch.tensor(gt_np).unsqueeze(0),
                                 torch.tensor(pred_np).unsqueeze(0),
                                 crop=0.0,
                                 show_disparities=False)

    avg = lambda x: x / count if count > 0 else float('nan')

    print(f"\n[Sequence Evaluation]")
    print(f"MAE (GT - Prediction): {avg(total_mae_pred):.4f}")
    print(f"MSE (GT - Prediction): {avg(total_mse_pred):.4f}")
    print(f"MAE (GT - Input t0):   {avg(total_mae_input):.4f}")
    print(f"MSE (GT - Input t0):   {avg(total_mse_input):.4f}")

    return {
        "mae_pred": avg(total_mae_pred),
        "mse_pred": avg(total_mse_pred),
        "mae_input": avg(total_mae_input),
        "mse_input": avg(total_mse_input),
    }


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
            frame_spacing=3, target_size=(128,128),
        )
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
    # viz_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_params/1e6:.2f}M\n")

    if os.path.exists(os.path.join(cache_dir, "best_unet_depth.pth")):
        print("Loading cached model...")
        model.load_state_dict(torch.load(os.path.join(cache_dir, "best_unet_depth.pth")))
    else:
        max_epochs      = 20
        patience        = 5     
        best_val_loss   = float('inf')
        no_improve_cnt  = 0
        train_losses = []
        val_losses   = []

        print("Training model with early stopping...")
        for epoch in range(1, max_epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            for seqs, targets in train_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                optimizer.zero_grad()
                preds = model(seqs)
                loss  = indexed_masked_mse_loss(preds, targets, crop=0.1)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train = train_loss / len(train_loader)
            train_losses.append(avg_train)

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seqs, targets in val_loader:
                    seqs, targets = seqs.to(device), targets.to(device)
                    preds = model(seqs)
                    val_loss += indexed_masked_mse_loss(preds, targets, crop=0.1).item()
            avg_val = val_loss / len(val_loader)
            val_losses.append(avg_val)

            print(f"Epoch {epoch:02d} — Train Loss: {avg_train:.4f} — Val Loss: {avg_val:.4f}")

            # Early stopping logic
            if avg_val < best_val_loss:
                best_val_loss  = avg_val
                no_improve_cnt = 0
                # save the best checkpoint
                torch.save(model.state_dict(), os.path.join(cache_dir, "best_unet_depth.pth"))
                print("  ↳ New best model saved.")
            else:
                no_improve_cnt += 1
                print(f"  ↳ No improvement for {no_improve_cnt}/{patience} epochs.")

            if no_improve_cnt >= patience:
                print(f"Early stopping triggered (no improvement in {patience} epochs).")
                break

        print("Training complete.")

        # # 3) Plot curves
        # plt.figure()
        # plt.plot(train_losses, label="Train Loss")
        # plt.plot(val_losses,   label="Val Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.title("Learning Curves")
        # plt.show()


    model.eval()

    create_sequence_gif(model, viz_loader, device, gif_path='saved_data_8/vis.gif', fps=8, crop=0.1)

    all_seqs = []
    all_gts = []
    all_preds = []

    with torch.no_grad():
        for seqs, gts in test_loader:

            seqs = seqs.to(device)
            gts  = gts.to(device)
            preds = model(seqs)

            all_seqs.append(seqs.cpu())
            all_gts.append(gts.cpu())
            all_preds.append(preds.cpu())

    # Concatenate all batches
    all_seqs = torch.cat(all_seqs, dim=0)   # shape: [N, 3, H, W]
    all_gts = torch.cat(all_gts, dim=0)     # shape: [N, 1, H, W]
    all_preds = torch.cat(all_preds, dim=0) # shape: [N, 1, H, W]

    # Evaluate the full test set
    results = evaluate_sequence(
        input_seqs=all_seqs,
        gt_seqs=all_gts,
        pred_seqs=all_preds,
        crop=0.1,
        mask_value=0.0,
        visualize_indices=[2]  # or whatever you want to visualize
    )

    print("Evaluation Results:")
    print("Prediction:")
    print(f"  MAE: {results['mae_pred']:.4f}")
    print(f"  MSE: {results['mse_pred']:.4f}")
    print("Input t0:")
    print(f"  MAE: {results['mae_input']:.4f}")
    print(f"  MSE: {results['mse_input']:.4f}")


