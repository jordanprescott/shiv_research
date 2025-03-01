import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------
# UNet Model for Depth Prediction
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        current_in = in_channels
        # Encoder: Downsampling path
        for feature in features:
            self.downs.append(DoubleConv(current_in, feature))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_in = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder: Upsampling path
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.up_convs.append(DoubleConv(feature*2, feature))
        
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for up, conv in zip(self.ups, self.up_convs):
            x = up(x)
            skip = skip_connections.pop(0)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = conv(x)
        return self.final_conv(x)

# ---------------------------
# Preprocessing Function (Normal Order)
# ---------------------------
def preprocess_depth_sequence(parent_sequence_path, frame_spacing=10, target_size=(128,128)):
    """
    For each sequence folder in normal sorted order:
      - "dp": input depth images
      - "depth": ground truth depth maps
    We create triplets (i, i+frame_spacing, i+2*frame_spacing) from dp_files,
    and for each triplet, we use depth_files[i+frame_spacing] as ground truth.
    Only pairs up data when there are enough frames in both dp and depth.
    """
    all_depth_inputs = []
    all_depth_gts = []
    for sequence_folder in os.listdir(parent_sequence_path):
        sequence_path = os.path.join(parent_sequence_path, sequence_folder)
        dp_folder = os.path.join(sequence_path, "dp")
        depth_folder = os.path.join(sequence_path, "depth")
        if not (os.path.exists(dp_folder) and os.path.exists(depth_folder)):
            continue

        dp_files = sorted([os.path.join(dp_folder, f) for f in os.listdir(dp_folder) if f.endswith(".png")])
        depth_files = sorted([os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith(".png")])

        # Ensure that we have enough frames:
        # - For dp: we need index i + 2*frame_spacing to exist.
        # - For depth: we need index i + frame_spacing to exist.
        max_idx = min(len(dp_files) - 2 * frame_spacing, len(depth_files) - frame_spacing)
        for i in range(max_idx):
            img1 = cv2.imread(dp_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
            img2 = cv2.imread(dp_files[i + frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)
            img3 = cv2.imread(dp_files[i + 2 * frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)
            # Resize and normalize (example normalization: divide by 5000.0)
            img1 = cv2.resize(img1, target_size) / 5000.0
            img2 = cv2.resize(img2, target_size) / 5000.0
            img3 = cv2.resize(img3, target_size) / 5000.0
            all_depth_inputs.append((img1, img2, img3))

            gt_img = cv2.imread(depth_files[i + frame_spacing], cv2.IMREAD_UNCHANGED).astype(np.float32)
            gt_img = cv2.resize(gt_img, target_size) / 5000.0
            all_depth_gts.append(gt_img)
    return all_depth_inputs, all_depth_gts


# ---------------------------
# Dataset Class
# ---------------------------
class CachedDepthDataset(Dataset):
    def __init__(self, depth_inputs, depth_gts):
        self.depth_inputs = depth_inputs
        self.depth_gts = depth_gts

    def __len__(self):
        return len(self.depth_inputs)

    def __getitem__(self, idx):
        d1, d2, d3 = self.depth_inputs[idx]
        d1 = np.asarray(d1, dtype=np.float32)
        d2 = np.asarray(d2, dtype=np.float32)
        d3 = np.asarray(d3, dtype=np.float32)
        input_tensor = torch.tensor(np.stack([d1, d2, d3], axis=0), dtype=torch.float32)
        gt_tensor = torch.tensor(np.asarray(self.depth_gts[idx], dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        return input_tensor, gt_tensor

# ---------------------------
# Training & Evaluation Functions
# ---------------------------
def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for inputs, gt in loader:
        inputs, gt = inputs.to(device), gt.to(device)
        optimizer.zero_grad()
        pred = model(inputs)
        loss = loss_fn(pred, gt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for inputs, gt in loader:
            inputs, gt = inputs.to(device), gt.to(device)
            pred = model(inputs)
            loss = loss_fn(pred, gt)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

def visualize_predictions(model, loader, device, num_samples=6):
    model.eval()
    inputs, gt = next(iter(loader))
    inputs, gt = inputs.to(device), gt.to(device)
    with torch.no_grad():
        pred = model(inputs)
    
    inputs = inputs.cpu().numpy()  # shape: [B, 3, H, W]
    gt = gt.cpu().numpy()          # shape: [B, 1, H, W]
    pred = pred.cpu().numpy()      # shape: [B, 1, H, W]
    
    n_examples = min(num_samples, inputs.shape[0])
    fig, axs = plt.subplots(n_examples, 3, figsize=(18, 4 * n_examples))
    if n_examples == 1:
        axs = np.expand_dims(axs, axis=0)
    
    for i in range(n_examples):
        # Display the middle input depth map (channel index 1)
        im_input = axs[i, 0].imshow(inputs[i, 1, :, :], cmap='viridis')
        axs[i, 0].set_title("Input Depth (Middle dp)")
        axs[i, 0].axis("off")
        fig.colorbar(im_input, ax=axs[i, 0])
        
        # Display Ground Truth depth map
        im_gt = axs[i, 1].imshow(gt[i, 0, :, :], cmap='viridis')
        axs[i, 1].set_title("Ground Truth Depth")
        axs[i, 1].axis("off")
        fig.colorbar(im_gt, ax=axs[i, 1])
        
        # Display Predicted depth map
        im_pred = axs[i, 2].imshow(pred[i, 0, :, :], cmap='viridis')
        axs[i, 2].set_title("Predicted Depth")
        axs[i, 2].axis("off")
        fig.colorbar(im_pred, ax=axs[i, 2])
    
    plt.tight_layout()
    plt.show()

# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(data_dir, "saved_data_7")
    os.makedirs(save_folder, exist_ok=True)
    parent_sequence_path = os.path.join(data_dir, "tnt_data")

    input_cache_path = os.path.join(save_folder, "depth_input_seq.npy")
    gt_cache_path = os.path.join(save_folder, "depth_gt_seq.npy")

    if os.path.exists(input_cache_path) and os.path.exists(gt_cache_path):
        print("Loading cached depth data...")
        depth_input_seq = np.load(input_cache_path, allow_pickle=True)
        depth_gt_seq = np.load(gt_cache_path, allow_pickle=True)
    else:
        print("Preprocessing depth sequences from TUM data...")
        depth_input_seq, depth_gt_seq = preprocess_depth_sequence(
            parent_sequence_path, frame_spacing=10, target_size=(128, 128)
        )
        np.save(input_cache_path, np.array(depth_input_seq, dtype=object))
        np.save(gt_cache_path, np.array(depth_gt_seq, dtype=object))

    dataset = CachedDepthDataset(depth_input_seq, depth_gt_seq)
    total_samples = len(dataset)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    checkpoint_path = os.path.join(save_folder, "unet_depth_model.pth")
    skip_training = False
    if os.path.exists(checkpoint_path):
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        skip_training = True

    if not skip_training:
        num_epochs = 100
        patience = 10
        epochs_without_improvement = 0
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss = evaluate(model, val_loader, loss_fn, device)
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best model at epoch {epoch+1} with val loss: {val_loss:.4f}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}")
    visualize_predictions(model, test_loader, device, num_samples=2)
