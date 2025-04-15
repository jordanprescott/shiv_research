import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------
# Define the U-Net model
# -------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(UNet, self).__init__()
        features = init_features

        # Encoder (downsampling path)
        self.encoder1 = UNet._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNet._block(features * 8, features * 16)

        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features)

        # Final 1x1 convolution to get desired output channels
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, features):
        """Creates a convolutional block with two Conv-BatchNorm-ReLU layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

# -------------------------
# Define the masked L1 loss
# -------------------------
def masked_l1_loss(pred, target, mask_value=0.0, epsilon=1e-6):
    """
    Compute L1 loss while ignoring pixels where the ground truth equals mask_value.
    This prevents the network from learning zeros that represent missing data.
    """
    mask = (target != mask_value).float()
    valid_pixels = mask.sum()
    if valid_pixels < epsilon:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum') / valid_pixels
    return loss

# -------------------------
# Define the dataset loader
# -------------------------
class DepthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Expects a root_dir containing multiple dataset folders (e.g., rgbd_dataset_freiburg1_desk).
        Each dataset folder should contain two folders:
          - 'dp': folder with Depth Pro (input) depth maps.
          - 'depth': folder with ground truth depth maps.
        This loader creates samples by taking three consecutive images from 'dp' (using the middle image as the reference)
        and matching it by index with the corresponding ground truth image from 'depth'.
        """
        self.samples = []
        self.transform = transform if transform is not None else transforms.ToTensor()
        # Iterate over each dataset folder
        dataset_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))]
        print("Found dataset folders:", dataset_folders)
        for ds in dataset_folders:
            dp_folder = os.path.join(ds, "dp")
            gt_folder = os.path.join(ds, "depth")
            if not os.path.exists(dp_folder) or not os.path.exists(gt_folder):
                continue
            dp_files = glob.glob(os.path.join(dp_folder, "*"))
            gt_files = glob.glob(os.path.join(gt_folder, "*"))
            n = min(len(dp_files), len(gt_files))
            # Ensure there are at least 3 images in the smaller folder for a valid sample
            if n < 3:
                continue
            # Use index matching to form samples.
            # Input: three consecutive dp images; GT: corresponding ground truth for the middle frame.
            # Also, return the file name from the middle dp image.
            for i in range(1, n - 1):
                file_name = os.path.basename(dp_files[i])
                self.samples.append((dp_files[i-1], dp_files[i], dp_files[i+1], gt_files[i], file_name))
        print("Number of samples found:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dp1_path, dp2_path, dp3_path, gt_path, file_name = self.samples[idx]
        # Open images and convert them to grayscale
        dp1 = Image.open(dp1_path).convert('L')
        dp2 = Image.open(dp2_path).convert('L')
        dp3 = Image.open(dp3_path).convert('L')
        gt = Image.open(gt_path).convert('L')
        # Apply transformation (e.g., conversion to a tensor)
        dp1 = self.transform(dp1)
        dp2 = self.transform(dp2)
        dp3 = self.transform(dp3)
        gt = self.transform(gt)
        # Concatenate the three dp images along the channel dimension: shape [3, H, W]
        input_tensor = torch.cat([dp1, dp2, dp3], dim=0)
        return input_tensor, gt, file_name

# -------------------------
# Function to plot input, ground truth, and output
# -------------------------
def plot_sample(input_tensor, gt, output, file_name="Unknown"):
    """
    Plot the three input depth maps, the ground truth, and the network output,
    including the file name in the plot.
    
    Args:
        input_tensor (Tensor): A tensor of shape [3, H, W] containing the three input depth maps.
        gt (Tensor): A tensor of shape [1, H, W] for the ground truth depth.
        output (Tensor): A tensor of shape [1, H, W] for the network predicted depth.
        file_name (str): The file name associated with this sample.
    """
    # Convert tensors to numpy arrays and squeeze extra dimensions.
    dp1 = input_tensor[0].cpu().numpy().squeeze()
    dp2 = input_tensor[1].cpu().numpy().squeeze()
    dp3 = input_tensor[2].cpu().numpy().squeeze()
    gt_np = gt.cpu().numpy().squeeze()
    output_np = output.cpu().detach().numpy().squeeze()
    
    # Create a figure with 5 subplots.
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(dp1, cmap="gray")
    axs[0].set_title("Input dp1")
    axs[1].imshow(dp2, cmap="gray")
    axs[1].set_title("Input dp2")
    axs[2].imshow(dp3, cmap="gray")
    axs[2].set_title("Input dp3")
    axs[3].imshow(gt_np, cmap="gray")
    axs[3].set_title("Ground Truth")
    axs[4].imshow(output_np, cmap="gray")
    axs[4].set_title("Output")
    for ax in axs:
        ax.axis('off')
        
    # Include the file name as a super title for the plot.
        
    print("File Name:", file_name)

    fig.suptitle(f"File: {file_name}", fontsize=16)
    plt.tight_layout()
    plt.show()

# -------------------------
# Main training/demo loop
# -------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Path where your datasets are stored
    dataset_path = "/home/jordanprescott/shiv_research/tnt_data"
    dataset = DepthDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize the model
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Process one batch for demonstration
    for inputs, gt, file_names in dataloader:
        inputs = inputs.to(device)
        gt = gt.to(device)
        outputs = model(inputs)
        loss = masked_l1_loss(outputs, gt)
        print("Batch Loss:", loss.item())
        
        # Plot the first sample from the batch, including the file name in the title.
        plot_sample(inputs[0], gt[0], outputs[0], file_name=file_names[0])
        break
