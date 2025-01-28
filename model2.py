import torch
import torch.nn as nn

class DepthPoseEstimationNN(nn.Module):
    def __init__(self):
        super(DepthPoseEstimationNN, self).__init__()
        
        # Update: 3 input channels (for 3 depth maps)
        self.conv_block = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output shape: [B, 16, 64, 64]
            
            # Conv Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output shape: [B, 32, 32, 32]
            
            # Conv Block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output shape: [B, 64, 16, 16]
            
            # Conv Block 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # Output shape: [B, 128, 8, 8]
        )
        
        # Fully connected layers remain unchanged
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7D Pose: [tx, ty, tz, qx, qy, qz, qw]
        )
        
    def forward(self, d1, d2, d3, pose):
        """
        Args:
            d1 (torch.Tensor): Depth map 1, shape [B, 1, 128, 128]
            d2 (torch.Tensor): Depth map 2, shape [B, 1, 128, 128]
            d3 (torch.Tensor): Depth map 3, shape [B, 1, 128, 128]
            pose (torch.Tensor): 7D pose [B, 7] (currently unused)
        
        Returns:
            pred_pose (torch.Tensor): shape [B, 7]
        """
        # Concatenate depth maps along the channel dimension
        # shape: [B, 3, 128, 128]
        depth_input = torch.cat((d1, d2, d3), dim=1)
        
        # Pass through conv layers
        # shape after final conv block: [B, 128, 8, 8]
        conv_out = self.conv_block(depth_input)
        
        # Flatten
        # shape: [B, 128*8*8] = [B, 8192]
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        # Pass flattened features through FC layers -> [B, 7]
        pred_pose = self.fc(conv_out)
        
        return pred_pose
