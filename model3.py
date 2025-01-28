import torch
import torch.nn as nn

class DepthPoseEstimationNN(nn.Module):
    def __init__(self):
        super(DepthPoseEstimationNN, self).__init__()
        
        # Convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 3 input channels for 3 depth maps
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Fully connected layers for pose prediction
        self.pose_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # Pose: [tx, ty, tz, qx, qy, qz, qw]
        )
        
        # Fully connected layers for sequential prediction
        self.seq_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Binary output: 1 for sequential, 0 for not sequential
            nn.Sigmoid()       # Sigmoid activation for binary classification
        )

    def forward(self, d1, d2, d3, pose=None):
        """
        Args:
            d1, d2, d3: Depth maps (shape: [B, 1, 128, 128]) for three inputs.
            pose (optional): Ground truth pose (shape: [B, 7]), not used in forward pass.

        Returns:
            pred_pose: Predicted 7D pose (shape: [B, 7]).
            is_sequential: Binary classification output for sequential indicator (shape: [B, 1]).
        """
        # Concatenate depth maps along channel dimension -> [B, 3, 128, 128]
        depth_input = torch.cat((d1, d2, d3), dim=1)
        
        # Pass through convolutional layers
        conv_out = self.conv_block(depth_input)
        
        # Flatten for fully connected layers
        conv_out = conv_out.view(conv_out.size(0), -1)  # Shape: [B, 128 * 8 * 8]
        
        # Pose prediction
        pred_pose = self.pose_fc(conv_out)
        
        # Sequential indicator prediction
        is_sequential = self.seq_fc(conv_out)
        
        return pred_pose, is_sequential
