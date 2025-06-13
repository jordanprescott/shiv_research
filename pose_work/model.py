import torch
import torch.nn as nn

class DepthPoseEstimationNN(nn.Module):
    def __init__(self):
        super(DepthPoseEstimationNN, self).__init__()
        
        # 4 Convolutional blocks, each followed by a 2x2 max pool (stride=2)
        # Input: [B, 2, 128, 128]
        
        self.conv_block = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
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
        
        # After the 4th max pool, the spatial resolution is 8x8
        # So the feature map is [B, 128, 8, 8].
        # Flatten to [B, 128 * 8 * 8 = 8192] before the final FC layers.
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7D Pose: [tx, ty, tz, qx, qy, qz, qw]
        )
        
    def forward(self, d1, d2, pose):
        """
        Args:
            d1 (torch.Tensor): Depth map 1, shape [B, 1, 128, 128]
            d2 (torch.Tensor): Depth map 2, shape [B, 1, 128, 128]
            pose (torch.Tensor): 7D pose [B, 7] (currently unused)
        
        Returns:
            pred_pose (torch.Tensor): shape [B, 7]
        """
        # Concatenate depth maps along channel dimension
        # shape: [B, 2, 128, 128]
        depth_input = torch.cat((d1, d2), dim=1)
        
        # Pass through conv layers
        # shape after final conv block: [B, 128, 8, 8]
        conv_out = self.conv_block(depth_input)
        
        # Flatten
        # shape: [B, 128*8*8] = [B, 8192]
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        # Pass flattened features through FC layers -> [B, 7]
        pred_pose = self.fc(conv_out)
        
        return pred_pose




# import torch
# import torch.nn as nn

# class DepthPoseEstimationNN(nn.Module):
#     def __init__(self):
#         super(DepthPoseEstimationNN, self).__init__()
#         # Simple convolutional layers
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),  # Input: [B, 2, H, W]
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((4, 4))  # Output: [B, 512, 4, 4]
#         )

#         # Fully connected layers for pose regression (no pose embedding)
#         self.fc = nn.Sequential(
#             nn.Linear(512 * 4 * 4, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 7)  # Output: [tx, ty, tz, qx, qy, qz, qw]
#         )

#     def forward(self, d1, d2, pose):
#         """
#         Args:
#             d1  (torch.Tensor): Depth map 1 [B, 1, H, W]
#             d2  (torch.Tensor): Depth map 2 [B, 1, H, W]
#             pose (torch.Tensor): 7D pose [B, 7] (currently unused in this model)
        
#         Returns:
#             pred_pose (torch.Tensor): Predicted pose [B, 7]
#         """
#         # Combine the two depth maps
#         depth_input = torch.cat((d1, d2), dim=1)  # [B, 2, H, W]
        
#         # Extract features using the simple convolutional layers
#         depth_features = self.conv_layers(depth_input)  # [B, 512, 4, 4]
#         depth_features = depth_features.view(depth_features.size(0), -1)  # Flatten [B, 512*4*4]

#         # Predict pose directly from depth features
#         pred_pose = self.fc(depth_features)  # [B, 7]

#         return pred_pose



# import torch
# import torch.nn as nn
# from torchvision.models import resnet18

# class DepthPoseEstimationNN(nn.Module):
#     def __init__(self):
#         super(DepthPoseEstimationNN, self).__init__()
#         # Feature extractor (ResNet18 backbone)
#         self.feature_extractor = resnet18(pretrained=True)
#         self.feature_extractor.conv1 = nn.Conv2d(
#             2, 64, kernel_size=7, stride=2, padding=3, bias=False
#         )  # Input: Concatenated depth maps [B, 2, H, W]
#         self.feature_extractor.avgpool = nn.Identity()
#         self.feature_extractor.fc = nn.Identity()

#         # Fully connected layers for pose regression (no pose embedding)
#         self.fc = nn.Sequential(
#             nn.Linear(512 * 4 * 4, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 7)  # Output: [tx, ty, tz, qx, qy, qz, qw]
#         )

#     def forward(self, d1, d2, pose):
#         """
#         Args:
#             d1  (torch.Tensor): Depth map 1 [B, 1, H, W]
#             d2  (torch.Tensor): Depth map 2 [B, 1, H, W]
#             pose (torch.Tensor): 7D pose [B, 7] (currently unused in this model)
        
#         Returns:
#             pred_pose (torch.Tensor): Predicted pose [B, 7]
#         """
#         # Combine the two depth maps
#         depth_input = torch.cat((d1, d2), dim=1)  # [B, 2, H, W]
        
#         # Extract features using the modified ResNet18
#         depth_features = self.feature_extractor(depth_input)  # [B, 512, 4, 4]
#         depth_features = depth_features.view(depth_features.size(0), -1)  # Flatten [B, 512*4*4]

#         # Predict pose directly from depth features
#         pred_pose = self.fc(depth_features)  # [B, 7]

#         return pred_pose
