import torch
import torch.nn as nn
from torchvision.models import resnet18

# mess with number of epochs and learning rate
# change weights when there is no change in translation/rotation
# change to training on rotation
# maybe just try it with MoGe

class DepthPoseEstimationNN(nn.Module):
    def __init__(self):
        super(DepthPoseEstimationNN, self).__init__()
        # Feature extractor (ResNet18 backbone)
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # Input: Concatenated depth maps [B, 2, H, W]
        self.feature_extractor.avgpool = nn.Identity()
        self.feature_extractor.fc = nn.Identity()

        # Pose embedding (optional: encode the pose)
        self.pose_fc = nn.Sequential(
            nn.Linear(7, 128),  # Pose input [tx, ty, tz, qx, qy, qz, qw]
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Fully connected layers for pose regression
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4 + 128, 256),  # Concatenate depth features + pose embedding
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # Output: [tx, ty, tz, qx, qy, qz, qw]
        )

    def forward(self, d1, d2, pose):
        # Depth features
        depth_input = torch.cat((d1, d2), dim=1)  # Concatenate depth maps [B, 2, H, W]
        depth_features = self.feature_extractor(depth_input)  # [B, 512, 4, 4]
        depth_features = depth_features.view(depth_features.size(0), -1)  # Flatten [B, 512*4*4]

        # Pose embedding
        pose_embedding = self.pose_fc(pose)  # [B, 128]

        # Concatenate depth features and pose embedding
        combined_features = torch.cat((depth_features, pose_embedding), dim=1)

        # Predict pose
        pred_pose = self.fc(combined_features)  # [B, 7]
        return pred_pose
