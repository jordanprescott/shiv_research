import torch
import torch.nn as nn
from torchvision.models import resnet18

class DepthPoseEstimationNN(nn.Module):
    def __init__(self):
        super(DepthPoseEstimationNN, self).__init__()
        # Feature extractor (ResNet18 with modified input)
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # Adjust for 2-channel input
        self.feature_extractor.avgpool = nn.Identity()  # Remove adaptive pooling
        self.feature_extractor.fc = nn.Identity()  # Remove final classification layer

        # Fully connected layers for pose regression
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * (4 + 3), 256),  # Adjusted input size to include pose
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # Output: [tx, ty, tz]
        )

    def forward(self, d1, d2, pose):
        # Concatenate depth maps along the channel dimension
        input_depths = torch.cat((d1, d2), dim=1)  # [B, 2, 128, 128]
        
        # Extract features
        features = self.feature_extractor(input_depths)  # [B, 512, 4, 4]
        
        # Flatten features for the fully connected layers
        features = features.view(features.size(0), -1)  # [B, 512 * 4 * 4]
        
        # Concatenate features with pose
        features = torch.cat((features, pose), dim=1)  # [B, 512 * 4 * 4 + 3]
        
        # Pose regression
        pose_output = self.fc(features)
        return pose_output
