import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from torchsummary import summary


class SimpleBaselines(nn.Module):

    def __init__(self, num_keypoints):
        super().__init__()

        self.num_keypoints = num_keypoints

        # Load resnet
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.deconv_layers = nn.Sequential(
            DeconvLayer(in_channels=2048),
            DeconvLayer(),
            DeconvLayer()
        )

        self.hm = nn.Conv2d(in_channels=256, out_channels=num_keypoints, kernel_size=1)


    def forward(self, x):
        # Forward pass through resnet backbone
        x = self.backbone(x)    # 8x8x2048

        # Pass through deconvolution layers
        x = self.deconv_layers(x)   # 64x64x256

        # Generate heatmaps
        x = self.hm(x)

        return x


class DeconvLayer(nn.Module):

    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = F.relu(x)

        return x


if __name__ == "__main__":
    model = SimpleBaselines(num_keypoints=16)

    profile = summary(model, input_size=(3, 256, 256))
    