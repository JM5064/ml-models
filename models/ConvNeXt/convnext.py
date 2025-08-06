import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class ConvNeXtBlock(nn.Module):

    def __init__(self, in_channels) -> None:
        super().__init__()

        # depthwise 7x7, 96 -> 96
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding="same", groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # pointwise 1x1, 96 -> 384
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*4, kernel_size=1, padding="same")

        self.conv3 = nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels, kernel_size=1, padding="same")


    def forward(self, x):
        x_res = x

        x = self.bn1(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        x += x_res

        return x


class DownsampleLayer(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        # spatial downsampling
        self.lnDS = nn.BatchNorm2d(in_channels)
        self.downsample_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)


    def forward(self, x):
        x = self.lnDS(x)
        x = self.downsample_conv(x)

        return x


class ConvNeXt(nn.Module):

    def __init__(self, layer_distribution: list[int], num_classes: int) -> None:
        super().__init__()

        # patchified stem
        self.stem = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=4, padding=3)

        self.stage1 = self.stage(64, 64, layer_distribution[0], False)
        self.stage2 = self.stage(64, 128, layer_distribution[1], True)
        self.stage3 = self.stage(128, 256, layer_distribution[2], True)
        self.stage4 = self.stage(256, 512, layer_distribution[3], True)

        self.gblavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(in_features=512, out_features=num_classes)
#        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)


    def stage(self, in_channels, out_channels, num_layers, downsample):
        layers = []

        # update spatial dimension to new dimension (out_channels)
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same"))

        if downsample:
            layers.append(DownsampleLayer(in_channels=out_channels))

        for _ in range(num_layers):
            layers.append(ConvNeXtBlock(in_channels=out_channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gblavgpool(x)
        x = self.bn1(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
#        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = ConvNeXt(layer_distribution=[3, 3, 9, 3], num_classes=10)
    profile = summary(model, input_size=(3, 224, 224))