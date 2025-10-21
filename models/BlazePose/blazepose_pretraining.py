import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class BlazePose(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.bb1 = BlazeBlock(3, 16)        # 256 x 256 x 3 -> 128 x 128 x 16
        self.bb2 = BlazeBlock(16, 32)       # 128 x 128 x 16 -> 64 x 64 x 32
        self.bb3 = BlazeBlock(32, 64)       # 64 x 64 x 32 -> 32 x 32 x 64
        self.bb4 = BlazeBlock(64, 128)      # 32 x 32 x 64 -> 16 x 16 x 128
        self.bb5 = BlazeBlock(128, 192)     # 16 x 16 x 128 -> 8 x 8 x 128

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=192, out_features=num_classes)


    def forward(self, x):
        x = self.bb1(x)

        # Center BlazeBlock path
        x = self.bb2(x)
        x = self.bb3(x)
        x = self.bb4(x)
        x = self.bb5(x) # last shared layer between regression and heatmap paths
        
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x


class BlazeBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Downsample
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Regular convolution for good measure
        # self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same")
        # self.bn22 = nn.BatchNorm2d(out_channels)

        # Depthwise seperable convolution
        # Add batch norm after pointwise? (convnext doesnt do this)
        self.dw = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, groups=out_channels, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pw1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, padding='same')
        self.pw2 = nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels, kernel_size=1, padding='same')


    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu6(x)

        x_resid = x

        # x = self.conv2(x)
        # x = self.bn22(x)
        # x = F.relu6(x)

        x = self.dw(x)
        x = self.bn2(x)

        x = F.relu6(x)
        x = self.pw1(x)
        x = self.pw2(x)

        x = x + x_resid

        return x
    

if __name__ == "__main__":
    model = BlazePose(257)
    profile = summary(model, input_size=(3, 256, 256))