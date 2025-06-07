import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ----- Basic Building Blocks -----

class UpsamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                       output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class NonBottleneck1D(nn.Module):
    def __init__(self, channels, dropout_prob=0.1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1,
                               padding=(dilation, 0), bias=True, dilation=(dilation, 1))
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1,
                               padding=(0, dilation), bias=True, dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(channels)

        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(output)
        output = self.conv2(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3(output)
        output = F.relu(output)
        output = self.conv4(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(output + x)

# ----- ResNet Encoder -----

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=6, backbone='resnet18', pretrained=False):
        super().__init__()
        if backbone == 'resnet18':
            resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone}")

        if in_channels != 3:
            self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.initial_conv = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x  # (batch, 512, H/32, W/32)

# ----- Full Model -----

class ERFNetWithResNetEncoder(nn.Module):
    def __init__(self, in_channels=6, num_classes=1, backbone='resnet18', pretrained=False):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels, backbone=backbone, pretrained=pretrained)

        self.decoder = nn.Sequential(
            UpsamplerBlock(512, 256),
            NonBottleneck1D(256, 0.1, 1),
            UpsamplerBlock(256, 128),
            NonBottleneck1D(128, 0.1, 1),
            UpsamplerBlock(128, 64),
            NonBottleneck1D(64, 0.1, 1),
            UpsamplerBlock(64, 16),
            NonBottleneck1D(16, 0.1, 1)
        )

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final(x)
        return x

# ----- Helper function -----

def get_model(in_channels=6, num_classes=1, backbone='resnet18', pretrained=False):
    return ERFNetWithResNetEncoder(
        in_channels=in_channels,
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained
    )
