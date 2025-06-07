import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels-in_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        output = torch.cat([self.conv(x), self.pool(x)], 1)
        return self.bn(output)

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

class ERFNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=1):
        super().__init__()
        
        # Initial block
        self.initial_block = DownsamplerBlock(in_channels, 16)
        
        # Encoder
        self.encoder = nn.Sequential(
            DownsamplerBlock(16, 64),
            NonBottleneck1D(64, 0.03, 1),
            NonBottleneck1D(64, 0.03, 1),
            NonBottleneck1D(64, 0.03, 1),
            NonBottleneck1D(64, 0.03, 1),
            NonBottleneck1D(64, 0.03, 1),
            DownsamplerBlock(64, 128),
            NonBottleneck1D(128, 0.3, 2),
            NonBottleneck1D(128, 0.3, 4),
            NonBottleneck1D(128, 0.3, 8),
            NonBottleneck1D(128, 0.3, 16)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            UpsamplerBlock(128, 64),
            NonBottleneck1D(64, 0, 1),
            NonBottleneck1D(64, 0, 1),
            UpsamplerBlock(64, 16),
            NonBottleneck1D(16, 0, 1),
            NonBottleneck1D(16, 0, 1)
        )
        
        # Final conv layer
        self.final = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        # Initial block
        x = self.initial_block(x)
        
        # Encoder
        x = self.encoder(x)
        
        # Decoder
        x = self.decoder(x)
        
        # Final classification
        x = self.final(x)
        
        return x

# Modified train script to use ERFNet
def get_model(in_channels=4, num_classes=1):
    model = ERFNet(in_channels=in_channels, num_classes=num_classes)
    return model
