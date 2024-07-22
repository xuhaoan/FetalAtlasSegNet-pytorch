import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler


def convolution_block(in_chan, out_chan, ksize=3, pad=1, stride=1, bias=False):
    """
    Convolution Block
    Convolution + Normalization + NonLinear
    """
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=ksize, padding=pad, stride=stride, bias=bias),
        nn.BatchNorm3d(out_chan),
        nn.ReLU()
    )


def up_sample3d(x):
    """
    3D Up Sampling
    """
    return nn.Upsample(scale_factor=2, mode="trilinear",align_corners=True)(x)


class DoubleConv(nn.Module):
    """
    3D Res stage
    """

    def __init__(self, in_chan, out_chan):
        super(DoubleConv, self).__init__()
        self.conv1 = convolution_block(in_chan, out_chan)
        self.conv2 = convolution_block(out_chan, out_chan)

    def forward(self, x):
        out = self.conv2(self.conv1(x))

        return out


class Down(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(Down, self).__init__()
        self.doubleconv = DoubleConv(in_chan, out_chan)
        self.down = nn.MaxPool3d(2)

    def forward(self, x):
        out = self.doubleconv(self.down(x))

        return out



def out_stage(in_chan):
    return nn.Sequential(
        nn.Conv3d(in_chan, 1, kernel_size=1)
    )


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.channels = 16
        self.enc1 = DoubleConv(1, self.channels)
        self.enc2 = Down(self.channels, self.channels * 2)
        self.enc3 = Down(self.channels * 2, self.channels * 4)
        self.enc4 = Down(self.channels * 4, self.channels * 8)
        self.enc5 = Down(self.channels * 8, self.channels * 8)

        self.dec4 = DoubleConv(self.channels * 8 + self.channels * 8, self.channels * 4)
        self.dec3 = DoubleConv(self.channels * 4 + self.channels * 4, self.channels * 2)
        self.dec2 = DoubleConv(self.channels * 2 + self.channels * 2, self.channels)
        self.dec1 = DoubleConv(self.channels + self.channels, self.channels)

        self.out = out_stage(self.channels)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec4 = self.dec4(
            torch.cat((enc4, up_sample3d(enc5)), dim=1))
        dec3 = self.dec3(
            torch.cat((enc3, up_sample3d(dec4)), dim=1))
        dec2 = self.dec2(
            torch.cat((enc2, up_sample3d(dec3)), dim=1))
        dec1 = self.dec1(
            torch.cat((enc1, up_sample3d(dec2)), dim=1))
        out = self.out(dec1)

        if self.training:
            return out
        else:
            return torch.sigmoid(out)


if __name__ == '__main__':
    net = UNet().cuda()
    torch.save(net.state_dict(), "UNet.pth.gz")
