import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels

        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


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
    return nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)(x)
    # return F.interpolate(x, t.size()[2:], mode=mode, align_corners=False)


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
        nn.Conv3d(in_chan, 10, kernel_size=1)
    )


class SENet(nn.Module):
    def __init__(self):
        super(SENet, self).__init__()
        self.channels = 16
        self.enc1 = DoubleConv(1, self.channels)
        self.se1 = ChannelSpatialSELayer3D(self.channels)
        self.enc2 = Down(self.channels, self.channels * 2)
        self.se2 = ChannelSpatialSELayer3D(self.channels * 2)
        self.enc3 = Down(self.channels * 2, self.channels * 4)
        self.se3 = ChannelSpatialSELayer3D(self.channels * 4)
        self.enc4 = Down(self.channels * 4, self.channels * 8)
        self.se4 = ChannelSpatialSELayer3D(self.channels * 8)
        self.enc5 = Down(self.channels * 8, self.channels * 8)
        self.se5 = ChannelSpatialSELayer3D(self.channels * 8)

        self.dec4 = DoubleConv(self.channels * 8 + self.channels * 8, self.channels * 4)
        self.se6 = ChannelSpatialSELayer3D(self.channels * 4)
        self.dec3 = DoubleConv(self.channels * 4 + self.channels * 4, self.channels * 2)
        self.se7 = ChannelSpatialSELayer3D(self.channels * 2)
        self.dec2 = DoubleConv(self.channels * 2 + self.channels * 2, self.channels)
        self.se8 = ChannelSpatialSELayer3D(self.channels)
        self.dec1 = DoubleConv(self.channels + self.channels, self.channels)
        self.se9 = ChannelSpatialSELayer3D(self.channels)

        self.out = out_stage(self.channels)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1 = self.se1(enc1)
        enc2 = self.enc2(enc1)
        enc2 = self.se2(enc2)
        enc3 = self.enc3(enc2)
        enc3 = self.se3(enc3)
        enc4 = self.enc4(enc3)
        enc4 = self.se4(enc4)
        enc5 = self.enc5(enc4)
        enc5 = self.se5(enc5)

        dec4 = self.dec4(
            torch.cat((enc4, up_sample3d(enc5)), dim=1))
        dec4 = self.se6(dec4)
        dec3 = self.dec3(
            torch.cat((enc3, up_sample3d(dec4)), dim=1))
        dec3 = self.se7(dec3)
        dec2 = self.dec2(
            torch.cat((enc2, up_sample3d(dec3)), dim=1))
        dec2 = self.se8(dec2)
        dec1 = self.dec1(
            torch.cat((enc1, up_sample3d(dec2)), dim=1))
        dec1 = self.se9(dec1)
        out = self.out(dec1)

        if self.training:
            return out
        else:
            return torch.softmax(out, 1)


if __name__ == '__main__':
    net = SENet().cuda()
    torch.save(net.state_dict(), "SENet.pth.gz")
