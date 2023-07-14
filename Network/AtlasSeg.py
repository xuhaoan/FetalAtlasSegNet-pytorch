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


def up_sample3d(x, t):
    """
    3D Up Sampling
    """
    return nn.ConvTranspose3d(x.size()[2:], t.size()[2:], kernel_size=2, stride=2)(x)

    # return nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)(x)
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
        nn.Conv3d(in_chan, 1, kernel_size=1)
    )


class MixConv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(MixConv, self).__init__()
        kernel_size_group = [1, 3, 5, 7]
        self.grouped_conv = nn.ModuleList()
        for i in range(len(kernel_size_group)):
            self.grouped_conv.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_chan,
                        out_chan // 4,
                        kernel_size_group[i],
                        stride=1,
                        padding=(kernel_size_group[i] - 1) // 2,
                        bias=True
                    ),
                    nn.BatchNorm3d(out_chan // 4),
                    nn.PReLU()
                )
            )

    def forward(self, x):
        x = [conv(x) for conv in self.grouped_conv]
        x = torch.cat(x, dim=1)
        return x


class AttentionJoint(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionJoint, self).__init__()
        self.seg_conv = MixConv(in_chan, out_chan)
        self.atlas_conv = MixConv(in_chan, out_chan // 4)
        self.conv = nn.Conv3d(out_chan + out_chan // 4, out_chan)

    def forward(self, seg, atlas):
        seg = self.seg_conv(seg)
        atlas = self.atlas_conv(atlas)
        att_map = F.sigmoid(self.conv(torch.cat([seg, atlas], dim=1)))

        return att_map


class Attention(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Attention, self).__init__()
        self.atlas_conv = MixConv(in_chan, out_chan)
        self.conv = nn.Conv3d(out_chan, 1, kernel_size=1, padding=0, stride=1)

    def forward(self, atlas):
        atlas = self.atlas_conv(atlas)
        att_map = F.sigmoid(self.conv(atlas))

        return att_map

class AtlasConditioner(nn.Module):
    def __init__(self):
        super(AtlasConditioner, self).__init__()
        self.channels = 4

        self.enc1 = DoubleConv(2, self.channels)
        self.att1 = Attention(self.channels, self.channels)
        self.enc2 = Down(self.channels, self.channels * 2)
        self.att2 = Attention(self.channels * 2, self.channels * 2)
        self.enc3 = Down(self.channels * 2, self.channels * 4)
        self.att3 = Attention(self.channels * 4, self.channels * 4)
        self.enc4 = Down(self.channels * 4, self.channels * 8)
        self.att4 = Attention(self.channels * 8, self.channels * 8)
        self.enc5 = Down(self.channels * 8, self.channels * 8)
        self.att5 = Attention(self.channels * 8, self.channels * 8)

        self.dec4 = DoubleConv(self.channels * 8 + self.channels * 8, self.channels * 4)
        self.att6 = Attention(self.channels * 4, self.channels * 4)
        self.dec3 = DoubleConv(self.channels * 4 + self.channels * 4, self.channels * 2)
        self.att7 = Attention(self.channels * 2, self.channels * 2)
        self.dec2 = DoubleConv(self.channels * 2 + self.channels * 2, self.channels)
        self.att8 = Attention(self.channels, self.channels)
        self.dec1 = DoubleConv(self.channels + self.channels, self.channels)
        self.att9 = Attention(self.channels, self.channels)

    def forward(self, atlas_image, atlas_label):
        x = torch.cat((atlas_image, atlas_label), dim=1)
        enc1 = self.enc1(x)
        att1 = self.att1(enc1)
        enc2 = self.enc2(enc1)
        att2 = self.att2(enc2)
        enc3 = self.enc3(enc2)
        att3 = self.att3(enc3)
        enc4 = self.enc4(enc3)
        att4 = self.att4(enc4)
        enc5 = self.enc5(enc4)
        att5 = self.att5(enc5)

        dec4 = self.dec4(
            torch.cat((enc4, up_sample3d(enc5, enc4)), dim=1))
        att6 = self.att6(dec4)
        dec3 = self.dec3(
            torch.cat((enc3, up_sample3d(dec4, enc3)), dim=1))
        att7 = self.att7(dec3)
        dec2 = self.dec2(
            torch.cat((enc2, up_sample3d(dec3, enc2)), dim=1))
        att8 = self.att8(dec2)
        dec1 = self.dec1(
            torch.cat((enc1, up_sample3d(dec2, enc1)), dim=1))
        att9 = self.att9(dec1)

        attention_maps = (att1, att2, att3, att4, att5, att6, att7, att8, att9)

        return attention_maps


class AtlasSegmentor(nn.Module):
    def __init__(self):
        super(AtlasSegmentor, self).__init__()
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

    def forward(self, x, attention_maps):
        att1, att2, att3, att4, att5, att6, att7, att8, att9 = attention_maps
        enc1 = self.enc1(x)
        enc1 = torch.mul(enc1, att1)
        enc2 = self.enc2(enc1)
        enc2 = torch.mul(enc2, att2)
        enc3 = self.enc3(enc2)
        enc3 = torch.mul(enc3, att3)
        enc4 = self.enc4(enc3)
        enc4 = torch.mul(enc4, att4)
        enc5 = self.enc5(enc4)
        enc5 = torch.mul(enc5, att5)

        dec4 = self.dec4(
            torch.cat((enc4, up_sample3d(enc5, enc4)), dim=1))
        dec4 = torch.mul(dec4, att6)
        dec3 = self.dec3(
            torch.cat((enc3, up_sample3d(dec4, enc3)), dim=1))
        dec3 = torch.mul(dec3, att7)
        dec2 = self.dec2(
            torch.cat((enc2, up_sample3d(dec3, enc2)), dim=1))
        dec2 = torch.mul(dec2, att8)
        dec1 = self.dec1(
            torch.cat((enc1, up_sample3d(dec2, enc1)), dim=1))
        dec1 = torch.mul(dec1, att9)
        out = self.out(dec1)

        return torch.sigmoid(out)


class AtlasSegNet(nn.Module):
    def __init__(self):
        super(AtlasSegNet, self).__init__()
        self.conditioner = AtlasConditioner()
        self.segmentor = AtlasSegmentor()

    def forward(self, x):
        input, atlas_image, atlas_label = torch.split(x, [1, 1, 1], dim=1)
        attention_maps = self.conditioner(atlas_image, atlas_label)
        output = self.segmentor(input, attention_maps)

        return output


if __name__ == '__main__':
    net = AtlasSegNet().cuda()
    torch.save(net.state_dict(), "AtlasSegNet.pth.gz")
