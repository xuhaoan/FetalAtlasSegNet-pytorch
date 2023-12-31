import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from torch.autograd import Variable

def convolution_block(in_chan, out_chan, ksize=3, pad=1, stride=1, bias=False):
    """
    Convolution Block
    Convolution + Normalization + NonLinear
    """
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=ksize, padding=pad, stride=stride, bias=bias),
        nn.BatchNorm3d(out_chan),
        nn.PReLU()
    )


def up_sample3d(x, t, channel):
    """
    3D Up Sampling
    """
    return nn.ConvTranspose3d(channel, channel, kernel_size=3, stride=2)(x)

    # return nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)(x)
    # return F.interpolate(x, t.size()[2:], mode="trilinear", align_corners=False)


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


class Up(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(Up, self).__init__()
        self.trans_conv = nn.ConvTranspose3d(in_chan, in_chan, kernel_size=2, stride=2)
        self.activation = nn.PReLU()
        self.doubleconv = DoubleConv(in_chan * 2, out_chan)

    def forward(self, x1, x2):
        x1 = self.activation(self.trans_conv(x1))
        out = self.doubleconv(torch.cat((x1, x2), dim=1))

        return out


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
        self.GA_conv = MixConv(in_chan // 4, out_chan // 4)
        self.conv = convolution_block(out_chan + out_chan // 4, out_chan + out_chan // 4, ksize=1, pad=0, stride=1)

    def forward(self, seg, atlas):
        seg = self.seg_conv(seg)
        GA = self.GA_conv(atlas)
        att_map = torch.sigmoid(self.conv(torch.cat([seg, GA], dim=1)))

        return att_map


class GAPredictor(nn.Module):
    def __init__(self):
        super(GAPredictor, self).__init__()
        self.channels = 4

        self.enc1 = DoubleConv(1, self.channels)
        self.enc2 = Down(self.channels, self.channels * 2)
        self.enc3 = Down(self.channels * 2, self.channels * 4)
        self.enc4 = Down(self.channels * 4, self.channels * 8)
        self.enc5 = Down(self.channels * 8, self.channels * 8)

        self.dec4 = Up(self.channels * 8, self.channels * 4)
        self.dec3 = Up(self.channels * 4, self.channels * 2)
        self.dec2 = Up(self.channels * 2, self.channels)
        self.dec1 = Up(self.channels, self.channels)
        self.maxpool = nn.MaxPool3d(kernel_size=6, stride=6)
        self.dense1 = nn.Linear(in_features=16384, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec4 = self.dec4(enc5, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        GA = self.maxpool(dec1)
        GA = GA.view(GA.size(0), -1)
        GA = torch.nn.ELU()(self.dense1(GA))
        GA = torch.nn.ELU()(self.dense2(GA))

        feature_maps = (enc1, enc2, enc3, enc4, enc5, dec4, dec3, dec2, dec1)

        return feature_maps, GA


class Segmentor(nn.Module):
    def __init__(self):
        super(Segmentor, self).__init__()
        self.channels = 16
        self.jointchannels = 20
        self.enc1 = DoubleConv(1, self.channels)
        self.att1 = AttentionJoint(self.channels, self.channels)
        self.enc2 = Down(self.jointchannels, self.channels * 2)
        self.att2 = AttentionJoint(self.channels * 2, self.channels * 2)
        self.enc3 = Down(self.jointchannels * 2, self.channels * 4)
        self.att3 = AttentionJoint(self.channels * 4, self.channels * 4)
        self.enc4 = Down(self.jointchannels * 4, self.channels * 8)
        self.att4 = AttentionJoint(self.channels * 8, self.channels * 8)
        self.enc5 = Down(self.jointchannels * 8, self.channels * 8)
        self.att5 = AttentionJoint(self.channels * 8, self.channels * 8)

        self.dec4 = Up(self.jointchannels * 8, self.channels * 4)
        self.att6 = AttentionJoint(self.channels * 4, self.channels * 4)
        self.dec3 = Up(self.jointchannels * 4, self.channels * 2)
        self.att7 = AttentionJoint(self.channels * 2, self.channels * 2)
        self.dec2 = Up(self.jointchannels * 2, self.channels)
        self.att8 = AttentionJoint(self.channels, self.channels)
        self.dec1 = Up(self.jointchannels, self.channels)
        self.att9 = AttentionJoint(self.channels, self.channels)

        self.out = nn.Conv3d(self.jointchannels, 1, kernel_size=1)

    def forward(self, x, feature_maps):
        atlas_enc1, atlas_enc2, atlas_enc3, atlas_enc4, atlas_enc5, atlas_dec4, atlas_dec3, atlas_dec2, atlas_dec1 = feature_maps

        enc1 = self.enc1(x)
        att1 = self.att1(enc1, atlas_enc1)
        enc1 = torch.cat([enc1, atlas_enc1], dim=1)
        enc1 = torch.mul(enc1, att1)

        enc2 = self.enc2(enc1)
        att2 = self.att2(enc2, atlas_enc2)
        enc2 = torch.cat([enc2, atlas_enc2], dim=1)
        enc2 = torch.mul(enc2, att2)

        enc3 = self.enc3(enc2)
        att3 = self.att3(enc3, atlas_enc3)
        enc3 = torch.cat([enc3, atlas_enc3], dim=1)
        enc3 = torch.mul(enc3, att3)

        enc4 = self.enc4(enc3)
        att4 = self.att4(enc4, atlas_enc4)
        enc4 = torch.cat([enc4, atlas_enc4], dim=1)
        enc4 = torch.mul(enc4, att4)

        enc5 = self.enc5(enc4)
        att5 = self.att5(enc5, atlas_enc5)
        enc5 = torch.cat([enc5, atlas_enc5], dim=1)
        enc5 = torch.mul(enc5, att5)

        dec4 = self.dec4(enc5, enc4)
        att6 = self.att6(dec4, atlas_dec4)
        dec4 = torch.cat([dec4, atlas_dec4], dim=1)
        dec4 = torch.mul(dec4, att6)

        dec3 = self.dec3(dec4, enc3)
        att7 = self.att7(dec3, atlas_dec3)
        dec3 = torch.cat([dec3, atlas_dec3], dim=1)
        dec3 = torch.mul(dec3, att7)

        dec2 = self.dec2(dec3, enc2)
        att8 = self.att8(dec2, atlas_dec2)
        dec2 = torch.cat([dec2, atlas_dec2], dim=1)
        dec2 = torch.mul(dec2, att8)

        dec1 = self.dec1(dec2, enc1)
        att9 = self.att9(dec1, atlas_dec1)
        dec1 = torch.cat([dec1, atlas_dec1], dim=1)
        dec1 = torch.mul(dec1, att9)

        out = self.out(dec1)

        if not self.training:
            return torch.sigmoid(out)
        else:
            return out


class GASegNet(nn.Module):
    def __init__(self):
        super(GASegNet, self).__init__()
        self.GApredictor = GAPredictor()
        self.segmentor = Segmentor()

    def forward(self, x):
        input = x
        feature_maps, output_GA = self.GApredictor(input)
        output_seg = self.segmentor(input, feature_maps)

        return [output_seg, output_GA]

def loss_mae(predict, label, GA, weight=None):
    bce_loss = F.binary_cross_entropy_with_logits(predict[0], label, weight)
    mae = torch.nn.SmoothL1Loss(reduction='mean')(predict[1], GA)
    return bce_loss + mae

if __name__ == '__main__':
    device=torch.device('cpu')
    net = GASegNet().to(device)
    net.train()
    image=np.zeros([3,1,96,96,96]).astype(np.float32)
    image=Variable(torch.from_numpy(image))
    predict = net(image)
    label=image
    GA=Variable(torch.from_numpy(np.zeros([3,1,1]).astype(np.float32)))
    loss = loss_mae(predict, label, GA)
    torch.save(net.state_dict(), "GASegNet.pth.gz")
