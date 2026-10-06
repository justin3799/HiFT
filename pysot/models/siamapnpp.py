# SiamAPN++ network, inference only.
#
# Copied from the authors' release (github.com/vision4robotics/SiamAPN,
# SiamAPN++/pysot/models: backbone/alexnet.py, utile_adapn.py and the track
# path of model_builder_adapn.py), Apache License 2.0. Layer definitions and
# forward passes are unchanged so the released weights (general++.pth) load
# identically; training-only losses and label assignment are left out.
#
# Fu, C., Cao, Z., Li, Y., Ye, J. and Feng, C. (2021). Siamese Anchor Proposal
# Network for High-Speed Aerial Tracking. ICRA. / Cao, Z., Fu, C., Ye, J.,
# Li, B. and Li, Y. (2021). SiamAPN++: Siamese Attentional Aggregation Network
# for Real-Time UAV Tracking. IROS.
#
# Note: the released checkpoint also holds weights for layers this released
# code does not build (e.g. grader.point, grader.channel, grader.fusion2/3,
# adcat.bat/gamma2); like the authors' test.py, loading ignores them.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch as t
import torch.nn as nn
import torch.nn.functional as F

APN_CHANNEL = 384
CLS_LOC_CHANNEL = 256
GROUP_CHANNEL = 32


class AlexNet(nn.Module):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self):
        super(AlexNet, self).__init__()
        c = AlexNet.configs
        self.layer1 = nn.Sequential(nn.Conv2d(c[0], c[1], kernel_size=11, stride=2),
                                    nn.BatchNorm2d(c[1]), nn.MaxPool2d(kernel_size=3, stride=2),
                                    nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(c[1], c[2], kernel_size=5), nn.BatchNorm2d(c[2]),
                                    nn.MaxPool2d(kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(c[2], c[3], kernel_size=3), nn.BatchNorm2d(c[3]),
                                    nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(c[3], c[4], kernel_size=3), nn.BatchNorm2d(c[4]),
                                    nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(nn.Conv2d(c[4], c[5], kernel_size=3), nn.BatchNorm2d(c[5]))

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(x)))
        x1 = self.layer4(x)
        return x1, self.layer5(x1)


def xcorr_depthwise(x, kernel):
    batch, channel = kernel.size(0), kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    return out.view(batch, channel, out.size(2), out.size(3))


def conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def conv_gn_relu(c):
    return [nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1), nn.GroupNorm(GROUP_CHANNEL, c),
            nn.ReLU(inplace=True)]


class selfpointbranch(nn.Module):
    def __init__(self, in_dim):
        super(selfpointbranch, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(t.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)
        k = self.key_conv(x).view(b, -1, w * h)
        attention = self.softmax(t.bmm(q, k))
        v = self.value_conv(x).view(b, -1, w * h)
        out = t.bmm(v, attention.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x


class selfchannelbranch(nn.Module):
    def __init__(self, in_dim):
        super(selfchannelbranch, self).__init__()
        # defined (and in the checkpoint) but unused in the released forward pass
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 256, kernel_size=7, stride=1),
                                   nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.gamma2 = nn.Parameter(t.zeros(1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        return x + self.gamma2 * self.sigmoid(max_out + avg_out) * x


class adcat(nn.Module):
    def __init__(self, in_dim):
        super(adcat, self).__init__()
        self.add = nn.ConvTranspose2d(in_dim * 2, in_dim, kernel_size=1, stride=1)
        self.fc1 = nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)
        self.gamma1 = nn.Parameter(t.zeros(1))
        self.gamma3 = nn.Parameter(t.zeros(1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, z):
        w = self.sigmoid(self.fc2(self.relu1(self.fc1(self.avg_pool(z)))))
        c2 = self.relu1(self.add(t.cat((x, z), 1)))
        return x + self.gamma1 * c2 + self.gamma3 * w * x


class APN(nn.Module):
    """Anchor proposal network: per-cell anchor shapes from the similarity map."""

    def __init__(self):
        super(APN, self).__init__()
        cr = CLS_LOC_CHANNEL
        self.conv_shape = conv_bn_relu(cr, cr)
        self.anchor = nn.Conv2d(cr, 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv_bn_relu(APN_CHANNEL, cr)
        self.conv5 = conv_bn_relu(cr, cr)
        self.conv6 = conv_bn_relu(cr, cr)
        self.adcat = adcat(256)

    def forward(self, x, z):
        res2 = self.conv3(xcorr_depthwise(x[0], z[0]))
        ress = xcorr_depthwise(self.conv5(x[1]), self.conv6(z[1]))
        ress = self.adcat(ress, res2)
        return self.anchor(self.conv_shape(ress)), ress


class clsandloc(nn.Module):
    """Classification (three branches) and anchor refinement."""

    def __init__(self):
        super(clsandloc, self).__init__()
        c = CLS_LOC_CHANNEL
        self.conv1 = conv_bn_relu(c, c)
        self.conv2 = conv_bn_relu(c, c)
        self.conv4 = conv_bn_relu(c, c)
        self.convloc = nn.Sequential(*sum([conv_gn_relu(c) for _ in range(4)], []),
                                     nn.Conv2d(c, 4, kernel_size=3, stride=1, padding=1))
        self.convcls = nn.Sequential(*sum([conv_gn_relu(c) for _ in range(2)], []))
        self.channel = selfchannelbranch(c)
        self.point = selfpointbranch(c)
        self.cls1 = nn.Conv2d(c, 2, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(c, 2, kernel_size=3, stride=1, padding=1)
        self.cls3 = nn.Conv2d(c, 1, kernel_size=3, stride=1, padding=1)
        self.adcat = adcat(c)

    def forward(self, x, z, ress):
        res = xcorr_depthwise(self.conv1(x[1]), self.conv2(z[1]))
        channel = self.conv4(self.channel(self.point(res)))
        res = self.adcat(channel, ress)
        cls = self.convcls(res)
        return self.cls1(cls), self.cls2(cls), self.cls3(cls), self.convloc(res)


class SiamAPNpp(nn.Module):
    """Same attribute names as ModelBuilderADAPN, so released weights load as-is."""

    def __init__(self):
        super(SiamAPNpp, self).__init__()
        self.backbone = AlexNet()
        self.grader = APN()
        self.new = clsandloc()

    def forward_head(self, xf, zf):
        """Batched track step: returns anchor shapes and the four head outputs."""
        anchors, ress = self.grader(xf, zf)
        cls1, cls2, cls3, loc = self.new(xf, zf, ress)
        return {'anchors': anchors, 'cls1': cls1, 'cls2': cls2, 'cls3': cls3, 'loc': loc}
