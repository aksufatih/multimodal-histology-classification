import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel=3, stride=2):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel, stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(nn.Conv3d(inplanes, planes, 1, 2), nn.BatchNorm3d(planes))

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class FusionBlock(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.conv_ct = nn.Conv3d(inplanes, 1, 1)
        self.conv_pt = nn.Conv3d(inplanes, 1, 1)
        self.bn_ct = nn.BatchNorm3d(1)
        self.bn_pt = nn.BatchNorm3d(1)

    def forward(self, ct, pt):
        id_ct, id_pt = ct, pt

        ct = self.conv_ct(ct)
        pt = self.conv_pt(pt)
        ct = self.bn_ct(ct)
        pt = self.bn_pt(pt)

        fused = ct*pt

        return id_ct + fused, id_pt + fused


class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ct = UniNet()
        self.pt = UniNet()
        self.fusion1 = FusionBlock(16)
        self.fusion2 = FusionBlock(32)
        self.fusion3 = FusionBlock(64)
        self.fusion4 = FusionBlock(128)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, 2)


    def forward(self, x):

        ct, pt = x[0], x[1]

        ct = self.ct.layer1(ct)
        pt = self.pt.layer1(pt)

        ct, pt = self.fusion1(ct, pt)

        ct = self.ct.layer2(ct)
        pt = self.pt.layer2(pt)

        ct, pt = self.fusion2(ct, pt)

        ct = self.ct.layer3(ct)
        pt = self.pt.layer3(pt)

        ct, pt = self.fusion3(ct, pt)

        ct = self.ct.layer4(ct)
        pt = self.pt.layer4(pt)

        ct, pt = self.fusion4(ct, pt)

        out = torch.cat((ct, pt), dim=1)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out

class UniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicBlock(1, 16)
        self.layer2 = BasicBlock(16, 32)
        self.layer3 = BasicBlock(32, 64)
        self.layer4 = BasicBlock(64, 128)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = self.avgpool(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out

