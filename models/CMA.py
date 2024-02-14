import torch.nn as nn
import torch
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d


class CMA(nn.Module):
    def __init__(self, plane):
        super(CMA, self).__init__()
        inter_plane = plane // 2
        self.v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1))  # ,BatchNorm2d(plane)

    def forward(self, x):

        v = self.v(x)
        k = self.k(x)
        q = self.q(x)
        b, c, h, w = v.size()

        v = v.view(b, c, -1).permute(0, 2, 1)
        q = q.view(b, c, -1)
        k = k.view(b, c, -1).permute(0, 2, 1)
        FMCM = torch.bmm(q, k)
        AWM = self.softmax(FMCM)
        AV = torch.bmm(v, AWM)
        AV = AV.transpose(1, 2).contiguous()
        Output = self.conv_wg(AV)
        Output = self.bn_wg(Output)
        Output = Output.view(b, c, h, -1)
        Output = F.relu_(self.out(Output) + x)
        return Output

