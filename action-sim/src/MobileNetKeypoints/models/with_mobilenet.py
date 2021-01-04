import torch
from torch import nn

from src.MobileNetKeypoints.modules.conv import conv, conv_pw, conv_dw_no_bn, conv1


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]

class InvertedResidual(nn.Module):
    def __init__(self, in_channels,out_channels,stride,expand_ratio):
        super(InvertedResidual,self).__init__()

        self.stride=stride
        assert stride in [1,2]
        
        hidden_dim = round(in_channels*expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                #depthwise convolution
                nn.Conv2d(hidden_dim,hidden_dim,3,stride,1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #pointwise convolution
                nn.Conv2d(hidden_dim,out_channels,1,1,0,bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                #pointwise convolution
                nn.Conv2d(in_channels,hidden_dim,1,1,0,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #depthwise convolution
                nn.Conv2d(hidden_dim,hidden_dim,3,stride,1,groups=hidden_dim,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #pointwise convolution
                nn.Conv2d(hidden_dim,out_channels,1,1,0,bias=False),
                nn.BatchNorm2d(out_channels),
            )
            
    def forward(self,x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        block = InvertedResidual
        in_chan = 32
        out_chan = 512
        width_mult = 1.0
        in_chan = int(in_chan * width_mult)
        self.out_chan = int(out_chan * width_mult) if width_mult > 1.0 else out_chan
        
        self.model = nn.Sequential(
            conv1(3,in_chan,2),
            block(in_chan, 32, 1, expand_ratio=1),
            #block(16, 24, 1, expand_ratio=6),
            #block(24, 24, 1, expand_ratio=6),
            block(32, 32, 2, expand_ratio=6),
            block(32, 32, 1, expand_ratio=6),
            block(32, 32, 2, expand_ratio=6),
            block(32, 64, 1, expand_ratio=6),
            block(64, 64, 1, expand_ratio=6),
            block(64, 64, 1, expand_ratio=6),
            block(64, 64, 1, expand_ratio=6),
            block(64, 96, 1, expand_ratio=6),
            block(96, 96, 1, expand_ratio=6),
            block(96, 96, 1, expand_ratio=6),
            block(96, 160, 1, expand_ratio=6),
            block(160, 160, 1, expand_ratio=6),
            block(160, 320, 1, expand_ratio=6),
            conv_pw(320,512),
        )
        """
        self.model = nn.Sequential(
            conv_bn(3,in_chan,2),
            block(in_chan, 16, 1, expand_ratio=1),
            block(16, 24, 1, expand_ratio=6),
            block(24, 24, 1, expand_ratio=6),
            block(24, 32, 2, expand_ratio=6),
            block(32, 32, 1, expand_ratio=6),
            block(32, 32, 2, expand_ratio=6),
            block(32, 64, 1, expand_ratio=6),
            block(64, 64, 1, expand_ratio=6),
            block(64, 64, 1, expand_ratio=6),
            block(64, 64, 1, expand_ratio=6),
            block(64, 96, 1, expand_ratio=6),
            block(96, 96, 1, expand_ratio=6),
            block(96, 96, 1, expand_ratio=6),
            block(96, 160, 1, expand_ratio=6),
            block(160, 160, 1, expand_ratio=6),
            block(160, 320, 1, expand_ratio=6),
            conv_pw(320,512),
        )
        """
        self.cpm = Cpm(512, num_channels)
        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))
    def forward(self, x):
        backbone_features = self.model(x)
        #print("Backbone", backbone_features.shape)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))
            #print(len(stages_output))
        return stages_output
