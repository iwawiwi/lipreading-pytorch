# -- Class for main block for LRW network
import math

import numpy as np
import torch
import torch.nn as nn

from ...utils.lrw_utils import threeD_to_2D_tensor
from .resnet import BasicBlock, ResNet
from .shufflenet import ShuffleNetV2
from .temporal_conv_net import TCN


class LRWTCNet(nn.Module):
    """LRWTCNet."""

    def __init__(
        self,
        visual: bool = True,
        backbone: str = "resnet",
        hidden_size: int = 256,
        num_classes=500,
        relu_type="prelu",
        tcn_options={},
        width_mult=1.0,
        extract_feats=False,
    ) -> None:
        super(LRWTCNet, self).__init__()
        self.extract_feats = extract_feats
        self.backbone = backbone
        self.visual = visual

        if self.visual:
            if self.backbone == "resnet":
                self.frontend_out = 64
                self.backend_out = 256
                self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
            elif self.backbone == "shufflenet":
                assert width_mult in [
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                ], "width_mult must be 0.5, 1.0, 1.5, or 2.0"
                shufflenet = ShuffleNetV2(input_size=96, width_mult=width_mult)
                self.trunk = nn.Sequential(
                    shufflenet.features,
                    shufflenet.conv_last,
                    shufflenet.globalpool,
                )
                self.frontend_out = 24
                self.backend_out = 1024 if width_mult != 2.0 else 2048
                self.stage_out_channels = shufflenet.stage_out_channels[-1]

            # -- using 3d convolution as front end
            frontend_relu = (
                nn.PReLU(num_parameters=self.frontend_out) if relu_type == "prelu" else nn.ReLU()
            )
            self.frontend3D = nn.Sequential(
                nn.Conv3d(
                    1,
                    self.frontend_out,
                    kernel_size=(5, 7, 7),
                    stride=(1, 2, 2),
                    padding=(2, 3, 3),
                    bias=False,
                ),
                nn.BatchNorm3d(self.frontend_out),
                frontend_relu,
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            )
        # -- audio backend is not implemented yet
        else:
            raise NotImplementedError("Audio features are not implemented yet")

        # -- using TCN as classifier
        tcn_class = TCN if isinstance(tcn_options["kernel_size"], int) else NotImplemented
        self.tcn = tcn_class(
            input_size=self.backend_out,
            num_channels=[
                hidden_size * 1
                if isinstance(tcn_options["kernel_size"], int)
                else len(tcn_options["kernel_size"]) * tcn_options["width_mult"]
            ]
            * tcn_options["num_layers"],
            num_classes=num_classes,
            dropout=tcn_options["dropout"],
            relu_type=relu_type,
            options=tcn_options,
            dwpw=tcn_options["dwpw"],
        )

        self.__init_weights_random()

    def forward(self, x):
        if self.visual:
            (
                B,
                C,
                T,
                H,
                W,
            ) = x.size()
            x = self.frontend3D(x)
            Tnew = x.shape[2]  # output should be B x C2 x Tnew x H x W
            x = threeD_to_2D_tensor(x)
            x = self.trunk(x)
            if self.backbone_type == "shufflenet":
                x = x.view(-1, self.stage_out_channels)
            x = x.view(B, Tnew, x.size(1))
        else:
            raise NotImplementedError("Audio features are not implemented yet")

        return x if self.extract_feats else NotImplemented

    def __init_weights_random(self):
        use_sqrt = True
        if use_sqrt:

            def f(n):
                return math.sqrt(2.0 / float(n))

        else:

            def f(n):
                return 2.0 / float(n)

        # -- init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif (
                isinstance(m, nn.BatchNorm3d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.BatchNorm1d)
            ):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))  # TODO: why not inplace?
