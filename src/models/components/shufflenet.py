import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inplane, outplane, stride):
    return nn.Sequential(
        nn.Conv2d(inplane, outplane, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outplane),
        nn.ReLU(inplace=True),
    )


def conv1x1_bn(inplane, outplane):
    return nn.Sequential(
        nn.Conv2d(inplane, outplane, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outplane),
        nn.ReLU(inplace=True),
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inplane, outplane, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2], "stride must be 1 or 2"

        output_inc = outplane // 2

        if self.benchmodel == 1:
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inplane, output_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(output_inc, output_inc, 3, stride, 1, groups=output_inc, bias=False),
                nn.BatchNorm2d(output_inc),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(output_inc, output_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inplane, inplane, 3, stride, 1, groups=inplane, bias=False),
                nn.BatchNorm2d(inplane),
                # pw-linear
                nn.Conv2d(inplane, output_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inplane, output_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(output_inc, output_inc, 3, stride, 1, groups=output_inc, bias=False),
                nn.BatchNorm2d(output_inc),
                # pw-linear
                nn.Conv2d(output_inc, output_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.benchmodel == 1:
            x1 = x[:, : (x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2) :, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif self.benchmodel == 2:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    """ShuffleNet v2.0 implementation."""

    def __init__(self, input_size=224, num_classes=1000, width_mult=2.0):
        super(ShuffleNetV2, self).__init__()

        assert input_size % 32 == 0, "Input size must be divided by 32"
        # self.out_dim = 24

        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} width multiplier is not supported for 1x1 Grouped Convolutions.
                Should be in [0.5, 1.0, 1.5, 2.0]""".format(
                    width_mult
                )
            )

        # First convolution layer
        input_channel = self.stage_out_channels[1]
        self.firstconv = nn.Sequential(
            conv_bn(3, input_channel, 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.features = []
        # -- building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel

        # -- feature extraction layer as nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last layers
        self.conv_last = conv1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size / 32)))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], num_classes),
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x
