import torch
import torch.nn as nn

from ...utils.lrw_utils import average_batch


class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN) https://arxiv.org/pdf/1803.01271.pdf."""

    def __init__(
        self,
        input_size,
        num_channels,
        num_classes,
        dropout,
        relu_type,
        options,
        dwpw=False,
    ) -> None:
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(
            input_size, num_channels, dropout, relu_type, options, dwpw
        )
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = average_batch
        self.has_aux_losses = False

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func(x, lengths, B)
        return self.tcn_output(x)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,
        dropout=0.2,
        relu_type="relu",
        options={},
        dwpw=False,
    ) -> None:
        super(TemporalConvNet, self).__init__()
        self.ksize = (
            options["kernel_size"][0]
            if isinstance(options["kernel_size"], list)
            else options["kernel_size"]
        )

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    self.ksize,
                    stride=1,
                    padding=(self.ksize - 1) * dilation_size,
                    dilation=dilation_size,
                    dropout=dropout,
                    symm_chomp=True,
                    no_padding=False,
                    relu_type=relu_type,
                    dwpw=dwpw,
                ),
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ----------------------------------------------------------------------------------------------------------------------


# -------------------- STANDARD VERSION (single branch) --------------------
class TemporalBlock(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        kernel_size,
        stride,
        padding,
        dilation,
        dropout=0.2,
        symm_chomp=False,  # symmetric chomp
        no_padding=False,  # no padding
        relu_type="relu",
        dwpw=False,  # double-wide-pathway
    ) -> None:
        super(TemporalBlock, self).__init__()

        self.no_padding = no_padding
        if self.no_padding:
            downsample_chomp_size = 2 * padding - 4
            padding = 1  # hacky way so that we can use 3 layers

        if dwpw:
            self.net = nn.Sequential(
                # -- first layer within the block
                # -- dw
                nn.Conv1d(
                    num_inputs,
                    num_inputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=num_inputs,
                    bias=False,
                ),
                nn.BatchNorm1d(num_inputs),
                Chomp1d(padding, True),  # symmetric chomp
                nn.PReLU(num_parameters=num_inputs)
                if relu_type == "prelu"
                else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d(num_inputs, num_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(num_outputs),
                nn.PReLU(num_parameters=num_outputs)
                if relu_type == "prelu"
                else nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                # -- second layer within the block
                # -- dw
                nn.Conv1d(
                    num_outputs,
                    num_outputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=num_outputs,
                    bias=False,
                ),
                nn.BatchNorm1d(num_outputs),
                Chomp1d(padding, True)
                if symm_chomp
                else Chomp1d(padding, False),  # TODO: different from implementation
                nn.PReLU(num_parameters=num_outputs)
                if relu_type == "prelu"
                else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d(num_outputs, num_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(num_outputs),
                nn.PReLU(num_parameters=num_outputs)
                if relu_type == "prelu"
                else nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        else:
            self.conv1 = nn.Conv1d(
                num_inputs,
                num_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            self.bn1 = nn.BatchNorm1d(num_outputs)
            self.chomp1 = Chomp1d(padding, symm_chomp) if not self.no_padding else None
            if relu_type == "prelu":
                self.relu1 = nn.PReLU(num_parameters=num_outputs)
            else:
                self.relu1 = nn.ReLU(inplace=True)
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(
                num_outputs,
                num_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            self.bn2 = nn.BatchNorm1d(num_outputs)
            self.chomp2 = Chomp1d(padding, symm_chomp) if not self.no_padding else None
            if relu_type == "prelu":
                self.relu2 = nn.PReLU(num_parameters=num_outputs)
            else:
                self.relu2 = nn.ReLU(inplace=True)
            self.dropout2 = nn.Dropout(dropout)

            if self.no_padding:
                self.net = nn.Sequential(
                    self.conv1,
                    self.bn1,
                    self.relu1,
                    self.dropout1,
                    self.conv2,
                    self.bn2,
                    self.relu2,
                    self.dropout2,
                )
            else:
                self.net = nn.Sequential(
                    self.conv1,
                    self.bn1,
                    self.chomp1,
                    self.relu1,
                    self.dropout1,
                    self.conv2,
                    self.bn2,
                    self.chomp2,
                    self.relu2,
                    self.dropout2,
                )

        self.downsample = (
            nn.Conv1d(num_inputs, num_outputs, 1) if num_inputs != num_outputs else None
        )
        if self.no_padding:
            self.downsample_chomp = Chomp1d(downsample_chomp_size, True)
        self.relu = nn.PReLU(num_parameters=num_outputs) if relu_type == "prelu" else nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        if self.no_padding:
            x = self.downsample_chomp(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# -------------------- CHOMP Block --------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp

        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "chomp size must be even for symmetric chomp"

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size // 2 : -self.chomp_size // 2].contiguous()
        else:
            return x[:, :, : -self.chomp_size].contiguous()
