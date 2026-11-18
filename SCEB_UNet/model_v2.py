"""
ACC-UNet architecture using PyTorch
"""

from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SimAM(torch.nn.Module):
    """Parameter-free spatial attention from SimAM (ICML 2021)."""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        stats = x.float() if getattr(self, "force_float32_statistics", False) else x
        energy = self.energy(stats, self.e_lambda)
        gate = self.activation(energy)
        return x * (gate.to(dtype=x.dtype)
                    if getattr(self, "force_float32_statistics", False) else gate)

    @staticmethod
    def energy(x, e_lambda=1e-4):
        height, width = x.shape[-2:]
        normalizer = max(height * width - 1, 1)
        deviation = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        return deviation / (
            4 * (deviation.sum(dim=(2, 3), keepdim=True) / normalizer + e_lambda)
        ) + 0.5


class ScaleCoupledSimAM(torch.nn.Module):
    """Couple SimAM's global energy to the HANC neighborhood scale.

    The learnable scalar is zero-initialized, so this module starts as exact
    SimAM. Training may then calibrate the energy temperature according to the
    corresponding HANC level without material spatial-computation overhead.
    """

    def __init__(self, context_scale, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.context_scale = max(1, int(context_scale))
        self.scale_temperature = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        stats = x.float() if getattr(self, "force_float32_statistics", False) else x
        energy = SimAM.energy(stats, self.e_lambda)
        log_scale = energy.new_tensor(float(self.context_scale)).log()
        temperature = torch.exp(torch.tanh(self.scale_temperature) * log_scale)
        gate = torch.sigmoid(energy / temperature)
        return x * (gate.to(dtype=x.dtype)
                    if getattr(self, "force_float32_statistics", False) else gate)


class DualContextSimAM(torch.nn.Module):
    """Dual-context calibrated SimAM (DC-SimAM).

    The original SimAM energy provides image-wide contrast.  A local energy
    term, computed from a 3x3 neighbourhood without learned convolutions,
    complements it around fine structures and boundaries.  Both the local
    contribution and temperature are zero-initialized so the module begins as
    exact SimAM and learns only the correction required by the data.
    """

    def __init__(
        self,
        context_scale=1,
        e_lambda=1e-4,
        local_limit=1.0,
        temperature_limit=1.0,
    ):
        super().__init__()
        self.e_lambda = float(e_lambda)
        self.context_scale = max(1, int(context_scale))
        self.local_limit = float(local_limit)
        self.temperature_limit = float(temperature_limit)
        self.local_gain = torch.nn.Parameter(torch.zeros(1))
        self.temperature_gain = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Float32 statistics avoid variance underflow when AMP is enabled.
        stats = x.float()
        global_energy = SimAM.energy(stats, self.e_lambda)

        local_mean = F.avg_pool2d(
            stats, kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        )
        local_deviation = (stats - local_mean).pow(2)
        local_variance = F.avg_pool2d(
            local_deviation, kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        )
        local_energy = local_deviation / (
            4.0 * (local_variance + self.e_lambda)
        ) + 0.5

        energy = global_energy + self.local_limit * torch.tanh(self.local_gain) * (
            local_energy - 0.5
        )
        log_scale = energy.new_tensor(float(self.context_scale)).log()
        temperature = torch.exp(
            self.temperature_limit
            * torch.tanh(self.temperature_gain)
            * log_scale
        )
        gate = torch.sigmoid(energy / temperature).to(dtype=x.dtype)
        return x * gate


class BoundedDualContextSimAM(DualContextSimAM):
    """Risk-controlled local/global SimAM refinement.

    The learned local and scale corrections are capped at five percent of the
    unconstrained DC-SimAM range.  ``scale_temperature`` preserves the trained
    ScaleCoupledSimAM path of the predecessor; the two zero-initialized gains
    then add only bounded residual corrections.  This makes checkpoint
    migration exactly function-preserving instead of resetting a useful scale
    calibration learned on the same training split.
    """

    def __init__(self, context_scale=1, e_lambda=1e-4):
        super().__init__(
            context_scale=context_scale,
            e_lambda=e_lambda,
            local_limit=0.05,
            temperature_limit=0.05,
        )
        self.scale_temperature = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        stats = x.float()
        global_energy = SimAM.energy(stats, self.e_lambda)
        local_mean = F.avg_pool2d(
            stats, kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        )
        local_deviation = (stats - local_mean).pow(2)
        local_variance = F.avg_pool2d(
            local_deviation, kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        )
        local_energy = local_deviation / (
            4.0 * (local_variance + self.e_lambda)
        ) + 0.5
        energy = global_energy + self.local_limit * torch.tanh(self.local_gain) * (
            local_energy - 0.5
        )

        log_scale = energy.new_tensor(float(self.context_scale)).log()
        inherited_temperature = torch.tanh(self.scale_temperature)
        residual_temperature = self.temperature_limit * torch.tanh(
            self.temperature_gain
        )
        temperature = torch.exp(
            (inherited_temperature + residual_temperature) * log_scale
        )
        gate = torch.sigmoid(energy / temperature).to(dtype=x.dtype)
        return x * gate


class SingleTemperatureBoundedDualContextSimAM(torch.nn.Module):
    """Bounded dual-context SimAM with one identifiable scale temperature.

    The predecessor used two independently optimized scalars on the same
    ``log(context_scale)`` term.  This candidate retains the bounded local
    correction and the established scale temperature while removing the
    redundant residual temperature degree of freedom.
    """

    def __init__(self, context_scale=1, e_lambda=1e-4, local_limit=0.05):
        super().__init__()
        self.e_lambda = float(e_lambda)
        self.context_scale = max(1, int(context_scale))
        self.local_limit = float(local_limit)
        self.local_gain = torch.nn.Parameter(torch.zeros(1))
        self.scale_temperature = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        stats = x.float()
        global_energy = SimAM.energy(stats, self.e_lambda)
        local_mean = F.avg_pool2d(
            stats, kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        )
        local_deviation = (stats - local_mean).pow(2)
        local_variance = F.avg_pool2d(
            local_deviation, kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        )
        local_energy = local_deviation / (
            4.0 * (local_variance + self.e_lambda)
        ) + 0.5
        energy = global_energy + self.local_limit * torch.tanh(self.local_gain) * (
            local_energy - 0.5
        )
        log_scale = energy.new_tensor(float(self.context_scale)).log()
        temperature = torch.exp(
            torch.tanh(self.scale_temperature) * log_scale
        )
        gate = torch.sigmoid(energy / temperature).to(dtype=x.dtype)
        return x * gate


class ChannelSELayer(torch.nn.Module):
    """Use the original SE layer or SimAM at an ACC-UNet attention site."""

    def __init__(
        self,
        num_channels,
        attention_type="se",
        scale_coupled=False,
        context_scale=1,
    ):
        super().__init__()
        if attention_type not in {
            "se", "simam", "dc_simam", "bdc_simam", "bdc_simam_single"
        }:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        self.attention_type = attention_type
        if attention_type in {
            "simam", "dc_simam", "bdc_simam", "bdc_simam_single"
        }:
            self.simam = (
                SingleTemperatureBoundedDualContextSimAM(
                    context_scale if scale_coupled else 1
                )
                if attention_type == "bdc_simam_single"
                else (
                    BoundedDualContextSimAM(context_scale if scale_coupled else 1)
                    if attention_type == "bdc_simam"
                    else (
                        DualContextSimAM(context_scale if scale_coupled else 1)
                        if attention_type == "dc_simam"
                        else (
                            ScaleCoupledSimAM(context_scale)
                            if scale_coupled else SimAM()
                        )
                    )
                )
            )
            return

        num_channels_reduced = max(1, num_channels // 8)
        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)

    def forward(self, inp):
        if self.attention_type in {
            "simam", "dc_simam", "bdc_simam", "bdc_simam_single"
        }:
            return self.simam(inp)

        batch_size, num_channels, _, _ = inp.size()
        out = self.act(self.fc1(self.gp_avg_pool(inp).reshape(batch_size, num_channels)))
        out = self.sigmoid(self.fc2(out))
        out = torch.mul(inp, out.reshape(batch_size, num_channels, 1, 1))
        return self.act(self.bn(out))


class UncertaintyGuidedBoundaryFusion(torch.nn.Module):
    """Refine a skip connection only around uncertain predicted boundaries.

    A coarse decoder prediction supplies semantic uncertainty, while a shallow
    auxiliary edge head supplies local boundary evidence.  The module keeps a
    residual path for both inputs, so disabling or weakly activating the gate
    cannot erase the original ACC-UNet features.
    """

    def __init__(self, hidden_channels=8, residual_limit=1.0):
        super().__init__()
        self.residual_limit = float(residual_limit)
        if not 0.0 < self.residual_limit <= 1.0:
            raise ValueError("UGBF residual_limit must be in (0, 1]")
        self.gate = torch.nn.Sequential(
            torch.nn.Conv2d(6, hidden_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(1, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        # Start as an exact identity mapping, then learn signed enhancement or
        # suppression around boundaries.
        torch.nn.init.zeros_(self.gate[-1].weight)
        torch.nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _resize(tensor, size):
        if tensor.shape[-2:] == size:
            return tensor
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    def forward(self, decoder, skip, coarse_logits, edge_logits):
        if decoder.shape[-2:] != skip.shape[-2:]:
            raise ValueError(
                f"Decoder/skip spatial mismatch: {decoder.shape[-2:]} vs {skip.shape[-2:]}"
            )

        size = skip.shape[-2:]
        # The auxiliary heads provide a supervised spatial prior. Detaching the
        # prior prevents the main segmentation objective from gaming the
        # uncertainty/edge maps, which otherwise overfits the small GlaS split.
        probability = torch.sigmoid(self._resize(coarse_logits, size)).detach()
        uncertainty = 4.0 * probability * (1.0 - probability)
        edge_probability = torch.sigmoid(self._resize(edge_logits, size)).detach()
        boundary_focus = uncertainty * edge_probability

        descriptors = torch.cat(
            [
                decoder.mean(dim=1, keepdim=True),
                decoder.amax(dim=1, keepdim=True),
                skip.mean(dim=1, keepdim=True),
                skip.amax(dim=1, keepdim=True),
                uncertainty,
                edge_probability,
            ],
            dim=1,
        )
        decoder_gate, skip_gate = torch.tanh(self.gate(descriptors)).chunk(2, dim=1)
        decoder = decoder * (
            1.0 + self.residual_limit * decoder_gate * boundary_focus
        )
        skip = skip * (
            1.0 + self.residual_limit * skip_gate * boundary_focus
        )
        return torch.cat([decoder, skip], dim=1), boundary_focus



class HANCLayer(torch.nn.Module):
    """
    Implements Hierarchical Aggregation of Neighborhood Context operation
    """

    def __init__(self, in_chnl, out_chnl, k):
        """
        Initialization

        Args:
            in_chnl (int): number of input channels
            out_chnl (int): number of output channels
            k (int): value of k in HANC
        """

        super(HANCLayer, self).__init__()

        if k not in {1, 2, 3, 4, 5}:
            raise ValueError("HANC k must be an integer from 1 to 5")
        self.k = k
        self.pool_scales = tuple(2 ** index for index in range(1, k))

        self.cnv = torch.nn.Conv2d((2 * k - 1) * in_chnl, out_chnl, kernel_size=(1, 1))
        self.act = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm2d(out_chnl)


    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        average_branches = []
        maximum_branches = []
        for scale in self.pool_scales:
            average = F.interpolate(
                F.avg_pool2d(inp, scale), size=(H, W), mode="nearest"
            )
            maximum = F.interpolate(
                F.max_pool2d(inp, scale), size=(H, W), mode="nearest"
            )
            average_branches.append(average)
            maximum_branches.append(maximum)

        branches = [inp, *average_branches, *maximum_branches]
        x = torch.concat(branches, dim=2) if len(branches) > 1 else inp

        x = x.reshape(batch_size, num_channels * (2 * self.k - 1), H, W)

        x = self.act(self.bn(self.cnv(x)))

        return x



class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        activation="LeakyReLU",
        attention_type="se",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters, attention_type)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.sqe(self.activation(x))


class Conv2d_channel(torch.nn.Module):
    """
    2D pointwise Convolutional layers        
    """

    def __init__(self, num_in_filters, num_out_filters, attention_type="se"):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=(1, 1),
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters, attention_type)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        return self.sqe(self.activation(x))



class HANCBlock(torch.nn.Module):
    """
    Encapsulates HANC block
    """

    def __init__(
        self,
        n_filts,
        out_channels,
        k=3,
        inv_fctr=3,
        attention_type="se",
        context_energy=False,
        context_residual=None,
        scale_calibration=None,
    ):
        """
        Initialization

        Args:
            n_filts (int): number of filters
            out_channels (int): number of output channel
            activation (str, optional): activation function. Defaults to 'LeakyReLU'.
            k (int, optional): k in HANC. Defaults to 1.
            inv_fctr (int, optional): inv_fctr in HANC. Defaults to 4.
        """

        super().__init__()

        self.conv1 = torch.nn.Conv2d(n_filts, n_filts * inv_fctr, kernel_size=1)
        self.norm1 = torch.nn.BatchNorm2d(n_filts * inv_fctr)

        self.conv2 = torch.nn.Conv2d(
            n_filts * inv_fctr,
            n_filts * inv_fctr,
            kernel_size=3,
            padding=1,
            groups=n_filts * inv_fctr,
        )
        self.norm2 = torch.nn.BatchNorm2d(n_filts * inv_fctr)

        if context_residual is None:
            context_residual = context_energy
        if scale_calibration is None:
            scale_calibration = context_energy

        self.hnc = HANCLayer(n_filts * inv_fctr, n_filts, k)
        self.context_residual = bool(context_residual)
        self.scale_calibration = bool(scale_calibration)
        if self.context_residual:
            # Controls how much HANC context is injected when it disagrees with
            # the identity feature. Zero initialization reproduces ACC exactly.
            self.context_gain = torch.nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("context_gain", None)

        self.norm = torch.nn.BatchNorm2d(n_filts)

        self.conv3 = torch.nn.Conv2d(n_filts, out_channels, kernel_size=1)
        self.norm3 = torch.nn.BatchNorm2d(out_channels)

        self.sqe = ChannelSELayer(
            out_channels,
            attention_type,
            scale_coupled=(
                self.scale_calibration
                and attention_type in {
                    "simam", "dc_simam", "bdc_simam", "bdc_simam_single"
                }
                and k > 1
            ),
            context_scale=k,
        )

        self.activation = torch.nn.LeakyReLU()


    def forward(self, inp):

        x = self.conv1(inp)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.hnc(x)
        if self.context_residual:
            stable_stats = getattr(self, "force_float32_statistics", False)
            disagreement = ((x.float() - inp.float()).pow(2)
                            if stable_stats else (x - inp).pow(2))
            normalizer = max(x.shape[-2] * x.shape[-1] - 1, 1)
            variance = disagreement.sum(dim=(2, 3), keepdim=True) / normalizer
            energy = disagreement / (4.0 * (variance + 1e-4)) + 0.5
            gate = 2.0 * torch.sigmoid(energy) - 1.0
            modulation = 1.0 + torch.tanh(self.context_gain) * gate
            # Keep scalar residual modulation in FP32 rather than rounding
            # small learned corrections back to one in BF16.
            x = x * modulation

        x = self.norm(x + inp)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.sqe(x)

        return x



class ResPath(torch.nn.Module):
    """
    Implements ResPath-like modified skip connection

    """

    def __init__(self, in_chnls, n_lvl, attention_type="se"):
        """
        Initialization

        Args:
            in_chnls (int): number of input channels
            n_lvl (int): number of blocks or levels
        """

        super(ResPath, self).__init__()

        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        self.sqes = torch.nn.ModuleList([])

        self.bn = torch.nn.BatchNorm2d(in_chnls)
        self.act = torch.nn.LeakyReLU()
        self.sqe = torch.nn.BatchNorm2d(in_chnls)

        for i in range(n_lvl):
            self.convs.append(
                torch.nn.Conv2d(in_chnls, in_chnls, kernel_size=(3, 3), padding=1)
            )
            self.bns.append(torch.nn.BatchNorm2d(in_chnls))
            self.sqes.append(ChannelSELayer(in_chnls, attention_type))


    def forward(self, x):

        for i in range(len(self.convs)):
            x = x + self.sqes[i](self.act(self.bns[i](self.convs[i](x))))

        return self.sqe(self.act(self.bn(x)))



class MLFC(torch.nn.Module):
    """
    Implements Multi Level Feature Compilation

    """

    def __init__(
        self,
        in_filters1,
        in_filters2,
        in_filters3,
        in_filters4,
        lenn=1,
        attention_type="se",
    ):
        """
        Initialization

        Args:
            in_filters1 (int): number of channels in the first level
            in_filters2 (int): number of channels in the second level
            in_filters3 (int): number of channels in the third level
            in_filters4 (int): number of channels in the fourth level
            lenn (int, optional): number of repeats. Defaults to 1.
        """

        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.in_filters = (
            in_filters1 + in_filters2 + in_filters3 + in_filters4
        )  # total number of channels

        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling
        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):
            self.cnv_blks1.append(
                Conv2d_batchnorm(self.in_filters, in_filters1, (1, 1), attention_type=attention_type)
            )
            self.cnv_mrg1.append(Conv2d_batchnorm(2 * in_filters1, in_filters1, (1, 1), attention_type=attention_type))
            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))
            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))

            self.cnv_blks2.append(
                Conv2d_batchnorm(self.in_filters, in_filters2, (1, 1), attention_type=attention_type)
            )
            self.cnv_mrg2.append(Conv2d_batchnorm(2 * in_filters2, in_filters2, (1, 1), attention_type=attention_type))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                Conv2d_batchnorm(self.in_filters, in_filters3, (1, 1), attention_type=attention_type)
            )
            self.cnv_mrg3.append(Conv2d_batchnorm(2 * in_filters3, in_filters3, (1, 1), attention_type=attention_type))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

            self.cnv_blks4.append(
                Conv2d_batchnorm(self.in_filters, in_filters4, (1, 1), attention_type=attention_type)
            )
            self.cnv_mrg4.append(Conv2d_batchnorm(2 * in_filters4, in_filters4, (1, 1), attention_type=attention_type))
            self.bns4.append(torch.nn.BatchNorm2d(in_filters4))
            self.bns_mrg4.append(torch.nn.BatchNorm2d(in_filters4))

        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1, attention_type)
        self.sqe2 = ChannelSELayer(in_filters2, attention_type)
        self.sqe3 = ChannelSELayer(in_filters3, attention_type)
        self.sqe4 = ChannelSELayer(in_filters4, attention_type)


    def forward(self, x1, x2, x3, x4):

        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape
        _, _, h4, w4 = x4.shape

        for i in range(len(self.cnv_blks1)):
            x_c1 = self.act(
                self.bns1[i](
                    self.cnv_blks1[i](
                        torch.cat(
                            [
                                x1,
                                self.no_param_up(x2),
                                self.no_param_up(self.no_param_up(x3)),
                                self.no_param_up(self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c2 = self.act(
                self.bns2[i](
                    self.cnv_blks2[i](
                        torch.cat(
                            [
                                self.no_param_down(x1),
                                (x2),
                                (self.no_param_up(x3)),
                                (self.no_param_up(self.no_param_up(x4))),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c3 = self.act(
                self.bns3[i](
                    self.cnv_blks3[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(x1)),
                                self.no_param_down(x2),
                                (x3),
                                (self.no_param_up(x4)),
                            ],
                            dim=1,
                        )
                    )
                )
            )
            x_c4 = self.act(
                self.bns4[i](
                    self.cnv_blks4[i](
                        torch.cat(
                            [
                                self.no_param_down(self.no_param_down(self.no_param_down(x1))),
                                self.no_param_down(self.no_param_down(x2)),
                                self.no_param_down(x3),
                                x4,
                            ],
                            dim=1,
                        )
                    )
                )
            )

            x_c1 = self.act(
                self.bns_mrg1[i](
                    self.cnv_mrg1[i](
                        torch.cat([x_c1, x1], dim=2).reshape(batch_size, 2 * self.in_filters1, h1, w1)
                    )
                    + x1
                )
            )
            x_c2 = self.act(
                self.bns_mrg2[i](
                    self.cnv_mrg2[i](
                        torch.cat([x_c2, x2], dim=2).reshape(batch_size, 2 * self.in_filters2, h2, w2)
                    )
                    + x2
                )
            )
            x_c3 = self.act(
                self.bns_mrg3[i](
                    self.cnv_mrg3[i](
                        torch.cat([x_c3, x3], dim=2).reshape(batch_size, 2 * self.in_filters3, h3, w3)
                    )
                    + x3
                )
            )
            x_c4 = self.act(
                self.bns_mrg4[i](
                    self.cnv_mrg4[i](
                        torch.cat([x_c4, x4], dim=2).reshape(batch_size, 2 * self.in_filters4, h4, w4)
                    )
                    + x4
                )
            )

        x1 = self.sqe1(x_c1)
        x2 = self.sqe2(x_c2)
        x3 = self.sqe3(x_c3)
        x4 = self.sqe4(x_c4)

        return x1, x2, x3, x4


class ReliabilityCalibratedMultiScaleFusion(torch.nn.Module):
    """Reliability-Calibrated Multi-Level Fusion (RCMF).

    The proven MLFC transform is retained as a stable path. A tiny router uses
    per-image mean and dispersion summaries to calibrate the reliability of
    each source scale before fusion. Its zero-initialized output makes the
    initial forward mapping exactly MLFC, and a five-percent bound prevents the
    adaptive branch from dominating the static multi-level representation.
    """

    def __init__(
        self,
        in_filters1,
        in_filters2,
        in_filters3,
        in_filters4,
        lenn=1,
        attention_type="se",
        calibration_limit=0.05,
    ):
        super().__init__()
        self.calibration_limit = float(calibration_limit)
        self.core = MLFC(
            in_filters1,
            in_filters2,
            in_filters3,
            in_filters4,
            lenn=lenn,
            attention_type=attention_type,
        )
        # Do not perturb the RNG stream used by the proven core or downstream
        # layers. This makes a seeded RCMF model share the exact same base
        # initialization as its MLFC counterpart.
        with torch.random.fork_rng(devices=[]):
            self.router = torch.nn.Sequential(
                torch.nn.Linear(8, 16),
                torch.nn.GELU(),
                torch.nn.Linear(16, 4),
            )
            torch.nn.init.zeros_(self.router[-1].weight)
            torch.nn.init.zeros_(self.router[-1].bias)

    def forward(self, x1, x2, x3, x4):
        features = (x1, x2, x3, x4)
        descriptors = []
        for feature in features:
            stats = feature.float()
            descriptors.extend([
                stats.mean(dim=(1, 2, 3)),
                stats.std(dim=(1, 2, 3), unbiased=False),
            ])
        descriptor = torch.stack(descriptors, dim=1)
        calibration = 1.0 + self.calibration_limit * torch.tanh(
            self.router(descriptor)
        )
        calibrated = [
            feature * calibration[:, index].to(feature.dtype).reshape(-1, 1, 1, 1)
            for index, feature in enumerate(features)
        ]
        return self.core(*calibrated)


class RecurrentScaleRoutedFusion(torch.nn.Module):
    """Recurrent Scale-Routed Fusion (RSRF).

    Unlike MLFC's fixed dense concatenation, RSRF projects every encoder level
    into one compact latent space and predicts a sample-specific 4x4 routing
    matrix.  The aligned features are fused by normalized weighted summation,
    spatially refined with depth-wise convolution, and returned through a
    guarded residual path.  Reusing the same parameters for several recurrent
    exchanges preserves iterative cross-scale reasoning at low parameter cost.
    """

    def __init__(
        self,
        in_filters1,
        in_filters2,
        in_filters3,
        in_filters4,
        steps=3,
        attention_type="se",
        latent_channels=None,
    ):
        super().__init__()
        self.channels = (
            in_filters1, in_filters2, in_filters3, in_filters4
        )
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError("RSRF steps must be at least one")
        latent = latent_channels or max(16, min(self.channels))
        self.latent_channels = int(latent)

        self.project = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(channels, latent, kernel_size=1, bias=False),
                torch.nn.GroupNorm(1, latent),
                torch.nn.GELU(),
            )
            for channels in self.channels
        ])
        self.refine = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    latent, latent, kernel_size=3, padding=1,
                    groups=latent, bias=False,
                ),
                torch.nn.GroupNorm(1, latent),
                torch.nn.GELU(),
                torch.nn.Conv2d(latent, latent, kernel_size=1, bias=False),
            )
            for _ in self.channels
        ])
        self.restore = torch.nn.ModuleList([
            torch.nn.Conv2d(latent, channels, kernel_size=1, bias=False)
            for channels in self.channels
        ])
        self.attention = torch.nn.ModuleList([
            ChannelSELayer(channels, attention_type)
            for channels in self.channels
        ])

        descriptor_channels = 2 * latent * len(self.channels)
        router_hidden = max(16, latent)
        self.router = torch.nn.Sequential(
            torch.nn.Linear(descriptor_channels, router_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(
                router_hidden, len(self.channels) * len(self.channels)
            ),
        )
        # Begin with content-preserving, self-dominant routes. The final router
        # layer then learns deviations specific to each image.
        torch.nn.init.zeros_(self.router[-1].weight)
        torch.nn.init.zeros_(self.router[-1].bias)
        self.register_buffer(
            "identity_route_prior",
            2.0 * torch.eye(len(self.channels)),
            persistent=False,
        )
        # A small initial residual contribution trains every branch from the
        # first step while keeping the original skip features dominant.
        self.step_gains = torch.nn.Parameter(
            torch.full((self.steps, len(self.channels)), -2.0)
        )

    @staticmethod
    def _align(source, target_size):
        if source.shape[-2:] == target_size:
            return source
        if source.shape[-2] > target_size[0]:
            return F.interpolate(source, size=target_size, mode="area")
        return F.interpolate(
            source, size=target_size, mode="bilinear", align_corners=False
        )

    def _route(self, latent_features):
        descriptors = []
        for feature in latent_features:
            descriptors.extend([
                feature.mean(dim=(2, 3)),
                feature.std(dim=(2, 3), unbiased=False),
            ])
        descriptor = torch.cat(descriptors, dim=1)
        route_logits = self.router(descriptor).reshape(
            descriptor.shape[0], len(self.channels), len(self.channels)
        )
        route_logits = route_logits + self.identity_route_prior
        return torch.softmax(route_logits, dim=-1)

    def forward(self, x1, x2, x3, x4):
        features = [x1, x2, x3, x4]
        for step in range(self.steps):
            latent_features = [
                projection(feature)
                for projection, feature in zip(self.project, features)
            ]
            routes = self._route(latent_features)
            updated = []
            for target_index, target in enumerate(features):
                aligned = torch.stack([
                    self._align(source, target.shape[-2:])
                    for source in latent_features
                ], dim=1)
                weights = routes[:, target_index, :].reshape(
                    routes.shape[0], len(self.channels), 1, 1, 1
                )
                fused = (aligned * weights).sum(dim=1)
                residual = self.restore[target_index](
                    self.refine[target_index](fused)
                )
                gain = torch.sigmoid(
                    self.step_gains[step, target_index]
                )
                updated.append(
                    self.attention[target_index](target + gain * residual)
                )
            features = updated
        return tuple(features)


class ACC_UNet(torch.nn.Module):
    """
    ACC-UNet model
    """

    def __init__(
        self,
        n_channels,
        n_classes,
        n_filts=32,
        attention_type="simam",
        attention_scope="all",
        boundary_fusion=False,
        context_energy=False,
        context_residual=None,
        scale_calibration=None,
        feature_fusion_type="mlfc",
        rcmf_calibration_limit=0.05,
        activation_checkpointing=False,
        ugbf_residual_limit=1.0,
        preserve_boundary_rng=False,
        numerical_protocol="legacy_probability_v1",
    ):
        """
        Initialization

        Args:
            n_channels (int): number of channels of the input image.
            n_classes (int): number of output classes
            n_filts (int, optional): multiplier of the number of filters throughout the model.
                                     Increase this to make the model wider.
                                     Decrease this to make the model ligher.
                                     Defaults to 32.
        """

        super().__init__()
        if attention_type not in {
            "se", "simam", "dc_simam", "bdc_simam", "bdc_simam_single"
        }:
            raise ValueError(
                "attention_type must be se, simam, dc_simam, bdc_simam, "
                "or bdc_simam_single"
            )
        if attention_scope not in {"all", "encoder", "skip", "decoder"}:
            raise ValueError("attention_scope must be all, encoder, skip, or decoder")
        if feature_fusion_type not in {"mlfc", "rsrf", "rcmf"}:
            raise ValueError("feature_fusion_type must be mlfc, rsrf, or rcmf")
        if boundary_fusion and n_classes != 1:
            raise ValueError("UGBF currently supports binary segmentation only")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filts = n_filts
        self.attention_type = attention_type
        self.attention_scope = attention_scope
        self.feature_fusion_type = feature_fusion_type
        self.rcmf_calibration_limit = float(rcmf_calibration_limit)
        self.boundary_fusion = boundary_fusion
        self.ugbf_residual_limit = float(ugbf_residual_limit)
        self.preserve_boundary_rng = bool(preserve_boundary_rng)
        if numerical_protocol not in {"legacy_probability_v1", "stable_logits_v2"}:
            raise ValueError("Unsupported numerical protocol")
        self.numerical_protocol = numerical_protocol
        self.activation_checkpointing = bool(activation_checkpointing)
        if context_residual is None:
            context_residual = context_energy
        if scale_calibration is None:
            scale_calibration = context_energy
        self.context_residual = bool(context_residual)
        self.scale_calibration = bool(scale_calibration)

        def attention_for(section):
            if attention_type == "se":
                return "se"
            if attention_type in {
                "dc_simam", "bdc_simam", "bdc_simam_single"
            }:
                return (
                    attention_type
                    if attention_scope in {"all", section}
                    else "simam"
                )
            return attention_type if attention_scope in {"all", section} else "se"

        encoder_attention = attention_for("encoder")
        skip_attention = attention_for("skip")
        decoder_attention = attention_for("decoder")

        self.pool = torch.nn.MaxPool2d(2)

        self.cnv11 = HANCBlock(n_channels, n_filts, k=3, inv_fctr=3, attention_type=encoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)
        self.cnv12 = HANCBlock(n_filts, n_filts, k=3, inv_fctr=3, attention_type=encoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)

        self.cnv21 = HANCBlock(n_filts, n_filts * 2, k=3, inv_fctr=3, attention_type=encoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)
        self.cnv22 = HANCBlock(n_filts * 2, n_filts * 2, k=3, inv_fctr=3, attention_type=encoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)

        self.cnv31 = HANCBlock(n_filts * 2, n_filts * 4, k=3, inv_fctr=3, attention_type=encoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)
        self.cnv32 = HANCBlock(n_filts * 4, n_filts * 4, k=3, inv_fctr=3, attention_type=encoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)

        self.cnv41 = HANCBlock(n_filts * 4, n_filts * 8, k=2, inv_fctr=3, attention_type=encoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)
        self.cnv42 = HANCBlock(n_filts * 8, n_filts * 8, k=2, inv_fctr=3, attention_type=encoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)

        self.cnv51 = HANCBlock(n_filts * 8, n_filts * 16, k=1, inv_fctr=3, attention_type=encoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)
        self.cnv52 = HANCBlock(n_filts * 16, n_filts * 16, k=1, inv_fctr=3, attention_type=encoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)

        self.rspth1 = ResPath(n_filts, 4, attention_type=skip_attention)
        self.rspth2 = ResPath(n_filts * 2, 3, attention_type=skip_attention)
        self.rspth3 = ResPath(n_filts * 4, 2, attention_type=skip_attention)
        self.rspth4 = ResPath(n_filts * 8, 1, attention_type=skip_attention)

        if feature_fusion_type == "rsrf":
            self.rsrf = RecurrentScaleRoutedFusion(
                n_filts, n_filts * 2, n_filts * 4, n_filts * 8,
                steps=3, attention_type=skip_attention,
            )
        else:
            fusion_class = (
                ReliabilityCalibratedMultiScaleFusion
                if feature_fusion_type == "rcmf"
                else MLFC
            )
            fusion_kwargs = {
                "lenn": 1,
                "attention_type": skip_attention,
            }
            if feature_fusion_type == "rcmf":
                fusion_kwargs["calibration_limit"] = self.rcmf_calibration_limit
            self.mlfc1 = fusion_class(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, **fusion_kwargs)
            self.mlfc2 = fusion_class(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, **fusion_kwargs)
            self.mlfc3 = fusion_class(n_filts, n_filts * 2, n_filts * 4, n_filts * 8, **fusion_kwargs)

        self.up6 = torch.nn.ConvTranspose2d(n_filts * 16, n_filts * 8, kernel_size=(2, 2), stride=2)
        self.cnv61 = HANCBlock(n_filts * 8 + n_filts * 8, n_filts * 8, k=2, inv_fctr=3, attention_type=decoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)
        self.cnv62 = HANCBlock(n_filts * 8, n_filts * 8, k=2, inv_fctr=3, attention_type=decoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)

        self.up7 = torch.nn.ConvTranspose2d(n_filts * 8, n_filts * 4, kernel_size=(2, 2), stride=2)
        self.cnv71 = HANCBlock(n_filts * 4 + n_filts * 4, n_filts * 4, k=3, inv_fctr=3, attention_type=decoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)
        self.cnv72 = HANCBlock(n_filts * 4, n_filts * 4, k=3, inv_fctr=34, attention_type=decoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)

        self.up8 = torch.nn.ConvTranspose2d(n_filts * 4, n_filts * 2, kernel_size=(2, 2), stride=2)
        self.cnv81 = HANCBlock(n_filts * 2 + n_filts * 2, n_filts * 2, k=3, inv_fctr=3, attention_type=decoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)
        self.cnv82 = HANCBlock(n_filts * 2, n_filts * 2, k=3, inv_fctr=3, attention_type=decoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)

        self.up9 = torch.nn.ConvTranspose2d(n_filts * 2, n_filts, kernel_size=(2, 2), stride=2)
        self.cnv91 = HANCBlock(n_filts + n_filts, n_filts, k=3, inv_fctr=3, attention_type=decoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)
        self.cnv92 = HANCBlock(n_filts, n_filts, k=3, inv_fctr=3, attention_type=decoder_attention, context_residual=self.context_residual, scale_calibration=self.scale_calibration)

        if boundary_fusion:
            # Keep added branches from advancing the global RNG when a paired
            # architecture comparison requests matched common initialization.
            # RCMF already applies the same isolation to its router.
            rng_context = (
                torch.random.fork_rng(devices=[])
                if self.preserve_boundary_rng else nullcontext()
            )
            with rng_context:
                # x2 retains the finest spatial detail after ResPath/MLFC fusion.
                self.boundary_head = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        n_filts, n_filts, kernel_size=3, padding=1,
                        groups=n_filts, bias=False,
                    ),
                    torch.nn.GroupNorm(1, n_filts),
                    torch.nn.GELU(),
                    torch.nn.Conv2d(n_filts, 1, kernel_size=1),
                )
                self.coarse_head_56 = torch.nn.Conv2d(n_filts * 4, 1, kernel_size=1)
                self.coarse_head_112 = torch.nn.Conv2d(n_filts * 2, 1, kernel_size=1)
                self.ugbf_112 = UncertaintyGuidedBoundaryFusion(
                    residual_limit=self.ugbf_residual_limit
                )
                self.ugbf_224 = UncertaintyGuidedBoundaryFusion(
                    residual_limit=self.ugbf_residual_limit
                )

        if n_classes == 1:
            self.out = torch.nn.Conv2d(n_filts, n_classes, kernel_size=(1, 1))
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Conv2d(n_filts, n_classes + 1, kernel_size=(1, 1))
            self.last_activation = None

        if self.numerical_protocol == "stable_logits_v2":
            for module in self.modules():
                if isinstance(module, (SimAM, ScaleCoupledSimAM, HANCBlock)):
                    module.force_float32_statistics = True

    @staticmethod
    @contextmanager
    def _suspend_bn_running_updates(module):
        """Prevent checkpoint recomputation from updating BatchNorm twice."""
        states = []
        for child in module.modules():
            if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
                states.append((child, child.momentum,
                               child.num_batches_tracked.clone()
                               if child.num_batches_tracked is not None else None))
                child.momentum = 0.0
        try:
            yield
        finally:
            for child, momentum, batches in states:
                child.momentum = momentum
                if batches is not None:
                    child.num_batches_tracked.copy_(batches)

    def _run(self, module, *inputs):
        if not self.activation_checkpointing or not self.training:
            return module(*inputs)

        def context_fn():
            return nullcontext(), self._suspend_bn_running_updates(module)

        return checkpoint(
            module,
            *inputs,
            use_reentrant=False,
            context_fn=context_fn,
        )

    def forward(self, x, return_aux=False, return_logits=False):

        x1 = x

        x2 = self._run(self.cnv11, x1)
        x2 = self._run(self.cnv12, x2)

        x2p = self.pool(x2)

        x3 = self._run(self.cnv21, x2p)
        x3 = self._run(self.cnv22, x3)

        x3p = self.pool(x3)

        x4 = self._run(self.cnv31, x3p)
        x4 = self._run(self.cnv32, x4)

        x4p = self.pool(x4)

        x5 = self._run(self.cnv41, x4p)
        x5 = self._run(self.cnv42, x5)

        x5p = self.pool(x5)

        x6 = self._run(self.cnv51, x5p)
        x6 = self._run(self.cnv52, x6)

        x2 = self._run(self.rspth1, x2)
        x3 = self._run(self.rspth2, x3)
        x4 = self._run(self.rspth3, x4)
        x5 = self._run(self.rspth4, x5)

        if self.feature_fusion_type == "rsrf":
            x2, x3, x4, x5 = self._run(self.rsrf, x2, x3, x4, x5)
        else:
            x2, x3, x4, x5 = self._run(self.mlfc1, x2, x3, x4, x5)
            x2, x3, x4, x5 = self._run(self.mlfc2, x2, x3, x4, x5)
            x2, x3, x4, x5 = self._run(self.mlfc3, x2, x3, x4, x5)

        x7 = self.up6(x6)
        x7 = self._run(self.cnv61, torch.cat([x7, x5], dim=1))
        x7 = self._run(self.cnv62, x7)

        x8 = self.up7(x7)
        x8 = self._run(self.cnv71, torch.cat([x8, x4], dim=1))
        x8 = self._run(self.cnv72, x8)

        x9 = self.up8(x8)
        if self.boundary_fusion:
            edge_logits = self.boundary_head(x2)
            coarse_logits_56 = self.coarse_head_56(x8)
            x9_fusion, boundary_focus_112 = self.ugbf_112(
                x9, x3, coarse_logits_56, edge_logits
            )
        else:
            x9_fusion = torch.cat([x9, x3], dim=1)
        x9 = self._run(self.cnv81, x9_fusion)
        x9 = self._run(self.cnv82, x9)

        x10 = self.up9(x9)
        if self.boundary_fusion:
            coarse_logits_112 = self.coarse_head_112(x9)
            x10_fusion, boundary_focus_224 = self.ugbf_224(
                x10, x2, coarse_logits_112, edge_logits
            )
        else:
            x10_fusion = torch.cat([x10, x2], dim=1)
        x10 = self._run(self.cnv91, x10_fusion)
        x10 = self._run(self.cnv92, x10)

        raw_logits = self.out(x10)
        if return_logits or self.last_activation is None:
            output = raw_logits
        elif self.numerical_protocol == "stable_logits_v2":
            output = torch.sigmoid(raw_logits.float())
        else:
            output = self.last_activation(raw_logits)

        if return_aux:
            auxiliary = {}
            if self.boundary_fusion:
                auxiliary = {
                    "boundary_logits": edge_logits,
                    "coarse_logits": (coarse_logits_56, coarse_logits_112),
                    "boundary_focus": (boundary_focus_112, boundary_focus_224),
                }
            return output, auxiliary
        return output
