import torch
from torch import nn
from torch.nn import functional as F
from model.networks_other import init_weights

## from github: https://github.com/ozan-oktay/Attention-Gated-Networks
class _GridAttentionBlockND_TORR(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2, mode='concatenation_sigmoid',
                 sub_sample_factor=(1, 1), bn_layer=True, use_W=True, use_phi=True, use_theta=True, use_psi=True, nonlinearity1='relu'):
        super(_GridAttentionBlockND_TORR, self).__init__()

        assert dimension == 2, "Only 2D is supported in this simplified version."

        self.mode = mode
        self.dimension = dimension
        self.sub_sample_factor = sub_sample_factor
        self.sub_sample_kernel_size = sub_sample_factor

        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels or max(1, in_channels // 2)

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d
        self.upsample_mode = 'bilinear'

        self.W = nn.Sequential(
            conv_nd(in_channels, in_channels, kernel_size=1),
            bn(in_channels),
        ) if use_W else lambda x: x

        self.theta = conv_nd(in_channels, self.inter_channels, kernel_size=self.sub_sample_kernel_size,
                             stride=self.sub_sample_factor, padding=0, bias=False) if use_theta else lambda x: x

        self.phi = conv_nd(gating_channels, self.inter_channels, kernel_size=self.sub_sample_kernel_size,
                           stride=self.sub_sample_factor, padding=0, bias=False) if use_phi else lambda x: x

        self.psi = conv_nd(self.inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True) if use_psi else lambda x: x

        self.nl1 = F.relu if nonlinearity1 == 'relu' else lambda x: x

        self.operation_function = self._concatenation

        for m in self.children():
            init_weights(m, init_type='kaiming')

        if use_psi and self.mode == 'concatenation_sigmoid':
            nn.init.constant_(self.psi.bias, 3.0)

    def forward(self, x, g):
        return self.operation_function(x, g)

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        phi_g = F.interpolate(self.phi(g), size=theta_x.shape[2:], mode=self.upsample_mode)

        f = self.nl1(theta_x + phi_g)
        psi_f = self.psi(f)
        sigm_psi_f = F.sigmoid(psi_f)

        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock2D_TORR(_GridAttentionBlockND_TORR):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation_sigmoid',
                 sub_sample_factor=(1, 1), bn_layer=True, use_W=True, use_phi=True, use_theta=True, use_psi=True,
                 nonlinearity1='relu'):
        super(GridAttentionBlock2D_TORR, self).__init__(
            in_channels=in_channels,
            gating_channels=gating_channels,
            inter_channels=inter_channels,
            dimension=2,
            mode=mode,
            sub_sample_factor=sub_sample_factor,
            bn_layer=bn_layer,
            use_W=use_W,
            use_phi=use_phi,
            use_theta=use_theta,
            use_psi=use_psi,
            nonlinearity1=nonlinearity1
        )
