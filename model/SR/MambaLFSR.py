import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from einops import repeat, rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

class get_model(nn.Module):
    def __init__(self,
                 args,
                 img_size=160,
                 patch_size=1,
                 depths=(6, 6, 6, 6),
                 drop_ratio=0.,
                 d_state=10,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv'):
        super(get_model, self).__init__()

        self.angRes = args.angRes_in
        self.channels = args.channels
        self.mlp_ratio = mlp_ratio
        self.n_groups = args.n_groups
        self.upscale_factor = args.scale_factor

        ######################## Initial Feature Extraction #######################
        # self.conv_init = nn.Sequential(
        #     nn.Conv2d(1, self.channels, kernel_size=3, padding=1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=2, dilation=2, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=int(self.angRes), dilation=int(self.angRes), bias=False)
        # )

        self.conv_init = FEM(channels=self.channels, angRes=self.angRes)


        # --------------------- Deep Spatial-Angular Correlation Learning ----------------- #
        modules_body = [
            AltFilter(angRes=self.angRes,
                      channels=self.channels,
                      d_state=d_state,
                      drop_path=0.,
                      mlp_ratio=2,
                      norm_layer=norm_layer,
                      is_light_sr=False) \
            for _ in range(self.n_groups)
        ]

        self.body = nn.Sequential(*modules_body)

        # -------------------------- Up-sampling ------------------------------------#
        modules_tail = [
            Upsampler(self.upscale_factor, self.channels, kernel_size=3, stride=1, dilation=1, padding=1, act=False),
            nn.Conv2d(self.channels, 1, kernel_size=1, stride=1, dilation=1, padding=0, bias=True)]

        self.tail = nn.Sequential(*modules_tail)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, info=None):

        # Bicubic Upsample
        x_upscale = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)



        x = SAI2MacPI(x, self.angRes)
        x = self.conv_init(x)

        x = MacPI2SAI(x, self.angRes)

        x = self.body(x)
        x = self.tail(x)

        x += x_upscale

        return x


class FEM(nn.Module):
    def __init__(self, channels, angRes):
        super(FEM, self).__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=2, padding=2, bias=False)
        self.lrelu2 = nn.LeakyReLU(0.2, True)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False)

    def forward(self, x):
        buffer_init = self.conv1(x)
        buffer_init = self.lrelu1(buffer_init)
        buffer = self.conv2(buffer_init)
        buffer = self.lrelu2(buffer)
        buffer = self.conv3(buffer)

        return buffer + buffer_init

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr:
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(num_feat // compress_ratio, num_feat // compress_ratio, kernel_size=3, stride=1, padding=1, groups=num_feat // compress_ratio),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=2, groups=num_feat, dilation=2),
                ChannelAttention(num_feat, squeeze_factor)
            )
        else:
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, kernel_size=3, stride=1, padding=1),
                ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.f1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.f2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.f1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.f2(x)
        x = self.drop(x)
        return x


class SS2D(nn.Module):
    def __init__(self,
                 d_model,
                 d_state=16,
                 d_conv=3,
                 expand=2.,
                 dt_rank="auto",
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 dropout=0.,
                 conv_bias=True,
                 bias=False,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SS2D, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv -1 ) // 2,
            **factory_kwargs
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        # self.selective_scan = partial(selective_scan_fn, backend="torch")
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(self,
                 hidden_dim=0,
                 drop_path=0.,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0.,
                 d_state=16,
                 expand=2.,
                 is_light_sr=False,
                 **kwargs):
        super(VSSBlock, self).__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class AltFilter(nn.Module):
    def __init__(self,
                 angRes,
                 channels,
                 d_state=16,
                 drop_path=0.,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 is_light_sr=False):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.dim = channels
        self.mlp_ratio = mlp_ratio
        self.epi_mamba = VSSBlock(
            hidden_dim=self.dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=0,
            d_state=d_state,
            expand=self.mlp_ratio,
            is_light_sr=is_light_sr
        )

        self.horizontal_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.vertical_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, buffer):
        shortcut = buffer
        buffer = rearrange(buffer, 'b c (u h) (v w) -> b c u v h w', c=self.dim, u=self.angRes, v=self.angRes)
        b, c, u, v, h, w = buffer.shape
        buffer = buffer.permute(0, 3, 5, 2, 4, 1).contiguous()       # (b, v, w, u, h, c)
        buffer = buffer.view(b * v * w, u * h, c)

        # Horizontal
        horizontal_size = (u, h)
        buffer = self.epi_mamba(buffer, horizontal_size).transpose(1, 2).view(b * v * w, c, u, h)
        buffer = self.horizontal_conv(buffer).view(b, v, w, c, u, h).permute(0, 3, 4, 5, 1, 2).contiguous()
        buffer = rearrange(buffer, 'b c u h v w -> b c (u h) (v w)', u=self.angRes, v=self.angRes)
        buffer += shortcut

        # Vertical
        vertical_size = (v, w)
        buffer = buffer.view(b, c, u, h, v, w).permute(0, 2, 3, 4, 5, 1).contiguous().view(b * u * h, v * w, c)
        buffer = self.epi_mamba(buffer, vertical_size).transpose(1, 2).view(b * u * h, c, v, w)
        buffer = self.vertical_conv(buffer).view(b, u, h, c, v, w).permute(0, 3, 1, 2, 4, 5).contiguous()
        buffer = rearrange(buffer, 'b c u h v w -> b c (u h) (v w)')
        buffer += shortcut

        return buffer


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feat,kernel_size, stride, dilation, padding,  bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feat, 4 * n_feat, kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding, bias=True))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(nn.Conv2d(n_feat, 9 * n_feat, kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding, bias=True))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)

        return loss