import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import models.utils as util

from torch.autograd import Variable
from torchvision.utils import save_image
from models.MCB import MCB
from models.ResNeSt import SplitAttention

"""
Impressed by
https://github.com/sjmoran/CURL
https://arxiv.org/pdf/1911.13175.pdf
@misc{moran2019curl,
    title={CURL: Neural Curve Layers for Global Image Enhancement},
    author={Sean Moran and Steven McDonagh and Gregory Slabaugh},
    year={2019},
    eprint={1911.13175},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}

https://github.com/leaderj1001/Attention-Augmented-Conv2d
Attention Augmented Convolutional Networks Paper
Author, Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens
Quoc V.Le Google Brain
"""


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class Flatten(nn.Module):

    def forward(self, x):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = x.size(0)
        out = x.view(batch_size, -1)
        return out  # (batch_size, *size)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1, padding=None):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride

        if padding is not None:
            self.padding = padding
        else:
            self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None and x.shape[0] != 1:  # error occurs when batch size is 1.
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class Block(nn.Module):

    def __init__(self):
        """Initialisation for a lower-level conv block
        :returns: N/A
        :rtype: N/A
        """
        super(Block, self).__init__()

    def conv3x3(self, in_channels, out_channels, stride=1):
        """Represents a convolution of shape 3x3
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: the convolution stride
        :returns: convolution function with the specified parameterisation
        :rtype: function
        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=True)


class ConvBlock(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block
        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        :returns:
        :rtype:
        """
        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        """ Forward function for the higher level convolution block
        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor
        """
        img_out = self.lrelu(self.conv(x))
        return img_out


class ConvBlock_tanh(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block
        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        :returns:
        :rtype:
        """
        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """ Forward function for the higher level convolution block
        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor
        """
        img_out = self.tanh(self.conv(x))
        return img_out


class ConvBlock_1X1(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block
        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        :returns:
        :rtype:
        """
        super(Block, self).__init__()
        self.conv = nn.Conv2d(num_in_channels, num_out_channels, kernel_size=1, bias=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        """ Forward function for the higher level convolution block
        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor
        """
        img_out = self.lrelu(self.conv(x))
        return img_out


class ConvBlock_tanh_1X1(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block
        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        :returns:
        :rtype:
        """
        super(Block, self).__init__()
        self.conv = nn.Conv2d(num_in_channels, num_out_channels, kernel_size=1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """ Forward function for the higher level convolution block
        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor
        """
        img_out = self.tanh(self.conv(x))
        return img_out


class ConvBlock_InstanceNorm(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):

        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=1)
        self.instance_norm = nn.InstanceNorm2d(num_out_channels, affine=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):

        return self.lrelu(self.instance_norm(self.conv(x)))


class MaxPoolBlock(Block, nn.Module):

    def __init__(self):
        """Initialise function for the max pooling block
        :returns: N/A
        :rtype: N/A
        """
        super(Block, self).__init__()

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        """ Forward function for the max pooling block
        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor
        """
        x_width, x_height = x.shape[2], x.shape[3]

        if x_width == 1 or x_height == 1:
            img_out = self.max_pool_2(x)
        else:
            img_out = self.max_pool_1(x)
        return img_out


class AvgPoolBlock(Block, nn.Module):

    def __init__(self):
        """Initialise function for the max pooling block
        :returns: N/A
        :rtype: N/A
        """
        super(Block, self).__init__()

        self.avg_pool_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        """ Forward function for the max pooling block
        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor
        """
        x_width, x_height = x.shape[2], x.shape[3]

        if x_width == 1 or x_height == 1:
            img_out = self.avg_pool_2(x)
        else:
            img_out = self.avg_pool_1(x)
        return img_out


class GlobalPoolingBlock(Block, nn.Module):

    def __init__(self, receptive_field):
        """Implementation of the global pooling block. Takes the average over a 2D receptive field.
        :param receptive_field:
        :returns: N/A
        :rtype: N/A
        """
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Forward function for the high-level global pooling block
        :param x: Tensor of shape BxCxAxA
        :returns: Tensor of shape BxCx1x1, where B is the batch size
        :rtype: Tensor
        """
        out = self.avg_pool(x)
        return out


class CUD_Loss(nn.Module):
    def __init__(self, ssim_window_size=5, alpha=0.5):
        """This class is greatly impressed by https://github.com/sjmoran/CURL
        :param ssim_window_size: size of averaging window for SSIM
        :param alpha: interpolation paramater for L1 and SSIM parts of the loss
        :returns: N/A
        :rtype: N/A
        """
        super(CUD_Loss, self).__init__()
        self.alpha = alpha
        self.ssim_window_size = ssim_window_size

    def create_window(self, window_size, num_channel):
        """Window creation function for SSIM metric. Gaussian weights are applied to the window.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param window_size: size of the window to compute statistics
        :param num_channel: number of channels
        :returns: Tensor of shape Cx1xWindow_sizexWindow_size
        :rtype: Tensor
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(
            num_channel, 1, window_size, window_size).contiguous())
        return window

    def gaussian(self, window_size, sigma):
        """
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param window_size: size of the SSIM sampling window e.g. 11
        :param sigma: Gaussian variance
        :returns: 1xWindow_size Tensor of Gaussian weights
        :rtype: Tensor
        """
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        """Computes the structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float
        """
        (_, num_channel, _, _) = img1.size()
        window = self.create_window(self.ssim_window_size, num_channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
            window = window.type_as(img1)

        mu1 = F.conv2d(
            img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
        mu2 = F.conv2d(
            img2, window, padding=self.ssim_window_size // 2, groups=num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        ssim_map2 = ((mu1_sq.cuda() + mu2_sq.cuda() + C1) *
                     (sigma1_sq.cuda() + sigma2_sq.cuda() + C2))
        ssim_map = ssim_map1.cuda() / ssim_map2.cuda()

        v1 = 2.0 * sigma12.cuda() + C2
        v2 = sigma1_sq.cuda() + sigma2_sq.cuda() + C2
        cs = torch.mean(v1 / v2)

        return ssim_map.mean(), cs

    def compute_msssim(self, img1, img2):
        """Computes the multi scale structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float
        """

        # img1 = Variable(img1)
        # img2 = Variable(img2)

        if img1.size() != img2.size():
            raise RuntimeError('Input images must have the same shape (%s vs. %s).' % (
                img1.size(), img2.size()))
        if len(img1.size()) != 4:
            raise RuntimeError(
                'Input images must have four dimensions, not %d' % len(img1.size()))

        # if type(img1) is not Variable or type(img2) is not Variable:
        #     raise RuntimeError(
        #         'Input images must be Variables, not %s' % img1.__class__.__name__)

        weights = Variable(torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))
        # weights = Variable(torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0]))
        if img1.is_cuda:
            weights = weights.cuda(img1.get_device())

        levels = weights.size()[0]
        mssim = []
        mcs = []
        for _ in range(levels):
            sim, cs = self.compute_ssim(img1, img2)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

        prod = (torch.prod(mcs[0:levels - 1] ** weights[0:levels - 1])
                * (mssim[levels - 1] ** weights[levels - 1]))
        return prod

    def forward(self, input_image_batch, predicted_img_batch, target_img_batch, is_identity=False):
        num_images = target_img_batch.shape[0]
        target_img_batch = target_img_batch

        ssim_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        lab_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        histogram_loss_lab = Variable(
            torch.cuda.FloatTensor(torch.zeros(1).cuda()))

        for i in range(0, num_images):
            target_img_rgb = target_img_batch[i, :, :, :].cuda()
            predicted_img_rgb = predicted_img_batch[i, :, :, :].cuda()

            predicted_img_lab = torch.clamp(
                util.ImageProcessing.rgb_to_lab(predicted_img_rgb.squeeze(0)), 0, 1)
            target_img_lab = torch.clamp(
                util.ImageProcessing.rgb_to_lab(target_img_rgb.squeeze(0)), 0, 1)

            predict_hist_lab, target_hist_lab = util.get_histogram(predicted_img_lab, target_img_lab)

            # compute cos similarity for each dimension. Similarity should be same.
            histogram_loss_lab += torch.mean(F.l1_loss(predict_hist_lab, target_hist_lab)).cuda()
            histogram_loss_lab = (histogram_loss_lab + 1) / 2   # [-1, 1] to [0, 1]

            input_img_rgb = input_image_batch[i, :, :, :].cuda()
            input_img_lab = torch.clamp(
                util.ImageProcessing.rgb_to_lab(input_img_rgb.squeeze(0)), 0, 1)

            if not is_identity:
                predicted_img_lab = util.variational_prediction(_input=input_img_lab[:3, :, :],
                                                                _output=predicted_img_lab,
                                                                _target=target_img_lab)

            target_img_L_ssim = target_img_lab[0, :, :].unsqueeze(0)
            predicted_img_L_ssim = predicted_img_lab[0, :, :].unsqueeze(0)
            target_img_L_ssim = target_img_L_ssim.unsqueeze(0)
            predicted_img_L_ssim = predicted_img_L_ssim.unsqueeze(0)

            ssim_value = self.compute_msssim(
                predicted_img_L_ssim, target_img_L_ssim)
            ssim_loss_value += (1.0 - ssim_value)
            lab_loss_value += F.l1_loss(predicted_img_lab, target_img_lab)

        lab_loss_value = lab_loss_value/num_images
        ssim_loss_value = ssim_loss_value/num_images
        histogram_loss_lab = histogram_loss_lab/num_images

        _, _, img_width, img_height = input_image_batch.shape
        scaler = img_width * img_height / 20
        histogram_loss_lab /= scaler

        lab_loss = lab_loss_value + histogram_loss_lab
        ssim_loss = ssim_loss_value

        return (lab_loss.requires_grad_(),
                ssim_loss.requires_grad_())


class CUD_NET(nn.Module):

    def __init__(self,
                 num_in_channels=3,
                 num_out_channels=64,
                 num_points=64,
                 save_figures=False,
                 clip_threshold=False):

        super(CUD_NET, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.num_points = num_points
        self.save_figures = save_figures
        self.clip_threshold = clip_threshold
        self.batch_counter = 0
        self.interval = 500
        self.bottle_neck_size = 512

        conv_layer = [
            ConvBlock_tanh(num_in_channels=self.num_in_channels, num_out_channels=16),
            AvgPoolBlock(),
            ConvBlock_tanh(num_in_channels=16, num_out_channels=32),
            AvgPoolBlock(),
            ConvBlock_tanh(num_in_channels=32, num_out_channels=64),
            AvgPoolBlock(),
            ConvBlock_tanh(num_in_channels=64, num_out_channels=128),
            AvgPoolBlock(),
            ConvBlock_tanh(num_in_channels=128, num_out_channels=256),
            AvgPoolBlock(),
            ConvBlock_tanh(num_in_channels=256, num_out_channels=512),
            AvgPoolBlock(),
            ConvBlock_tanh_1X1(num_in_channels=512, num_out_channels=self.bottle_neck_size),
            AvgPoolBlock(),
            ConvBlock_tanh_1X1(num_in_channels=self.bottle_neck_size, num_out_channels=self.bottle_neck_size),
            GlobalPoolingBlock(2)
        ]

        self.conv_layer = nn.Sequential(*conv_layer)
        self.res_layer = ConvBlock_tanh_1X1(num_in_channels=self.bottle_neck_size * 3,
                                            num_out_channels=self.bottle_neck_size)

        self.mcb = MCB(self.bottle_neck_size, self.bottle_neck_size)

        radix = 16
        cardinality = 4
        self.split_attention = SplitAttention(self.bottle_neck_size, radix, cardinality)

        regression_layer = [
            nn.Dropout(0.5),
            nn.Linear(self.bottle_neck_size * 2, self.num_points),
        ]
        self.regression_layer = nn.Sequential(*regression_layer)

    def forward(self, x, fn=None):
        x.contiguous()
        x = torch.clamp(x, 0, 1)
        img = x[:, 0:3]
        img_deu = x[:, 3:6]
        img_diff = x[:, 6:9]
        torch.cuda.empty_cache()

        # save_image(img, 'img.jpg')
        # save_image(img_deu, 'img_deu.jpg')
        # save_image(img_diff, 'img_diff.jpg')

        # --------------- HSV layer ---------------
        img_hsv = util.ImageProcessing.rgb_to_hsv(img.squeeze(0))
        img_hsv = torch.clamp(img_hsv, 0, 1)

        feat_img = self.conv_layer(img)
        feat_deu = self.conv_layer(img_deu)
        feat_diff = self.conv_layer(img_diff)
        feat_img = feat_img.view(feat_img.size()[0], -1)
        feat_deu = feat_deu.view(feat_deu.size()[0], -1)
        feat_diff = feat_diff.view(feat_diff.size()[0], -1)

        feat_img_deu = self.mcb(feat_img, feat_deu)
        feat_img_diff = self.mcb(feat_img, feat_diff)
        feat_fusion = self.mcb(feat_img_deu, feat_img_diff)
        feat_final = self.mcb(feat_img, feat_fusion)
        feat_final = feat_final.view(feat_final.size()[0], -1, 1, 1)
        feat_final_att = self.split_attention(feat_final)

        feat_img = feat_img.view(feat_img.size()[0], -1, 1, 1)
        feat_cat = torch.cat((feat_img, feat_final_att), dim=1)

        feat_cat = feat_cat.view(feat_cat.size()[0], -1)
        H = self.regression_layer(feat_cat)
        H = H.view(H.size()[0], -1)
        if self.batch_counter % self.interval == 0:
            print('H', H[0, :50])

        if self.save_figures:
            curve_SS = np.array(torch.exp(H[0, int((self.num_points / 2) * 0): int((self.num_points / 2) * 1)].cpu().detach()))
            curve_VV = np.array(torch.exp(H[0, int((self.num_points / 2) * 1): int((self.num_points / 2) * 2)].cpu().detach()))

            curve_list = [curve_SS, curve_VV]
            curve_list_labels = ['curve_SS', 'curve_VV']
            if fn is not None:
                util.save_plt_figures(curve_list, labels=curve_list_labels, fn=fn + '_curve_HSV')
            else:
                util.save_plt_figures(curve_list, labels=curve_list_labels, fn='curve_HSV')

            util.save_plt_figure(curve_SS, fn=fn + '_curve_SS')
            util.save_plt_figure(curve_VV, fn=fn + '_curve_VV')

        img_hsv_out, _ = util.ImageProcessing.adjust_sv(
            img_hsv, H[0, 0:self.num_points])
        img_hsv_out = torch.clamp(img_hsv_out, 0, 1)

        if self.clip_threshold:
            # clip S channel
            img_hsv_out[1, :, :] = util.clip_by_threshold(img_hsv[1, :, :], img_hsv_out[1, :, :], threshold=0.1)

        img_rgb_out = torch.clamp(util.ImageProcessing.hsv_to_rgb(
           img_hsv_out.squeeze(0)), 0, 1)

        self.batch_counter += 1

        return img_rgb_out
