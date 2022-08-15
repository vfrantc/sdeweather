import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QuaternionConv

__all__ = ['QuaternionBatchNorm2d', 'QuaternionInstanceNorm2d', 'ShareSepConv', 'SmoothDilatedResidualBlock']

class QuaternionBatchNorm2d(nn.Module):
    r"""Applies a 2D Quaternion Batch Normalization to the incoming data.
        """

    def __init__(self, num_features, gamma_init=1., beta_param=True, training=True):
        super(QuaternionBatchNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.training = training
        self.eps = torch.tensor(1e-5)

    def reset_parameters(self):
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):
        quat_components = torch.chunk(input, 4, dim=1)
        r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
        delta_r, delta_i, delta_j, delta_k = r - torch.mean(r), i - torch.mean(i), j - torch.mean(j), k - torch.mean(k)
        quat_variance = torch.mean((delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2))
        denominator = torch.sqrt(quat_variance + self.eps)

        # Normalize
        r_normalized = delta_r / denominator
        i_normalized = delta_i / denominator
        j_normalized = delta_j / denominator
        k_normalized = delta_k / denominator

        beta_components = torch.chunk(self.beta, 4, dim=1)

        # Multiply gamma (stretch scale) and add beta (shift scale)
        new_r = (self.gamma * r_normalized) + beta_components[0]
        new_i = (self.gamma * i_normalized) + beta_components[1]
        new_j = (self.gamma * j_normalized) + beta_components[2]
        new_k = (self.gamma * k_normalized) + beta_components[3]

        new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)

        return new_input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma) \
               + ', beta=' + str(self.beta) \
               + ', eps=' + str(self.eps) + ')'

class QuaternionInstanceNorm2d(nn.Module):
    def __init__(self, num_features, gamma_init=1., beta_param=True, training=True):
        super(QuaternionInstanceNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.training = training
        self.eps = torch.tensor(1e-5)

    def reset_parameters(self):
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):
        quat_components = torch.chunk(input, 4, dim=1)
        r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
        delta_r, delta_i, delta_j, delta_k = r - torch.mean(r, axis=[1, 2, 3], keepdim=True), i - torch.mean(i, axis=[1, 2, 3], keepdim=True), j - torch.mean(j, axis=[1, 2, 3], keepdim=True), k - torch.mean(k, axis=[1, 2, 3], keepdim=True)
        quat_variance = torch.mean((delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2))
        denominator = torch.sqrt(quat_variance + self.eps)

        # Normalize
        r_normalized = delta_r / denominator
        i_normalized = delta_i / denominator
        j_normalized = delta_j / denominator
        k_normalized = delta_k / denominator

        beta_components = torch.chunk(self.beta, 4, dim=1)

        # Multiply gamma (stretch scale) and add beta (shift scale)
        new_r = (self.gamma * r_normalized) + beta_components[0]
        new_i = (self.gamma * i_normalized) + beta_components[1]
        new_j = (self.gamma * j_normalized) + beta_components[2]
        new_k = (self.gamma * k_normalized) + beta_components[3]

        new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)

        return new_input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma) \
               + ', beta=' + str(self.beta) \
               + ', eps=' + str(self.eps) + ')'

class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)

class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1, use_bn=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilatation=dilation, groups=group, bias=False)
        self.norm1 = get_norm(channel_num, use_bn)
        self.conv2 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilatation=dilation, groups=group, bias=False)
        self.norm2 = get_norm(channel_num, use_bn)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)

class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1, use_bn=False):
        super(SmoothDilatedResidualBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilatation=dilation, groups=group, bias=False)
        self.norm1 = get_norm(channel_num, use_bn=use_bn)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilatation=dilation, groups=group, bias=False)
        self.norm2 = get_norm(channel_num, use_bn=use_bn)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
        y = self.norm2(self.conv2(self.pre_conv2(y)))
        return F.relu(x+y)


def get_norm(num_channels, use_bn=True):
    if use_bn:
        return QuaternionBatchNorm2d(num_features=num_channels)
    else:
        return QuaternionInstanceNorm2d(num_features=num_channels)

