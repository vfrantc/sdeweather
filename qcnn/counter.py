from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

from .base import QuaternionTransposeConv
from .base import QuaternionConv
from .base import QuaternionLinearAutograd
from .base import QuaternionLinear
from .layers import QuaternionBatchNorm2d
from .layers import QuaternionInstanceNorm2d
from .layers import ShareSepConv
from .layers import SmoothDilatedResidualBlock
from .layers import ResidualBlock

def count_QuaternionTransposeConv(m: QuaternionTransposeConv, x: (torch.Tensor), y: (torch.Tensor)):
    x = x[0]
    m.total_ops += torch.DoubleTensor([int(0)])
    m.total_params = torch.DoubleTensor([int(0)])
    m.total_params += (m.r_weight.numel() + m.i_weight.numel() + m.j_weight.numel() + m.k_weight.numel()) // 4
    p = (m.r_weight.numel() + m.i_weight.numel() + m.j_weight.numel() + m.k_weight.numel()) // 4
    if hasattr(m, 'bias') and (m.bias is not None):
        m.total_params += m.bias.numel() // 4
        p = p + m.bias.numel() // 4

    m.real_params = 0
    m.quat_params = p


def count_QuaternionConv(m: QuaternionConv, x: (torch.Tensor), y: (torch.Tensor)):
    x = x[0]
    m.total_ops += torch.DoubleTensor([int(0)])
    m.total_params = torch.DoubleTensor([int(0)])
    m.total_params += (m.r_weight.numel() + m.i_weight.numel() + m.j_weight.numel() + m.k_weight.numel()) // 4
    a = 0
    if hasattr(m, 'bias') and (m.bias is not None):
        m.total_params += m.bias.numel() // 4
        a += m.bias.numel() // 4
    if hasattr(m, 'scale_param') and (m.scale_param is not None):
        m.total_params += m.scale_param.numel() // 4
        a += m.scale_param.numel() // 4
    m.real_params = 0
    m.quat_params = a + (m.r_weight.numel() + m.i_weight.numel() + m.j_weight.numel() + m.k_weight.numel()) // 4


def count_QuaternionLinearAutograd(m: QuaternionLinearAutograd, x: (torch.Tensor), y: (torch.Tensor)):
    x = x[0]
    m.total_ops += torch.DoubleTensor([int(0)])

    m.total_params = torch.DoubleTensor([int(0)])
    m.total_params += (m.r_weight.numel() + m.i_weight.numel() + m.j_weight.numel() + m.k_weight.numel()) // 4
    p = 0
    p += (m.r_weight.numel() + m.i_weight.numel() + m.j_weight.numel() + m.k_weight.numel()) // 4
    if hasattr(m, 'bias') and (m.bias is not None):
        m.total_params += m.bias.numel() // 4
        p += m.bias.numel() // 4

    r = 0
    if hasattr(m, 'scale_param') and (m.scale_param is not None):
        m.total_params += m.scale_param.numel()
        r += m.scale_param.numel()

    if hasattr(m, 'zero_kernel') and (m.zero_kernel is not None):
        m.total_params += m.zero_kernel.numel()
        r += m.zero_kernel.numel()

    m.real_params = r
    m.quat_params = p


def count_QuaternionLinear(m: QuaternionLinear, x: (torch.Tensor), y: (torch.Tensor)):
    x = x[0]
    m.total_ops += torch.DoubleTensor([int(0)])

    m.total_params = torch.DoubleTensor([int(0)])
    m.total_params += (m.r_weight.numel() + m.i_weight.numel() + m.j_weight.numel() + m.k_weight.numel()) // 4

    p = (m.r_weight.numel() + m.i_weight.numel() + m.j_weight.numel() + m.k_weight.numel()) // 4

    if hasattr(m, 'bias') and (m.bias is not None):
        m.total_params += m.bias.numel() // 4
        p += m.bias.numel() // 4

    m.real_params = 0
    m.quat_params = p


def count_QuaternionInstanceNorm2d(m: QuaternionInstanceNorm2d, x: (torch.Tensor), y: (torch.Tensor)):
    x = x[0]
    m.total_ops += torch.DoubleTensor([int(0)])

    m.total_params = torch.DoubleTensor([int(0)])
    p = 0
    r = 0
    if hasattr(m, 'gamma'):
        m.total_params += m.gamma.numel()
        r += m.gamma.numel()

    if hasattr(m, 'beta'):
        m.total_params += m.beta.numel()
        r += m.beta.numel()
    m.real_params = r
    m.quat_params = p


def count_ShareSepConv(m: ShareSepConv, x: (torch.Tensor), y: (torch.Tensor)):
    x = x[0]
    m.total_ops += torch.DoubleTensor([int(0)])
    m.total_params = torch.DoubleTensor([int(0)])
    m.total_params += m.weight.numel()
    m.real_params = m.weight.numel()
    m.quat_params = 0


def count_SmoothDilatedResidualBlock(m: SmoothDilatedResidualBlock, x: (torch.Tensor), y: (torch.Tensor)):
    x = x[0]
    m.total_ops += torch.DoubleTensor([int(0)])
    m.total_params = torch.DoubleTensor([int(0)])

    m.total_params += m.pre_conv1.total_params
    m.total_params += m.conv1.total_params
    m.total_params += m.norm1.total_params
    m.total_params += m.pre_conv2.total_params
    m.total_params += m.conv2.total_params
    m.total_params += m.norm2.total_params

    m.real_params = 0
    m.quat_params = 0
    for item in [m.pre_conv1, m.conv1, m.norm1, m.pre_conv2, m.conv2, m.norm2]:
        if hasattr(item, 'real_params'):
            m.real_params += item.real_params
            m.quat_params += item.quat_params
        else:
            m.real_params += item.total_params


def count_ResidualBlock(m: ResidualBlock, x: (torch.Tensor), y: (torch.Tensor)):
    x = x[0]
    m.total_ops += torch.DoubleTensor([int(0)])
    m.total_params = torch.DoubleTensor([int(0)])

    m.total_params += m.conv1.total_params
    m.total_params += m.norm1.total_params
    m.total_params += m.conv2.total_params
    m.total_params += m.norm2.total_params

    m.real_params = 0
    m.quat_params = 0
    for item in [m.conv1, m.norm1, m.conv2, m.norm2]:
        if hasattr(item, 'real_params'):
            m.real_params += item.real_params
            m.quat_params += item.quat_params
        else:
            m.real_params += item.total_params


def summary(net):
    input = torch.randn(1, 4, 224, 224)
    macs, params = profile(net, inputs=(input,),
                           custom_ops={QuaternionTransposeConv: count_QuaternionTransposeConv,
                                       QuaternionConv: count_QuaternionConv,
                                       QuaternionLinearAutograd: count_QuaternionLinearAutograd,
                                       QuaternionLinear: count_QuaternionLinear,
                                       QuaternionInstanceNorm2d: count_QuaternionInstanceNorm2d,
                                       ShareSepConv: count_ShareSepConv,
                                       SmoothDilatedResidualBlock: count_SmoothDilatedResidualBlock,
                                       ResidualBlock: count_ResidualBlock})

    macs, params = clever_format([macs, params], "%.3f")
    print('MACs: {}', macs)
    print('Params: {}', params)

    total_real = 0
    total_quat = 0
    for name, m in net.named_modules():
        print(name, end=' ')
        if hasattr(m, 'real_params'):
            total_real += m.real_params
            print('\t\t{}\t\t'.format(m.real_params), end='')

        if hasattr(m, 'quat_params'):
            total_quat += m.quat_params
            print('\t\t{}\t\t'.format(m.quat_params))

    print(f'Total real: {total_real} Total quat: {total_quat}')
    num_elements = 0
    for param in net.parameters():
        num_elements += param.numel()
    print(num_elements)


if __name__ == '__main__':
    pass