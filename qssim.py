import os
import math

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from qcnn.ops import q_normalize
from qcnn.ops import get_r
from qcnn.ops import get_i
from qcnn.ops import get_j
from qcnn.ops import get_k
from qcnn.ops import get_modulus
from qcnn.ops import get_normalized
from qcnn.ops import quaternion_conv
from qcnn.ops import hamilton_product

def q_conj(input, channel=1):
    '''conjugate every quaternion in the feature'''
    r =  get_r(input)
    i = -get_i(input)
    j = -get_j(input)
    k = -get_k(input)
    return torch.cat([r, i, j, k], dim=channel)

def img_to_q(input, channel=-1):
    '''Convert an image '''
    b, c, h, w = input.size()
    real = torch.zeros((b, 1, h, w))
    if input.is_cuda:
        real = real.cuda(real.get_device())
    x = torch.cat([real, input], dim=channel)
    return x


def get_L(input, channel=1):
    red, green, blue = torch.split(input, 1, dim=1)
    L = red/3 + green/3 + blue/3
    luminance = torch.cat([L, L, L], dim=1)
    return luminance

def gaussian(window_size, sigma):
    '''Generate a Tensor with the gaussian thing'''
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window_size, size_average=True):
    ch = 1

    window3 = create_window(window_size, channel=3)
    if img1.is_cuda:
        window3 = window3.cuda(img1.get_device())
    window3 = window3.type_as(img1)

    window4 = create_window(window_size, channel=4)
    if img1.is_cuda:
        window4 = window4.cuda(img1.get_device())
    window4 = window4.type_as(img1)

    # for ch=1 it does nothing
    img1_L = get_L(img1)
    img1_ch = img1 - img1_L
    img1 = img1_ch * ch + img1_L

    img2_L = get_L(img2)
    img2_ch = img2 - img2_L
    img2 = img2_ch * ch + img2_L

    img1_Q = img_to_q(img1)
    img2_Q = img_to_q(img2)

    C1 = torch.tensor([[[[0.01 ** 2]], [[0.]], [[0.]], [[0.]]]])
    C2 = torch.tensor([[[[0.03 ** 2]], [[0.]], [[0.]], [[0.]]]])

    mu1 = F.conv2d(img1, window3, padding=window_size // 2, groups=3)
    mu2 = F.conv2d(img2, window3, padding=window_size // 2, groups=3)
    mu1_Q = img_to_q(mu1)
    mu2_Q = img_to_q(mu2)
    mu1_sq_Q = hamilton_product(mu1_Q, q_conj(mu1_Q))
    mu2_sq_Q = hamilton_product(mu2_Q, q_conj(mu2_Q))
    mu1_mu2_Q = hamilton_product(mu1_Q, q_conj(mu2_Q))

    img1_hue_sq_Q = hamilton_product(img1_Q, q_conj(img1_Q))
    img2_hue_sq_Q = hamilton_product(img2_Q, q_conj(img2_Q))
    img1_img2_hue_Q = hamilton_product(img1_Q, q_conj(img2_Q))

    sigma1_sq_Q = F.conv2d(img1_hue_sq_Q, window4, padding=window_size // 2, groups=4) - mu1_sq_Q
    sigma2_sq_Q = F.conv2d(img2_hue_sq_Q, window4, padding=window_size // 2, groups=4) - mu2_sq_Q
    sigma12_Q = F.conv2d(img1_img2_hue_Q, window4, padding=window_size // 2, groups=4) - mu1_mu2_Q

    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    #  numerator1 = 2*mu1_mu2 + C1;
    #  numerator2 = 2*sigma12 + C2;
    # denominator1 = mu1_sq + mu2_sq + C1;
    #  denominator2 = sigma1_sq + sigma2_sq + C2;
    #  qssim_map = ones(size(mu1));
    #  index = (denominator1.*denominator2 > 0);
    #  qssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
    #  index = (denominator1 ~= 0) & (denominator2 == 0);
    #  qssim_map(index) = numerator1(index)./denominator1(index);

    qssim_map_Q = ((2 * mu1_mu2_Q + C1) * (2 * q_conj(
        sigma12_Q) + C2))  # / ((mu1_sq_Q + mu2_sq_Q + C1) * (q_conj(sigma1_sq_Q + sigma2_sq_Q) + C2))
    ssim_map = get_modulus(qssim_map_Q)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    img1=torch.clamp(img1,min=0,max=1)
    img2=torch.clamp(img2,min=0,max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window_size, size_average)


def run_ssim(img1_name, img2_name, cuda=False):
    img1 = cv2.imread(img1_name, 1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img1 = np.transpose(img1, (2, 0, 1))
    img1 = np.expand_dims(img1, axis=0)
    img1 = Variable(torch.FloatTensor(torch.from_numpy(img1)))
    if cuda:
        img1 = img1.cuda()

    img2 = cv2.imread(img2_name, 1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img2 = np.transpose(img2, (2, 0, 1))
    img2 = np.expand_dims(img2, axis=0)
    img2 = Variable(torch.FloatTensor(torch.from_numpy(img2)))
    if cuda:
        img2 = img2.cuda()
    return ssim(img1, img2)



if __name__ == '__main__':
    #TODO: Make sure it is equivalent to the original implmentation
    #TODO: Make a loss function out of this
    import cv2
    import numpy as np

    print(run_ssim('qssim/image1.jpg', 'qssim/image2.jpg'))

    img1 = cv2.imread('qssim/image1.jpg', 1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img1 = np.transpose(img1, (2, 0, 1))
    img1 = np.expand_dims(img1, axis=0)
    img1 = Variable(torch.FloatTensor(torch.from_numpy(img1)))