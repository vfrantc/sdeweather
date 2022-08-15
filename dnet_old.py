import os
import time
import random
from glob import glob
from PIL import Image
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from box import Box

from qcnn import QuaternionConv
from qcnn import get_r, get_i, get_j, get_k

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = QuaternionConv(8, channel, kernel_size * 3, stride=1, padding=4)
        # Activated layers!
        self.net1_convs = nn.Sequential(QuaternionConv(channel, channel, kernel_size, stride=1, padding=1),
                                        nn.ReLU(),
                                        QuaternionConv(channel, channel, kernel_size, stride=1, padding=1),
                                        nn.ReLU(),
                                        QuaternionConv(channel, channel, kernel_size, stride=1, padding=1),
                                        nn.ReLU(),
                                        QuaternionConv(channel, channel, kernel_size, stride=1, padding=1),
                                        nn.ReLU(),
                                        QuaternionConv(channel, channel, kernel_size, stride=1, padding=1),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = QuaternionConv(channel, 8, kernel_size, stride=1, padding=1)

    def edge_compute(self, x):
        x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
        x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])

        y = x.new(x.size())
        y.fill_(0)
        y[:,:,:,1:] += x_diffx
        y[:,:,:,:-1] += x_diffx
        y[:,:,1:,:] += x_diffy
        y[:,:,:-1,:] += x_diffy
        #y = torch.sum(y,1,keepdim=True)/3
        y /= 4
        return y

    def forward(self, input_im):
        b, c, h, w = input_im.shape
        real = torch.zeros((b, 1, h, w)).cuda()

        edges = self.edge_compute(input_im)
        edges = torch.cat((real, edges), dim=1)
        input_im = torch.cat((real, input_im), dim=1)

        input = torch.cat((get_r(edges), get_r(input_im), get_i(edges), get_i(input_im), get_j(edges), get_j(input_im), get_k(edges), get_k(input_im)), dim=1)
        feats0   = self.net1_conv0(input)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)

        b = outs[:, 0, :, :].unsqueeze(1)
        R        = torch.sigmoid(torch.cat((outs[:, 0, :, :].unsqueeze(1), outs[:, 2, :, :].unsqueeze(1), outs[:, 4, :, :].unsqueeze(1), outs[:, 6, :, :].unsqueeze(1)), dim=1))
        L        = torch.sigmoid(torch.cat((outs[:, 1, :, :].unsqueeze(1), outs[:, 3, :, :].unsqueeze(1), outs[:, 5, :, :].unsqueeze(1), outs[:, 7, :, :].unsqueeze(1)), dim=1))
        return R, L

def train():
    opt = Box({'epochs': 200,
               'batch_size': 16,
               'patch_size': 128,
               'lr': 0.0005})

    net = DecomNet()
    net = net.cuda()
    lr = opt.lr * np.ones([opt.epochs])
    lr[20:] = lr[0] / 10.0

    train_input_data_names = glob('./input/*.png')
    train_input_data_names.sort()
    train_slow_data_names = glob('./slow/*.png')
    train_slow_data_names.sort()
    train_fast_data_names = glob('./fast/*.png')
    train_fast_data_names.sort()

    train_op = optim.Adam(net.parameters(), lr=lr[0], betas=(0.9, 0.999))

    image_id = 0
    numBatch = len(train_input_data_names) // int(opt.batch_size)
    start_time = time.time()
    epoch = 0
    iter_num = 0
    for epoch in range(0, opt.epochs):
        slr = lr[epoch]

        # Adjust learning rate
        for param_group in train_op.param_groups:
            param_group['lr'] = slr

        for batch_id in range(0, numBatch):
            # Generate training data for a batch
            batch_input = np.zeros((opt.batch_size, 3, opt.patch_size, opt.patch_size,), dtype="float32")
            batch_slow = np.zeros((opt.batch_size, 3, opt.patch_size, opt.patch_size,), dtype="float32")
            batch_fast = np.zeros((opt.batch_size, 3, opt.patch_size, opt.patch_size,), dtype="float32")

            for patch_id in range(opt.batch_size):
                # Load images
                train_input_img = Image.open(train_input_data_names[image_id])
                train_input_img = 2 * np.array(train_input_img, dtype='float32') / 255.0 - 1
                train_slow_img = Image.open(train_slow_data_names[image_id])
                train_slow_img = np.array(train_slow_img, dtype='float32') / 255.0
                train_fast_img = Image.open(train_fast_data_names[image_id])
                train_fast_img = np.array(train_fast_img, dtype='float32') / 255.0

                # Take random crops
                h, w, _ = train_input_img.shape
                x = random.randint(0, h - opt.patch_size)
                y = random.randint(0, w - opt.patch_size)

                train_input_img = train_input_img[x: x + opt.patch_size, y: y + opt.patch_size, :]
                train_slow_img = train_slow_img[x: x + opt.patch_size, y: y + opt.patch_size, :]
                train_fast_img = train_fast_img[x: x + opt.patch_size, y: y + opt.patch_size, :]

                # Data augmentation
                if random.random() < 0.5:
                    train_input_img = np.flipud(train_input_img)
                    train_slow_img = np.flipud(train_slow_img)
                    train_fast_img = np.flipud(train_fast_img)
                if random.random() < 0.5:
                    train_input_img = np.fliplr(train_input_img)
                    train_slow_img = np.fliplr(train_slow_img)
                    train_fast_img = np.fliplr(train_fast_img)
                rot_type = random.randint(1, 4)
                if random.random() < 0.5:
                    train_input_img = np.rot90(train_input_img, rot_type)
                    train_slow_img = np.rot90(train_slow_img, rot_type)
                    train_fast_img = np.rot90(train_fast_img, rot_type)

                # Permute the images to tensor format
                train_input_img = np.transpose(train_input_img, (2, 0, 1))
                train_slow_img = np.transpose(train_slow_img, (2, 0, 1))
                train_fast_img = np.transpose(train_fast_img, (2, 0, 1))

                # Prepare the batch
                batch_input[patch_id, :, :, :] = train_input_img
                batch_slow[patch_id, :, :, :] = train_slow_img
                batch_fast[patch_id, :, :, :] = train_fast_img

                image_id = (image_id + 1) % len(train_input_data_names)
                if image_id == 0:
                    tmp = list(zip(train_input_data_names, train_slow_data_names, train_fast_data_names))
                    random.shuffle(list(tmp))
                    train_input_data_names, train_slow_data_names, train_fast_data_names = zip(*tmp)

            input = Variable(torch.FloatTensor(torch.from_numpy(batch_input))).cuda()
            b, c, h, w = input.shape
            real = torch.zeros((b, 1, h, w)).cuda()
            target_slow = Variable(torch.FloatTensor(torch.from_numpy(batch_slow))).cuda()
            target_slow = torch.cat((real, target_slow), dim=1)
            input_target = torch.cat((real, input), dim=1)

            target_fast = Variable(torch.FloatTensor(torch.from_numpy(batch_fast))).cuda()
            target_fast = torch.cat((real, target_fast), dim=1)

            out_fast, out_slow = net(input)
            train_op.zero_grad()
            loss = F.l1_loss(out_fast * out_slow, (input_target + 1) / 2) \
                   + F.l1_loss(out_slow, target_slow) \
                   + 100 * F.l1_loss(out_fast, target_fast) \
                   + 0.04 * loss_network(out_slow[:, 1:, :, :], target_slow[:, 1:, :, :]) \
                   + 0.04 * loss_network(out_fast[:, 1:, :, :], target_fast[:, 1:, :, :])

            loss.backward()
            train_op.step()

            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" % (
            epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss.item()))
            iter_num += 1

    print("Finished training...")


class TVLoss(nn.Module):
    '''
    Define Total Variance Loss for images
    which is used for smoothness regularization
    '''

    def __init__(self):
        super(TVLoss, self).__init__()

    def __call__(self, input):
        # Tensor with shape (n_Batch, C, H, W)
        # this is very simple and very smart implementation
        origin = input[:, :, :-1, :-1] # original image
        right = input[:, :, :-1, 1:]
        down = input[:, :, 1:, :-1]

        tv = torch.mean(torch.abs(origin-right)) + torch.mean(torch.abs(origin-down))
        return tv * 0.5

'''
self.r = r
self.eps = eps
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
# this is the way to implement the boxfilter
self.boxfilter = nn.AvgPool2d(kernel_size=2*self.r+1, stride=1,padding=self.r)
'''

'''
def norm(a):
  return np.sqrt(np.sum(a**2))

def convbox(I, r):
    f = np.ones((1, 2*r+1))
    f = f / sum(f.ravel())
    J = np.pad(I, [(r, r), (r, r)], 'edge')
    J = scipy.signal.convolve(J, f, mode='valid', method='auto')
    J = scipy.signal.convolve(J, f.T, mode='valid', method='auto')
    return J

alpha=0.001
beta=0.0001
pI=1.5
pR=0.5
varreps = 0.01
r = 3
K=20
debug=True
eps = 0.000001

I=S
R=np.ones_like(S)
preI = I
preR = R
I = S / R
h, w = s.shape[:2]
hw = h * w
uvx = uvx.ravel()
uvy = uvy.ravel()
ux = np.pad()
'''

if __name__ == '__main__':
    #TODO: write that feature extraction as in star
    #TODO: write a neural network that could be scaled
    #TODO: use TotalVariation loss to regularize the image