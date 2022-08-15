import sys
sys.path.insert(0,'/content/deweather/QTransWeather-dehaze/')

import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import vgg16
import kornia as K

import cv2
import numpy as np
from tqdm.notebook import tqdm
from glob import glob
from PIL import Image
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

'''
    lr = args.lr * np.ones([args.epochs])
    lr[20:] = lr[0] / 10.0

    train_low_data_names = glob(args.data_dir + '/data/our485/low/*.png') + \
                           glob(args.data_dir + '/data/syn/low/*.png')
    train_low_data_names.sort()
    train_high_data_names= glob(args.data_dir + '/data/our485/high/*.png') + \
                           glob(args.data_dir + '/data/syn/high/*.png')
    train_high_data_names.sort()
    eval_low_data_names  = glob(args.data_dir + '/eval/low/*.*')
    eval_low_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('Number of training data: %d' % len(train_low_data_names))

    model.train(train_low_data_names,
                train_high_data_names,
                eval_low_data_names,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                epoch=args.epochs,
                lr=lr,
                vis_dir=args.vis_dir,
                ckpt_dir=args.ckpt_dir,
                eval_every_epoch=10,
                train_phase="Decom")

class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()
        self.DecomNet   = DecomNet() # this network is used for decomposition
        self.RelightNet = RelightNet()

    def forward(self, input_low, input_high):
        # Forward DecompNet
        # so we take the images, and this network needs
        # input_low image and input_high image
        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).cuda()
        input_high= Variable(torch.FloatTensor(torch.from_numpy(input_high))).cuda()
        # reflection, illumination
        R_low, I_low   = self.DecomNet(input_low) # input_low
        # R, I
        R_high, I_high = self.DecomNet(input_high) # input high
        # SO we decompose both parts
        # AND both parts has this components R and I, AND we know that Rs are equal

        # Forward RelightNet
        I_delta = self.RelightNet(I_low, R_low) # so I run the RelightNet, just like this, Relight is delta!!!!!
        # And the thing is that I_delta (illumination is only delta for the illumination)

        # Other variables
        I_low_3  = torch.cat((I_low, I_low, I_low), dim=1) # then I connect to each all the lows, Three of them, But they are the same
        # it's a computational trick
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1) # this is the high component, again, I thnk we use it only to compute
        # the loss-function, there is nothing wierd here
        I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1) # this is my delta

        # Compute losses
        # loss function for the reconstruction of the low -part
        self.recon_loss_low  = F.l1_loss(R_low * I_low_3,  input_low) #
        self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high) # R_high * I_high_3
        self.recon_loss_mutal_low  = F.l1_loss(R_high * I_low_3, input_low) # R_high * I_low_3
        self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high) # R_low * I_high_3
        self.equal_R_loss = F.l1_loss(R_low,  R_high.detach())#
        self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high)

        self.Ismooth_loss_low   = self.smooth(I_low, R_low)
        self.Ismooth_loss_high  = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        self.loss_Decom = self.recon_loss_low + \
                          self.recon_loss_high + \
                          0.001 * self.recon_loss_mutal_low + \
                          0.001 * self.recon_loss_mutal_high + \
                          0.1 * self.Ismooth_loss_low + \
                          0.1 * self.Ismooth_loss_high + \
                          0.01 * self.equal_R_loss
        self.loss_Relight = self.relight_loss + \
                            3 * self.Ismooth_loss_delta

        self.output_R_low   = R_low.detach().cpu()
        self.output_I_low   = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S       = R_low.detach().cpu() * I_delta_3.detach().cpu()

    def gradient(self, input_tensor, direction):
        # so it's very good implementation of the gradient
        # along one of the axis
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        # so we can use this thing to compute the average gradient
        # F.avg_pool2d to average it
        #
        return F.avg_pool2d(self.gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        # So as an input, input_I and input_R
        # so we take one channel input_R 0.299*input_R
        # so this way we convert reflection to grayscale!
        # then we add one dimension to it
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        # !!!!! And it uses exponentiated gradient
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data_names)):
            eval_low_img   = Image.open(eval_low_data_names[idx])
            eval_low_img   = np.array(eval_low_img, dtype="float32")/255.0
            eval_low_img   = np.transpose(eval_low_img, (2, 0, 1))
            input_low_eval = np.expand_dims(eval_low_img, axis=0)

            if train_phase == "Decom":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                input    = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                cat_image= np.concatenate([input, result_1, result_2], axis=2)
            if train_phase == "Relight":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                result_3 = np.squeeze(result_3)
                result_4 = np.squeeze(result_4)
                cat_image= np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png' %
                       (train_phase, idx + 1, epoch_num))
            im.save(filepath[:-4] + '.jpg')


    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name= save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == 'Relight':
            torch.save(self.RelightNet.state_dict(),save_name)

    def load(self, ckpt_dir):
        load_dir   = ckpt_dir + '/' + self.train_phase + '/'
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts)>0:
                load_ckpt  = load_ckpts[-1]
                global_step= int(load_ckpt[:-4])
                ckpt_dict  = torch.load(load_dir + load_ckpt)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                elif self.train_phase == 'Relight':
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0


    def train(self,
              train_low_data_names,
              train_high_data_names,
              eval_low_data_names,
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)

        # Create the optimizers
        self.train_op_Decom   = optim.Adam(self.DecomNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999))
        self.train_op_Relight = optim.Adam(self.RelightNet.parameters(),
                                           lr=lr[0], betas=(0.9, 0.999))

        # Initialize a network if its checkpoint is available
        self.train_phase= train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num    = global_step
            start_epoch = global_step // numBatch
            start_step  = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num    = 0
            start_epoch = 0
            start_step  = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
             (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id   = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group['lr'] = self.lr
            for param_group in self.train_op_Relight.param_groups:
                param_group['lr'] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                batch_input_high= np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                for patch_id in range(batch_size):
                    # Load images
                    train_low_img = Image.open(train_low_data_names[image_id])
                    train_low_img = np.array(train_low_img, dtype='float32')/255.0
                    train_high_img= Image.open(train_high_data_names[image_id])
                    train_high_img= np.array(train_high_img, dtype='float32')/255.0
                    # Take random crops
                    h, w, _        = train_low_img.shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
                    train_low_img = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    train_high_img= train_high_img[x: x + patch_size, y: y + patch_size, :]
                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img = np.flipud(train_low_img)
                        train_high_img= np.flipud(train_high_img)
                    if random.random() < 0.5:
                        train_low_img = np.fliplr(train_low_img)
                        train_high_img= np.fliplr(train_high_img)
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img = np.rot90(train_low_img, rot_type)
                        train_high_img= np.rot90(train_high_img, rot_type)
                    # Permute the images to tensor format
                    train_low_img = np.transpose(train_low_img, (2, 0, 1))
                    train_high_img= np.transpose(train_high_img, (2, 0, 1))
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, :] = train_low_img
                    batch_input_high[patch_id, :, :, :]= train_high_img
                    self.input_low = batch_input_low
                    self.input_high= batch_input_high

                    image_id = (image_id + 1) % len(train_low_data_names)
                    if image_id == 0:
                        tmp = list(zip(train_low_data_names, train_high_data_names))
                        random.shuffle(list(tmp))
                        train_low_data_names, train_high_data_names = zip(*tmp)


                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low,  self.input_high)
                if self.train_phase == "Decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    loss = self.loss_Decom.item()
                elif self.train_phase == "Relight":
                    self.train_op_Relight.zero_grad()
                    self.loss_Relight.backward()
                    self.train_op_Relight.step()
                    loss = self.loss_Relight.item()

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data_names,
                              vis_dir=vis_dir, train_phase=train_phase)
                self.save(iter_num, ckpt_dir)

        print("Finished training for phase %s." % train_phase)
'''

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


# calculate the five-point positive definite Laplacian matrix

'''
# check how this padarray does work in matlab
ux = np.padarray(uvx, h, 'pre')
ux = ux[:-h]
uy = np.padarray(uvy, 1, 'pre')
uy = uy[:-1]
D = uvx + ux + uvy + uy
# what is spdiags in the matlab
T = np.spdiags([-uvx, -uvy],[-h,-1],hw,hw)
# calculate the variable of linear system
MN = T + T.T + spdiags(D, 0, hw, hw);               # M in Eq.(12) or N in Eq.(13)
ir2 = ir**2                                         # R^{T}R in Eq.(12) or I^{T}I in Eq.(13)
ir2 = np.spdiags(ir2(:),0,hw,hw)
DEN = ir2 + alphabet * MN + lmbd * np.speye(hw, hw) # denominator in Eq.(12) or Eq.(13)
NUM = ir.*s + lmbd * b                              # numerator in Eq.(12) or Eq.(13)
L = ichol(DEN,struct('michol','on'))
[dst,~] = pcg(DEN, NUM(:), 0.01, 40, L, L.T)
dst = np.reshape(dst, (h, w))
'''

if __name__ == '__main__':
    #TODO: write that feature extraction
    #TODO: write