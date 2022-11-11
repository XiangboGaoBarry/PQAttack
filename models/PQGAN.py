import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from torchvision import utils
import numpy as np
from tqdm import tqdm
import os

from .fid import load_images as fid_load, calculate_fid
from .networks import GeneratorStage1, GeneratorStage2, GeneratorStage34, \
	DiscriminatorStage1, DiscriminatorStage2, DiscriminstorStage34

class PQGAN(object):

    def __init__(self, args, eval=False):
        if eval:
            self.eval_init()
            return None
        print("WGAN_GradientPenalty init model.")
        self.args = args

        self.G1 = GeneratorStage1(args)
        self.D1 = DiscriminatorStage1(args)
        self.G2 = GeneratorStage2(args)
        self.D2 = DiscriminatorStage2(args)
        self.G34 = GeneratorStage34(args)
        self.D34 = DiscriminstorStage34(args)

        print("self.args.load_model:", self.args.load_model)
        if self.args.load_model: self.load_model()

        for net in [self.G1, self.D1, self.G2, self.D2, self.G34, self.D34]:
            self.train_setup(net)

        if torch.cuda.device_count() > 1:
            self.G1 = nn.DataParallel(self.G1)
            self.D1 = nn.DataParallel(self.D1)
            self.G2 = nn.DataParallel(self.G2)
            self.D2 = nn.DataParallel(self.D2)
            self.G34 = nn.DataParallel(self.G34)
            self.D34 = nn.DataParallel(self.D34)

        self.check_cuda(args.cuda)

        # WGAN values from paper
        self.learning_rate = args.lr
        self.b1 = 0
        self.b2 = 0.9
        self.batch_size = args.batch_size

        # WGAN_gradient penalty uses ADAM
        self.d1_optimizer = optim.Adam(self.D1.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g1_optimizer = optim.Adam(self.G1.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.d2_optimizer = optim.Adam(self.D2.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g2_optimizer = optim.Adam(self.G2.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.d34_optimizer = optim.Adam(self.D34.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g34_optimizer = optim.Adam(self.G34.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        self.d1_scheduler = optim.lr_scheduler.MultiStepLR(self.d1_optimizer, args.milestone, gamma=0.5)
        self.g1_scheduler = optim.lr_scheduler.MultiStepLR(self.g1_optimizer, args.milestone, gamma=0.5)
        self.d2_scheduler = optim.lr_scheduler.MultiStepLR(self.d2_optimizer, args.milestone, gamma=0.5)
        self.g2_scheduler = optim.lr_scheduler.MultiStepLR(self.g2_optimizer, args.milestone, gamma=0.5)
        self.d34_scheduler = optim.lr_scheduler.MultiStepLR(self.d34_optimizer, args.milestone, gamma=0.5)
        self.g34_scheduler = optim.lr_scheduler.MultiStepLR(self.g34_optimizer, args.milestone, gamma=0.5)

        self.generator_iters = args.generator_iters
        self.critic_iter = 5
        self.lambda_term = 10

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda()
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D1.cuda()
            self.G1.cuda()
            self.D2.cuda()
            self.G2.cuda()
            self.G34.cuda()
            self.D34.cuda()
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def train_setup(self, net):
        net_para = sum([np.prod(list(net.size())) for net in net.parameters()])
        print('Model {} : params: {:4f}M'.format(net._get_name(), net_para * 4 / 1024 / 1024))

    def change_requires_grad(self, mode_gen=False):
        for p in self.D1.parameters(): p.requires_grad = (not mode_gen)
        for p in self.D2.parameters(): p.requires_grad = (not mode_gen)
        for p in self.D34.parameters(): p.requires_grad = (not mode_gen)

        for p in self.G1.parameters(): p.requires_grad = mode_gen
        for p in self.G2.parameters(): p.requires_grad = mode_gen
        for p in self.G34.parameters(): p.requires_grad = mode_gen


    def gt_preparation(self, gt1, gt2, gt34, conditions_gt):
        gt1, gt2, gt34, conditions_gt = self.get_torch_variable(gt1), self.get_torch_variable(gt2), \
                self.get_torch_variable(gt34), self.get_torch_variable(conditions_gt)
        conditions_gt = conditions_gt.unsqueeze(1).view(
            conditions_gt.shape[0], 1, conditions_gt.shape[1],1,1).repeat(1,self.patch_count // 6,1,1,1)
        conditions_gt = conditions_gt.view(self.batch_size * self.patch_count // 6, 4, 1, 1)
        gt1 = gt1.view(gt1.shape[0] * gt1.shape[1], gt1.shape[2], gt1.shape[3], gt1.shape[4])
        gt2 = gt2.view(gt2.shape[0] * gt2.shape[1], gt2.shape[2], gt2.shape[3], gt2.shape[4])
        gt34 = gt34.view(gt34.shape[0] * gt34.shape[1], gt34.shape[2], gt34.shape[3], gt34.shape[4])
        return gt1, gt2, gt34, conditions_gt

    def stage_1(self, ctrlc=None):
        # if not ctrlz:
        z = self.get_torch_variable(torch.randn(self.batch_size * self.patch_count, self.args.nz, 1, 1))
        # else:
        # z = self.get_torch_variable(torch.randn(1, self.args.nz, 1, 1).repeat(self.batch_size * self.patch_count,1,1,1))
        if ctrlc:
            conditions_fake = torch.rand(1, 4, 1, 1).repeat(self.batch_size * self.patch_count,1,1,1)
        else:
            conditions_fake = torch.rand(self.batch_size * self.patch_count// 6,4,1,1) \
                                    .unsqueeze(0) \
                                    .repeat(6,1,1,1,1) \
                                    .view(self.batch_size * self.patch_count, 4, 1, 1)
        conditions_fake = self.get_torch_variable(conditions_fake)
        fake_images = self.G1(z, conditions_fake)
        return fake_images, conditions_fake

    def stage_2of3(self, fake_images, conditions_fake):
        conditions_fake = conditions_fake.reshape(6,self.batch_size * self.patch_count // 6,4,1,1)[:4]\
                        .view(4 * self.batch_size * self.patch_count // 6,4,1,1)
        # Stage 2
        z_s2 = self.get_torch_variable(torch.randn(self.batch_size * self.patch_count//3 * 2, self.args.nzp, 1, 1))
        # z_s2_right = self.get_torch_variable(torch.randn(self.batch_size * self.patch_count//3, self.args.nzp, 1, 1))
        fake_images = fake_images.view(6, self.batch_size * self.patch_count//6, 
                                        fake_images.shape[1], 
                                        fake_images.shape[2], 
                                        fake_images.shape[3])
        fake_images_stage2_top_left = torch.cat(
                    (fake_images[0], self.get_torch_variable(torch.zeros_like(fake_images[0])), fake_images[1]),
                    dim=-1)
        fake_images_stage2_top_right = torch.cat(
                    (fake_images[1], self.get_torch_variable(torch.zeros_like(fake_images[0])), fake_images[2]),
                    dim=-1)
        fake_images_stage2_bottom_left = torch.cat(
                    (fake_images[3], self.get_torch_variable(torch.zeros_like(fake_images[0])), fake_images[4]),
                    dim=-1)
        fake_images_stage2_bottom_right = torch.cat(
                    (fake_images[4], self.get_torch_variable(torch.zeros_like(fake_images[0])), fake_images[5]),
                    dim=-1)
        fake_images_stage2 = torch.cat((fake_images_stage2_top_left.unsqueeze(0),
                                    fake_images_stage2_top_right.unsqueeze(0),
                                    fake_images_stage2_bottom_left.unsqueeze(0),
                                    fake_images_stage2_bottom_right.unsqueeze(0)), 0)\
                                        .view(self.batch_size * self.patch_count//3 * 2,
                                            fake_images_stage2_top_left.shape[1], 
                                            fake_images_stage2_top_left.shape[2],
                                            fake_images_stage2_top_left.shape[3])
        fake_stage2_gen = self.G2(z_s2, fake_images_stage2, conditions_fake)
        fake_stage2_gen = fake_stage2_gen.view(4, 
                                            self.batch_size * self.patch_count//6, 
                                            fake_stage2_gen.shape[1],
                                            fake_stage2_gen.shape[2],
                                            fake_stage2_gen.shape[3])
        fake_stage2_top_left = torch.cat(
            (fake_images[0], fake_stage2_gen[0], fake_images[1]),
            dim=-1)
        fake_stage2_top_right = torch.cat(
            (fake_images[1], fake_stage2_gen[1], fake_images[2]),
            dim=-1)
        fake_stage2_bottom_left = torch.cat(
            (fake_images[3], fake_stage2_gen[2], fake_images[4]),
            dim=-1)
        fake_stage2_bottom_right = torch.cat(
            (fake_images[4], fake_stage2_gen[3], fake_images[5]),
            dim=-1)
        return (fake_stage2_top_left, fake_stage2_top_right, fake_stage2_bottom_left, fake_stage2_bottom_right), conditions_fake

    def stage_34(self, fake_stage2, conditions_fake):
        # Stage 3
        conditions_fake = conditions_fake.view(4,self.batch_size * self.patch_count // 6,4,1,1)[:2]
        conditions_fake = conditions_fake.view(2 * self.batch_size * self.patch_count // 6,4,1,1)
        z_s3 = self.get_torch_variable(torch.randn(self.batch_size * self.patch_count//3, self.args.nzp3, 1, 1))
        fake_images_stage3_left = \
            torch.cat((fake_stage2[0],
                    self.get_torch_variable(torch.zeros_like(fake_stage2[0])),
                    fake_stage2[1]), dim=-2)
        fake_images_stage3_right = \
            torch.cat((fake_stage2[2],
                    self.get_torch_variable(torch.zeros_like(fake_stage2[2])),
                    fake_stage2[3]), dim=-2)
        fake_images_stage3 = torch.cat((fake_images_stage3_left.unsqueeze(0),
                                    fake_images_stage3_right.unsqueeze(0)), 0)\
                                        .view(self.batch_size * self.patch_count//3,
                                            fake_images_stage3_left.shape[1], 
                                            fake_images_stage3_left.shape[2],
                                            fake_images_stage3_left.shape[3])
        fake_stage3_gen = self.G34(z_s3, fake_images_stage3, conditions_fake)
        fake_stage3_gen = fake_stage3_gen.view(2, 
                                            self.batch_size * self.patch_count//6, 
                                            fake_stage3_gen.shape[1],
                                            fake_stage3_gen.shape[2],
                                            fake_stage3_gen.shape[3])

        # Stage 4
        conditions_fake = conditions_fake.reshape(2,self.batch_size * self.patch_count // 6,4,1,1)[0]
        z_s4 = self.get_torch_variable(torch.randn(self.batch_size * self.patch_count//6, self.args.nzp3, 1, 1))
        fake_image_stage4_top = torch.cat((fake_stage2[0][:, :, :, self.patch_size:],
                                        fake_stage2[1][:, :, :, self.patch_size:self.patch_size*2]), dim=-1)
        fake_image_stage4_bottom = torch.cat((fake_stage2[2][:, :, :, self.patch_size:],
                                        fake_stage2[3][:, :, :, self.patch_size:self.patch_size*2]), dim=-1)
        fake_image_stage4_middle = torch.cat((fake_stage3_gen[0],
                                            self.get_torch_variable(torch.zeros_like(fake_stage3_gen[0])),
                                            fake_stage3_gen[1]), dim=-1)
        fake_image_stage4 = torch.cat((fake_image_stage4_top, fake_image_stage4_middle, fake_image_stage4_bottom), dim=-2)
        fake_image_gen_stage4 = self.G34(z_s4, fake_image_stage4, conditions_fake)
        fake_image_stage4_middle = torch.cat((fake_stage3_gen[0],
                                            fake_image_gen_stage4,
                                            fake_stage3_gen[1]), dim=-1)
        fake_image_stage4 = torch.cat((fake_image_stage4_top, fake_image_stage4_middle, fake_image_stage4_bottom), dim=-2)
        return (fake_image_stage4, fake_image_gen_stage4), conditions_fake

    def generate_image(self, ctrlc=None):
        fake_images, conditions_fake_1 = self.stage_1()
        fake_stage2, conditions_fake_2  = self.stage_2of3(fake_images, conditions_fake_1)
        (fake_image_stage4, fake_image_gen_stage4), conditions_fake = self.stage_34(fake_stage2, conditions_fake_2)
        return (fake_images, conditions_fake_1), \
            (fake_stage2, conditions_fake_2), \
            (fake_image_stage4, fake_image_gen_stage4, conditions_fake)


    def train(self, train_loader):

        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()

        self.D1.train()
        self.D2.train()
        self.D34.train()
        pbar = tqdm(range(self.args.load_iter+1, self.generator_iters))
        fid0 = fid1 = fid2 = fid3 = fid4 = fid5 = 0
        for g_iter in pbar:

            self.change_requires_grad(False)

            for d_iter in range(self.critic_iter):
                self.D1.zero_grad()
                self.D2.zero_grad()
                self.D34.zero_grad()

                self.G1.train()
                self.G2.train()
                self.G34.train()

                gt1, gt2, gt34, conditions_gt = self.data.__next__()
                self.batch_size = conditions_gt.shape[0]
                self.patch_count = patch_count = gt34.shape[1] * 6
                self.patch_size = 64
                gt1, gt2, gt34, conditions_gt = self.gt_preparation(gt1, gt2, gt34, conditions_gt)

                # Grount truth
                d1_loss_real = self.D1(gt1, conditions_gt).mean()
                d2_loss_real = self.D2(gt2, conditions_gt).mean()
                d34_loss_real = self.D34(gt34, conditions_gt).mean()

                # Fake
                (fake_images, conditions_fake_1), \
                ((fake_stage2_top_left, fake_stage2_top_right, fake_stage2_bottom_left, fake_stage2_bottom_right), conditions_fake_2), \
                (fake_image_stage4, fake_image_gen_stage4, conditions_fake_34) = self.generate_image()
                d1_loss_fake = self.D1(fake_images, conditions_fake_1).mean()
                d2_loss_fake = self.D2(fake_stage2_top_left, conditions_fake_34).mean()
                d2_loss_fake += self.D2(fake_stage2_top_right, conditions_fake_34).mean()
                d2_loss_fake += self.D2(fake_stage2_bottom_left, conditions_fake_34).mean()
                d2_loss_fake += self.D2(fake_stage2_bottom_right, conditions_fake_34).mean()
                
                d2_loss_fake /= 4
                
                d1_loss_fake += self.D1(fake_stage2_top_left[:,:,:,self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
                d1_loss_fake += self.D1(fake_stage2_top_left[:,:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()
                
                d1_loss_fake += self.D1(fake_stage2_top_right[:,:,:,self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
                d1_loss_fake += self.D1(fake_stage2_top_right[:,:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()
                
                d1_loss_fake += self.D1(fake_stage2_bottom_left[:,:,:,self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
                d1_loss_fake += self.D1(fake_stage2_bottom_left[:,:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()
                
                d1_loss_fake += self.D1(fake_stage2_bottom_right[:,:,:,self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
                d1_loss_fake += self.D1(fake_stage2_bottom_right[:,:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()
                
                d34_loss_fake = self.D34(fake_image_stage4, conditions_fake_34).mean()
                d1_loss_fake += self.D1(fake_image_gen_stage4, conditions_fake_34).mean()
                d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size:self.patch_size * 2, 
                								self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
                d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size:self.patch_size * 2, 
                								self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()

                d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2:self.patch_size//2 + self.patch_size, 
                								self.patch_size:self.patch_size * 2], conditions_fake_34).mean()
                d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2, 
                								self.patch_size:self.patch_size * 2], conditions_fake_34).mean()

                d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2:self.patch_size//2 + self.patch_size, 
                								self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
                d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2:self.patch_size//2 + self.patch_size, 
                								self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()

                d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2, 
                								self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
                d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2, 
                								self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()
                

                gradient_penalty_1 = self.calculate_gp_with_c(gt1.data, fake_images[:conditions_gt.shape[0]].data, conditions_gt.data,
                                                                conditions_fake_1[:conditions_gt.shape[0]].data, self.D1)
                gradient_penalty_2 = self.calculate_gp_with_c(gt2.data, fake_stage2_top_left.data, conditions_gt.data,
                                                                conditions_fake_2[:conditions_gt.shape[0]].data, self.D2)
                gradient_penalty_34 = self.calculate_gp_with_c(gt34.data, fake_image_stage4.data, conditions_gt.data,
                                                                conditions_fake_34.data, self.D34)

                d1_loss_fake /= 18
                (d1_loss_real + d2_loss_real + d34_loss_real).backward(mone)
                (d1_loss_fake + d2_loss_fake + d34_loss_fake).backward(one)
                (gradient_penalty_1 + gradient_penalty_2 + gradient_penalty_34).backward()

                self.d1_optimizer.step()
                self.d2_optimizer.step()
                self.d34_optimizer.step()
            
            self.change_requires_grad(mode_gen=True)
            self.G1.zero_grad()
            self.G2.zero_grad()
            self.G34.zero_grad()

            # Fake
            (fake_images, conditions_fake_1), \
            ((fake_stage2_top_left, fake_stage2_top_right, fake_stage2_bottom_left, fake_stage2_bottom_right), conditions_fake_2), \
            (fake_image_stage4, fake_image_gen_stage4, conditions_fake_34) = self.generate_image()
            
            d1_loss_fake = self.D1(fake_images, conditions_fake_1).mean()
            d2_loss_fake = self.D2(fake_stage2_top_left, conditions_fake_34).mean()
            d2_loss_fake += self.D2(fake_stage2_top_right, conditions_fake_34).mean()
            d2_loss_fake += self.D2(fake_stage2_bottom_left, conditions_fake_34).mean()
            d2_loss_fake += self.D2(fake_stage2_bottom_right, conditions_fake_34).mean()
            
            d2_loss_fake /= 4
            
            d1_loss_fake += self.D1(fake_stage2_top_left[:,:,:,self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
            d1_loss_fake += self.D1(fake_stage2_top_left[:,:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()
            
            d1_loss_fake += self.D1(fake_stage2_top_right[:,:,:,self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
            d1_loss_fake += self.D1(fake_stage2_top_right[:,:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()
            
            d1_loss_fake += self.D1(fake_stage2_bottom_left[:,:,:,self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
            d1_loss_fake += self.D1(fake_stage2_bottom_left[:,:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()
            
            d1_loss_fake += self.D1(fake_stage2_bottom_right[:,:,:,self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
            d1_loss_fake += self.D1(fake_stage2_bottom_right[:,:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()
            
            d34_loss_fake = self.D34(fake_image_stage4, conditions_fake_34).mean()
            d1_loss_fake += self.D1(fake_image_gen_stage4, conditions_fake_34).mean()
            d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size:self.patch_size * 2, 
                                            self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
            d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size:self.patch_size * 2, 
                                            self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()

            d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2:self.patch_size//2 + self.patch_size, 
                                            self.patch_size:self.patch_size * 2], conditions_fake_34).mean()
            d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2, 
                                            self.patch_size:self.patch_size * 2], conditions_fake_34).mean()

            d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2:self.patch_size//2 + self.patch_size, 
                                            self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
            d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2:self.patch_size//2 + self.patch_size, 
                                            self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()

            d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2, 
                                            self.patch_size//2:self.patch_size//2 + self.patch_size], conditions_fake_34).mean()
            d1_loss_fake += self.D1(fake_image_stage4[:,:,self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2, 
                                            self.patch_size//2 + self.patch_size:self.patch_size//2 + self.patch_size * 2], conditions_fake_34).mean()
            
            d1_loss_fake /= 18
            
            (d1_loss_fake + d2_loss_fake + d34_loss_fake).backward(mone)
            self.g1_optimizer.step()
            self.g2_optimizer.step()
            self.g34_optimizer.step()
            

            if (g_iter) % self.args.save_per_iter == 0:

                self.G1.eval()
                self.G2.eval()
                self.G34.eval()
                self.save_model(g_iter)
                if not os.path.exists('training_result_images/'):
                    os.makedirs('training_result_images/')
                if not os.path.isdir('training_result_images/%s/' % self.args.save_file):
                    os.makedirs('training_result_images/%s/' % self.args.save_file)
                if not os.path.isdir('training_result_images/%s/stage_34' % self.args.save_file):
                    os.makedirs('training_result_images/%s/stage_34' % self.args.save_file)
                
                with torch.no_grad():
                    # z = self.get_torch_variable(torch.randn(1, self.args.nz, 1, 1).repeat(self.batch_size * self.patch_count,1,1,1))
                     _, _,(fake_image_stage4, fake_image_gen_stage4, conditions_fake) = self.generate_image(True)

                fake_image = fake_image_stage4
                fake_image = fake_image.view(-1, 1, fake_image.shape[2], fake_image.shape[3])
                grid = utils.make_grid(fake_image, nrow=6, padding=5, pad_value=0.2)
                utils.save_image(grid, 'training_result_images/{}/stage_34/img_generatori_iter_{}.png'.format(
                    self.args.save_file,
                    str(g_iter).zfill(3)))
                
            if (g_iter) % self.args.fid_per_iter == 0:
            # if False:
                image_path = 'training_result_images/%s/stage_34/images' % self.args.save_file
                if not os.path.isdir(image_path): os.mkdir(image_path)
                image_path_real = '%s/real' % image_path
                image_path_fake = '%s/fake' % image_path
                if not os.path.isdir(image_path_real): os.mkdir(image_path_real)
                if not os.path.isdir(image_path_fake): os.mkdir(image_path_fake)
                fid0, fid1, fid2, fid3, fid4, fid5 = self.cal_FID(image_path_real, image_path_fake)
                with open("%s/fids.txt" % image_path, 'a') as fid_log:
                    fid_log.write("%06d %.5f %.5f %.5f %.5f %.5f %.5f\n" % (g_iter, fid0, fid1, fid2, fid3, fid4, fid5))
                # pbar.set_description("fids:  %.5f %.5f %.5f %.5f %.5f %.5f" % (fid0, fid1, fid2, fid3, fid4, fid5))
            pbar.set_description("fids: %.1f %.1f %.1f %.1f %.1f %.1f | gp: %.3f %.3f %.3f" % \
                (fid0, fid1, fid2, fid3, fid4, fid5, gradient_penalty_1, gradient_penalty_2, gradient_penalty_34))

            self.d1_scheduler.step()
            self.g1_scheduler.step()
            self.d2_scheduler.step()
            self.g2_scheduler.step()
            self.d34_scheduler.step()
            self.g34_scheduler.step()

        self.save_model(-1)
        

    def evaluate(self):
        print("model loading...")
        self.load_model(eval=True)
        self.G1.eval()
        self.G2.eval()
        self.G34.eval()
        with torch.no_grad():
            fake_image_stage4 = self.generate_image()[-2]
        fake_image_stage4 = fake_image_stage4.view(-1, 1, fake_image_stage4.shape[2],
                                                                    fake_image_stage4.shape[3])
        grid = utils.make_grid(fake_image_stage4, nrow=6, padding=5, pad_value=20)
        utils.save_image(grid, 'generated.png')


    def calculate_gp_with_c(self, real_images, fake_images, condition_gt, condition_fake, D):
        batch_size = real_images.shape[0]
        eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        eta_c = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1)
        eta_c = eta_c.expand(batch_size, condition_gt.size(1), condition_gt.size(2), condition_gt.size(3))

        if self.cuda:
            eta = eta.cuda()
            eta_c = eta_c.cuda()
        else:
            eta = eta
            eta_c = eta_c
        interpolated = eta * real_images + ((1 - eta) * fake_images)
        interpolated_cond = eta_c * condition_gt + ((1 - eta_c) * condition_fake)

        if self.cuda:
            interpolated = interpolated.cuda()
            interpolated_cond = interpolated_cond.cuda()
        else:
            interpolated = interpolated
            interpolated_cond = interpolated_cond

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated_cond = Variable(interpolated_cond, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = D(interpolated, interpolated_cond)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=[interpolated, interpolated_cond],
                                    grad_outputs=torch.ones(
                                        prob_interpolated.size()).cuda() if self.cuda else torch.ones(
                                        prob_interpolated.size()),
                                    create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term * 10
        return grad_penalty


    def save_model(self, epoch):
        if not os.path.isdir('pretrained/'):
            os.makedirs('pretrained/')
        if not os.path.isdir('pretrained/%s/' % self.args.save_file):
            os.makedirs('pretrained/%s/' % self.args.save_file)
        if not os.path.isdir('pretrained/%s/stage_1' % self.args.save_file):
            os.makedirs('pretrained/%s/stage_1' % self.args.save_file)
        if not os.path.isdir('pretrained/%s/stage_2' % self.args.save_file):
            os.makedirs('pretrained/%s/stage_2' % self.args.save_file)
        if not os.path.isdir('pretrained/%s/stage_34' % self.args.save_file):
            os.makedirs('pretrained/%s/stage_34' % self.args.save_file)
        
        try:
            G1_stat_dict = self.G1.module.state_dict()
            D1_stat_dict = self.D1.module.state_dict()
            G2_stat_dict = self.G2.module.state_dict()
            D2_stat_dict = self.D2.module.state_dict()
            G34_stat_dict = self.G34.module.state_dict()
            D34_stat_dict = self.D34.module.state_dict()
        except AttributeError:
            D1_stat_dict = self.D1.state_dict()
            G1_stat_dict = self.G1.state_dict()
            D2_stat_dict = self.D2.state_dict()
            G2_stat_dict = self.G2.state_dict()
            G34_stat_dict = self.G34.state_dict()
            D34_stat_dict = self.D34.state_dict()
        torch.save(G1_stat_dict, './pretrained/%s/stage_1/generator_%06d.pkl' % (self.args.save_file, epoch))
        torch.save(D1_stat_dict, './pretrained/%s/stage_1/discriminator_%06d.pkl' % (self.args.save_file, epoch))
        torch.save(G2_stat_dict, './pretrained/%s/stage_2/generator_%06d.pkl' % (self.args.save_file, epoch))
        torch.save(D2_stat_dict, './pretrained/%s/stage_2/discriminator_%06d.pkl' % (self.args.save_file, epoch))
        torch.save(G34_stat_dict, './pretrained/%s/stage_34/generator_%06d.pkl' % (self.args.save_file, epoch))
        torch.save(D34_stat_dict, './pretrained/%s/stage_34/discriminator_%06d.pkl' % (self.args.save_file, epoch))

    def load_model(self, eval=False):
        G1_path = './pretrained/%s/stage_1/generator_%06d.pkl' % (self.args.load_path, self.args.load_iter_1)
        self.G1.load_state_dict(torch.load(G1_path))
        print('Generator model stage 1 loaded from %s in %d iterations.' % (self.args.load_path, self.args.load_iter_1))
        G2_path = './pretrained/%s/stage_2/generator_%06d.pkl' % (self.args.load_path, self.args.load_iter_2)
        self.G2.load_state_dict(torch.load(G2_path))
        print('Generator model stage 2 loaded from %s in %d iterations.' % (self.args.load_path, self.args.load_iter_2))
        G34_path = './pretrained/%s/stage_34/generator_%06d.pkl' % (self.args.load_path, self.args.load_iter_34)
        self.G34.load_state_dict(torch.load(G34_path))
        print('Generator model stage 34 loaded from %s in %d iterations.' % (self.args.load_path, self.args.load_iter_34))
        if not eval:
            D1_path = './pretrained/%s/stage_1/discriminator_%06d.pkl' % (self.args.load_path, self.args.load_iter_1)
            self.D1.load_state_dict(torch.load(D1_path))
            print('Discriminator model stage 1 loaded from %s in %d iterations.' % (self.args.load_path, self.args.load_iter_1))
            D2_path = './pretrained/%s/stage_2/discriminator_%06d.pkl' % (self.args.load_path, self.args.load_iter_2)
            self.D2.load_state_dict(torch.load(D2_path))
            print('Discriminator model loaded from %s in %d iterations.' % (self.args.load_path, self.args.load_iter_2))
            D34_path = './pretrained/%s/stage_34/discriminator_%06d.pkl' % (self.args.load_path, self.args.load_iter_34)
            self.D34.load_state_dict(torch.load(D34_path))
            print('Discriminator model stage 34 loaded from %s in %d iterations.' % (self.args.load_path, self.args.load_iter_34))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (gt1, gt2, gt34, conditions_gt) in enumerate(data_loader):
                yield gt1, gt2, gt34, conditions_gt

    def cal_FID(self, real_path, fake_path, iters=100):
        n = 0
        for _ in range(iters):
            _, _, gt, conditions_gt = self.data.__next__()
            gt = gt.view(-1, gt.shape[2], gt.shape[3], gt.shape[4])
            with torch.no_grad():
                (fake_images, _), \
                fake_stage2, \
                (fake_image_stage4, fake_image_gen_stage4, conditions_fake) = self.generate_image()
            gen = fake_image_stage4
            for idx in range(len(gen)):
                for depth in range(6):
                    real_path_depth = '%s/%d' % (real_path, depth)
                    fake_path_depth = '%s/%d' % (fake_path, depth)
                    if not os.path.isdir(real_path_depth): os.mkdir(real_path_depth)
                    if not os.path.isdir(fake_path_depth): os.mkdir(fake_path_depth)
                    utils.save_image(gt[idx][depth].unsqueeze(0), "%s/%06d.png" % (real_path_depth, n))
                    utils.save_image(gen[idx][depth].unsqueeze(0), "%s/%06d.png" % (fake_path_depth, n))
                n += 1
        fid_values = []
        for depth in range(6):
            images1 = fid_load('%s/%d' % (real_path, depth))
            images2 = fid_load('%s/%d' % (fake_path, depth))
            fid_values.append(calculate_fid(images1, images2, False, 8))
        return fid_values

        