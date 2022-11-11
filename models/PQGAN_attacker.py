from math import floor, ceil
from re import T
import torch
from torch.nn.functional import interpolate
from torch import nn, nuclear_norm
import sys
import numpy as np
from models.networks import GeneratorStage1, GeneratorStage2, GeneratorStage34

class PQGAN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.G1 = GeneratorStage1(args)
        self.G2 = GeneratorStage2(args)
        self.G34 = GeneratorStage34(args)

        if self.args.load_model: self.load_model()

        for net in [self.G1, self.G2, self.G34]:
            self.model_setup(net)

        if torch.cuda.device_count() > 1:
            self.G1 = nn.DataParallel(self.G1)
            self.G2 = nn.DataParallel(self.G2)
            self.G34 = nn.DataParallel(self.G34)
    
    def to_device(self, cuda):
        if cuda:
            self.G1 = self.G1.cuda()
            self.G2 = self.G2.cuda()
            self.G34 = self.G34.cuda()

    def model_setup(self, net):
        net_para = sum([np.prod(list(net.size())) for net in net.parameters()])
        print('Model {} : params: {:4f}M'.format(net._get_name(), net_para * 4 / 1024 / 1024))
    
    def load_model(self):
        G1_path = '%s/stage_1/generator_%06d.pkl' % (self.args.load_path, self.args.load_iter_1)
        self.G1.load_state_dict(torch.load(G1_path))
        print('Generator model stage 1 loaded from %s in %d iterations.' % (self.args.load_path, self.args.load_iter_1))
        G2_path = '%s/stage_2/generator_%06d.pkl' % (self.args.load_path, self.args.load_iter_2)
        self.G2.load_state_dict(torch.load(G2_path))
        print('Generator model stage 2 loaded from %s in %d iterations.' % (self.args.load_path, self.args.load_iter_2))
        G34_path = '%s/stage_34/generator_%06d.pkl' % (self.args.load_path, self.args.load_iter_34)
        self.G34.load_state_dict(torch.load(G34_path))
        print('Generator model stage 34 loaded from %s in %d iterations.' % (self.args.load_path, self.args.load_iter_34))

class PQGAN_attacker():

    def __init__(self, batch_size, img_size, patch_size, nz, args, nzp2=None, nzp3=None, nzp4=None, pattern_size=None):
        self.batch_size = batch_size
        assert len(img_size) == 2 and (not pattern_size or len(pattern_size) == 2), \
            "img_size and pattern_size must be tuple with length=2 or None"
        self.img_size = img_size
        self.patch_size = patch_size
        self.pattern_size = pattern_size if pattern_size else img_size
        self.nz = nz
        self.nzp2 = nzp2 if nzp2 else nz
        self.nzp3 = nzp3 if nzp3 else nz
        self.nzp4 = nzp4 if nzp4 else nz

        self.args = args
        self.pqgan = PQGAN(args)
        self.pqgan.to_device(args.cuda)
        # self.G1 = self.pqgan.G1
        # self.G2 = self.pqgan.G2
        # self.G34 = self.pqgan.G34
        self.pqgan.G1.eval()
        self.pqgan.G2.eval()
        self.pqgan.G34.eval()
        for param in self.pqgan.G1.parameters():
            param.requires_grad = False
        for param in self.pqgan.G2.parameters():
            param.requires_grad = False
        for param in self.pqgan.G34.parameters():
            param.requires_grad = False

    def set_image_size(self, img_size):
        self.img_size = img_size

    def set_pattern_size(self, pattern_size):
        self.pattern_size = pattern_size

    def step(self, loss):
        self.optimizer.zero_grad()
        self.optimizer_c.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        if not self.cond_fix:
            self.optimizer_c.step()
            self.scheduler_c.step()

    # condition: [length ratio, drops count, streak angle, rain radius]
    def init_para(self, cond_limit=[[1,1,1,1],[0,0,0,0]], cond_fix=None):
        self.pc_H = floor(ceil(self.pattern_size[0] / self.patch_size) / 2) + 1
        self.pc_W = floor(ceil(self.pattern_size[1] / self.patch_size) / 2) + 2
        self.z = torch.randn((self.batch_size * self.pc_H * self.pc_W, self.nz, 1, 1)).cuda()
        self.zp2 = torch.randn((self.batch_size * self.pc_H * (self.pc_W-1), self.nzp2, 1, 1)).cuda()
        self.zp3 = torch.randn((self.batch_size * (self.pc_H-1) * (self.pc_W-1), self.nzp3, 1, 1)).cuda()
        self.zp4 = torch.randn((self.batch_size * (self.pc_H-1) * (self.pc_W-2), self.nzp4, 1, 1)).cuda()
        self.z.requires_grad = True
        self.zp2.requires_grad = True
        self.zp3.requires_grad = True
        self.zp4.requires_grad = True
        
        eps = 1e-7
        conditions = torch.rand(self.batch_size, 4, 1, 1)
        if cond_fix:
            conditions = torch.Tensor(cond_fix).view(1,4,1,1)
        else:
            cond_limit = torch.Tensor(cond_limit)
            assert len(cond_limit.shape) == 2 and cond_limit.shape[0] == 2 and cond_limit.shape[1] == 4
            conditions = conditions * cond_limit[0].view(1,4,1,1) + cond_limit[1].view(1,4,1,1)

        self.conditions = torch.arctanh(((2-eps*2) * conditions) - 1 + eps).detach()
        conditions = conditions.detach()
        if cond_fix:
            conditions.requires_grad = False
            self.cond_fix = True
        else:
            conditions.requires_grad = True
            self.cond_fix = False

    def set_atk_para(self, iterations):
        self.optimizer_c = torch.optim.Adam([self.conditions], betas=[0.9, 0.999], lr=self.args.lr)
        self.optimizer = torch.optim.Adam([self.z, self.zp2, self.zp3, self.zp4], betas=[0.9, 0.999], lr=self.args.lr)
        self.scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_c, iterations, eta_min=0, last_epoch=-1, verbose=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, iterations, eta_min=0, last_epoch=-1, verbose=False)

    def generate_pattern(self):
        c = 1 / 2 * (torch.tanh(self.conditions) + 1).cuda()
        c1 = c.unsqueeze(1).repeat(1, self.pc_H * self.pc_W, 1, 1, 1).view(self.batch_size * self.pc_H * self.pc_W, 4, 1, 1)
        c2 = c.unsqueeze(1).repeat(1, self.pc_H * (self.pc_W-1), 1, 1, 1).view(self.batch_size * self.pc_H * (self.pc_W-1), 4, 1, 1)
        c3 = c.unsqueeze(1).repeat(1, (self.pc_H-1) * (self.pc_W-1), 1, 1, 1).view(self.batch_size * (self.pc_H-1) * (self.pc_W-1), 4, 1, 1)
        c4 = c.unsqueeze(1).repeat(1, (self.pc_H-1) * (self.pc_W-2), 1, 1, 1).view(self.batch_size * (self.pc_H-1) * (self.pc_W-2), 4, 1, 1)

        # Stage 1
        gen1 = self.pqgan.G1(self.z, c1)
        gen1 = gen1.view(self.batch_size, self.pc_H, self.pc_W, 6, self.patch_size, self.patch_size)
        gen2_img_cond_left = gen1[:,:,:-1]
        gen2_img_cond_right = gen1[:,:,1:]
        gen2_img_cond = torch.cat((gen2_img_cond_left, torch.zeros_like(gen2_img_cond_left).cuda(), gen2_img_cond_right), dim=-1)
        gen2_img_cond = gen2_img_cond.view(self.batch_size * self.pc_H * (self.pc_W-1), 6, self.patch_size, self.patch_size * 3)

        # Stage 2
        gen2 = self.pqgan.G2(self.zp2, gen2_img_cond, c2)
        gen2 = gen2.view(self.batch_size, self.pc_H, self.pc_W-1, 6, self.patch_size, self.patch_size)
        gen2_concat = torch.cat((gen2_img_cond_left, gen2, gen2_img_cond_right), dim=-1)
        gen3_img_cond_top = gen2_concat[:,:-1,:]
        gen3_img_cond_bottm = gen2_concat[:,1:,:]
        gen3_img_cond = torch.cat((gen3_img_cond_top, torch.zeros_like(gen3_img_cond_top).cuda(), gen3_img_cond_bottm), dim=-2)
        gen3_img_cond = gen3_img_cond.view(self.batch_size * (self.pc_H-1) * (self.pc_W-1), 6, self.patch_size * 3, self.patch_size * 3)

        # Stage 3
        gen3 = self.pqgan.G34(self.zp3, gen3_img_cond, c3)
        gen3 = gen3.view(self.batch_size, self.pc_H-1, self.pc_W-1, 6, self.patch_size, self.patch_size)
        gen3_concat_middle = torch.cat((torch.zeros_like(gen3).cuda(), gen3, torch.zeros_like(gen3).cuda()),-1)
        gen3_concat = torch.cat((gen3_img_cond_top, gen3_concat_middle, gen3_img_cond_bottm), dim=-2)
        gen4_img_cond_left = gen3_concat[:,:,:-1]
        gen4_img_cond_right = gen3_concat[:,:,1:]
        gen4_img_cond = torch.cat((gen4_img_cond_left[:,:,:,:,:,:self.patch_size*2], gen4_img_cond_right[:,:,:,:,:,self.patch_size*2:]), dim=-1)
        gen4_img_cond = gen4_img_cond.view(self.batch_size * (self.pc_H-1) * (self.pc_W-2), 6, self.patch_size*3, self.patch_size*3)

        # Stage 4
        gen4 = self.pqgan.G34(self.zp4, gen4_img_cond, c4)
        gen4 = gen4.view(self.batch_size, self.pc_H-1, self.pc_W-2, 6, self.patch_size, self.patch_size)

        # concatenation
        odd_lines = gen2_concat[:,:,:,:,:,self.patch_size:].permute((0,1,3,4,2,5))\
            .reshape((self.batch_size, self.pc_H, 6, self.patch_size, self.patch_size * 2 * (self.pc_W-1)))[:,:,:,:,:-self.patch_size]

        even_lines = torch.cat((gen3[:,:,:-1], gen4),-1).permute((0,1,3,4,2,5))\
            .reshape((self.batch_size, self.pc_H-1, 6, self.patch_size, self.patch_size * 2 * (self.pc_W-2)))
        even_lines = torch.cat((even_lines, gen3[:,:,-1]), -1)

        pattern = torch.cat((odd_lines[:,:-1], even_lines),-2).permute((0,2,1,3,4))\
            .reshape((self.batch_size, 6, self.patch_size * 2 * (self.pc_H-1), self.patch_size * (2 * (self.pc_W-2) + 1)))
        pattern = torch.cat((pattern, odd_lines[:,0]), -2)
        
        pattern = pattern[:, :, :self.pattern_size[0], :self.pattern_size[1]]
        self.pattern_resized = interpolate(pattern, size=self.img_size)

        return self.pattern_resized
        
        







    
