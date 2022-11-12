import matplotlib.pyplot as plt
import torch
from utils.utils import to_device
# from depth_prediction import midas_transforms, DepthPrediction

# O(x) = I(x)(1 - R(x) - A(x)) + R(x) + A_0*A(x)
# R(x) = R_pattern(x) * t_r(x)
# t_r(x) = e^(-alpha*max(d_1, d(x))
# A(x) = 1 - e^(-beta*d(x))
class RainSynthesis():

    def __init__(self, alpha=0.03, beta=0.03, r_r=2, a=0.5, dmax=10):
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.r_r = r_r
        self.dmax = dmax

    def synthesize(self, ori, d, rain):
        rain = rain.max(1,keepdim=True)[0]
        # dmax = torch.ones_like(d).to(d.device) * self.dmax
        tr = (-self.alpha * d).exp()
        R = rain * tr * self.r_r
        A = 1 - (- self.beta * d).exp()
        img = ori * (1 - R - A) + R + self.a * A
        return img


class RainSynthesisDepth():

    def __init__(self, alpha=0.03, beta=0.015, r_r=2, a=1):
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.r_r = r_r

    def synthesize(self, ori, d, rain_layers):
        # rain = torch.zeros_like(rain_layers)
        depth_map = torch.zeros_like(rain_layers)
        # print(d.shape)
        # print(depth_map[:, 0].shape)
        d_squeeze = d.view(d.shape[0], d.shape[2], d.shape[3])
        depth_map[:, 0][d_squeeze > 0] = to_device(torch.exp(torch.Tensor([-self.alpha * 5]))) * self.r_r
        depth_map[:, 1][d_squeeze > 5] = to_device(torch.exp(torch.Tensor([-self.alpha * 10]))) * self.r_r
        depth_map[:, 2][d_squeeze > 10] = to_device(torch.exp(torch.Tensor([-self.alpha * 15]))) * self.r_r
        depth_map[:, 3][d_squeeze > 15] = to_device(torch.exp(torch.Tensor([-self.alpha * 20]))) * self.r_r
        depth_map[:, 4][d_squeeze > 20] = to_device(torch.exp(torch.Tensor([-self.alpha * 25]))) * self.r_r
        depth_map[:, 5][d_squeeze > 25] = to_device(torch.exp(torch.Tensor([-self.alpha * 30]))) * self.r_r

        R = (rain_layers * depth_map)
        R = R.max(1).values
        R = to_device(R.unsqueeze(1))
        A = 1 - (-self.beta * d).exp()
        img = ori * (1 - R - A) + R + self.a * A
        # tr = (-self.alpha * d).exp()
        # R = rain * tr * self.r_r
        # A = 1 - (-self.beta * d).exp()
        # img = ori * (1 - R - A) + R + self.a * A
        return img


