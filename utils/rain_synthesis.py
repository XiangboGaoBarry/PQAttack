import matplotlib.pyplot as plt
import torch
from utils.utils import to_device
# from depth_prediction import midas_transforms, DepthPrediction

# O(x) = I(x)(1 - R(x) - A(x)) + R(x) + A_0*A(x)
# R(x) = R_pattern(x) * t_r(x)
# t_r(x) = e^(-alpha*max(d_1, d(x))
# A(x) = 1 - e^(-beta*d(x))
class RainSynthesis():

    def __init__(self, alpha=0.04, beta=0.02, r_r=2, a=0.5):
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.r_r = r_r

    def synthesize(self, ori, d, rain):
        tr = (-self.alpha * d).exp()

        R = rain * tr * self.r_r
        A = 1 - (-self.beta * d).exp()
        img = ori * (1 - R - A) + R + self.a * A
        return img


class RainSynthesisDepth():

    def __init__(self, alpha=0.02, beta=0.00, r_r=1, a=1):
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
        depth_map[:, 1][d_squeeze > 10] = to_device(torch.exp(torch.Tensor([-self.alpha * 15]))) * self.r_r
        depth_map[:, 2][d_squeeze > 20] = to_device(torch.exp(torch.Tensor([-self.alpha * 25]))) * self.r_r
        depth_map[:, 3][d_squeeze > 30] = to_device(torch.exp(torch.Tensor([-self.alpha * 35]))) * self.r_r
        depth_map[:, 4][d_squeeze > 40] = to_device(torch.exp(torch.Tensor([-self.alpha * 45]))) * self.r_r
        depth_map[:, 5][d_squeeze > 50] = to_device(torch.exp(torch.Tensor([-self.alpha * 55]))) * self.r_r

        # print("depth_map_3", "max", depth_map[:, 3].max().item(), "min", depth_map[:, 3].min().item())
        # print("depth_map_2", "max", depth_map[:, 2].max().item(), "min", depth_map[:, 2].min().item())
        # print("depth_map_1", "max", depth_map[:, 1].max().item(), "min", depth_map[:, 1].min().item())
        # print("depth_map_4", "max", depth_map[:, 4].max().item(), "min", depth_map[:, 4].min().item())
        # print("depth_map_5", "max", depth_map[:, 5].max().item(), "min", depth_map[:, 5].min().item())

        R = (rain_layers * depth_map)

        # print("R_1", "max", R[:, 1].max().item(), "min", R[:, 1].min().item())
        # print("R_2", "max", R[:, 2].max().item(), "min", R[:, 2].min().item())
        # print("R_3", "max", R[:, 3].max().item(), "min", R[:, 3].min().item())
        # print("R_4", "max", R[:, 4].max().item(), "min", R[:, 4].min().item())
        # print("R_5", "max", R[:, 5].max().item(), "min", R[:, 5].min().item())

        R = R.max(1).values

        # print("rain_layers", "max", rain_layers.max().item(), "min", rain_layers.min().item())
        #
        # print("max ", R.max().item(), "min ", R.min().item())

        R = to_device(R.unsqueeze(1))
        A = 1 - (-self.beta * d).exp()
        img = ori * (1 - R - A) + R + self.a * A
        # tr = (-self.alpha * d).exp()
        # R = rain * tr * self.r_r
        # A = 1 - (-self.beta * d).exp()
        # img = ori * (1 - R - A) + R + self.a * A
        return img


