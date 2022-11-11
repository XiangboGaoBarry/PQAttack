import torch
import torch.nn as nn

def LayerConvT(in_channels, out_channels, kernel_size, stride, padding):
    return (nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True))

def LayerConv(in_channels, out_channels, kernel_size, stride, padding):
    return (nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
			nn.InstanceNorm2d(out_channels, affine=True),
			nn.LeakyReLU(0.2, inplace=True))

# class GeneratorStage1(torch.nn.Module):
# 	def __init__(self, args):
# 		super().__init__()
# 		self.layer_latent = nn.Sequential(
# 			*LayerConvT(args.nz, 128 * args.ndf, 4, 1, 0), 
# 			*LayerConvT(128 * args.ndf, 64 * args.ndf, 4, 2, 1), 
# 		)
# 		self.layer_condition = nn.Sequential(
# 			*LayerConvT(args.cond_channels, 128 * args.ndf, 4, 1, 0), 
# 			*LayerConvT(128 * args.ndf, 64 * args.ndf, 4, 2, 1), 
# 		)
# 		self.main_module = nn.Sequential(
# 			*LayerConvT(128 * args.ndf, 64 * args.ndf, 4, 2, 1), 
# 			*LayerConvT(64 * args.ndf, 32 * args.ndf, 4, 2, 1), 
# 			nn.ConvTranspose2d(in_channels=32 * args.ndf, out_channels=6, kernel_size=4, stride=2, padding=1))
# 		self.output = nn.Tanh()

# 	def forward(self, latent, condition):
# 		latent = self.layer_latent(latent)
# 		condition = self.layer_condition(condition)
# 		x = torch.cat((latent, condition), 1)
# 		x = self.main_module(x)
# 		x = self.output(x)
# 		# print(self.__class__.__name__, x.shape)
# 		return x

class GeneratorStage1(torch.nn.Module):
	def __init__(self, args):
		super().__init__()
		self.layer_latent = nn.Sequential(
			*LayerConvT(args.nz, 4 * args.ndf, 4, 1, 0), 
			*LayerConvT(4 * args.ndf, 2 * args.ndf, 4, 2, 1), 
		)
		self.layer_condition = nn.Sequential(
			*LayerConvT(args.cond_channels, 4 * args.ndf, 4, 1, 0), 
			*LayerConvT(4 * args.ndf, 2 * args.ndf, 4, 2, 1), 
		)
		self.main_module = nn.Sequential(
			*LayerConvT(4 * args.ndf, 2 * args.ndf, 6, 4, 1), 
			nn.ConvTranspose2d(in_channels=2 * args.ndf, out_channels=6, kernel_size=4, stride=2, padding=1))
		self.output = nn.Sigmoid()

	def forward(self, latent, condition):
		latent = self.layer_latent(latent)
		condition = self.layer_condition(condition)
		x = torch.cat((latent, condition), 1)
		x = self.main_module(x)
		x = self.output(x)
		# print(self.__class__.__name__, x.shape)
		return x


class DiscriminatorStage1(torch.nn.Module):
	def __init__(self, args):
		super().__init__()
		self.layer_img = nn.Sequential(
			*LayerConv(6, 1 * args.ndf, 4, 2, 1),
			*LayerConv(1 * args.ndf, 2 * args.ndf, 4, 2, 1)
		)
		self.layer_condition = nn.Sequential(
			*LayerConv(args.cond_channels, 1 * args.ndf, 4, 2, 1),
			*LayerConv(1 * args.ndf, 2 * args.ndf, 4, 2, 1)
		)
		self.main_module = nn.Sequential(
			*LayerConv(4 * args.ndf, 8 * args.ndf, 6, 4, 1),
		)
		self.output = nn.Sequential(
			nn.Conv2d(in_channels=8 * args.ndf, out_channels=1, kernel_size=4, stride=1, padding=0))

	def forward(self, img, condition):
		B, C, H, W = img.shape
		condition = condition.expand(B, condition.shape[1], H, W)
		img = self.layer_img(img)
		condition = self.layer_condition(condition)
		x = torch.cat((img, condition), 1)
		x = self.main_module(x)
		x = self.output(x)
		# print(self.__class__.__name__, x.shape)
		return x


class GeneratorStage2(torch.nn.Module):
	def __init__(self, args):
		super().__init__()
		self.layer_latent = nn.Sequential(
			*LayerConvT(args.nz, 8 * args.ndf, 4, 1, 0), 
			*LayerConvT(8 * args.ndf, 4 * args.ndf, 4, 2, 1)
		)
		self.layer_condition = nn.Sequential(
			*LayerConvT(args.cond_channels, 8 * args.ndf, 4, 1, 0), 
			*LayerConvT(8 * args.ndf, 4 * args.ndf, 4, 2, 1)
		)
		self.layer_img = nn.Sequential(
            *LayerConv(6, 2 * args.ndf, 4, 2, 1),
			*LayerConv(2 * args.ndf, 4 * args.ndf, (3, 5), (1, 3), 1),
			*LayerConv(4 * args.ndf, 8 * args.ndf, 6, 4, 1),
		)
		self.main_module = nn.Sequential(
			*LayerConvT(16 * args.ndf, 4 * args.ndf, 6, 4, 1), 
			nn.ConvTranspose2d(in_channels= 4 * args.ndf, out_channels=6, kernel_size=4, stride=2, padding=1)
		)
		self.output = nn.Sigmoid()

	def forward(self, latent, img, condition):
		latent = self.layer_latent(latent)
		condition = self.layer_condition(condition)
		img = self.layer_img(img)
		x = torch.cat((latent, condition, img), 1)
		x = self.main_module(x)
		x = self.output(x)
		# print(self.__class__.__name__, x.shape)
		return x


class DiscriminatorStage2(torch.nn.Module):
	def __init__(self, args):
		super().__init__()
		self.layer_img = nn.Sequential(
			*LayerConv(6, 1 * args.ndf, 4, 2, 1),
			*LayerConv(1 * args.ndf, 2 * args.ndf, (3, 5), (1, 3), 1),
		)
		self.layer_condition = nn.Sequential(
			*LayerConv(args.cond_channels, 1 * args.ndf, 4, 2, 1),
			*LayerConv(1 * args.ndf, 2 * args.ndf, (3, 5), (1, 3), 1),
		)
		self.main_module = nn.Sequential(
			*LayerConv(4 * args.ndf, 8 * args.ndf, 4, 2, 1),
			*LayerConv(8 * args.ndf, 16 * args.ndf, 6, 4, 1),
		)
		self.output = nn.Sequential(
			nn.Conv2d(in_channels=16 * args.ndf, out_channels=1, kernel_size=4, stride=1, padding=0))

	def forward(self, img, condition):
		B, C, H, W = img.shape
		condition = condition.expand(B, condition.shape[1], H, W)
		img = self.layer_img(img)
		condition = self.layer_condition(condition)
		x = torch.cat((img, condition), 1)
		x = self.main_module(x)
		x = self.output(x)
		# print(self.__class__.__name__, x.shape)
		return x



class GeneratorStage34(torch.nn.Module):
	def __init__(self, args):
		super().__init__()
		self.layer_latent = nn.Sequential(
			*LayerConvT(args.nz, 8 * args.ndf, 4, 1, 0),
			*LayerConvT(8 * args.ndf, 4 * args.ndf, 4, 2, 1)
		)
		self.layer_condition = nn.Sequential(
			*LayerConvT(args.cond_channels, 8 * args.ndf, 4, 1, 0),
			*LayerConvT(8 * args.ndf, 4 * args.ndf, 4, 2, 1)
		)

		self.layer_img = nn.Sequential(
			*LayerConv(6, 2 * args.ndf, 5, 3, 1),
			*LayerConv(2 * args.ndf, 4 * args.ndf, 4, 2, 1),
			*LayerConv(4 * args.ndf, 8 * args.ndf, 6, 4, 1),
		)
		self.main_module = nn.Sequential(
			*LayerConvT(16 * args.ndf, 4 * args.ndf, 6, 4, 1),
			nn.ConvTranspose2d(in_channels=4 * args.ndf, out_channels=6, kernel_size=4, stride=2, padding=1))
		self.output = nn.Sigmoid()

	def forward(self, latent, img, condition):
		latent = self.layer_latent(latent)
		condition = self.layer_condition(condition)
		img = self.layer_img(img)
		# print(latent.shape, condition.shape, img.shape)
		x = torch.cat((latent, condition, img), 1)
		x = self.main_module(x)
		x = self.output(x)
		# print(self.__class__.__name__, x.shape)
		return x

class DiscriminstorStage34(torch.nn.Module):
	def __init__(self, args):
		super().__init__()
		self.layer_img = nn.Sequential(
			*LayerConv(6, 1 * args.ndf, 5, 3, 1),
			*LayerConv(1 * args.ndf, 2 * args.ndf, 4, 2, 1)
		)
		self.layer_condition = nn.Sequential(
			*LayerConv(args.cond_channels, 1 * args.ndf, 5, 3, 1),
			*LayerConv(1 * args.ndf, 2 * args.ndf, 4, 2, 1)
		)

		self.main_module = nn.Sequential(
			*LayerConv(4 * args.ndf, 8 * args.ndf, 4, 2, 1),
			*LayerConv(8 * args.ndf, 16 * args.ndf, 6, 4, 1),
		)
		self.output = nn.Sequential(
			nn.Conv2d(in_channels=16 * args.ndf, out_channels=1, kernel_size=4, stride=1, padding=0))

	def forward(self, img, condition):
		B, C, H, W = img.shape
		condition = condition.expand(B, condition.shape[1], H, W)
		img = self.layer_img(img)
		condition = self.layer_condition(condition)
		x = torch.cat((img, condition), 1)
		x = self.main_module(x)
		x = self.output(x)
		# print(self.__class__.__name__, x.shape)
		return x

