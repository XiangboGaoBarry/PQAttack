import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def get_perspective_matrix(shape, ratio=0.1):
	"""Get the perspective transformation matrix for the entire rain streak image

	Args:
		shape (tuple, list): image shape
		ratio (float): image rotation around the x axis. Defaults to 0.1.

	Returns:
		np.ndarry: perspective transformation matrix
	"""
	W, H = shape
	scaling = W * ratio
	if ratio > 0:
		bottom_left, top_left, bottom_right, top_right = \
			[-scaling, 0], [0, H], [W + scaling, 0], [W, H]
	else:
		bottom_left, top_left, bottom_right, top_right = \
			[0, 0], [scaling, H], [W, 0], [W - scaling, H]
	pts_src = np.float32([[0, 0], [0, H], [W, 0], [W, H]])
	pts_dst = np.float32([bottom_left, top_left, bottom_right, top_right])
	Tp = cv2.getPerspectiveTransform(pts_src, pts_dst)
	return Tp


class MultilayerConditionalRain:

	def __init__(self, size=(256, 256)):
		self.random = random.random
		self.rain = np.zeros(size)
		self.size = size
		# self.depth_ratio = [23, 26, 29, 32, 35, 38]  #ori
		self.depth_ratio = [13, 16, 19, 22, 25, 28]  #shallow
		self.depth_layer_num = len(self.depth_ratio)
		self.verbose = False

	def new_rain(self):
		self.depth_layers = np.zeros((self.depth_layer_num, self.size[0], self.size[1]))
		self.rain = np.zeros(self.size)

	def set_verbose(self, verbose=False):
		self.verbose = verbose

	def generate_rain(self, homo=True):
		drop_iter = tqdm(range(self.dc)) if self.verbose else range(self.dc)
		depth_layers = np.zeros((6, self.size[0], self.size[1]))
		for i in drop_iter:
			depth, thre = self.select_d()
			gen = self.generate_drop(depth)
			depth_layers[thre] = np.maximum(depth_layers[thre], gen)
			
		Tp = get_perspective_matrix(self.size, ratio=0)
		for i in range(6):
			depth_layers[i] = cv2.warpPerspective(depth_layers[i], Tp, self.size)
		self.rain = np.max(depth_layers, axis=0)

		return self.rain, depth_layers

	def normalize_0255(self, image):
		img_min = image.min()
		img_max = image.max()
		return (image / img_max) * 225

	def normalize(self, image):
		img_min = image.min()
		img_max = image.max()
		return (image / img_max) * 225

	def set_camera_para(self):
		self.camera_para = dict()
		self.model_para = dict()
		self.camera_para["frame_size"] = 0.024  # 24mm * 24mm   [36*24, 23.6*15.8, 18*13.5, 7.6*5.7, 6.1*4.6]
		self.camera_para["focal_length"] = 0.05  # 50mm           14 ~ 800mm
		self.zoom_ratio = 255 / self.camera_para["frame_size"]

	@staticmethod
	def line_intersect_sphere(x1, x2, c, r):
		D = x2 - x1
		a = D.dot(D)
		b = 2 * (x1 - c).dot(D)
		c = (x1 - c).dot(x1 - c) - r ** 2
		return MultilayerConditionalRain.solve_quadratics(a, b, c, x1, D)

	@staticmethod
	def solve_quadratics(a, b, c, x1, d):
		delta = b * b - 4 * a * c
		if delta < 0:
			return None
		if abs(delta) < 1e-6:
			t = -b / (2 * a)
			return np.array([x1 + t * d])
		else:
			t1 = (-b - np.sqrt(delta)) / (2 * a)
			t2 = (-b + np.sqrt(delta)) / (2 * a)
			return np.array([x1 + t1 * d, x1 + t2 * d])

	@staticmethod
	def get_distance(x, y, center):
		return ((center[0] - x) ** 2 + (center[1] - y) ** 2) ** (1 / 2)

	def get_boundary(self, f, r_v, drop_c):
		oc_x = np.array([drop_c[0] - self.image_center[0], -f])
		d_oc_x = np.linalg.norm(oc_x, ord=2)
		theta_poc = np.arcsin(r_v / d_oc_x)
		x_s = np.zeros(2)
		for idx, theta_poc in enumerate([theta_poc, -theta_poc]):
			d_op_x = np.cos(theta_poc) * d_oc_x
			sign = np.sign(oc_x[0])
			theta_xoc = np.arctan(abs(oc_x[1]) / abs(oc_x[0]))
			theta_xop = theta_xoc - theta_poc
			op_x = np.array([sign * d_op_x * np.cos(theta_xop), - d_op_x * np.sin(theta_xop)])
			x_s[idx] = op_x[0]

		x_s += self.image_center[0]
		left, right = np.min(x_s), np.max(x_s)

		oc_y = np.array([drop_c[1] - self.image_center[1], -f])
		d_oc_y = np.linalg.norm(oc_y, ord=2)

		theta_poc = np.arcsin(r_v / d_oc_y)

		y_s = np.zeros(2)
		for idx, theta_poc in enumerate([theta_poc, -theta_poc]):
			d_op_y = np.cos(theta_poc) * d_oc_y
			sign = np.sign(oc_y[0])
			theta_xoc = np.arctan(abs(oc_y[1]) / abs(oc_y[0]))
			theta_xop = theta_xoc - theta_poc
			op_y = np.array([sign * d_op_y * np.cos(theta_xop), - d_op_y * np.sin(theta_xop)])
			y_s[idx] = op_y[0]
		y_s += self.image_center[1]
		top, bottom = np.min(y_s), np.max(y_s)
		left = np.max([0, left])
		right = np.min([self.size[0] - 1, right])
		top = np.max([0, top])
		bottom = np.min([self.size[0] - 1, bottom])
		return left, right, top, bottom

	def draw_drop(self, d0, drop_layer, center, radius=0.01):

		self.image_center = np.array(self.size) / 2.
		drop_c = center * np.array([self.size[0], self.size[1]])  # virtual drop center
		f = self.camera_para['focal_length'] * self.zoom_ratio
		ratio = d0 / f  # real / frame
		r_v = radius

		left, right, top, bottom = self.get_boundary(f, r_v, drop_c)
		c_f = np.array([-f, drop_c[0], drop_c[1]])
		c = -c_f * ratio

		def distortion_mapping(s):
			o = np.array([0, 0, 0])
			points = self.line_intersect_sphere(o, s, c_f, r_v)
			return points[0] if type(points) == type(np.zeros(1)) else None

		def get_i(s):
			o = np.zeros(3)
			p = -s * ratio
			pc = c - p
			po = o - p
			i = np.pi - np.arccos((pc.dot(po) / (np.linalg.norm(pc, ord=2) * np.linalg.norm(po, ord=2))))
			return i

		def refraction_coefficient(i, u=1.33):
			Rs = ((np.cos(i) - u * np.sqrt(1 - ((1 / u) * np.sin(i)) ** 2)) /
				  (np.cos(i) + u * np.sqrt(1 - ((1 / u) * np.sin(i)) ** 2))) ** 2
			Rp = ((np.sqrt(1 - ((1 / u) * np.sin(i)) ** 2) - u * np.cos(i)) /
				  (np.sqrt(1 - ((1 / u) * np.sin(i)) ** 2) + u * np.cos(i))) ** 2
			return np.sqrt(Rs + Rp)

		def refraction(k, Le=1):
			return (1 - k) ** 2 * Le

		def reflection(k, Le=1):
			return k * Le

		def internal_reflection(k, Le=1, N=2):
			return (1 - k) ** 2 * k ** N * Le

		for x in range(int(left), int(right) + 1):
			for y in range(int(top), int(bottom) + 1):
				p_f = distortion_mapping(np.array([-f, x, y]))
				if not type(p_f) == type(np.zeros(1)):
					continue
				i = get_i(p_f)
				k = refraction_coefficient(i)
				Ln = refraction(k) + reflection(k) + internal_reflection(k)
				drop_layer[x, y] = Ln


	def select_d(self, minD=10, maxD=28, step=3):
		"""Randomly select a rain drop's depth from the view frustum

		Args:
			minD (int): minimum depth. Defaults to 10.
			maxD (int): maximum depth. Defaults to 28.
			step (int): step distance between depth levels. Defaults to 3.

		Returns:
			tuple(float, int): (rain drop's depth, rain drop's depth level)
		"""
		d = 0
		assert maxD > minD
		while d <= minD:
			d = np.sqrt(np.random.rand())
			d *= maxD
		thre = (d - minD) // step
		return d, int(thre)

	def set_condition(self, min_values=[15, -15, 5, 10], range=[15, 30, 10, 8]):
		"""Set the condition parameter for rain pattern generation

		Args:
			min_values (list): The min value of the parameter for rain streak length ratio, 
  										streak angle, rain density, and rain streak radius, respectively. 
         								Defaults to [15, -15, 5, 10].
			range (list): The range of the parameter for rain streak length ratio, 
  									streak angle, rain density, and rain streak radius, respectively. 
         							Defaults to [15, 30, 10, 8].

		Returns:
			list: An array of size 4. A condition parameter uniformly draw from the given range.
		"""
  		
		lr_para = [range[0], min_values[0]]
		angle_para = [range[1], min_values[1]]
		self.length_ratio = self.random() * lr_para[0] + lr_para[1]
		self.angle = self.random() * angle_para[0] + angle_para[1]

		dc_para = [int(self.size[0]**2 * range[2] / 1024), int(self.size[0]**2 * min_values[2] / 1024)]
		radius_para = [range[3], min_values[3]]
		dc_ = self.random() * dc_para[0] + dc_para[1]
		self.radius = self.random() * radius_para[0] + radius_para[1]
		self.dc = int(dc_)

		lr = (self.length_ratio - lr_para[1]) / lr_para[0]
		dc = (dc_ - dc_para[1]) / dc_para[0]
		angle = (self.angle - angle_para[1]) / angle_para[0]
		rr = (self.radius - radius_para[1]) / radius_para[0]
		return [lr, dc, angle, rr]

	def generate_drop(self, depth):
		drop_layer = np.zeros_like(self.rain)
		radius = self.radius / depth
		center = np.array([self.random(), self.random()])

		self.draw_drop(depth, drop_layer, center, radius)

		def make_streak(angle):
			length = int(radius * self.length_ratio)
			if length <= 1: return drop_layer

			M = cv2.getRotationMatrix2D((length // 2, length // 2), angle, 1)

			motion_blur_kernel = np.zeros((length, length))
			if length % 2 == 0:
				motion_blur_kernel[:, [length // 2 - 1, length // 2]] = 0.5
			else:
				motion_blur_kernel[:, length // 2] = 1

			motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (length, length))
			drop_layer_padding = cv2.copyMakeBorder(drop_layer, length // 2, length // 2, length // 2, length // 2,
													borderType=cv2.BORDER_CONSTANT, value=0)
			gen = cv2.filter2D(drop_layer_padding, -1, motion_blur_kernel)
			gen = gen[length // 2: length // 2 + self.size[0], length // 2: length // 2 + self.size[1]]

			gen = gen / int(length)
			return gen

		return make_streak(self.angle)



