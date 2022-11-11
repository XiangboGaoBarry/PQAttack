from random import random
import torch

import matplotlib.pyplot as plt
import time
from utils.utils import to_device






class DepthPrediction():
    
    @staticmethod
    def normalize(A):
        A_resized = A.reshape((A.shape[0], -1))
        min_A = A_resized.min(1).values
        max_A = A_resized.max(1).values
        return (A - min_A) / (max_A - min_A)

    def __init__(self, model_type = "DPT_Large"):
        print("Loading MiDaS...")
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        print("MiDaS Done")
        self.model = to_device(torch.hub.load("intel-isl/MiDaS", model_type))
        self.model.eval()

    def __call__(self, images, ret_shape):

        with torch.no_grad():
            prediction = self.model(images)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=ret_shape,
                mode="bicubic",
                align_corners=False,
            )

        return prediction.reshape((prediction.shape[0], -1)).max(1).values.reshape((-1, 1, 1, 1)) \
               - prediction



# if __name__ == '__main__':
#     import numpy as np
#     random_img = (np.random.rand(256,256,3) * 225).astype(np.uint8)
#     print(random_img.max(), random_img.min())
#     after_transform = self.midas_transforms(random_img)
#     print(after_transform.mean(), after_transform.max(), after_transform.min())
#     dp = DepthPrediction()
