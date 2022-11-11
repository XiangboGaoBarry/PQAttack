from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def save_images_from_torch(images, dir):
    for i, img in enumerate(images):
        print(img.shape)
        np_img = img.cpu().detach().numpy().transpose((1,2,0))
        img = Image.fromarray(np.uint8(np_img)*255)
        img.save("%s/%d%s" % (dir, i, '.jpg'))

def show_images_from_torch(images):
    for i, img in enumerate(images):
        print(img.shape)
        np_img = img.cpu().detach().numpy().transpose((1,2,0))
        plt.imshow(np_img)
        plt.show()

def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


to_tensor = lambda shape: transforms.Compose([transforms.Resize(shape), 
                                              transforms.ToTensor(), 
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

def center_crop(t, batched=False):
    if batched:
        return t[:, :, 16:-16, 16:-16]
    else:
        return t[:, 16:-16, 16:-16]


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps



class ImageFolderWithPath(ImageFolder):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path