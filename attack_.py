from models.PQGAN_attacker import PQGAN_attacker
import argparse
import os
from PIL import Image
import torch
from torchvision.models import resnet18, resnet50
from torchvision.transforms import ToTensor, Normalize, ToPILImage
from utils.depth_prediction import DepthPrediction
from utils.rain_synthesis import RainSynthesisDepth
from torchvision.datasets import ImageNet, ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
import numpy as np
import time
import pdb
import torch.nn.functional as F
from tqdm import tqdm
import glob
from utils.utils import ImageFolderWithPath

normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
inv_normalize = Normalize(mean=[1/0.485, 1/0.456, 1/0.406],
                                std=[1/0.229, 1/0.224, 1/0.225])

def save_adv_img(img, save_dir, name):
    img.shape
    to_save = torch.clamp(img * 255,0,255).to(torch.uint8).squeeze()
    to_save = ToPILImage()(to_save)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    save_path = os.path.join(save_dir, name)
    to_save.save(save_path)
    
def center_crop(t):
    return t[:, :, 16:-16, 16:-16]

parser = argparse.ArgumentParser(description="Implementation of PQAttack.")

# /home/bar/xb/dataset/imagenet/val
parser.add_argument('--data_path', type=str, default="path/to/imagenet/val")
parser.add_argument('--dataset', type=str, default="imagenet")
parser.add_argument('--load_path', type=str, default="models/pretrained/nz128")
parser.add_argument('--save_path', type=str, default="attack_results")
parser.add_argument('--load_model', type=str, default="True")
parser.add_argument('--load_iter', type=int, default=19000)
parser.add_argument('--load_iter_1', type=int, default=0)
parser.add_argument('--load_iter_2', type=int, default=0)
parser.add_argument('--load_iter_34', type=int, default=0)
parser.add_argument('--image_size', type=tuple, default=(256, 256), help='Rescale images to this size if fixed_size is True')

# architecture setup
parser.add_argument('--cond_channels', type=int, default=4)
parser.add_argument('--nz', type=int, default=128)
parser.add_argument('--nzp', type=int, default=128)
parser.add_argument('--nzp3', type=int, default=128)
parser.add_argument('--ndf', type=int, default=32)

# attack setup
parser.add_argument('--lr', type=float, default=0.03, help='learning rate used for adversarial attack')
parser.add_argument('--cuda', type=str, default='True', help='Availability of cuda')
parser.add_argument('--iters', type=int, default=300, help='number of attack iterations')
parser.add_argument('--early_stopping', type=str, default='True', help='stop once attack succeed. False for black box attack or PCA visualization')
parser.add_argument('--save_results', type=str, default='False', help='save results')
parser.add_argument('--fixed_scale', type=str, default='False', help='Set to True if you want to attack with a fixed scale')
parser.add_argument('--black_box', type=str, default='False', help='Set to True if you want to attack with a black box')

args = parser.parse_args()
args.cuda = True if args.cuda in [True, 'True'] else False
args.load_model = True if args.load_model in [True, 'True'] else False
args.early_stopping = True if args.early_stopping in [True, 'True'] else False
args.save_results = True if args.save_results in [True, 'True'] else False
args.fixed_scale = True if args.fixed_scale in [True, 'True'] else False
args.black_box = True if args.black_box in [True, 'True'] else False
if args.black_box: args.early_stopping = False

if args.load_iter:
    args.load_iter_1 = args.load_iter_2 = args.load_iter_34 = args.load_iter

if args.save_results:
    if not os.path.isdir(args.save_path): os.mkdir(args.save_path)


to_device = lambda x: x.cuda() if args.cuda else x.cpu()

classifier = to_device(resnet18(pretrained=True))
classifier.eval()

if args.black_box:
    black_box_classifier = to_device(resnet50(pretrained=True))
    black_box_classifier.eval()

attacker = PQGAN_attacker(batch_size=1, 
                         img_size=args.image_size, 
                         patch_size=64, 
                         nz=128, 
                         args=args, 
                         pattern_size=args.image_size)
depth_predicter = DepthPrediction()
rain_synthesizer = RainSynthesisDepth(alpha=0.015, beta=0.04, r_r=2, a=0.9)

imagenet = ImageFolderWithPath(
            args.data_path,
            Compose([
                # Resize((256,256)),
                ToTensor(),
            ]))
imagenet_loader = DataLoader(imagenet, shuffle=True)
succ_count = 0
succ_clean_count = 0
bb_succ_count = 0
bb_succ_clean_count = 0
resizer = Resize((256,256))

pbar = tqdm(enumerate(imagenet_loader), total=5000)
for idx, (image, label, filepath) in pbar:
    filename = filepath[0].split('/')[-1]
    if not args.fixed_scale:
        H, W = image.shape[2:]
        attacker.set_image_size((H, W))
        attacker.set_pattern_size((H, W))
    else:
        H, W = args.image_size
    image = Resize((H,W))(image)
    if idx >= 5000:
            break
    with torch.no_grad():
        img_np = np.array((image[0] * 255)).astype(np.uint8).transpose((1,2,0))
        depth = depth_predicter(to_device(depth_predicter.midas_transforms(img_np)), (H, W)).detach()
    image = to_device(image)
    gt = label.item()
    img = image
    attacker.init_para()
    attacker.set_atk_para(iterations=args.iters)
    syn_img = img
    outputs = classifier(center_crop(normalize(resizer(syn_img))))
    pred_clean = torch.argmax(outputs)
    succ_clean = pred_clean.item() != gt
    
    if args.black_box:
        outputs = black_box_classifier(center_crop(normalize(resizer(syn_img))))
        bb_pred_clean = torch.argmax(outputs)
        bb_succ_clean = bb_pred_clean.item() != gt

    if succ_clean:
        succ_clean_count += 1

    if args.black_box and bb_succ_clean:
        bb_succ_clean_count += 1

    for iter_i in range(args.iters):
        rain_pattern = attacker.generate_pattern()
        syn_img = rain_synthesizer.synthesize(img, depth, rain_pattern)
        syn_img = syn_img.clamp(0,1)
        outputs = classifier(center_crop(normalize(resizer(syn_img))))
        prediction = torch.argmax(outputs)
        adv_loss = -F.cross_entropy(outputs, to_device(torch.Tensor([gt]).long()))
        attacker.step(adv_loss)
        # pbar.set_description("Image %02d | Iter %02d | Adv loss=%.3f | gt: %03d | pred: %03d | status: %s" % \
        #                     (idx, iter_i, adv_loss, gt, prediction, "SUCC" if prediction != gt else "FAILD"))
        succ = prediction.item() != gt
        if succ and args.early_stopping:
            break
    if succ:
        succ_count += 1

    if args.black_box:
        outputs = black_box_classifier(center_crop(normalize(resizer(syn_img))))
        bb_prediction = torch.argmax(outputs)
        if bb_prediction.item() != gt:
            bb_succ_count += 1

    if args.save_results:
        image_path = os.path.join(args.save_path, args.dataset)
        if not os.path.isdir(image_path): os.mkdir(image_path)
        save_adv_img(syn_img, image_path, filename)
        with open(os.path.join(args.save_path, "meta.txt"), "a") as f:
            if args.black_box:
                f.write(f"{filename} {gt} {pred_clean} {prediction} {bb_pred_clean} {bb_prediction}\n")
            else:
                f.write(f"{filename} {gt} {pred_clean} {prediction}\n")

    if args.black_box:
        pbar.set_description(f"White Box ACC: Before Attack: {round((1 - succ_clean_count/(idx+1)) * 100, 2)}%, After Attack: {round((1 - succ_count/(idx+1)) * 100, 2)}% "
        f"| White Box ACC: Before Attack: {round((1 - bb_succ_clean_count/(idx+1)) * 100, 2)}%, After Attack: {round((1 - bb_succ_count/(idx+1)) * 100, 2)}%")
    else:
        pbar.set_description(f"White Box ACC: Before Attack: {round((1 - succ_clean_count/(idx+1)) * 100, 2)}%, After Attack: {round((1 - succ_count/(idx+1)) * 100, 2)}%")


