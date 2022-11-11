import argparse

parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

parser.add_argument('--is_train', type=str, default='True')
parser.add_argument('--load_D', type=str, default=None, help='Path for loading Discriminator network')
parser.add_argument('--load_G', type=str, default=None, help='Path for loading Generator network')
parser.add_argument('--load_model', type=str, default="False")
parser.add_argument('--save_file', type=str, default="P_GAN")
parser.add_argument('--data_path', type=str, default="raindata_correct")
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--save_per_iter', type=int, default=500)
parser.add_argument('--load_path', type=str, default="")
parser.add_argument('--load_iter', type=int, default=0)
parser.add_argument('--load_iter_1', type=int, default=190000)
parser.add_argument('--load_iter_2', type=int, default=0)
parser.add_argument('--load_iter_34', type=int, default=0)
parser.add_argument('--stage', type=int, default=1)
parser.add_argument('--fid_per_iter', type=int, default=10000)

# architecture setup
parser.add_argument('--cond_channels', type=int, default=4)
parser.add_argument('--nz', type=int, default=128)
parser.add_argument('--nzp', type=int, default=128)
parser.add_argument('--nzp3', type=int, default=128)
parser.add_argument('--ndf', type=int, default=1)

# training setup
parser.add_argument('--cuda', type=str, default='True', help='Availability of cuda')
parser.add_argument('--milestone', type=int, nargs='+', default=[])
parser.add_argument('--batch_size', type=int, default=2, help='The size of batch')
parser.add_argument('--generator_iters', type=int, default=15001,
                    help='The number of iterations for generator in WGAN model.')
parser.add_argument('--lr', type=float, default=5e-5)

args = parser.parse_args()

try:
    assert args.batch_size >= 1
except:
    print('Batch size must be larger than or equal to one')
args.cuda = True if args.cuda == 'True' else False
args.load_model = True if args.load_model == 'True' else False
if args.load_path == "":
    args.load_path = args.save_file
if args.load_model == False:
    args.load_iter = args.load_iter_1 = args.load_iter_2 = args.load_iter_3 = 0
