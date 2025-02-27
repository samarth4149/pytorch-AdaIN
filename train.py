import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
# from tensorboardX import SummaryWriter
import wandb
from torchvision import transforms
from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

import numpy as np

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--expt_name', default='tdw2in_adain')
parser.add_argument('--use_wandb', action='store_true')
# parser.add_argument('--log_dir', default='./logs',
#                     help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=0.01)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--num_imgs', type=int, default=10000)
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(args.save_dir) / args.expt_name
save_dir.mkdir(exist_ok=True, parents=True)
# log_dir = Path(args.log_dir)
# log_dir.mkdir(exist_ok=True, parents=True)
# writer = SummaryWriter(log_dir=str(log_dir))
if args.use_wandb:
    wandb.init(
        project='pytorch-adain', name=args.expt_name, 
        config=args, dir=str(save_dir), reinit=True)

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

RNG = np.random.RandomState(44) 
content_dataset = ImageFolder(args.content_dir, content_tf)
style_dataset = ImageFolder(args.style_dir, style_tf)
content_subset_idxs = RNG.permutation(len(content_dataset))[:args.num_imgs]
style_subset_idxs = RNG.permutation(len(style_dataset))[:args.num_imgs]
content_dataset = Subset(content_dataset, content_subset_idxs)
style_dataset = Subset(style_dataset, style_subset_idxs)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

with tqdm(total=args.max_iter) as pbar:
    for i in range(args.max_iter):
        adjust_learning_rate(optimizer, iteration_count=i)
        content_images, _ = next(content_iter)
        style_images, _ = next(style_iter)
        content_images = content_images.to(device)
        style_images = style_images.to(device)
        loss_c, loss_s = network(content_images, style_images)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f'LC:{loss_c.item():.4f} LS:{loss_s.item():.4f}')
        if args.use_wandb and (args.log_interval > 0 and (i+1) % args.log_interval == 0):
            wandb.log({
                'Iter' : i+1,
                'loss_content' : loss_c.item(),
                'loss_style' : loss_s.item(),
            })
            
        # writer.add_scalar('loss_content', loss_c.item(), i + 1)
        # writer.add_scalar('loss_style', loss_s.item(), i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = net.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                    'decoder_iter_{:d}.pth.tar'.format(i + 1))
        pbar.update(1)
# writer.close()
