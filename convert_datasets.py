import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import Subset
from typing import Optional, Callable, Any
import numpy as np


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

class ImageFolderwPaths(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, 
                 loader: Callable[[str], Any] = default_loader, is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=256,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=256,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

RNG = np.random.RandomState(44)

content_dataset = ImageFolderwPaths(args.content_dir, transform=test_transform(args.content_size, args.crop))
style_dataset = ImageFolderwPaths(args.style_dir, transform=test_transform(args.style_size, args.crop))
subset_idxs = RNG.randint(0, len(style_dataset), (len(content_dataset),))
style_dataset = Subset(style_dataset, subset_idxs)

# content_loader = torch.utils.data.DataLoader(
#     content_dataset, batch_size=args.batch_size, shuffle=False,
#     num_workers=args.num_workers, pin_memory=True)
# style_loader = torch.utils.data.DataLoader(
#     style_dataset, batch_size=args.batch_size, shuffle=False,
#     num_workers=args.num_workers, pin_memory=True)

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)


for i in tqdm(range(len(content_dataset)), total=len(content_dataset)):
    content, _, content_path = content_dataset[i]
    style, _, style_path = style_dataset[i]
    
    if args.preserve_color:
        style = coral(style, content)
    
    content = content.to(device).unsqueeze(0)
    style = style.to(device).unsqueeze(0)
    
    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style,
                                args.alpha)
    output = output.cpu()

    content_path = Path(content_path)
    output_name = output_dir / content_path.parent.name / (content_path.stem + args.save_ext)
    os.makedirs(output_name.parent, exist_ok=True)
    save_image(output, str(output_name))
