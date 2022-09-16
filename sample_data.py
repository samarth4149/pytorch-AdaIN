import os
import sys

sys.path.append(os.path.abspath('.'))
import shutil
import argparse
from pathlib import Path
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_base', type=str, default='')
    parser.add_argument('--dest_name', type=str, default='')
    parser.add_argument('--num_imgs', type=int, default=100)
    args = parser.parse_args()
    
    src_path = Path(args.source_base)
    if not src_path.exists():
        raise Exception(f'Source path {args.source_base} does not exist')
    
    RNG = np.random.RandomState(44)
    folder = ImageFolder(src_path)
    img_paths = [s[0] for s in folder.samples]
    idxs = RNG.permutation(len(img_paths))[:args.num_imgs]
    
    for idx in tqdm(idxs):
        sample = img_paths[idx]
        fpath = Path(sample)
        write_path = Path(args.dest_name) / fpath.name
        shutil.copyfile(fpath, write_path)
            