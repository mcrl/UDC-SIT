from argparse import ArgumentParser
import os
from tqdm import tqdm
import numpy as np
import os.path as osp

parser = ArgumentParser()
parser.add_argument("--prefetch-dir", required=True, type=str)

args = parser.parse_args()
dirpath = args.prefetch_dir
dirpath = osp.abspath(dirpath)
files = os.listdir(dirpath)

print(f"Prefetching {len(files)} files from {dirpath}...")
for file in tqdm(files):
    filepath = osp.join(dirpath, file)
    _ = np.load(filepath)
print(f"PREFETCHED {len(files)} files from {dirpath}")
