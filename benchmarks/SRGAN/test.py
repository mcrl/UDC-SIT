import rawpy as rp
import imageio
import os
from argparse import ArgumentParser
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from networks.SRGAN import *
from datasets.dataset_pairs_npy import my_dataset_eval
from util import setup_img_save_function

img_save_function = None


def test(
    model: nn.Module,
    test_loader: DataLoader,
    save_image=False,
    save_dir=None,
    experiment_name=None,
):
    model.eval()
    ddp_loss = torch.zeros(3).to(0)

    iter_bar = tqdm.tqdm(test_loader)
    with torch.no_grad():
        for data, target, img_name in iter_bar:
            img_name = img_name[0].split(".")[0]

            data, target = data.to(0), target.to(0)
            output = model(data)
            # if ouptut is list, take only the last one
            if isinstance(output, list):
                output = output[-1]
            psnr = peak_signal_noise_ratio(output, target)
            ddp_loss[0] += psnr
            ssim = structural_similarity_index_measure(output, target)
            ddp_loss[1] += ssim
            ddp_loss[2] += len(data)

            if save_image and save_dir is not None:
                filename = f"{experiment_name}_{img_name}_{psnr:.4f}_{ssim:.4f}.png"
                img_save_function(output, os.path.join(save_dir, filename))

    print(
        "Test set: Average PSNR: {:.4f}, Average SSIM: {:.4f}".format(
            ddp_loss[0] / ddp_loss[2], ddp_loss[1] / ddp_loss[2]
        )
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-input", type=str, default="data/test/input")
    parser.add_argument("--test-GT", type=str, default="data/test/GT")
    parser.add_argument("--channels", type=int, default=3)
    args = parser.parse_args()

    print("Creating test dataset...", end=" ", flush=True)
    test_dataset = my_dataset_eval(args.test_input, args.test_GT)
    print("Done!", flush=True)

    print("Setup image save function...", end=" ", flush=True)
    setup_img_save_function(args.channels)

    print("Creating model...", end=" ", flush=True)
    model = Generator(io_channels=args.channels).to(0)
    print("Done!", flush=True)

    print("Loading model...", end=" ", flush=True)
    model_path = args.model_path
    loaded = torch.load(model_path)
    to_load = loaded
    model.load_state_dict(to_load, strict=False)
    print("Done!", flush=True)

    print("Creating test loader...", end=" ", flush=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    print("Done!", flush=True)
    save_dir = f"results/{args.name}"
    os.makedirs(save_dir, exist_ok=True)
    test(
        model,
        test_loader,
        save_image=True,
        save_dir=save_dir,
        experiment_name=args.name,
    )


if __name__ == "__main__":
    main()
