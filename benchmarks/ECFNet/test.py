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

from networks.ECFNet import *
from datasets.dataset_pairs_npy import my_dataset_eval


def _save_4ch_npy_to_img(
    img_ts,
    img_path,
    dng_info="/home/n5/chanwoo/background.dng",
):
    if dng_info is None:
        raise RuntimeError(
            "DNG information for saving 4 channeled npy file not provided"
        )
    # npy file in hwc manner to cwh manner
    data = rp.imread(dng_info)
    ts = img_ts.permute(0, 1, 3, 2)
    ts = torch.clamp(ts, 0, 1) * 1023
    ts = ts.cpu().numpy()

    GR = data.raw_image[0::2, 0::2]
    R = data.raw_image[0::2, 1::2]
    B = data.raw_image[1::2, 0::2]
    GB = data.raw_image[1::2, 1::2]
    GB[:, :] = 0
    B[:, :] = 0
    R[:, :] = 0
    GR[:, :] = 0

    w, h = (1280, 1792)

    GR[:w, :h] = ts[0][0][:w][:h]
    R[:w, :h] = ts[0][1][:w][:h]
    B[:w, :h] = ts[0][2][:w][:h]
    GB[:w, :h] = ts[0][3][:w][:h]
    start = (0, 464)  # (448 , 0) # 1792 1280 ->   3584, 2560
    end = (3584, 3024)
    newData = data.postprocess()
    output = newData[start[0] : end[0], start[1] : end[1]]
    # output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    imageio.imsave(img_path, output)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    save_image=False,
    save_dir=None,
):
    model.eval()
    ddp_loss = torch.zeros(3).to(0)

    iter_bar = tqdm.tqdm(test_loader)
    with torch.no_grad():
        for data, target, img_name in iter_bar:
            img_name = img_name[0]

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
                filename = f"{img_name}_{psnr:.4f}_{ssim:.4f}.png"
                _save_4ch_npy_to_img(output, os.path.join(save_dir, filename))

    print(
        "Test set: Average PSNR: {:.4f}, Average SSIM: {:.4f}".format(
            ddp_loss[0] / ddp_loss[2], ddp_loss[1] / ddp_loss[2]
        )
    )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--loss-lambda", type=float, default=0.5, choices=[0.5, 0.1, 0.01]
    )
    parser.add_argument("--test-input", type=str, default="data/test/input")
    parser.add_argument("--test-GT", type=str, default="data/test/GT")
    args = parser.parse_args()

    # loss_lambda = args.loss_lambda
    loss_lambda = 0.5

    test_dataset = my_dataset_eval(args.test_input, args.test_GT)

    model = ECFNet(in_nc=4, out_nc=4).to(0)
    model_path = f"experiments/ECFNet-3-{loss_lambda}/model_latest.pth"
    loaded = torch.load(model_path)
    to_load = loaded
    model.load_state_dict(to_load, strict=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    save_dir = f"results/ECFNet-3-{loss_lambda}"
    os.makedirs(save_dir, exist_ok=True)
    test(model, test_loader, save_image=True, save_dir=save_dir)


if __name__ == "__main__":
    main()