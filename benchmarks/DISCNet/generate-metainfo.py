import os
import argparse
import tqdm
from pathlib import Path

META_DIR = "meta-info-files"


def generate_metainfo(dirpath, psf, mode):
    if mode not in ("train", "val", "test"):
        raise ValueError("Mode must be one of train, val, test")

    files = os.listdir(dirpath)
    files = [f for f in files if f.endswith(".npy")]
    files.sort(key=lambda x: int(x.split(".")[0]))

    psf_path = Path(psf)
    abspath = psf_path.absolute()

    with open(os.path.join(META_DIR, f"{mode}-metainfo.txt"), "w") as f:
        for file in tqdm.tqdm(files):
            f.write(f"{file} {abspath}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-input",
        type=str,
        required=True,
        help="Path to train ground truth directory",
    )
    parser.add_argument(
        "--val-input",
        type=str,
        required=True,
        help="Path to validation ground truth directory",
    )
    parser.add_argument(
        "--test-input",
        type=str,
        help="(Optional) Path to test input directory. As of 11.01.2023, we do not provide test GT. So, this is not required.",
    )
    parser.add_argument(
        "--psf-path", type=str, default="sit-train-psf.npy", help="Path to PSF file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.train_input):
        raise ValueError("Train ground truth file does not exist")
    if not os.path.exists(args.val_input):
        raise ValueError("Validation ground truth file does not exist")
    if args.test_input and not os.path.exists(args.test_input):
        raise ValueError("Validation ground truth file does not exist")
    if not os.path.exists(args.psf_path):
        raise ValueError("PSF file does not exist")
    if not args.psf_path.endswith(".npy"):
        raise ValueError("PSF file must be a numpy file")

    generate_metainfo(args.train_input, args.psf_path, "train")
    generate_metainfo(args.val_input, args.psf_path, "val")
    if args.test_input:
        generate_metainfo(args.test_input, args.psf_path, "test")


if __name__ == "__main__":
    main()
