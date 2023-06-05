import argparse
import datetime
import logging
import math
import os
import pdb
import random
import time
import torch
from mmcv.runner import get_dist_info, get_time_str, init_dist
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (
    MessageLogger,
    check_resume,
    get_env_info,
    get_root_logger,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    mkdir_and_rename,
    set_random_seed,
)
from basicsr.utils.options import dict2str, parse


def write_info_file(info_filepath: str, psf_kernel_path: str):
    with open(info_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [line.strip().split(" ") for line in lines]
    if len(lines[0]) > 1:
        return

    content = ""

    for line in lines:
        content += f"{line[0]} {psf_kernel_path}\n"

    with open(info_filepath, "w", encoding="utf-8") as f:
        f.write(content)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def load_resume_state(opt):
    resume_state_path = None
    if opt["auto_resume"]:
        state_path = osp.join("experiments", opt["name"], "training_states")
        if osp.isdir(state_path):
            states = list(
                scandir(state_path, suffix="state", recursive=False, full_path=False)
            )
            if len(states) != 0:
                states = [float(v.split(".state")[0]) for v in states]
                resume_state_path = osp.join(state_path, f"{max(states):.0f}.state")
                opt["path"]["resume_state"] = resume_state_path
    else:
        if opt["path"].get("resume_state"):
            resume_state_path = opt["path"]["resume_state"]

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id)
        )
        check_resume(opt, resume_state["iter"])
    return resume_state


def main():
    # options
    os.environ["WANDB_MODE"] = "dryrun"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to option YAML file."
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--auto-resume", action="store_true", default=False)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True)

    train_folders = opt["datasets"]["train"]["folders"]
    for folder in train_folders.keys():
        folder_opts = train_folders[folder]
        info_path = folder_opts["meta_info_file"]
        psf_path = folder_opts["psf_path"]
        write_info_file(info_path, psf_path)
    validation_folders = opt["datasets"]["val"]["folders"]
    for folder in validation_folders.keys():
        folder_opts = validation_folders[folder]
        info_path = folder_opts["meta_info_file"]
        psf_path = folder_opts["psf_path"]
        write_info_file(info_path, psf_path)

    # distributed training settings
    if args.launcher == "none":  # non-distributed training
        opt["dist"] = False
        print("Disable distributed training.", flush=True)
    else:
        opt["dist"] = True
        if args.launcher == "slurm" and "dist_params" in opt:
            init_dist(args.launcher, **opt["dist_params"])
        else:
            init_dist(args.launcher)

    rank, world_size = get_dist_info()
    opt["rank"] = rank
    opt["world_size"] = world_size
    opt["auto_resume"] = args.auto_resume

    # load resume states if exists
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    opt["root_path"] = opt.get("root_path", ".")
    if resume_state is None:
        make_exp_dirs(opt)
        if (
            opt["logger"].get("use_tb_logger")
            and "debug" not in opt["name"]
            and opt["rank"] == 0
        ):
            mkdir_and_rename(osp.join(opt["root_path"], "tb_logger", opt["name"]))

    log_file = osp.join(opt["path"]["log"], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name="basicsr", log_level=logging.INFO, log_file=log_file
    )
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize tensorboard logger and wandb logger
    tb_logger = None
    if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"]:
        log_dir = "./tb_logger/" + opt["name"]
        if resume_state is None and opt["rank"] == 0:
            mkdir_and_rename(log_dir)
        tb_logger = init_tb_logger(log_dir=log_dir)
    if (
        (opt["logger"].get("wandb") is not None)
        and (opt["logger"]["wandb"].get("project") is not None)
        and ("debug" not in opt["name"])
    ):
        assert (
            opt["logger"].get("use_tb_logger") is True
        ), "should turn on tensorboard when using wandb"
        init_wandb_logger(opt)

    # random seed
    seed = opt["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
        opt["manual_seed"] = seed
    logger.info(f"Random seed: {seed}")
    set_random_seed(seed + rank)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            dataset_enlarge_ratio = dataset_opt.get("dataset_enlarge_ratio", 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(
                train_set, world_size, rank, dataset_enlarge_ratio
            )
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt["num_gpu"],
                dist=opt["dist"],
                sampler=train_sampler,
                seed=seed,
            )

            # pdb.set_trace()
            num_iter_per_epoch = math.ceil(
                len(train_set)
                * dataset_enlarge_ratio
                / (dataset_opt["batch_size_per_gpu"] * opt["world_size"])
            )
            total_iters = int(opt["train"]["total_iter"])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                "Training statistics:"
                f"\n\tNumber of train images: {len(train_set)}"
                f"\n\tDataset enlarge ratio: {dataset_enlarge_ratio}"
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f"\n\tRequire iter number per epoch: {num_iter_per_epoch}"
                f"\n\tTotal epochs: {total_epochs}; iters: {total_iters}."
            )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt["num_gpu"],
                dist=opt["dist"],
                sampler=None,
                seed=seed,
            )
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f"{len(val_set)}"
            )
        else:
            raise ValueError(f"Dataset phase {phase} is not recognized.")
    assert train_loader is not None

    # create model
    if resume_state:
        check_resume(opt, resume_state["iter"])  # modify pretrain_model paths
    model = create_model(opt)

    # resume training
    if resume_state:
        logger.info(
            f"Resuming training from epoch: {resume_state['epoch']}, "
            f"iter: {resume_state['iter']}."
        )
        start_epoch = resume_state["epoch"]
        current_iter = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt["datasets"]["train"].get("prefetch_mode")
    if prefetch_mode is None or prefetch_mode == "cpu":
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == "cuda":
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f"Use {prefetch_mode} prefetch dataloader")
        if opt["datasets"]["train"].get("pin_memory") is not True:
            raise ValueError("Please set pin_memory=True for CUDAPrefetcher.")
    else:
        raise ValueError(
            f"Wrong prefetch_mode {prefetch_mode}."
            "Supported ones are: None, 'cuda', 'cpu'."
        )

    # torchinfo.summary(model.net_g, (8, 3, 256, 256))

    # training
    logger.info(f"Start training from epoch: {start_epoch}, iter: {current_iter}")
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt["train"].get("warmup_iter", -1)
            )
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt["logger"]["print_freq"] == 0:
                log_vars = {"epoch": epoch, "iter": current_iter}
                log_vars.update({"lrs": model.get_current_learning_rate()})
                log_vars.update({"time": iter_time, "data_time": data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                model.save(epoch, current_iter)

            # validation
            if (
                opt["val"]["val_freq"] is not None
                and current_iter % opt["val"]["val_freq"] == 0
            ):
                # model.validation(train_loader, current_iter, tb_logger, False)
                model.validation(
                    val_loader, current_iter, tb_logger, opt["val"]["save_img"]
                )

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"End of training. Time consumed: {consumed_time}")
    logger.info("Save the latest model.")
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt["val"]["val_freq"] is not None:
        model.validation(val_loader, current_iter, tb_logger, opt["val"]["save_img"])
    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    main()
