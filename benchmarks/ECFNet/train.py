import os
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import functools
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)


from torch.utils.data import DataLoader
from pathlib import Path

import tqdm

from networks.ECFNet import *
from datasets.dataset_pairs_npy import *
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


def print_root(rank, msg):
    if rank == 0:
        print(msg)


def interpolate_down(x):
    x_2 = F.interpolate(x, scale_factor=0.5)  # 1, 4, 128, 128
    x_4 = F.interpolate(x_2, scale_factor=0.5)  # 1, 4, 64, 64
    x_8 = F.interpolate(x_4, scale_factor=0.5)  # 1, 4, 32, 32
    return [x_8, x_4, x_2, x]


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / (2**30)
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def ecfnet_loss(lq, gt, ratio=0.5, eps=1e-12):
    charbonnier_loss = torch.mean(torch.sqrt((lq - gt) ** 2 + eps))
    fft_lq = abs(torch.fft.fftn(lq, dim=(-2, -1)))
    fft_gt = abs(torch.fft.fftn(gt, dim=(-2, -1)))
    fft_abs = F.l1_loss(fft_lq, fft_gt, reduction="mean")
    return charbonnier_loss + ratio * fft_abs


def ensemble_loss(lqs, gts, ratio=0.5, eps=1e-12):
    loss = 0
    for lq, gt in zip(lqs, gts):
        loss += ecfnet_loss(lq, gt, ratio, eps)
    return loss


def train(
    args,
    model,
    rank,
    world_size,
    train_loader,
    optimizer,
    epoch,
    loss_lambda=0.5,
    sampler=None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    train_bar = tqdm.tqdm(train_loader) if rank == 0 else train_loader
    for data, target in train_bar:
        data = data.to(rank)
        target = target.to(rank)
        optimizer.zero_grad()
        data.requires_grad = True
        output = model(data)
        targets = interpolate_down(target)
        loss = ensemble_loss(output, targets, ratio=loss_lambda)
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, ddp_loss[0] / ddp_loss[1]))


def save_image(output, save_path):
    print(f"Saving image to {save_path}")


def test(
    model: nn.Module,
    rank: int,
    test_loader: DataLoader,
    save_image=False,
    save_dir=None,
    epoch=None,
):
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)

    iter_bar = tqdm.tqdm(test_loader) if rank == 0 else test_loader
    with torch.no_grad():
        for data, target, img_name in iter_bar:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            # if ouptut is list, take only the last one
            if isinstance(output, list):
                output = output[-1]
            ddp_loss[0] += peak_signal_noise_ratio(output, target)
            ddp_loss[1] += structural_similarity_index_measure(output, target)
            ddp_loss[2] += len(data)
            if save_image and save_dir is not None:
                save_image(output, os.path.join(save_dir, img_name))

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0 and epoch is not None:
        print(
            "Test set: Average PSNR: {:.4f}, Average SSIM: {:.4f}".format(
                ddp_loss[0] / ddp_loss[2], ddp_loss[1] / ddp_loss[2]
            )
        )


def load_states(experiment_name: str):
    dir = Path("experiments") / experiment_name
    if not dir.exists():
        # make one
        dir.mkdir(parents=True, exist_ok=True)
        return None

    # check if model_latest, state_laste, and opt_latest exists
    model_latest = dir / "model_latest.pth"
    sched_latest = dir / "lr_scheduler_latest.pth"
    if not model_latest.exists() or not sched_latest.exists():
        return None

    # load the states
    model_state = torch.load(model_latest)
    sched_state = torch.load(sched_latest)
    return model_state, sched_state


def fsdp_main(rank, world_size, opt):
    setup(rank, world_size)
    print_root(rank, f"Running on rank {rank} of {world_size}")

    train_dataset = my_dataset(opt.train_input, opt.train_gt, opt.patch_size)
    print_root(rank, f"Training dataset path: {opt.train_input}")
    if opt.limit_train_batches > 1:
        train_dataset = Subset(
            train_dataset, np.arange(opt.limit_train_batches * world_size)
        )
        print_root(rank, f"Training dataset size: {len(train_dataset)}")
    else:
        print_root(rank, f"Training dataset size: {len(train_dataset)}")

    validation_dataset = my_dataset_eval(opt.val_input, opt.val_gt)
    print_root(rank, f"Validation dataset path: {opt.val_input}")
    if opt.limit_val_batches > 1:
        validation_dataset = Subset(
            validation_dataset, np.arange(opt.limit_val_batches * world_size)
        )
        print_root(rank, f"Validation dataset size: {len(validation_dataset)}")
    else:
        print_root(rank, f"Validation dataset size: {len(validation_dataset)}")

    train_sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    validation_sampler = DistributedSampler(
        validation_dataset, rank=rank, num_replicas=world_size
    )
    print_root(rank, f"Created Sampler")

    train_kwargs = {"batch_size": opt.batch_size, "sampler": train_sampler}
    test_kwargs = {"batch_size": 1, "sampler": validation_sampler}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    print_root(rank, f"Created train loader")
    validation_loader = torch.utils.data.DataLoader(validation_dataset, **test_kwargs)
    print_root(rank, f"Created validation loader")

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=128
    )
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = ECFNet(in_nc=4, out_nc=4).to(rank)
    # Activation Checkpoint Wrapper
    if opt.aggressive_checkpointing:
        blocks = [EBlock, DBlock, AFF1, AFF, SCM, FAM, SAM]
        check_fn = lambda m: isinstance(m, tuple(blocks))
        apply_activation_checkpointing(model, check_fn=check_fn)

    model = FSDP(
        model,
        device_id=rank,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        auto_wrap_policy=my_auto_wrap_policy,
    )
    print_root(rank, f"Created model")

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    print_root(rank, f"Created optimizer")
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.num_epochs, eta_min=1e-6
    )
    print_root(rank, f"Created lr scheduler")

    save_path = Path(f"experiments/{opt.experiment_name}")
    model_state, sched_state = None, None
    model_path = opt.pretrained_model
    if model_path is not None:
        model_path = Path(model_path)
        if model_path.exists():
            model_state = torch.load(model_path)
            print_root(rank, f"Loaded pretrained model from {model_path}")
    start_epoch = 1
    if opt.auto_resume:
        found_state = load_states(opt.experiment_name)
        if found_state is not None:
            model_state, sched_state = found_state
            start_epoch = model_state["current_epoch"] + 1
            print_root(rank, f"Resuming from previous state, epoch {start_epoch}")
    else:
        found_state = None
        print_root(rank, "Not resuming from previous state")

    if model_state is not None:
        if "params" in model_state:
            model_state = model_state["params"]
        model.load_state_dict(model_state, strict=False)
    if sched_state is not None:
        lr_scheduler.load_state_dict(sched_state)
        print_root(rank, f"Loaded previous state")

    init_start_event.record()
    for epoch in range(start_epoch, opt.num_epochs + 1):
        print_root(rank, f"Training at epoch {epoch}")
        train(
            opt,
            model,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            loss_lambda=opt.loss_lambda,
            sampler=train_sampler,
        )
        if epoch % opt.validation_interval == 0:
            print_root(rank, f"Testing at epoch {epoch}")
            test(model, rank, validation_loader, epoch=epoch)

        if epoch % opt.save_interval == 0:
            print_root(rank, f"Saving at epoch {epoch}")
            dist.barrier()
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.state_dict()
                cpu_lr_scheduler = lr_scheduler.state_dict()

            if rank == 0:
                cpu_state["current_epoch"] = epoch
                torch.save(
                    cpu_state,
                    os.path.join(save_path, f"model_epoch_{epoch:03}.pth"),
                )
                # save latest states
                torch.save(
                    cpu_state,
                    os.path.join(save_path, f"model_latest.pth"),
                )
                torch.save(
                    cpu_lr_scheduler,
                    os.path.join(save_path, f"lr_scheduler_latest.pth"),
                )

            dist.barrier()
        lr_scheduler.step()

    init_end_event.record()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    if rank == 0:
        # save final model
        print(f"Saving final model")
        torch.save(
            cpu_state,
            os.path.join(save_path, f"model_final.pth"),
        )

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--train-input", type=str, required=True)
    data_group.add_argument("--train-gt", type=str, required=True)
    data_group.add_argument("--val-input", type=str, required=True)
    data_group.add_argument("--val-gt", type=str, required=True)

    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument("--batch-size", type=int, default=1)
    train_group.add_argument("--num-epochs", type=int, default=150)
    train_group.add_argument("--lr", type=float, default=8 * 1e-6)
    train_group.add_argument("--patch-size", type=int, default=800)
    train_group.add_argument("--loss-lambda", type=float, default=0.5)

    trainer_group = parser.add_argument_group("Trainer Options")
    trainer_group.add_argument("--num-workers", type=int, default=8)
    trainer_group.add_argument("--validation-interval", type=int, default=5)
    trainer_group.add_argument("--limit-train-batches", type=int, default=1.0)
    trainer_group.add_argument("--limit-val-batches", type=int, default=1.0)
    trainer_group.add_argument("--aggressive-checkpointing", action="store_true")

    saving_group = parser.add_argument_group("Experiment Saving")
    saving_group.add_argument("--experiment-name", type=str, default="ECFNet")
    saving_group.add_argument("--save-interval", type=int, default=15)

    load_group = parser.add_argument_group("Experiment Loading")
    load_group.add_argument("--auto-resume", action="store_true")
    load_group.add_argument("--pretrained-model", type=str)

    args = parser.parse_args()

    torch.manual_seed(42)
    # torch.set_float32_matmul_precision("medium")
    torch.set_float32_matmul_precision("highest")

    WORLD_SIZE = 4  # in our case, we have 4 GPUs
    # WORLD_SIZE = torch.cuda.device_count()

    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
