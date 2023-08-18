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
import tqdm
import logging

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

from loss import GeneratorLoss
from networks.SRGAN import *
from datasets.dataset_pairs_npy import *
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


from util import setup_img_save_function

MSE_MODULE = nn.MSELoss()
IMG_SAVE = None
logger = None


def print_root(rank, msg):
    if rank == 0:
        print(msg)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(
    args,
    rank,
    world_size,
    train_loader,
    Gmodel,
    Gopt,
    epoch,
    sampler=None,
):
    global logger
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    bc = 0
    train_bar = tqdm.tqdm(train_loader) if rank == 0 else train_loader
    for data, target in train_bar:
        bc += 1
        data = data.to(rank)
        target = target.to(rank)

        Gmodel.zero_grad()
        output = Gmodel(data)
        mseloss = MSE_MODULE(output, target)
        mseloss.backward()

        Gopt.step()

        ddp_loss[0] += mseloss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        msg = f"Train Epoch: {epoch} \tG_Loss: {ddp_loss[0] / ddp_loss[1]}"
        logger.info(msg)


def test(
    Gmodel: nn.Module,
    rank: int,
    test_loader: DataLoader,
    save_image=None,
    save_dir=None,
    epoch=None,
):
    global logger
    Gmodel.eval()
    ddp_loss = torch.zeros(3).to(rank)

    iter_bar = tqdm.tqdm(test_loader) if rank == 0 else test_loader
    with torch.no_grad():
        for data, target, img_name in iter_bar:
            data, target = data.to(rank), target.to(rank)
            output = Gmodel(data)
            # if ouptut is list, take only the last one
            if isinstance(output, list):
                output = output[-1]
            ddp_loss[0] += peak_signal_noise_ratio(output, target)
            ddp_loss[1] += structural_similarity_index_measure(output, target)
            ddp_loss[2] += len(data)
            img_savename = f"{img_name[0].split('.')[0]}_epoch_{epoch}.png"
            if save_image and save_dir is not None:
                save_image(output, os.path.join(save_dir, img_savename))

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0 and epoch is not None:
        msg = f"Epoch {epoch}: Average PSNR: {ddp_loss[0] / ddp_loss[2]}, Average SSIM: {ddp_loss[1] / ddp_loss[2]}"
        logger.info(msg)


def load_states(experiment_name: str):
    dir = Path("experiments") / experiment_name
    if not dir.exists():
        # make one
        dir.mkdir(parents=True, exist_ok=True)
        return None, None

    # check if model_latest, state_laste, and opt_latest exists
    Gmodel_latest = dir / "Gmodel_latest.pth"
    Dmodel_latest = dir / "Dmodel_latest.pth"

    if not Gmodel_latest.exists() or not Dmodel_latest.exists():
        return None, None

    # load the states
    Gmodel_state = torch.load(Gmodel_latest)
    Dmodel_state = torch.load(Dmodel_latest)
    return Gmodel_state, Dmodel_state


def fsdp_main(rank, world_size, opt):
    setup(rank, world_size)
    print_root(rank, f"Running on rank {rank} of {world_size}")

    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    if rank == 0:
        handler = logging.FileHandler(f"logs/{opt.experiment_name}.log")
        logger.addHandler(handler)

    train_dataset = my_dataset(
        opt.train_input, opt.train_gt, opt.mapping, opt.patch_size
    )
    print_root(rank, f"Training dataset path: {opt.train_input}")
    if opt.limit_train_batches > 1:
        train_dataset = Subset(
            train_dataset, np.arange(opt.limit_train_batches * world_size)
        )
        print_root(rank, f"Training dataset size: {len(train_dataset)}")
    else:
        print_root(rank, f"Training dataset size: {len(train_dataset)}")

    validation_dataset = my_dataset_eval(opt.val_input, opt.val_gt, opt.mapping)
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

    train_kwargs = {
        "batch_size": opt.batch_size,
        "sampler": train_sampler,
    }
    test_kwargs = {
        "batch_size": 1,
        "sampler": validation_sampler,
    }
    cuda_kwargs = {
        "num_workers": 2,
        "pin_memory": True,
        "shuffle": False,
        "pin_memory_device": f"cuda:{rank}",
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    print_root(rank, f"Created train loader")
    validation_loader = torch.utils.data.DataLoader(validation_dataset, **test_kwargs)
    print_root(rank, f"Created validation loader")

    #
    # Creating Generator and Discriminator
    #

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=128
    )
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    global IMG_SAVE
    channels = opt.channels
    Gmodel = Generator(io_channels=channels).to(rank)
    IMG_SAVE = setup_img_save_function(channels)

    # Activation Checkpoint Wrapper
    if opt.aggressive_checkpointing:
        blocks = [ResidualBlock, UpsampleBLock, CLBLock, CBLBlock]
        check_fn = lambda m: isinstance(m, tuple(blocks))
        apply_activation_checkpointing(Gmodel, check_fn=check_fn)

    sharding_strategy = ShardingStrategy.NO_SHARD
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    Gmodel = FSDP(
        Gmodel,
        device_id=rank,
        sharding_strategy=sharding_strategy,
        backward_prefetch=backward_prefetch,
        forward_prefetch=True,
        auto_wrap_policy=my_auto_wrap_policy,
    )
    print_root(rank, f"Created model")

    #
    # Creating optimizer and lr schedulers for each models
    #

    Gopt = optim.Adam(Gmodel.parameters(), lr=opt.lr)
    print_root(rank, f"Created optimizer")

    save_path = Path(f"experiments/{opt.experiment_name}")
    checkpoint_path = save_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    final_path = save_path / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    model_dir = opt.pretrained_model_dir
    if model_dir is not None:
        model_dirpath = Path(model_dir)
        if model_dirpath.exists():
            Gmodel_path = model_dirpath / "Gmodel.pth"
            if Gmodel_path.exists():
                Gmodel_state = torch.load(Gmodel_path)
                Gmodel.load_state_dict(Gmodel_state, strict=False)
                print_root(rank, f"Loaded pretrained Generator model from {model_dir}")
    start_epoch = 1

    # calculate num epochs and iteration per epoch
    iterations = opt.num_iterations
    iterations_per_epoch = len(train_loader)
    num_epochs = iterations // iterations_per_epoch

    print_root(rank, f"Number of iterations: {iterations}")
    print_root(rank, f"Number of iterations per epoch: {iterations_per_epoch}")
    print_root(rank, f"Number of epochs: {num_epochs}")

    init_start_event.record()
    img_savepath = opt.img_savepath

    if rank == 0:
        if not os.path.exists(img_savepath):
            os.makedirs(img_savepath)

    for epoch in range(start_epoch, num_epochs + 1):
        print_root(rank, f"Training at epoch {epoch}")
        train(
            opt,
            rank,
            world_size,
            train_loader,
            Gmodel,
            Gopt,
            epoch,
            sampler=train_sampler,
        )
        if epoch % opt.validation_interval == 0:
            print_root(rank, f"Testing at epoch {epoch}")
            test(
                Gmodel,
                rank,
                validation_loader,
                epoch=epoch,
                save_dir=img_savepath,
                save_image=IMG_SAVE,
            )

        if epoch % opt.save_interval == 0:
            print_root(rank, f"Saving at epoch {epoch}")
            dist.barrier()
            save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
            with FSDP.state_dict_type(
                Gmodel, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = Gmodel.state_dict()
                gmodel_save_path = os.path.join(
                    checkpoint_path, f"Gmodel_epoch_{epoch:03}.pth"
                )
                if rank == 0:
                    torch.save(cpu_state, gmodel_save_path)
                    torch.save(
                        cpu_state, os.path.join(checkpoint_path, f"Gmodel_latest.pth")
                    )

            dist.barrier()

    init_end_event.record()
    save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
    with FSDP.state_dict_type(Gmodel, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = Gmodel.state_dict()
        if rank == 0:
            # save final model
            print(f"Saving final model")
            torch.save(
                cpu_state,
                os.path.join(final_path, "Gmodel.pth"),
            )

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--train-input", type=str, required=True)
    data_group.add_argument("--train-gt", type=str, required=True)
    data_group.add_argument("--val-input", type=str, required=True)
    data_group.add_argument("--val-gt", type=str, required=True)

    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument("--channels", type=int, default=3)

    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument("--batch-size", type=int, default=4)
    train_group.add_argument("--num-iterations", type=int, default=int(1e5))
    train_group.add_argument("--lr", type=float, default=1e-4)
    train_group.add_argument("--patch-size", type=int, default=256)
    train_group.add_argument(
        "--mapping", type=str, required=True, choices=["norm", "tonemap"]
    )

    trainer_group = parser.add_argument_group("Trainer Options")
    trainer_group.add_argument("--num-workers", type=int, default=8)
    trainer_group.add_argument("--validation-interval", type=int, default=100)
    trainer_group.add_argument("--limit-train-batches", type=int, default=1.0)
    trainer_group.add_argument("--limit-val-batches", type=int, default=1.0)
    trainer_group.add_argument("--aggressive-checkpointing", action="store_true")

    saving_group = parser.add_argument_group("Experiment Saving")
    saving_group.add_argument("--experiment-name", type=str, default="SRGAN")
    saving_group.add_argument("--save-interval", type=int, default=100)
    saving_group.add_argument("--img-savepath", type=str, required=True)

    load_group = parser.add_argument_group("Experiment Loading")
    load_group.add_argument(
        "--pretrained-model-dir",
        type=str,
        help="Directory that holds Generator model and Discriminator Model",
    )

    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")

    WORLD_SIZE = 4  # in our case, we have 4 GPUs
    # WORLD_SIZE = torch.cuda.device_count()

    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
