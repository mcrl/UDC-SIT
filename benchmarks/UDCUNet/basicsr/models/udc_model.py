import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import os
import math

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, tensor2npy
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

from random import random

import numpy as np
import torchinfo
import rawpy as rp
import imageio
import cv2


def tone_map(x, c=0.25):
    # Modified Reinhard tone mapping.
    mapped_x = x / (x + c)
    return mapped_x


@MODEL_REGISTRY.register()
class UDCModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(UDCModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # torchinfo.summary(self.net_g, (8, 3, 256, 256))

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            param_key = self.opt["path"].get("param_key_g", "params")
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key,
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt["path"].get("strict_load_g", True),
                    "params_ema",
                )
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get("perceptual_opt"):
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(
                self.device
            )
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("Both pixel and perceptual losses are None.")

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        # for ad-hoc csv logger
        self.init_csv_logger()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_g"].pop("type")
        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params, **train_opt["optim_g"]
        )
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)  # here we run forward pass

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        N, C, H, W = self.lq.shape
        # if H > 1000 or W > 1000:
        #   self.output = self.test_crop9()
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
            self.net_g_ema.train()
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_crop9(self):
        if hasattr(self, "net_g_ema"):
            return self.test_crop_ema()
        else:
            return self.test_crop_netg()

    def calculate_crop(self, shape):
        _, _, H, W = shape
        h, w = math.ceil(H / 3), math.ceil(W / 3)
        rf = 30
        return (
            (h, w, rf),
            (0, self.down(h + rf), 0, self.down(w + rf)),
            (self.down(h - rf), self.down(2 * h + rf), 0, self.down(w + rf)),
            (self.down(2 * h - rf), H, 0, self.down(w + rf)),
            (0, self.down(h + rf), self.down(w - rf), self.down(2 * w + rf)),
            (
                self.down(h - rf),
                self.down(2 * h + rf),
                self.down(w - rf),
                self.down(2 * w + rf),
            ),
            (self.down(2 * h - rf), H, self.down(w - rf), self.down(2 * w + rf)),
            (0, self.down(h + rf), self.down(2 * w - rf), W),
            (self.down(h - rf), self.down(2 * h + rf), self.down(2 * w - rf), W),
            (self.down(2 * h - rf), H, self.down(2 * w - rf), W),
        )

    def down(self, number, divisor=8):
        return (number // divisor) * divisor

    def test_crop_netg(self):
        self.net_g.eval()
        calculated = self.calculate_crop(self.lq.shape)
        h, w, rf = calculated[0]

        with torch.no_grad():
            imTL = self.net_g(
                self.lq[
                    :,
                    :,
                    calculated[1][0] : calculated[1][1],
                    calculated[1][2] : calculated[1][3],
                ]
            )[:, :, 0:h, 0:w]
            imML = self.net_g(
                self.lq[
                    :,
                    :,
                    calculated[2][0] : calculated[2][1],
                    calculated[2][2] : calculated[2][3],
                ]
            )[:, :, rf : (rf + h), 0:w]
            imBL = self.net_g(
                self.lq[
                    :,
                    :,
                    calculated[3][0] : calculated[3][1],
                    calculated[3][2] : calculated[3][3],
                ]
            )[:, :, rf:, 0:w]
            imTM = self.net_g(
                self.lq[
                    :,
                    :,
                    calculated[4][0] : calculated[4][1],
                    calculated[4][2] : calculated[4][3],
                ]
            )[:, :, 0:h, rf : (rf + w)]
            imMM = self.net_g(
                self.lq[
                    :,
                    :,
                    calculated[5][0] : calculated[5][1],
                    calculated[5][2] : calculated[5][3],
                ]
            )[:, :, rf : (rf + h), rf : (rf + w)]
            imBM = self.net_g(
                self.lq[
                    :,
                    :,
                    calculated[6][0] : calculated[6][1],
                    calculated[6][2] : calculated[6][3],
                ]
            )[:, :, rf:, rf : (rf + w)]
            imTR = self.net_g(
                self.lq[
                    :,
                    :,
                    calculated[7][0] : calculated[7][1],
                    calculated[7][2] : calculated[7][3],
                ]
            )[:, :, 0:h, rf:]
            imMR = self.net_g(
                self.lq[
                    :,
                    :,
                    calculated[8][0] : calculated[8][1],
                    calculated[8][2] : calculated[8][3],
                ]
            )[:, :, rf : (rf + h), rf:]
            imBR = self.net_g(
                self.lq[
                    :,
                    :,
                    calculated[9][0] : calculated[9][1],
                    calculated[9][2] : calculated[9][3],
                ]
            )[:, :, rf:, rf:]

        imT = torch.cat((imTL, imTM, imTR), 3)
        imM = torch.cat((imML, imMM, imMR), 3)
        imB = torch.cat((imBL, imBM, imBR), 3)
        output_cat = torch.cat((imT, imM, imB), 2)
        self.net_g.train()
        return output_cat

    def test_crop_ema(self):
        self.net_g_ema.eval()
        N, C, H, W = self.lq.shape
        h, w = math.ceil(H / 3), math.ceil(W / 3)
        rf = 30
        with torch.no_grad():
            imTL = self.net_g_ema(self.lq[:, :, 0 : h + rf, 0 : w + rf])[:, :, 0:h, 0:w]
            imML = self.net_g_ema(self.lq[:, :, h - rf : 2 * h + rf, 0 : w + rf])[
                :, :, rf : (rf + h), 0:w
            ]
            imBL = self.net_g_ema(self.lq[:, :, 2 * h - rf :, 0 : w + rf])[
                :, :, rf:, 0:w
            ]
            imTM = self.net_g_ema(self.lq[:, :, 0 : h + rf, w - rf : 2 * w + rf])[
                :, :, 0:h, rf : (rf + w)
            ]
            imMM = self.net_g_ema(
                self.lq[:, :, h - rf : 2 * h + rf, w - rf : 2 * w + rf]
            )[:, :, rf : (rf + h), rf : (rf + w)]
            imBM = self.net_g_ema(self.lq[:, :, 2 * h - rf :, w - rf : 2 * w + rf])[
                :, :, rf:, rf : (rf + w)
            ]
            imTR = self.net_g_ema(self.lq[:, :, 0 : h + rf, 2 * w - rf :])[
                :, :, 0:h, rf:
            ]
            imMR = self.net_g_ema(self.lq[:, :, h - rf : 2 * h + rf, 2 * w - rf :])[
                :, :, rf : (rf + h), rf:
            ]
            imBR = self.net_g_ema(self.lq[:, :, 2 * h - rf :, 2 * w - rf :])[
                :, :, rf:, rf:
            ]

        imT = torch.cat((imTL, imTM, imTR), 3)
        imM = torch.cat((imML, imMM, imMR), 3)
        imB = torch.cat((imBL, imBM, imBR), 3)
        output_cat = torch.cat((imT, imM, imB), 2)
        self.net_g_ema.train()
        return output_cat

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt["rank"] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        def _clamp(tensor, clamp_opts):
            if clamp_opts:
                return torch.clamp(tensor, clamp_opts["min"], clamp_opts["max"])
            return tensor

        def _save_4ch_npy_to_img(
            img_npy, img_path, dng_info, in_pxl=255.0, max_pxl=1023.0
        ):
            if dng_info is None:
                raise RuntimeError(
                    "DNG information for saving 4 channeled npy file not provided"
                )
            # npy file in hwc manner to cwh manner
            data = rp.imread(dng_info)
            npy = img_npy.transpose(2, 1, 0) / in_pxl * max_pxl

            GR = data.raw_image[0::2, 0::2]
            R = data.raw_image[0::2, 1::2]
            B = data.raw_image[1::2, 0::2]
            GB = data.raw_image[1::2, 1::2]
            GB[:, :] = 0
            B[:, :] = 0
            R[:, :] = 0
            GR[:, :] = 0

            w, h = npy.shape[1:]

            GR[:w, :h] = npy[0][:w][:h]
            R[:w, :h] = npy[1][:w][:h]
            B[:w, :h] = npy[2][:w][:h]
            GB[:w, :h] = npy[3][:w][:h]
            newData = data.postprocess()
            start = (0, 464)  # (448 , 0) # 1792 1280 ->   3584, 2560
            end = (3584, 3024)
            output = newData[start[0] : end[0], start[1] : end[1]]
            # output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
            imageio.imsave(img_path, output)

        def _save_image(img_npy, img_path, max_pxl=1023.0, dng_info=None):
            if img_npy.shape[2] == 3:
                mmcv.imwrite(img_npy, img_path)
            else:
                _save_4ch_npy_to_img(
                    img_npy, img_path, max_pxl=max_pxl, dng_info=dng_info
                )

        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        use_pbar = self.opt["val"].get("pbar", False)

        if save_img and not self.opt["is_train"]:
            img_dirpath = osp.join(self.opt["path"]["visualization"], dataset_name)
            os.makedirs(img_dirpath, exist_ok=True)

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {
                    metric: 0 for metric in self.opt["val"]["metrics"].keys()
                }
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        clamp_opts = self.opt["val"].get("clamp")
        dng_info = self.opt["val"].get("dng_info")
        max_pxl = self.opt["val"].get("max_pxl", 1023.0)

        _logger = get_root_logger()
        exp_name = self.opt["name"]
        save_img_ratio = self.opt["val"].get("save_img_ratio", 2)
        psnrs = []
        save_npy = self.opt["val"].get("save_npy", False)

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals["result"]])
            metric_data["img"] = sr_img
            if "gt" in visuals:
                gt_img_metric = tensor2img(visuals["gt"])
                # gt_img = tensor2img([visuals['gt']])
                metric_data["img2"] = gt_img_metric
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            image_metric = {}

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt["val"]["metrics"].items():
                    _metric = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += _metric
                    image_metric[name] = _metric
                    if name == "psnr":
                        psnrs.append(_metric)
            metric_message = (
                f"{img_name}_{image_metric.get('psnr', 0)}_{image_metric.get('ssim',0)}"
            )
            _logger.info(metric_message)

            if save_img:
                if self.opt["is_train"]:
                    if random() < save_img_ratio:
                        imgdir = osp.join(self.opt["path"]["visualization"], img_name)
                        os.makedirs(imgdir, exist_ok=True)
                        save_img_path = osp.join(
                            imgdir, f"{img_name}_{current_iter}.png"
                        )
                        _save_image(
                            sr_img, save_img_path, max_pxl=max_pxl, dng_info=dng_info
                        )
                        if save_npy:
                            save_npy_path = osp.join(
                                imgdir, f"{img_name}_{current_iter}.npy"
                            )
                            np.save(save_npy_path, sr_img.transpose(2, 1, 0))

                else:
                    if self.opt["val"]["suffix"]:
                        save_img_path = osp.join(
                            img_dirpath,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png',
                        )
                    else:
                        save_img_path = osp.join(
                            img_dirpath, f'{img_name}_{self.opt["name"]}.png'
                        )
                    _save_image(
                        sr_img, save_img_path, max_pxl=max_pxl, dng_info=dng_info
                    )
                    if save_npy:
                        save_npy_path = osp.join(
                            img_dirpath, f"{img_name}_{current_iter}.npy"
                        )
                        np.save(save_npy_path, sr_img.transpose(2, 1, 0))

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx + 1
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            log = f"{max(psnrs)} {sum(psnrs)/len(psnrs)} {len(psnrs)}"
            _logger.info(log)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )
        if self._adhoc_csv_enabled():
            self._log_csv(current_iter)

    def _log_csv(self, current_iter):
        line = f"{current_iter}"
        for metric in self.metric_names:
            value = self.metric_results[metric]
            line += f",{value:.6f}"
        with open(self.__ad_hoc_csv_filename, "a") as f:
            f.write("\n" + line)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, "net_g_ema"):
            self.save_network(
                [self.net_g, self.net_g_ema],
                "net_g",
                current_iter,
                param_key=["params", "params_ema"],
            )
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)

    def init_csv_logger(self):
        # print(self.opt["ad_hoc_logger"])
        if not self._adhoc_csv_enabled():
            return
        filename = self.opt["ad_hoc_logger"].get("path") + ".csv"
        self.metric_names = self.opt["val"]["metrics"].keys()

        header = "iter,"
        for metric_name in self.metric_names:
            header += f"{metric_name},"

        header = header[:-1]

        with open(filename, "w") as f:
            f.write(header)
            self.__ad_hoc_csv_filename = filename

    def _adhoc_csv_enabled(self):
        return (
            self.opt.get("ad_hoc_logger", False)
            and self.opt["ad_hoc_logger"].get("type", False) == "csv"
        )
