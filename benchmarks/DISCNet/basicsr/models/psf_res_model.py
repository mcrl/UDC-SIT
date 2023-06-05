import importlib
import mmcv
import torch
import math
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
import numpy as np
import rawpy as rp
import imageio
import os
from random import random
from basicsr.models import networks as networks
from basicsr.models.base_model import BaseModel
from basicsr.utils import ProgressBar, get_root_logger, tensor2img, tensor2raw, tensor2npy
import cv2

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class PSFResModel(BaseModel):
    """Base model for PSF-aware restoration."""

    def __init__(self, opt):
        super().__init__(opt)

        # define network
        self.net_g = networks.define_net_g(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if 'psf_code' in data:
            self.psf_code = data['psf_code'].to(self.device)
        else:
            self.psf_code = None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.psf_code)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        # here: we call vgg19 here for feature extraction, and then calculate loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            N, C, H, W = self.lq.shape
            if H > 1000 or W > 1000:
                self.output = self.test_crop9()
            else:
                self.output = self.net_g(self.lq, self.psf_code)
        self.net_g.train()

    def test_crop9(self):
        N, C, H, W = self.lq.shape
        h, w = math.ceil(H / 3), math.ceil(W / 3)
        rf = 30
        imTL = self.net_g(
            self.lq[:, :, 0:h + rf, 0:w + rf], self.psf_code)[:, :, 0:h, 0:w]
        imML = self.net_g(self.lq[:, :, h - rf:2 * h + rf,
                          0:w + rf], self.psf_code)[:, :, rf:(rf + h), 0:w]
        imBL = self.net_g(
            self.lq[:, :, 2 * h - rf:, 0:w + rf], self.psf_code)[:, :, rf:, 0:w]
        imTM = self.net_g(
            self.lq[:, :, 0:h + rf, w - rf:2 * w + rf], self.psf_code)[:, :, 0:h, rf:(rf + w)]
        imMM = self.net_g(self.lq[:, :, h - rf:2 * h + rf, w - rf:2 *
                          w + rf], self.psf_code)[:, :, rf:(rf + h), rf:(rf + w)]
        imBM = self.net_g(
            self.lq[:, :, 2 * h - rf:, w - rf:2 * w + rf], self.psf_code)[:, :, rf:, rf:(rf + w)]
        imTR = self.net_g(
            self.lq[:, :, 0:h + rf, 2 * w - rf:], self.psf_code)[:, :, 0:h, rf:]
        imMR = self.net_g(self.lq[:, :, h - rf:2 * h + rf,
                          2 * w - rf:], self.psf_code)[:, :, rf:(rf + h), rf:]
        imBR = self.net_g(
            self.lq[:, :, 2 * h - rf:, 2 * w - rf:], self.psf_code)[:, :, rf:, rf:]

        imT = torch.cat((imTL, imTM, imTR), 3)
        imM = torch.cat((imML, imMM, imMR), 3)
        imB = torch.cat((imBL, imBM, imBR), 3)
        output_cat = torch.cat((imT, imM, imB), 2)
        return output_cat

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        if self.opt.get("rank", 0) == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):

        def _clamp(tensor, clamp_opts):
            if clamp_opts:
                return torch.clamp(tensor, clamp_opts["min"], clamp_opts["max"])
            return tensor

        def _save_4ch_npy_to_img(img_npy, img_path, dng_info, in_pxl=255., max_pxl=1023.):
            if dng_info is None:
                raise RuntimeError(
                    "DNG information for saving 4 channeled npy file not provided")
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
            imageio.imsave(img_path, output)

        def _save_image(img_npy, img_path, max_pxl=1023., dng_info=None):
            if img_npy.shape[2] == 3:
                mmcv.imwrite(img_npy, img_path)
            else:
                _save_4ch_npy_to_img(
                    img_npy, img_path, max_pxl=max_pxl, dng_info=dng_info)

        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = ProgressBar(len(dataloader))

        clamp_opts = self.opt['val'].get("clamp")
        dng_info = self.opt["val"].get("dng_info")
        max_pxl = self.opt["val"].get("max_pxl", 1023.)

        if save_img and not self.opt["is_train"]:
            img_dirpath = osp.join(
                self.opt['path']['visualization'], dataset_name)
            os.makedirs(img_dirpath, exist_ok=True)

        logger = get_root_logger()
        exp_name = self.opt["name"]
        save_img_ratio = self.opt["val"].get("save_img_ratio")

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            visuals["result"] = _clamp(visuals["result"], clamp_opts)

            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                # gt_img = tensor2raw([visuals['gt']]) # replace for raw data.
                gt_img = _clamp(visuals["gt"], clamp_opts)
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            image_metric = {}
            
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    # replace for raw data.
                    # self.metric_results[name] += getattr(
                    #     metric_module, metric_type)(sr_img*255, gt_img*255, **opt_)
                    
                    val = getattr(
                        metric_module, metric_type)(sr_img, gt_img, **opt_)
                    self.metric_results[name] += val
                    image_metric[name] = val

            metric_message = f"{img_name}_{image_metric.get('psnr', 0)}_{image_metric.get('ssim',0)}"
            logger.info(metric_message)

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    if random() < save_img_ratio:
                        imgdir = osp.join(self.opt['path']['visualization'],
                                        img_name)
                        os.makedirs(imgdir, exist_ok=True)
                        save_img_path = osp.join(
                            imgdir, f'{img_name}_{current_iter}.png')
                        _save_image(sr_img, save_img_path,
                            max_pxl=max_pxl, dng_info=dng_info)
                else:
                    if self.opt['val'].get("suffix"):
                        save_img_path = osp.join(
                            img_dirpath, f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            img_dirpath, f'{img_name}_{exp_name}.png')
                # np.save(save_img_path.replace('.png', '.npy'), sr_img) # replace for raw data.
                    _save_image(sr_img, save_img_path,
                                max_pxl=max_pxl, dng_info=dng_info)
                # mmcv.imwrite(gt_img, save_img_path.replace('syn_val', 'gt'))

            save_npy = self.opt['val'].get('save_npy', None)
            if save_npy:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.npy')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.npy')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.npy')

                # saving as .npy format.
                np.save(save_img_path, tensor2npy([visuals['result']]))

            pbar.update(f'Test {img_name}')

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
