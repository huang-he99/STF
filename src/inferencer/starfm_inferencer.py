from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage import io

from src.logger.txt_logger import FusionLogger

import shutil
from torch.utils.tensorboard import SummaryWriter

def save_temp(img,key):
    from src.utils.img.process import truncated_linear_stretch
    img = img[0].cpu().numpy().transpose(1, 2, 0)*255
    img = img.astype(np.uint8)
    img = truncated_linear_stretch(img,2)
    io.imsave('{}_temp.png'.format(key), img[:,:,(3,2,1)])

class Inferencer:
    def __init__(
        self,
        congfig_path,
        inference_root_dir_path,
        inference_dir_prefix_list=[
            'txt_logs',
            'backend_logs',
            'imgs',
            'configs',
        ],
        #
        txt_logger_name='fusion',
        txt_logger_level='INFO',
        #
        test_dataloader=None,
        #
        model=None,
        #
        metric_list=None,
        #
        patch_info_dict=None,
    ):
        self.init_inference_dir(
            inference_root_dir_path, inference_dir_prefix_list, congfig_path
        )
        self.init_logger(txt_logger_name, txt_logger_level)
        self.init_dataloader(test_dataloader)
        self.init_model(model)
        self.init_metric(metric_list)
        self.get_patch_info(patch_info_dict)

    def init_inference_dir(
        self, inference_root_dir_path, inference_dir_prefix_list, congfig_path
    ):
        self.inference_root_dir_path = Path(inference_root_dir_path)
        self.inference_dir_prefix_list = inference_dir_prefix_list
        self.mkdir()
        shutil.copy(congfig_path, self.inference_configs_dir)

    def init_model(self, model):
        self.model = model.to('cuda')

    def mkdir(self):
        for inference_dir_prefix in self.inference_dir_prefix_list:
            inference_dir = self.inference_root_dir_path / inference_dir_prefix
            inference_dir.mkdir(parents=True, exist_ok=True)
            setattr(self, f'inference_{inference_dir_prefix}_dir', inference_dir)

    def init_logger(self, txt_logger_name, txt_logger_level):
        self.txt_logger = FusionLogger(
            logger_name=txt_logger_name,
            log_file=self.inference_txt_logs_dir / 'log.log',
            log_level=txt_logger_level,
        )
        self.backend_logger = SummaryWriter(self.inference_backend_logs_dir)

    def init_dataloader(self, test_dataloader):
        self.test_dataloader = test_dataloader

    def init_metric(self, metric_list):
        for metric in metric_list:
            metric.to('cuda')
        self.metric_list = metric_list

    def get_patch_info(self, patch_info_dict):
        self.patch_size = patch_info_dict['patch_size']
        self.patch_stride = patch_info_dict['patch_stride']
        self.window_size = patch_info_dict['window_size']
        self.virtual_patch_size = self.patch_size + self.window_size - 1

    def inference(self):
        self.model.eval()
        for iter_idx, data_per_batch in enumerate(self.test_dataloader):
            (
                prior_coarse_img_patches_series,
                prior_fine_img_patches_series,
                pred_coarse_img_patches,
                gt_fine_img_patches,
                img_padding_h,
                img_padding_w,
                img_padding_pixel_tuple,
                show_img_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            ) = self.before_inference_iter(data_per_batch)
            model_output_patches = self.inference_iter(
                iter_idx,
                prior_coarse_img_patches_series,
                prior_fine_img_patches_series,
                pred_coarse_img_patches,
                gt_fine_img_patches,
                dataset_name,
            )
            model_output, gt_fine_img = self.after_inference_iter(
                model_output_patches,
                gt_fine_img_patches,
                img_padding_h,
                img_padding_w,
                img_padding_pixel_tuple,
            )
            msg = f'dataset: {dataset_name}, inference iter idx: {iter_idx} '
            for metric in self.metric_list:
                metric_value = metric(model_output, gt_fine_img)
                metric_name = metric.__name__
                msg = msg + f', {metric_name}: {metric_value.item():.4f}'
                # self.val_tracker.update(metric_name, metric_value.item())
            self.txt_logger.info(msg)

            save_dir_prefix = f'{dataset_name}/save_img'
            save_dir_path = self.inference_imgs_dir / save_dir_prefix
            save_name = key.split('-')[-1]
            save_name = save_name[:8] + '_save_img_' + save_name[9:] + '.tif'
            if 'STIL' in save_dir_prefix:
                save_name = key + '.tif'
            self.img_save(
                model_output,
                save_dir_path,
                save_name,
                normalize_scale,
                normalize_mode,
            )

            show_dir_prefix = f'{dataset_name}/show_img'
            show_dir_path = self.inference_imgs_dir / show_dir_prefix
            show_name = key.split('-')[-1]
            show_name = show_name[:8] + '_show_img_' + show_name[9:] + '.png'
            if 'STIL' in save_dir_prefix:
                show_name = key + '.png'
            self.img_show(
                show_img_list,
                model_output,
                show_dir_path,
                show_name,
                normalize_mode,
            )

    def before_inference_iter(self, data_per_batch):
        (
            prior_coarse_img_patches_series,
            prior_fine_img_patches_series,
            pred_coarse_img_patches,
        ) = self.get_model_input(data_per_batch)
        gt_fine_img_patches = self.get_gt(data_per_batch)
        img_padding_h, img_padding_w, img_padding_pixel_tuple = self.get_padding_info(
            data_per_batch
        )
        show_img_list = self.get_show_img(data_per_batch)

        key = data_per_batch['key'][0]
        dataset_name = data_per_batch['dataset_name'][0]
        normalize_scale = data_per_batch['normalize_scale'][0].numpy()
        normalize_mode = data_per_batch['normalize_mode'][0].numpy()
        return (
            prior_coarse_img_patches_series,
            prior_fine_img_patches_series,
            pred_coarse_img_patches,
            gt_fine_img_patches,
            img_padding_h,
            img_padding_w,
            img_padding_pixel_tuple,
            show_img_list,
            key,
            dataset_name,
            normalize_scale,
            normalize_mode,
        )

    def get_model_input(self, data_per_batch):
        key_list = data_per_batch.keys()

        prior_coarse_img_key_list = [
            key
            for key in key_list
            if 'coarse_img' in key
            and len(key.split('_')) == 3
            and key != 'coarse_img_02'
        ]
        prior_fine_img_key_list = [
            key
            for key in key_list
            if 'fine_img' in key and len(key.split('_')) == 3 and key != 'fine_img_02'
        ]
        pred_coarse_img_key_list = ['coarse_img_02']

        prior_coarse_img_patches_series = self.process1(
            data_per_batch,
            prior_coarse_img_key_list,
        )
        prior_fine_img_patches_series = self.process1(
            data_per_batch, prior_fine_img_key_list
        )
        pred_coarse_img_patches_series = self.process1(
            data_per_batch, pred_coarse_img_key_list
        )
        pred_coarse_img_patches = pred_coarse_img_patches_series[:, :, 0]

        return (
            prior_coarse_img_patches_series,
            prior_fine_img_patches_series,
            pred_coarse_img_patches,
        )

    def process1(self, data_per_batch, img_key_list):
        img_patches_list = []
        for img_key in img_key_list:
            img = data_per_batch[img_key].to('cuda')
            img = self._pad(img)
            b, c, _, _ = img.shape
            img_patches = torch.nn.functional.unfold(
                img, self.virtual_patch_size, stride=self.patch_stride
            )
            img_patches = img_patches.view(
                b, c, self.virtual_patch_size, self.virtual_patch_size, -1
            )
            img_patches = img_patches.unsqueeze(2)
            img_patches_list.append(img_patches)
        img_patches_series = torch.cat(img_patches_list, dim=2)
        return img_patches_series

    def _pad(self, img):
        img_h, img_w = img.shape[-2:]
        img_padding_h, img_padding_w = self.cal_img_padding_hw(img_h, img_w)
        img_padding_pixel_tuple = self.cal_img_padding_pixel_num_hw(
            img_padding_h, img_padding_w, img_h, img_w
        )
        img = torch.nn.functional.pad(img, img_padding_pixel_tuple, mode='reflect')
        return img

    def cal_img_padding_hw(self, img_h, img_w):
        img_virtual_h = img_h + self.window_size - 1
        img_virtual_w = img_w + self.window_size - 1
        img_patch_num_h = self.cal_patch_num(img_virtual_h)
        img_patch_num_w = self.cal_patch_num(img_virtual_w)
        img_padding_h = self.cal_img_padding(img_patch_num_h)
        img_padding_w = self.cal_img_padding(img_patch_num_w)
        return (img_padding_h, img_padding_w)

    def cal_img_padding_pixel_num_hw(self, img_padding_h, img_padding_w, img_h, img_w):
        img_padding_pixel_top = (img_padding_h - img_h) // 2
        img_padding_pixel_bottom = img_padding_h - img_h - img_padding_pixel_top
        img_padding_pixel_left = (img_padding_w - img_w) // 2
        img_padding_pixel_right = img_padding_w - img_w - img_padding_pixel_left
        return (
            img_padding_pixel_left,
            img_padding_pixel_right,
            img_padding_pixel_top,
            img_padding_pixel_bottom,
        )

    def cal_patch_num(self, img_size):
        is_divide_exactly = (
            img_size - self.virtual_patch_size
        ) % self.patch_stride == 0
        if is_divide_exactly:
            patch_num = (img_size - self.virtual_patch_size) // self.patch_stride + 1
        else:
            patch_num = (img_size - self.virtual_patch_size) // self.patch_stride + 2
        return patch_num

    def cal_img_padding(self, img_patch_num):
        img_padding_pixel = (
            img_patch_num - 1
        ) * self.patch_stride + self.virtual_patch_size
        return img_padding_pixel

    def get_gt(self, data_per_batch):
        gt_fine_img_key_list = ['fine_img_02']
        gt_fine_img_padding_patches_series = self.process1(
            data_per_batch, gt_fine_img_key_list
        )
        half_window_size = self.window_size // 2
        gt_fine_img_patches = gt_fine_img_padding_patches_series[
            :,
            :,
            0,
            half_window_size:-half_window_size,
            half_window_size:-half_window_size,
        ]
        return gt_fine_img_patches

    def get_padding_info(self, data_per_batch):
        img = data_per_batch['fine_img_02']
        img_h, img_w = img.shape[-2:]
        img_padding_h, img_padding_w = self.cal_img_padding_hw(img_h, img_w)
        img_padding_pixel_tuple = self.cal_img_padding_pixel_num_hw(
            img_padding_h, img_padding_w, img_h, img_w
        )

        return img_padding_h, img_padding_w, img_padding_pixel_tuple

    def get_show_img(self, data_per_batch):
        key_list = data_per_batch.keys()
        coarse_img_key_list = [
            key for key in key_list if 'coarse_img' in key and len(key.split('_')) == 3
        ]
        fine_img_key_list = [
            key for key in key_list if 'fine_img' in key and len(key.split('_')) == 3
        ]
        show_img_key_list = [*coarse_img_key_list, *fine_img_key_list]
        show_img_list = [data_per_batch[key] for key in show_img_key_list]
        return show_img_list

    def inference_iter(
        self,
        iter_idx,
        prior_coarse_img_patches_series,
        prior_fine_img_patches_series,
        pred_coarse_img_patches,
        gt_fine_img_patches,
        dataset_name,
    ):
        b, _, _, _, _, patch_num = prior_coarse_img_patches_series.shape
        model_output_patch_list = []
        for patch_idx in range(patch_num):
            prior_coarse_img_patch_series = prior_coarse_img_patches_series[
                ..., patch_idx
            ]
            prior_fine_img_patch_series = prior_fine_img_patches_series[..., patch_idx]
            pred_coarse_img_patch = pred_coarse_img_patches[..., patch_idx]
            gt_fine_img_patch = gt_fine_img_patches[..., patch_idx]
            model_output_patch = self.inference_patch_based_iter(
                iter_idx,
                patch_idx,
                prior_coarse_img_patch_series,
                prior_fine_img_patch_series,
                pred_coarse_img_patch,
                gt_fine_img_patch,
                patch_num,
                dataset_name,
            )
            model_output_patch_list.append(model_output_patch.unsqueeze(-1))
        model_output_patches = torch.cat(model_output_patch_list, dim=-1)
        model_output_patches = model_output_patches.view(b, -1, patch_num)
        return model_output_patches

    def after_inference_iter(
        self,
        model_output_patches,
        gt_fine_img_patches,
        img_padding_h,
        img_padding_w,
        img_padding_pixel_tuple,
    ):
        img_h = img_padding_h - self.window_size + 1
        img_w = img_padding_w - self.window_size + 1
        b, l, n = model_output_patches.shape
        cnt = torch.ones_like(model_output_patches)
        cnt = torch.nn.functional.fold(
            cnt,
            (img_h, img_w),
            (self.patch_size, self.patch_size),
            stride=(self.patch_stride, self.patch_stride),
        )
        cnt[cnt == 0] = 1
        model_output = (
            torch.nn.functional.fold(
                model_output_patches,
                (img_h, img_w),
                (self.patch_size, self.patch_size),
                stride=(self.patch_stride, self.patch_stride),
            )
            / cnt
        )
        gt_fine_img_patches = gt_fine_img_patches.reshape(b, -1, n)
        gt_fine_img = (
            torch.nn.functional.fold(
                gt_fine_img_patches,
                (img_h, img_w),
                (self.patch_size, self.patch_size),
                stride=(self.patch_stride, self.patch_stride),
            )
            / cnt
        )

        # padding_left = img_padding_pixel_tuple[0] - self.window_size // 2
        # padding_right = img_padding_pixel_tuple[1] - self.window_size // 2
        # padding_top = img_padding_pixel_tuple[2] - self.window_size // 2
        # padding_bottom = img_padding_pixel_tuple[3] - self.window_size // 2

        # model_output = model_output[
        #     ..., padding_top:-padding_bottom, padding_left:-padding_right
        # ]
        # gt_fine_img = gt_fine_img[
        #     ..., padding_top:-padding_bottom, padding_left:-padding_right
        # ]
        return model_output, gt_fine_img

    def inference_patch_based_iter(
        self,
        iter_idx,
        patch_idx,
        prior_coarse_img_patch_series,
        prior_fine_img_patch_series,
        pred_coarse_img_patch,
        gt_fine_img_patch,
        patch_num,
        dataset_name,
    ):
        with torch.no_grad():
            model_input_list = [
                prior_coarse_img_patch_series,
                prior_fine_img_patch_series,
                pred_coarse_img_patch,
            ]
            model_output = self.model(*model_input_list)

            msg = f'dataset: {dataset_name}, inference iter: {iter_idx}, patch idx/patch_num: {patch_idx}/{patch_num} '

            for metric in self.metric_list:
                metric_value = metric(model_output, gt_fine_img_patch)
                metric_name = metric.__name__
                msg = msg + f', {metric_name}: {metric_value.item():.4f}'
                # self.val_tracker.update(metric_name, metric_value.item())

        self.txt_logger.info(msg)

        return model_output

    def img_save(
        self,
        save_tensor,
        save_dir_path,
        save_name,
        normalize_scale,
        normalize_mode,
    ):
        save_img = save_tensor[0].cpu().numpy().transpose(1, 2, 0)
        _, _, c = save_img.shape
        if normalize_mode == 1:
            save_img = save_img * normalize_scale
        elif normalize_mode == 2:
            save_img = (save_img + 1.0) / 2.0 * normalize_scale
        save_img = np.clip(save_img, 0, normalize_scale)
        if normalize_scale == 255:
            save_img = save_img.astype(np.uint8)
        elif normalize_scale == 1:
            save_img = save_img.astype(np.float32)
        elif normalize_scale == 10000:
            save_img = save_img.astype(np.uint16)
        save_img_path = save_dir_path / save_name
        save_dir_path.mkdir(parents=True, exist_ok=True)
        tifffile.imsave(save_img_path, save_img)

    def img_show(
        self,
        show_img_list,
        pred_tensor,
        show_dir_path,
        show_name,
        normalize_mode,
        img_interval=10,
    ):
        show_len = len(show_img_list)
        h_num, w_num = 3, show_len // 2
        _, c, h, w = pred_tensor.shape
        show_img = np.zeros(
            (
                (h + img_interval) * h_num + img_interval,
                (w + img_interval) * w_num + img_interval,
                3,
            ),
            dtype=np.uint8,
        )
        for h_index in range(h_num - 1):
            for w_index in range(w_num):
                show_sub_img = (
                    show_img_list[h_index * w_num + w_index][0]
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                )
                if normalize_mode == 1:
                    show_sub_img = show_sub_img * 255.0
                elif normalize_mode == 2:
                    show_sub_img = (show_sub_img + 1.0) / 2.0 * 255.0
                if c == 6:
                    show_sub_img = show_sub_img[:, :, (3, 2, 1)]
                show_sub_img = np.clip(show_sub_img, 0, 255).astype(np.uint8)
                show_img[
                    img_interval * (h_index + 1)
                    + h_index * h : img_interval * (h_index + 1)
                    + (h_index + 1) * h,
                    img_interval * (w_index + 1)
                    + w_index * w : img_interval * (w_index + 1)
                    + (w_index + 1) * w,
                    :,
                ] = show_sub_img

        show_sub_img = pred_tensor[0].cpu().numpy().transpose(1, 2, 0)
        if normalize_mode == 1:
            show_sub_img = show_sub_img * 255.0
        elif normalize_mode == 2:
            show_sub_img = (show_sub_img + 1.0) / 2.0 * 255.0
        if c == 6:
            show_sub_img = show_sub_img[:, :, (3, 2, 1)]
        show_sub_img = np.clip(show_sub_img, 0, 255).astype(np.uint8)
        show_img[
            img_interval * h_num + h * (h_num - 1) : img_interval * h_num + h * h_num,
            img_interval : img_interval + w,
            :,
        ] = show_sub_img

        show_img_path = show_dir_path / show_name
        show_dir_path.mkdir(parents=True, exist_ok=True)

        io.imsave(show_img_path, show_img)
