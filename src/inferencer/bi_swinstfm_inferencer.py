import shutil
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from skimage import io
from torch.utils.tensorboard import SummaryWriter

from src.logger import FusionLogger, Tracker
from src.utils import EPSILON
import cv2


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
        txt_logger_name='inference',
        txt_logger_level='INFO',
        #
        test_dataloader=None,
        #
        model=None,
        checkpoint_path=None,
        #
        metric_list=None,
    ):
        self.init_inference_params()
        self.init_inference_dir(
            inference_root_dir_path, inference_dir_prefix_list, congfig_path
        )
        self.init_logger(txt_logger_name, txt_logger_level)
        self.init_dataloader(test_dataloader)
        self.init_model(model, checkpoint_path)
        self.init_metric(metric_list)

        self.init_tracker()

    def init_inference_params(self):
        self.current_epoch = 0
        self.max_epoch = 1000
        self.val_interal = 10
        self.current_inference_step = 0
        self.patch_size = 256
        self.stride_size = 256

    def init_inference_dir(
        self, inference_root_dir_path, inference_dir_prefix_list, congfig_path
    ):
        self.inference_root_dir_path = Path(inference_root_dir_path)
        self.inference_dir_prefix_list = inference_dir_prefix_list
        self.mkdir()
        shutil.copy(congfig_path, self.inference_configs_dir)

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

    def init_model(self, model, checkpoint_path):
        self.device = 'cuda'
        self.model = model.to(self.device)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)

    def init_metric(self, metric_list):
        for metric in metric_list:
            metric.to(self.device)
        self.metric_list = metric_list

    def init_tracker(self):
        key_list = [
            'd_loss',
            'g_loss',
        ] + [metric.__name__ for metric in self.metric_list]
        self.inference_tracker = Tracker(*key_list)

    def inference(self):
        self.model.eval()
        self.inference_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.test_dataloader):
            (
                model_input_list_1,
                model_input_list_2,
                gt,
                show_img_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            ) = self.before_inference_iter(data_per_batch)
            self.inference_iter(
                iter_idx,
                model_input_list_1,
                model_input_list_2,
                gt,
                show_img_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            )
            self.current_inference_step += 1
        msg = f'inference epoch: {self.current_epoch}'
        for key, value in self.inference_tracker.results.items():
            msg = msg + f', {key}: {value:.4f}'
            self.backend_logger.add_scalar(f'val/{key}', value, self.current_epoch)
        self.txt_logger.info(msg)

    def before_inference_iter(self, data_per_batch):
        model_input_list_1, model_input_list_2 = self.get_model_input(data_per_batch)
        gt = data_per_batch['fine_img_02'].to('cuda')

        show_img_list = self.get_img_show_list(data_per_batch)

        key = data_per_batch['key'][0]
        dataset_name = data_per_batch['dataset_name'][0]
        normalize_scale = data_per_batch['normalize_scale'][0].numpy()
        normalize_mode = data_per_batch['normalize_mode'][0].numpy()
        return (
            model_input_list_1,
            model_input_list_2,
            gt,
            show_img_list,
            key,
            dataset_name,
            normalize_scale,
            normalize_mode,
        )

    def get_model_input(self, data_per_batch):
        coarse_img_01 = data_per_batch['coarse_img_01'].to('cuda')
        coarse_img_02 = data_per_batch['coarse_img_02'].to('cuda')
        fine_img_01 = data_per_batch['fine_img_01'].to('cuda')

        coarse_img_01_patches = F.unfold(
            coarse_img_01, self.patch_size, stride=self.stride_size
        ).view(1, 6, 256, 256, -1)
        coarse_img_02_patches = F.unfold(
            coarse_img_02, self.patch_size, stride=self.stride_size
        ).view(1, 6, 256, 256, -1)
        fine_img_01_patches = F.unfold(
            fine_img_01, self.patch_size, stride=self.stride_size
        ).view(1, 6, 256, 256, -1)

        # coarse_img_01_patches = F.unfold(
        #     coarse_img_01, self.patch_size, stride=self.stride_size
        # ).view(-1, 6, 256, 256)
        # coarse_img_02_patches = F.unfold(
        #     coarse_img_02, self.patch_size, stride=self.stride_size
        # ).view(-1, 6, 256, 256)
        # fine_img_01_patches = F.unfold(
        #     fine_img_01, self.patch_size, stride=self.stride_size
        # ).view(-1, 6, 256, 2561)

        model_input_list_1 = [
            coarse_img_01_patches,
            coarse_img_02_patches,
            fine_img_01_patches,
        ]

        coarse_img_03 = data_per_batch['coarse_img_03'].to('cuda')
        coarse_img_02 = data_per_batch['coarse_img_02'].to('cuda')
        fine_img_03 = data_per_batch['fine_img_03'].to('cuda')

        coarse_img_03_patches = F.unfold(coarse_img_03, 256, stride=256).view(
            1, 6, 256, 256, -1
        )
        coarse_img_02_patches = F.unfold(coarse_img_02, 256, stride=256).view(
            1, 6, 256, 256, -1
        )
        fine_img_03_patches = F.unfold(fine_img_03, 256, stride=256).view(
            1, 6, 256, 256, -1
        )

        model_input_list_2 = [
            coarse_img_03_patches,
            coarse_img_02_patches,
            fine_img_03_patches,
        ]

        return model_input_list_1, model_input_list_2

    def get_img_show_list(self, data_per_batch):
        coarse_img_01 = data_per_batch['coarse_img_01']
        coarse_img_02 = data_per_batch['coarse_img_02']
        coarse_img_03 = data_per_batch['coarse_img_03']
        fine_img_01 = data_per_batch['fine_img_01']
        fine_img_02 = data_per_batch['fine_img_02']
        fine_img_03 = data_per_batch['fine_img_03']
        img_show_list = [
            coarse_img_01,
            coarse_img_02,
            coarse_img_03,
            fine_img_01,
            fine_img_02,
            fine_img_03,
        ]

        return img_show_list

    def inference_iter(
        self,
        iter_idx,
        model_input_patches_list_1,
        model_input_patches_list_3,
        gt,
        show_img_list,
        key,
        dataset_name,
        normalize_scale,
        normalize_mode,
    ):
        patch_num = model_input_patches_list_1[0].shape[-1]
        model_output_patches_1 = torch.zeros_like(model_input_patches_list_1[0])
        model_output_patches_3 = torch.zeros_like(model_input_patches_list_1[0])
        cnt = torch.ones_like(model_input_patches_list_1[0])
        msg = (
            f'val epoch: {self.current_epoch}, iter: {iter_idx}, patch_num: {patch_num}'
        )
        for patch_idx in range(patch_num):
            corase_img_01_patch = model_input_patches_list_1[0][..., patch_idx]
            corase_img_02_patch = model_input_patches_list_1[1][..., patch_idx]
            fine_img_01_patch = model_input_patches_list_1[2][..., patch_idx]
            model_input_list_1 = [
                corase_img_01_patch,
                corase_img_02_patch,
                fine_img_01_patch,
            ]
            model_output_patch_1 = self.inference_patch(
                iter_idx,
                model_input_list_1,
            )
            model_output_patches_1[..., patch_idx] = model_output_patch_1

            corase_img_03_patch = model_input_patches_list_3[0][..., patch_idx]
            corase_img_02_patch = model_input_patches_list_3[1][..., patch_idx]
            fine_img_03_patch = model_input_patches_list_3[2][..., patch_idx]
            model_input_list_3 = [
                corase_img_03_patch,
                corase_img_02_patch,
                fine_img_03_patch,
            ]
            model_output_patch_3 = self.inference_patch(
                iter_idx,
                model_input_list_3,
            )
            model_output_patches_3[..., patch_idx] = model_output_patch_3

        model_output_patches_1 = model_output_patches_1.view(1, -1, patch_num)
        model_output_patches_3 = model_output_patches_3.view(1, -1, patch_num)
        cnt = cnt.view(1, -1, patch_num)

        model_output_1 = F.fold(
            model_output_patches_1,
            gt.shape[-2:],
            self.patch_size,
            stride=self.stride_size,
        )
        model_output_3 = F.fold(
            model_output_patches_3,
            gt.shape[-2:],
            self.patch_size,
            stride=self.stride_size,
        )
        cnt = F.fold(cnt, gt.shape[-2:], self.patch_size, stride=self.stride_size)

        model_output_1 = model_output_1 / cnt
        model_output_3 = model_output_3 / cnt
        model_output = (model_output_1 + model_output_3) / 2.0

        for metric in self.metric_list:
            metric_value = metric(
                (model_output + 1.0) / 2.0,
                (gt + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value.item():.4f}'
            self.inference_tracker.update(metric_name, metric_value.item())
            self.backend_logger.add_scalar(
                f'inference_running/{metric_name}',
                metric_value.item(),
                self.current_inference_step,
            )
        self.txt_logger.info(msg)

        ## TODO

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

        show_dir_prefix = f'{dataset_name}/show_img/fine_img'
        show_dir_path = self.inference_imgs_dir / show_dir_prefix
        show_name = key.split('-')[-1]
        show_name = show_name[:8] + '_show_fine_img_' + show_name[9:] + '.png'
        if 'STIL' in save_dir_prefix:
            show_name = key + '.png'
        self.img_show(
            show_img_list,
            model_output,
            show_dir_path,
            show_name,
            normalize_mode,
        )

    def inference_patch(self, iter_idx, model_input_list):
        with torch.no_grad():
            model_output = self.model(*model_input_list)
        return model_output

        # TODO

    def img_save(
        self,
        save_tensor,
        save_dir_path,
        save_name,
        normalize_scale,
        normalize_mode,
    ):
        save_img = save_tensor[0].cpu().numpy().transpose(1, 2, 0)
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
                show_sub_img = cv2.resize(
                    show_sub_img, (w, h), interpolation=cv2.INTER_NEAREST
                )
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
        show_sub_img = cv2.resize(show_sub_img, (w, h), interpolation=cv2.INTER_NEAREST)
        show_img[
            img_interval * h_num + h * (h_num - 1) : img_interval * h_num + h * h_num,
            img_interval : img_interval + w,
            :,
        ] = show_sub_img

        show_img_path = show_dir_path / show_name
        show_dir_path.mkdir(parents=True, exist_ok=True)

        io.imsave(show_img_path, show_img)
