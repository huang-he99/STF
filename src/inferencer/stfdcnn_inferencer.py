from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from skimage import io
from src.logger import FusionLogger, Tracker
from torch.utils.tensorboard import SummaryWriter
import shutil
from src.utils.constant import EPSILON
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
        model_stage_1=None,
        model_stage_2=None,
        checkpoint_stage_1_path=None,
        checkpoint_stage_2_path=None,
        device_info=None,
        #
        metric_list=None,
    ):
        self.init_inference_params()
        self.init_inference_dir(
            inference_root_dir_path, inference_dir_prefix_list, congfig_path
        )
        self.init_logger(txt_logger_name, txt_logger_level)
        self.init_dataloader(test_dataloader)
        self.init_model(
            model_stage_1,
            model_stage_2,
            checkpoint_stage_1_path,
            checkpoint_stage_2_path,
        )
        self.init_metric(metric_list)

        self.init_tracker()

    def init_inference_params(self):
        self.current_epoch = 0
        self.max_epoch = 1000
        self.val_interal = 10

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

    def init_model(
        self,
        model_stage_1,
        model_stage_2,
        checkpoint_stage_1_path,
        checkpoint_stage_2_path,
    ):
        self.device = 'cuda'

        self.model_stage_1 = model_stage_1.to(self.device)
        checkpoint_stage_1 = torch.load(checkpoint_stage_1_path)
        self.model_stage_1.load_state_dict(checkpoint_stage_1)

        self.model_stage_2 = model_stage_2.to(self.device)
        checkpoint_stage_2 = torch.load(checkpoint_stage_2_path)
        self.model_stage_2.load_state_dict(checkpoint_stage_2)

    def init_metric(self, metric_list):
        for metric in metric_list:
            metric.to(self.device)
        self.metric_list = metric_list

    def init_tracker(self):
        key_list = ['mse_loss'] + [metric.__name__ for metric in self.metric_list]
        self.train_tracker = Tracker(*key_list)
        self.val_tracker = Tracker(*key_list)

    def inference(self):
        self.model_stage_1.eval()
        self.model_stage_2.eval()
        for iter_idx, data_per_batch in enumerate(self.test_dataloader):
            (
                cnn_stage_1_input_list,
                fusion_stage_1_input_list,
                cnn_stage_2_input_list,
                fusion_stage_2_input_list,
                loss_gt,
                metrics_gt_stage_1,
                metrics_gt_stage_2,
                show_img_stage_1_list,
                show_img_stage_2_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            ) = self.before_inference_iter(data_per_batch)
            self.inference_iter(
                iter_idx,
                cnn_stage_1_input_list,
                fusion_stage_1_input_list,
                cnn_stage_2_input_list,
                fusion_stage_2_input_list,
                metrics_gt_stage_1,
                metrics_gt_stage_2,
                show_img_stage_1_list,
                show_img_stage_2_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            )

    def before_inference_iter(self, data_per_batch):
        (
            cnn_stage_1_input_list,
            fusion_stage_1_input_list,
            cnn_stage_2_input_list,
            fusion_stage_2_input_list,
        ) = self.get_model_input(data_per_batch)
        loss_gt = self.get_loss_gt(data_per_batch)
        metrics_gt_stage_1, metrics_gt_stage_2 = self.get_metrics_gt(data_per_batch)
        show_img_stage_1_list, show_img_stage_2_list = self.get_show_img(data_per_batch)
        key = data_per_batch['key'][0]
        dataset_name = data_per_batch['dataset_name'][0]
        normalize_scale = data_per_batch['normalize_scale'][0].numpy()
        normalize_mode = data_per_batch['normalize_mode'][0].numpy()
        return (
            cnn_stage_1_input_list,
            fusion_stage_1_input_list,
            cnn_stage_2_input_list,
            fusion_stage_2_input_list,
            loss_gt,
            metrics_gt_stage_1,
            metrics_gt_stage_2,
            show_img_stage_1_list,
            show_img_stage_2_list,
            key,
            dataset_name,
            normalize_scale,
            normalize_mode,
        )

    def get_model_input(self, data_per_batch):
        coarse_img_01_stage_1 = data_per_batch['coarse_img_01_stage_1'].to(self.device)
        coarse_img_02_stage_1 = data_per_batch['coarse_img_02_stage_1'].to(self.device)
        coarse_img_03_stage_1 = data_per_batch['coarse_img_03_stage_1'].to(self.device)

        cnn_stage_1_input_list = [
            coarse_img_01_stage_1,
            coarse_img_02_stage_1,
            coarse_img_03_stage_1,
        ]

        fine_img_01_stage_1 = data_per_batch['fine_img_01_stage_1'].to(self.device)
        fine_img_03_stage_1 = data_per_batch['fine_img_03_stage_1'].to(self.device)

        fusion_stage_1_input_list = [fine_img_01_stage_1, fine_img_03_stage_1]

        coarse_img_01_stage_2 = data_per_batch['coarse_img_01_stage_2'].to(self.device)
        coarse_img_03_stage_2 = data_per_batch['coarse_img_03_stage_2'].to(self.device)

        cnn_stage_2_input_list = [coarse_img_01_stage_2, None, coarse_img_03_stage_2]

        fine_img_01_stage_2 = data_per_batch['fine_img_01_stage_2'].to(self.device)
        fine_img_03_stage_2 = data_per_batch['fine_img_03_stage_2'].to(self.device)

        fusion_stage_2_input_list = [fine_img_01_stage_2, fine_img_03_stage_2]

        return (
            cnn_stage_1_input_list,
            fusion_stage_1_input_list,
            cnn_stage_2_input_list,
            fusion_stage_2_input_list,
        )

    def get_loss_gt(self, data_per_batch):
        loss_gt_key = 'fine_img'
        return data_per_batch['fine_img_02_stage_1'].to(self.device)

    def get_metrics_gt(self, data_per_batch):
        metrics_gt_stage_1 = data_per_batch['fine_img_02_stage_1'].to(self.device)
        metrics_gt_stage_2 = data_per_batch['fine_img_02_stage_2'].to(self.device)
        return metrics_gt_stage_1, metrics_gt_stage_2

    def get_show_img(self, data_per_batch):
        coarse_img_01_stage_1 = data_per_batch['coarse_img_01_stage_1']
        coarse_img_02_stage_1 = data_per_batch['coarse_img_02_stage_1']
        coarse_img_03_stage_1 = data_per_batch['coarse_img_03_stage_1']
        fine_img_01_stage_1 = data_per_batch['fine_img_01_stage_1']
        fine_img_02_stage_1 = data_per_batch['fine_img_02_stage_1']
        fine_img_03_stage_1 = data_per_batch['fine_img_03_stage_1']

        show_img_stage_1_list = [
            coarse_img_01_stage_1,
            coarse_img_02_stage_1,
            coarse_img_03_stage_1,
            fine_img_01_stage_1,
            fine_img_02_stage_1,
            fine_img_03_stage_1,
        ]

        coarse_img_01_stage_2 = data_per_batch['coarse_img_01_stage_2']
        coarse_img_02_stage_2 = data_per_batch['coarse_img_02_stage_2']
        coarse_img_03_stage_2 = data_per_batch['coarse_img_03_stage_2']
        fine_img_01_stage_2 = data_per_batch['fine_img_01_stage_2']
        fine_img_02_stage_2 = data_per_batch['fine_img_02_stage_2']
        fine_img_03_stage_2 = data_per_batch['fine_img_03_stage_2']

        show_img_stage_2_list = [
            coarse_img_01_stage_2,
            None,
            coarse_img_03_stage_2,
            fine_img_01_stage_2,
            fine_img_02_stage_2,
            fine_img_03_stage_2,
        ]
        return show_img_stage_1_list, show_img_stage_2_list

    def inference_iter(
        self,
        iter_idx,
        cnn_stage_1_input_list,
        fusion_stage_1_input_list,
        cnn_stage_2_input_list,
        fusion_stage_2_input_list,
        metrics_gt_stage_1,
        metrics_gt_stage_2,
        show_img_stage_1_list,
        show_img_stage_2_list,
        key,
        dataset_name,
        normalize_scale,
        normalize_mode,
    ):
        with torch.no_grad():
            transitional_img_stage_1_list = self.cnn_stage_1(*cnn_stage_1_input_list)
            fusion_stage_1_input_list = (
                fusion_stage_1_input_list + transitional_img_stage_1_list
            )
            outputs_stage_1 = self.fusion(*fusion_stage_1_input_list)

            cnn_stage_2_input_list[1] = outputs_stage_1
            transitional_img_stage_2_list = self.cnn_stage_2(*cnn_stage_2_input_list)
            fusion_stage_2_input_list = (
                fusion_stage_2_input_list + transitional_img_stage_2_list
            )
            outputs_stage_2 = self.fusion(*fusion_stage_2_input_list)

        msg = f'inference iter: {iter_idx}, dataset: {dataset_name}'
        msg += '\nstage_1'
        for metric in self.metric_list:
            metric_value = metric(
                (outputs_stage_1 + 1.0) / 2.0,
                (metrics_gt_stage_1 + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value:.4f}'

        msg += '\nstage_2'
        for metric in self.metric_list:
            metric_value = metric(
                (outputs_stage_2 + 1.0) / 2.0,
                (metrics_gt_stage_2 + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value:.4f}'

        self.txt_logger.info(msg)

        ## TODO
        save_dir_prefix = f'stage_1/{dataset_name}/save_img'
        save_dir_path = self.inference_imgs_dir / save_dir_prefix
        save_name = key.split('-')[-1]
        save_name = save_name[:8] + '_save_img_stage_1_' + save_name[9:] + '.tif'
        if 'STIL' in save_dir_prefix:
            save_name = key + '_save_img_stage_1.tif'
        self.img_save(
            outputs_stage_1,
            save_dir_path,
            save_name,
            normalize_scale,
            normalize_mode,
        )

        save_dir_prefix = f'stage_2/{dataset_name}/save_img'
        save_dir_path = self.inference_imgs_dir / save_dir_prefix
        save_name = key.split('-')[-1]
        save_name = save_name[:8] + '_save_img_stage_2_' + save_name[9:] + '.tif'
        if 'STIL' in save_dir_prefix:
            save_name = key + '_save_img_stage_2.tif'
        self.img_save(
            outputs_stage_2,
            save_dir_path,
            save_name,
            normalize_scale,
            normalize_mode,
        )

        show_dir_prefix = f'stage_1/{dataset_name}/show_img'
        show_dir_path = self.inference_imgs_dir / show_dir_prefix
        show_name = key.split('-')[-1]
        show_name = show_name[:8] + '_show_img_stage_1_' + show_name[9:] + '.png'
        if 'STIL' in save_dir_prefix:
            show_name = key + '_show_img_stage_1.png'
        self.img_show(
            show_img_stage_1_list,
            outputs_stage_1,
            show_dir_path,
            show_name,
            normalize_mode,
        )

        show_dir_prefix = f'stage_2/{dataset_name}/show_img'
        show_dir_path = self.inference_imgs_dir / show_dir_prefix
        show_name = key.split('-')[-1]
        show_name = show_name[:8] + '_show_img_stage_2_' + show_name[9:] + '.png'
        show_img_stage_2_list[1] = outputs_stage_1
        if 'STIL' in save_dir_prefix:
            show_name = key + '_show_img_stage_2.png'
        self.img_show(
            show_img_stage_2_list,
            outputs_stage_2,
            show_dir_path,
            show_name,
            normalize_mode,
        )

    def cnn_stage_1(
        self, coarse_img_01_stage_1, coarse_img_02_stage_1, coarse_img_03_stage_1
    ):
        transitional_img_01_stage_1 = self.model_stage_1(coarse_img_01_stage_1)
        transitional_img_02_stage_1 = self.model_stage_1(coarse_img_02_stage_1)
        transitional_img_03_stage_1 = self.model_stage_1(coarse_img_03_stage_1)
        return [
            transitional_img_01_stage_1,
            transitional_img_02_stage_1,
            transitional_img_03_stage_1,
        ]

    def fusion(
        self,
        fine_img_01,
        fine_img_03,
        transitional_img_01,
        transitional_img_02,
        transitional_img_03,
    ):
        fine_img_21 = fine_img_01 + (transitional_img_02 - transitional_img_01)
        fine_img_23 = fine_img_03 + (transitional_img_02 - transitional_img_03)
        w = (1 / (torch.abs(transitional_img_02 - transitional_img_01) + EPSILON)) / (
            (1 / (torch.abs(transitional_img_02 - transitional_img_01) + EPSILON))
            + (1 / (torch.abs(transitional_img_02 - transitional_img_03) + EPSILON))
        )
        outputs = w * fine_img_21 + (1 - w) * fine_img_23
        return outputs

    def cnn_stage_2(
        self, coarse_img_01_stage_2, coarse_img_02_stage_2, coarse_img_03_stage_2
    ):
        transitional_img_01_stage_2 = self.model_stage_2(coarse_img_01_stage_2)
        transitional_img_02_stage_2 = self.model_stage_2(coarse_img_02_stage_2)
        transitional_img_03_stage_2 = self.model_stage_2(coarse_img_03_stage_2)
        return [
            transitional_img_01_stage_2,
            transitional_img_02_stage_2,
            transitional_img_03_stage_2,
        ]

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

    def img_diff_show(
        self,
        show_img_diff,
        pred_tensor,
        show_dir_path,
        show_name,
        img_interval=10,
    ):
        h_num, w_num = 1, 2
        _, c, h, w = pred_tensor.shape
        show_img = np.zeros(
            (
                (h + img_interval) * h_num + img_interval,
                (w + img_interval) * w_num + img_interval,
                3,
            ),
            dtype=np.uint8,
        )
        show_sub_img = show_img_diff[0].cpu().numpy().transpose(1, 2, 0)
        show_sub_img = (
            (show_sub_img - np.min(show_sub_img))
            / (np.max(show_sub_img) - np.min(show_sub_img))
            * 255.0
        )
        show_sub_img = show_sub_img.astype(np.uint8)
        if c == 6:
            show_sub_img = show_sub_img[:, :, (3, 2, 1)]
        show_sub_img = np.clip(show_sub_img, 0, 255).astype(np.uint8)
        show_sub_img = cv2.resize(show_sub_img, (w, h), interpolation=cv2.INTER_NEAREST)
        show_img[
            img_interval : img_interval + h,
            img_interval : img_interval + w,
            :,
        ] = show_sub_img

        show_sub_img = pred_tensor[0].cpu().numpy().transpose(1, 2, 0)
        show_sub_img = (
            (show_sub_img - np.min(show_sub_img))
            / (np.max(show_sub_img) - np.min(show_sub_img))
            * 255.0
        )
        show_sub_img = show_sub_img.astype(np.uint8)
        if c == 6:
            show_sub_img = show_sub_img[:, :, (3, 2, 1)]
        show_sub_img = np.clip(show_sub_img, 0, 255).astype(np.uint8)
        show_sub_img = cv2.resize(show_sub_img, (w, h), interpolation=cv2.INTER_NEAREST)
        show_img[
            img_interval : img_interval + h,
            img_interval * 2 + w : img_interval * 2 + w * 2,
            :,
        ] = show_sub_img

        show_img_path = show_dir_path / show_name
        show_dir_path.mkdir(parents=True, exist_ok=True)

        io.imsave(show_img_path, show_img)
