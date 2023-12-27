from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from skimage import io
from src.logger import FusionLogger, Tracker
from torch.utils.tensorboard import SummaryWriter
import shutil

epsilon = 1e-10


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
            # model_input_list, _, metrics_gt = self.before_inference_iter(data_per_batch)
            # self.inference_iter(iter_idx, model_input_list, metrics_gt, data_per_batch)
            self.inference_iter(iter_idx, data_per_batch)

    def before_inference_iter(self, data_per_batch):
        model_input_list = self.get_model_input(data_per_batch)
        loss_gt = self.get_loss_gt(data_per_batch)
        metrics_gt = self.get_metrics_gt(data_per_batch)
        return model_input_list, loss_gt, metrics_gt

    def get_model_input(self, data_per_batch):
        model_input_key_list = [
            'coarse_img',
        ]
        model_input_list = []
        for model_input_key in model_input_key_list:
            model_input_list.append(data_per_batch[model_input_key].to(self.device))
        return model_input_list

    def get_loss_gt(self, data_per_batch):
        loss_gt_key = 'fine_img'
        return data_per_batch[loss_gt_key].to(self.device)

    def get_metrics_gt(self, data_per_batch):
        metrics_gt_key = 'fine_img'
        return data_per_batch[metrics_gt_key].to(self.device)

    # def inference_iter(self, iter_idx, model_input_list, metrics_gt, data_per_batch):
    def inference_iter(self, iter_idx, data_per_batch):
        with torch.no_grad():
            fine_img_01_stage_1 = data_per_batch['fine_img_01_stage_1'].to(self.device)
            fine_img_02_stage_1 = data_per_batch['fine_img_02_stage_1'].to(self.device)
            fine_img_03_stage_1 = data_per_batch['fine_img_03_stage_1'].to(self.device)
            coarse_img_01_stage_1 = data_per_batch['coarse_img_01_stage_1'].to(
                self.device
            )
            coarse_img_02_stage_1 = data_per_batch['coarse_img_02_stage_1'].to(
                self.device
            )
            coarse_img_03_stage_1 = data_per_batch['coarse_img_03_stage_1'].to(
                self.device
            )
            fine_img_01_stage_2 = data_per_batch['fine_img_01_stage_2'].to(self.device)
            fine_img_02_stage_2 = data_per_batch['fine_img_02_stage_2'].to(self.device)
            fine_img_03_stage_2 = data_per_batch['fine_img_03_stage_2'].to(self.device)
            coarse_img_01_stage_2 = data_per_batch['coarse_img_01_stage_2'].to(
                self.device
            )
            coarse_img_02_stage_2 = data_per_batch['coarse_img_02_stage_2'].to(
                self.device
            )
            coarse_img_03_stage_2 = data_per_batch['coarse_img_03_stage_2'].to(
                self.device
            )

            transitional_img_01_stage_1 = self.model_stage_1(coarse_img_01_stage_1)
            transitional_img_02_stage_1 = self.model_stage_1(coarse_img_02_stage_1)
            transitional_img_03_stage_1 = self.model_stage_1(coarse_img_03_stage_1)

            fine_img_21_stage_1 = fine_img_01_stage_1 + (
                transitional_img_02_stage_1 - transitional_img_01_stage_1
            )
            fine_img_23_stage_1 = fine_img_03_stage_1 + (
                transitional_img_02_stage_1 - transitional_img_03_stage_1
            )
            w_stage_1 = (
                1
                / (
                    torch.abs(transitional_img_02_stage_1 - transitional_img_01_stage_1)
                    + epsilon
                )
            ) / (
                (
                    1
                    / (
                        torch.abs(
                            transitional_img_02_stage_1 - transitional_img_01_stage_1
                        )
                        + epsilon
                    )
                )
                + (
                    1
                    / (
                        torch.abs(
                            transitional_img_02_stage_1 - transitional_img_03_stage_1
                        )
                        + epsilon
                    )
                )
            )
            outputs_stage_1 = (
                w_stage_1 * fine_img_21_stage_1 + (1 - w_stage_1) * fine_img_23_stage_1
            )

            transitional_img_01_stage_2 = self.model_stage_2(coarse_img_01_stage_2)
            transitional_img_02_stage_2 = self.model_stage_2(outputs_stage_1)
            transitional_img_03_stage_2 = self.model_stage_2(coarse_img_03_stage_2)

            fine_img_21_stage_2 = fine_img_01_stage_2 + (
                transitional_img_02_stage_2 - transitional_img_01_stage_2
            )
            fine_img_23_stage_2 = fine_img_03_stage_2 + (
                transitional_img_02_stage_2 - transitional_img_03_stage_2
            )
            w_stage_2 = (
                1
                / (
                    torch.abs(transitional_img_02_stage_2 - transitional_img_01_stage_2)
                    + epsilon
                )
            ) / (
                (
                    1
                    / (
                        torch.abs(
                            transitional_img_02_stage_2 - transitional_img_01_stage_2
                        )
                        + epsilon
                    )
                )
                + (
                    1
                    / (
                        torch.abs(
                            transitional_img_02_stage_2 - transitional_img_03_stage_2
                        )
                        + epsilon
                    )
                )
            )
            outputs_stage_2 = (
                w_stage_2 * fine_img_21_stage_2 + (1 - w_stage_2) * fine_img_23_stage_2
            )

        msg = f'inference iter: {iter_idx}'
        msg += '\nstage_1'
        for metric in self.metric_list:
            metric_value = metric(
                (outputs_stage_1 + 1.0) / 2.0,
                (fine_img_02_stage_1 + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value:.4f}'

        msg += '\nstage_2'
        for metric in self.metric_list:
            metric_value = metric(
                (outputs_stage_2 + 1.0) / 2.0,
                (fine_img_02_stage_2 + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value:.4f}'

        self.txt_logger.info(msg)

        ## TODO
        save_dir_prefix = f'stage_1/save_img'
        save_img_name = data_per_batch['key'][0].split('-')[-1] + f'_{iter_idx}.tif'
        normalize_scale = data_per_batch['normalize_scale'][0].numpy()
        normalize_mode = data_per_batch['normalize_mode'][0].numpy()
        self.img_save(
            outputs_stage_1,
            save_dir_prefix,
            save_img_name,
            normalize_scale,
            normalize_mode,
        )

        save_dir_prefix = f'stage_2/save_img'
        save_img_name = data_per_batch['key'][0].split('-')[-1] + f'_{iter_idx}.tif'
        normalize_scale = data_per_batch['normalize_scale'][0].numpy()
        normalize_mode = data_per_batch['normalize_mode'][0].numpy()
        self.img_save(
            outputs_stage_2,
            save_dir_prefix,
            save_img_name,
            normalize_scale,
            normalize_mode,
        )

        # show_dir_prefix = f'stage_1/show_img'
        # show_img_name = data_per_batch['key'][0].split('-')[-1] + f'_{iter_idx}.png'
        # self.img_show(
        #     data_per_batch,
        #     outputs,
        #     show_dir_prefix,
        #     show_img_name,
        #     normalize_mode,
        # )

    def img_save(
        self,
        save_tensor,
        save_dir_prefix,
        save_name,
        normalize_scale,
        normalize_mode,
    ):
        save_img = save_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        _, _, c = save_img.shape

        if normalize_mode == 1:
            save_img = save_img * normalize_scale
        elif normalize_mode == 2:
            save_img = (save_img + 1.0) / 2.0 * normalize_scale
        save_img = np.clip(save_img, 0, normalize_scale)
        if normalize_scale == 255:
            save_img = save_img.astype(np.uint8)
        else:
            save_img = save_img.astype(np.uint16)
        save_img_path = self.inference_imgs_dir / save_dir_prefix / save_name
        save_img_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imsave(save_img_path, save_img)

    def img_show(
        self,
        data_per_batch,
        pred_tensor,
        show_dir_prefix,
        show_name,
        normalize_mode,
        img_interval=10,
    ):
        show_key_list = [
            ['coarse_img', 'fine_img'],
        ]
        _, c, h, w = pred_tensor.shape
        show_img = np.zeros(
            (
                (h + img_interval) * 1 + img_interval,
                (w + img_interval) * 3 + img_interval,
                3,
            ),
            dtype=np.uint8,
        )
        for row_index in range(1):
            for col_index in range(2):
                show_sub_img = (
                    data_per_batch[show_key_list[row_index][col_index]][0]
                    .detach()
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
                    img_interval * (row_index + 1)
                    + h * row_index : img_interval * (row_index + 1)
                    + h * (row_index + 1),
                    img_interval * (col_index + 1)
                    + w * col_index : img_interval * (col_index + 1)
                    + w * (col_index + 1),
                    :,
                ] = show_sub_img

        show_sub_img = pred_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        if normalize_mode == 1:
            show_sub_img = show_sub_img * 255.0
        elif normalize_mode == 2:
            show_sub_img = (show_sub_img + 1.0) / 2.0 * 255.0
        if c == 6:
            show_sub_img = show_sub_img[:, :, (3, 2, 1)]
        show_sub_img = np.clip(show_sub_img, 0, 255).astype(np.uint8)
        show_img[
            img_interval * 1 + h * 0 : img_interval * 1 + h * 1,
            img_interval * 3 + w * 2 : img_interval * 3 + w * 3,
            :,
        ] = show_sub_img

        show_img_path = self.inference_imgs_dir / show_dir_prefix / show_name
        show_img_path.parent.mkdir(parents=True, exist_ok=True)

        io.imsave(show_img_path, show_img)
