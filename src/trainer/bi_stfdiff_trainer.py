from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from skimage import io
from src.logger import FusionLogger, Tracker
from torch.utils.tensorboard import SummaryWriter
import shutil
import cv2
from ema_pytorch import EMA


class Trainer:
    def __init__(
        self,
        congfig_path,
        train_root_dir_path,
        train_dir_prefix_list=[
            'checkpoints',
            'txt_logs',
            'backend_logs',
            'imgs',
            'configs',
        ],
        #
        txt_logger_name='fusion',
        txt_logger_level='INFO',
        #
        train_dataloader=None,
        val_dataloader=None,
        #
        model=None,
        device_info=None,
        #
        optimizer=None,
        scheduler=None,
        #
        metric_list=None,
    ):
        # self.accelerator = Accelerator(split_batches=True, mixed_precision='no')

        self.init_train_params()
        self.init_train_dir(train_root_dir_path, train_dir_prefix_list, congfig_path)
        self.init_logger(txt_logger_name, txt_logger_level)
        self.init_dataloader(train_dataloader, val_dataloader)
        self.init_model(model)
        self.init_optimizer(optimizer)
        self.init_scheduler(scheduler)
        self.init_metric(metric_list)

        self.init_tracker()

    def init_train_params(self):
        self.current_epoch = 0
        self.max_epoch = 10000
        self.val_interal = 500
        self.current_val_step = 0
        self.current_train_step = 0
        self.save_interal = 50

    def init_train_dir(self, train_root_dir_path, train_dir_prefix_list, congfig_path):
        self.train_root_dir_path = Path(train_root_dir_path)
        self.train_dir_prefix_list = train_dir_prefix_list
        self.mkdir()
        shutil.copy(congfig_path, self.train_configs_dir)

    def mkdir(self):
        for train_dir_prefix in self.train_dir_prefix_list:
            train_dir = self.train_root_dir_path / train_dir_prefix
            train_dir.mkdir(parents=True, exist_ok=True)
            setattr(self, f'train_{train_dir_prefix}_dir', train_dir)

    def init_logger(self, txt_logger_name, txt_logger_level):
        self.txt_logger = FusionLogger(
            logger_name=txt_logger_name,
            log_file=self.train_txt_logs_dir / 'log.log',
            log_level=txt_logger_level,
        )
        self.backend_logger = SummaryWriter(self.train_backend_logs_dir)

    def init_dataloader(self, train_dataloader, val_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def init_model(self, model):
        self.device = 'cuda'
        self.model = model.to(self.device)
        self.ema = EMA(self.model, beta=0.995, update_every=1).to(self.device)

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer(params=self.model.parameters())

    def init_scheduler(self, scheduler):
        self.scheduler = scheduler(self.optimizer) if scheduler is not None else None

    def init_metric(self, metric_list):
        for metric in metric_list:
            metric.to(self.device)
        self.metric_list = metric_list

    def init_tracker(self):
        key_list = ['loss'] + [metric.__name__ for metric in self.metric_list]
        self.train_tracker = Tracker(*key_list)
        self.val_tracker = Tracker(*key_list)

    def train(self):
        while self.current_epoch < self.max_epoch:
            if self.scheduler is not None:
                self.scheduler.step()
            self.train_epoch()
            if self.current_epoch % self.save_interal == 0:
                self.save_checkpoint()
            if (self.current_epoch + 1) % self.val_interal == 0:
                self.val()
            self.current_epoch += 1

    def train_epoch(self):
        self.model.train()
        self.train_dataloader.sampler.set_epoch(self.current_epoch)
        self.train_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.train_dataloader):
            # if iter_idx > 3:
            #     break
            model_input_list = self.before_train_iter(data_per_batch)
            self.train_iter(iter_idx, model_input_list)
            self.current_train_step += 1
        self.ema.update()
        msg = f'epoch: {self.current_epoch}'
        for key, value in self.train_tracker.results.items():
            msg += f', {key}: {value:.4f}'
            self.backend_logger.add_scalar(f'train/{key}', value, self.current_epoch)
        self.txt_logger.info(msg)

    def before_train_iter(self, data_per_batch):
        model_input_list = self.get_model_input(data_per_batch)
        return model_input_list

    def get_model_input(self, data_per_batch):
        coarse_img_01 = data_per_batch['coarse_img_01'].to(self.device)
        coarse_img_02 = data_per_batch['coarse_img_02'].to(self.device)
        coarse_img_03 = data_per_batch['coarse_img_03'].to(self.device)
        fine_img_01 = data_per_batch['fine_img_01'].to(self.device)
        fine_img_02 = data_per_batch['fine_img_02'].to(self.device)
        fine_img_03 = data_per_batch['fine_img_03'].to(self.device)
        model_input_list = [
            coarse_img_01,
            coarse_img_02,
            coarse_img_03,
            fine_img_01,
            fine_img_02,
            fine_img_03,
        ]
        return model_input_list

    def get_loss_gt(self, data_per_batch):
        return data_per_batch['fine_img_02'].to(self.device)

    def get_metrics_gt(self, data_per_batch):
        return data_per_batch['fine_img_02'].to(self.device)

    def train_iter(self, iter_idx, model_input_list):
        self.optimizer.zero_grad()
        loss = self.model(*model_input_list)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        msg = f'epoch: {self.current_epoch}, iter: {iter_idx}, loss: {loss.item():.4e}'
        self.train_tracker.update('loss', loss.item())
        self.backend_logger.add_scalar(
            'train_running/loss', loss.item(), self.current_train_step
        )
        self.txt_logger.info(msg)

    def val(self):
        self.ema.ema_model.eval()
        self.val_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.val_dataloader):
            # if iter_idx % 20 != 0:
            #     continue
            (
                model_sampling_input_list,
                loss_gt,
                metrics_gt,
                show_img_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            ) = self.before_val_iter(data_per_batch)
            self.val_iter(
                iter_idx,
                model_sampling_input_list,
                loss_gt,
                metrics_gt,
                show_img_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            )
            self.current_val_step += 1
        msg = f'val epoch: {self.current_epoch}'
        for key, value in self.val_tracker.results.items():
            msg += f', {key}: {value:.4f}'
            self.backend_logger.add_scalar(f'val/{key}', value, self.current_epoch)
        self.txt_logger.info(msg)

    def before_val_iter(self, data_per_batch):
        model_sampling_input_list = self.get_model_sampling_input(data_per_batch)
        loss_gt = self.get_loss_gt(data_per_batch)
        metrics_gt = self.get_metrics_gt(data_per_batch)
        show_img_list = [
            data_per_batch['coarse_img_01'],
            data_per_batch['coarse_img_02'],
            data_per_batch['coarse_img_03'],
            data_per_batch['fine_img_01'],
            data_per_batch['fine_img_02'],
            data_per_batch['fine_img_03'],
        ]
        key = data_per_batch['key']
        dataset_name = data_per_batch['dataset_name'][0]
        normalize_scale = data_per_batch['normalize_scale'][0].numpy()
        normalize_mode = data_per_batch['normalize_mode'][0].numpy()

        return (
            model_sampling_input_list,
            loss_gt,
            metrics_gt,
            show_img_list,
            key,
            dataset_name,
            normalize_scale,
            normalize_mode,
        )

    def get_model_sampling_input(self, data_per_batch):
        coarse_img_01 = data_per_batch['coarse_img_01'].to(self.device)
        coarse_img_02 = data_per_batch['coarse_img_02'].to(self.device)
        coarse_img_03 = data_per_batch['coarse_img_03'].to(self.device)
        fine_img_01 = data_per_batch['fine_img_01'].to(self.device)
        fine_img_03 = data_per_batch['fine_img_03'].to(self.device)

        model_sampling_input_list = [
            coarse_img_01,
            coarse_img_02,
            coarse_img_03,
            fine_img_01,
            fine_img_03,
        ]
        return model_sampling_input_list

    def val_iter(
        self,
        iter_idx,
        model_sampling_input_list,
        loss_gt,
        metrics_gt,
        show_img_lists,
        keys,
        dataset_name,
        normalize_scale,
        normalize_mode,
    ):
        msg = f'val epoch: {self.current_epoch}, iter: {iter_idx}'
        with torch.no_grad():
            outputs = self.ema.ema_model.sample(*model_sampling_input_list)
            pixel_loss = F.mse_loss(outputs, loss_gt)
            self.val_tracker.update('loss', pixel_loss.item())
            self.backend_logger.add_scalar(
                'val_running/loss', pixel_loss.item(), iter_idx
            )
            msg += f', loss: {pixel_loss.item():.4e}'
        for metric in self.metric_list:
            metric_value = metric(
                (outputs + 1.0) / 2.0,
                (metrics_gt + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value.item():.4f}'
            self.backend_logger.add_scalar(
                f'val_runinng/{metric_name}', metric_value.item(), self.current_val_step
            )
            self.val_tracker.update(metric_name, metric_value.item())

        self.txt_logger.info(msg)

        ## TODO
        batch_num = outputs.shape[0]

        for batch_idx in range(batch_num):
            key = keys[batch_idx]
            show_img_list = [
                show_img_list[batch_idx : batch_idx + 1]
                for show_img_list in show_img_lists
            ]
            output = outputs[batch_idx : batch_idx + 1]
            save_dir_prefix = f'{dataset_name}/{self.current_epoch}/save_img'
            save_dir_path = self.train_imgs_dir / save_dir_prefix
            save_name = key.split('-')[-1]
            save_name = save_name[:8] + '_save_img_' + save_name[9:] + '.tif'
            if 'STIL' in save_dir_prefix:
                save_name = key + '.tif'
            self.img_save(
                output,
                save_dir_path,
                save_name,
                normalize_scale,
                normalize_mode,
            )

            show_dir_prefix = f'{dataset_name}/{self.current_epoch}/show_img/fine_img'
            show_dir_path = self.train_imgs_dir / show_dir_prefix
            show_name = key.split('-')[-1]
            show_name = show_name[:8] + '_show_fine_img_' + show_name[9:] + '.png'
            if 'STIL' in save_dir_prefix:
                show_name = key + '.png'
            self.img_show(
                show_img_list,
                output,
                show_dir_path,
                show_name,
                normalize_mode,
            )

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

    def save_checkpoint(self):
        data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema.state_dict(),
        }

        torch.save(
            data,
            self.train_checkpoints_dir / f'model_epoch_{self.current_epoch}.pth',
        )
