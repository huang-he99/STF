from cmath import pi
from pathlib import Path
from src.logger.txt_logger import FusionLogger
import torch
import torch.nn.functional as F
import tifffile
import numpy as np
from skimage import io
from src.logger import FusionLogger, Tracker
from torch.utils.tensorboard import SummaryWriter
import shutil


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
        txt_logger_name='fusion',
        txt_logger_level='INFO',
        train_dataloader=None,
        val_dataloader=None,
        model_generator=None,
        model_discriminator=None,
        optimizer_generator=None,
        optimizer_discriminator=None,
        scheduler_generator=None,
        scheduler_discriminator=None,
        metric_list=None,
    ):
        self.init_train_params()
        self.init_train_dir(train_root_dir_path, train_dir_prefix_list, congfig_path)
        self.init_logger(txt_logger_name, txt_logger_level)
        self.init_dataloader(train_dataloader, val_dataloader)
        self.init_model(model_generator, model_discriminator)
        self.init_optimizer(optimizer_generator, optimizer_discriminator)
        self.init_scheduler(scheduler_generator, scheduler_discriminator)
        self.init_metric(metric_list)
        self.init_tracker()

    def init_train_params(self):
        self.current_val_step = 0
        self.current_train_step = 0
        self.current_epoch = 0
        self.max_epoch = 1000
        self.max_warm_up_epoch = 2
        self.val_interal = 50

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

    def init_model(self, model_generator, model_dicriminator):
        self.model_generator = model_generator.to('cuda')
        self.model_dicriminator = model_dicriminator.to('cuda')

    def init_optimizer(self, optimizer_generator, optimizer_dicriminator):
        self.optimizer_generator = optimizer_generator(
            params=self.model_generator.parameters()
        )
        self.optimizer_dicriminator = optimizer_dicriminator(
            params=self.model_dicriminator.parameters()
        )

    def init_scheduler(self, scheduler_generator, scheduler_discriminator):
        self.scheduler_generator = scheduler_generator
        self.scheduler_discriminator = scheduler_discriminator

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

    def init_metric(self, metric_list):
        for metric in metric_list:
            metric.to('cuda')
        self.metric_list = metric_list

    def init_tracker(self):
        key_list = [
            'phase_d_fake_loss',
            'phase_d_real_loss',
            'phase_d_gan_loss',
            'phase_g_fake_loss',
            'pixel_loss',
            'phase_g_loss',
        ] + [metric.__name__ for metric in self.metric_list]
        self.train_tracker = Tracker(*key_list)
        self.val_tracker = Tracker(*key_list)

    def train(self):
        while self.current_epoch < self.max_warm_up_epoch:
            self.train_warm_up_epoch()
            self.val_epoch()
            self.current_epoch += 1
            self.save_checkpoint()
        while self.current_epoch < self.max_epoch:
            self.train_epoch()
            if (
                self.current_epoch + 1
            ) % self.val_interal == 0 or self.current_epoch == 0:
                self.val_epoch()
                self.save_checkpoint()
            self.current_epoch += 1

    def train_warm_up_epoch(self):
        self.model_generator.train()
        self.train_dataloader.sampler.set_epoch(self.current_epoch)
        for iter_idx, data_per_batch in enumerate(self.train_dataloader):
            self.train_warm_up_iter(iter_idx, data_per_batch)
            self.current_train_step += 1
            # break
        msg = f'warm up epoch: {self.current_epoch}'
        for key, value in self.train_tracker.results.items():
            msg += f', {key}: {value:.4f}'
            self.backend_logger.add_scalar(f'train/{key}', value, self.current_epoch)
        self.txt_logger.info(msg)

    def train_warm_up_iter(self, iter_idx, data_per_batch):
        model_generator_inputs = [
            data_per_batch['coarse_img_01'].to('cuda'),
            data_per_batch['coarse_img_02'].to('cuda'),
            data_per_batch['coarse_img_03'].to('cuda'),
            data_per_batch['fine_img_01'].to('cuda'),
            data_per_batch['fine_img_03'].to('cuda'),
        ]
        self.optimizer_generator.zero_grad()
        phase_g_outputs = self.model_generator(*model_generator_inputs)
        pixel_loss = F.mse_loss(
            phase_g_outputs, data_per_batch['fine_img_02'].to('cuda')
        )
        pixel_loss.backward()
        self.optimizer_generator.step()
        msg = f'warm up epoch: {self.current_epoch}, iter: {iter_idx}, pixel_loss: {pixel_loss.item():.4f}'

        self.train_tracker.update('pixel_loss', pixel_loss.item())

        for metric in self.metric_list:
            metric_value = metric(
                phase_g_outputs, data_per_batch['fine_img_02'].to('cuda')
            )
            metric_name = metric.__name__
            msg += f', {metric_name}: {metric_value.item():.4f}'
            self.backend_logger.add_scalar(
                f'train_running/{metric_name}',
                metric_value.item(),
                self.current_train_step,
            )
            self.train_tracker.update(metric_name, metric_value.item())
        self.txt_logger.info(msg)

    def train_epoch(self):
        self.model_generator.train()
        self.model_dicriminator.train()
        self.train_dataloader.sampler.set_epoch(self.current_epoch)
        self.train_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.train_dataloader):
            self.train_iter(iter_idx, data_per_batch)
            self.current_train_step += 1
        msg = f'epoch: {self.current_epoch}'
        for key, value in self.train_tracker.results.items():
            msg += f', {key}: {value:.4f}'
            self.backend_logger.add_scalar(f'train/{key}', value, self.current_epoch)
        self.txt_logger.info(msg)

    def train_iter(self, iter_idx, data_per_batch):
        model_generator_inputs = [
            data_per_batch['coarse_img_01'].to('cuda'),
            data_per_batch['coarse_img_02'].to('cuda'),
            data_per_batch['coarse_img_03'].to('cuda'),
            data_per_batch['fine_img_01'].to('cuda'),
            data_per_batch['fine_img_03'].to('cuda'),
        ]
        # Train Discriminator
        for parms in self.model_dicriminator.parameters():
            parms.requires_grad = True
        self.optimizer_dicriminator.zero_grad()
        with torch.no_grad():
            phase_d_fake_output_g = self.model_generator(*model_generator_inputs)
        phase_d_fake_output_d = self.model_dicriminator(phase_d_fake_output_g)
        phase_d_real_output_d = self.model_dicriminator(
            data_per_batch['fine_img_02'].to('cuda')
        )

        phase_d_fake_loss = torch.mean(phase_d_fake_output_d)
        phase_d_real_loss = -torch.mean(phase_d_real_output_d)
        phase_d_gan_loss = phase_d_fake_loss + phase_d_real_loss
        phase_d_gan_loss.backward()
        self.optimizer_dicriminator.step()
        msg = f'epoch: {self.current_epoch}, iter: {iter_idx}, phase_d_fake_loss: {phase_d_fake_loss.item():.4f}, phase_d_real_loss: {phase_d_real_loss.item():.4f}, phase_d_gan_loss: {phase_d_gan_loss.item():.4f}'

        self.train_tracker.update('phase_d_fake_loss', phase_d_fake_loss.item())
        self.train_tracker.update('phase_d_real_loss', phase_d_real_loss.item())
        self.train_tracker.update('phase_d_gan_loss', phase_d_gan_loss.item())

        self.backend_logger.add_scalar(
            f'train_running/phase_d_fake_loss',
            phase_d_fake_loss.item(),
            self.current_train_step,
        )
        self.backend_logger.add_scalar(
            f'train_running/phase_d_real_loss',
            phase_d_real_loss.item(),
            self.current_train_step,
        )
        self.backend_logger.add_scalar(
            f'train_running/phase_d_gan_loss',
            phase_d_gan_loss.item(),
            self.current_train_step,
        )

        for parms in self.model_dicriminator.parameters():
            parms.data.clamp_(-0.01, 0.01)

        # Train Generator
        for parms in self.model_dicriminator.parameters():
            parms.requires_grad = False
        self.optimizer_generator.zero_grad()
        phase_g_fake_output_g = self.model_generator(*model_generator_inputs)
        phase_g_fake_output_d = self.model_dicriminator(phase_g_fake_output_g)
        phase_g_fake_loss = -torch.mean(phase_g_fake_output_d) * 1e-3

        phase_g_gan_loss = phase_g_fake_loss
        pixel_loss = F.mse_loss(
            phase_g_fake_output_g, data_per_batch['fine_img_02'].to('cuda')
        )
        phase_g_loss = phase_g_gan_loss + pixel_loss
        phase_g_loss.backward()
        self.optimizer_generator.step()

        msg += f', phase_g_fake_loss: {phase_g_fake_loss.item():.4f}, pixel_loss: {pixel_loss.item():.4f}, phase_g_loss: {phase_g_loss.item():.4f}'

        self.train_tracker.update('phase_g_fake_loss', phase_g_fake_loss.item())
        self.train_tracker.update('pixel_loss', pixel_loss.item())
        self.train_tracker.update('phase_g_loss', phase_g_loss.item())

        self.backend_logger.add_scalar(
            f'train_running/phase_g_fake_loss',
            phase_g_fake_loss.item(),
            self.current_train_step,
        )
        self.backend_logger.add_scalar(
            f'train_running/pixel_loss', pixel_loss.item(), self.current_train_step
        )
        self.backend_logger.add_scalar(
            f'train_running/phase_g_loss', phase_g_loss.item(), self.current_train_step
        )

        for metric in self.metric_list:
            metric_value = metric(
                (phase_g_fake_output_g + 1.0) / 2.0,
                (data_per_batch['fine_img_02'].to('cuda') + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value.item():.4f}'
            self.train_tracker.update(metric_name, metric_value.item())
            self.backend_logger.add_scalar(
                f'train_running/{metric_name}',
                metric_value.item(),
                self.current_train_step,
            )

        self.txt_logger.info(msg)

    def val_epoch(self):
        self.model_generator.eval()
        self.model_dicriminator.eval()
        self.val_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.val_dataloader):
            self.val_iter(iter_idx, data_per_batch)
            self.current_val_step += 1
        msg = f'val epoch: {self.current_epoch}'
        for key, value in self.val_tracker.results.items():
            msg = msg + f', {key}: {value:.4f}'
            self.backend_logger.add_scalar(f'val/{key}', value, self.current_epoch)
        self.txt_logger.info(msg)

    def val_iter(self, iter_idx, data_per_batch):
        model_generator_inputs = [
            data_per_batch['coarse_img_01'].to('cuda'),
            data_per_batch['coarse_img_02'].to('cuda'),
            data_per_batch['coarse_img_03'].to('cuda'),
            data_per_batch['fine_img_01'].to('cuda'),
            data_per_batch['fine_img_03'].to('cuda'),
        ]
        with torch.no_grad():
            phase_g_outputs = self.model_generator(*model_generator_inputs)
            pixel_loss = F.mse_loss(
                phase_g_outputs, data_per_batch['fine_img_02'].to('cuda')
            )

        self.val_tracker.update('pixel_loss', pixel_loss.item())
        self.backend_logger.add_scalar(
            f'val_running/pixel_loss', pixel_loss.item(), self.current_val_step
        )
        msg = f'val epoch: {self.current_epoch}, iter: {iter_idx}'
        for metric in self.metric_list:
            metric_value = metric(
                (phase_g_outputs + 1.0) / 2.0,
                (data_per_batch['fine_img_02'].to('cuda') + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value.item():.4f}'
            self.val_tracker.update(metric_name, metric_value.item())
            self.backend_logger.add_scalar(
                f'val_running/{metric_name}', metric_value.item(), self.current_val_step
            )
        self.txt_logger.info(msg)

        ## TODO
        save_dir_prefix = f'{self.current_epoch}/save_img'
        save_img_name = data_per_batch['key'][0].split('-')[-1] + f'_{iter_idx}.tif'
        normalize_scale = data_per_batch['normalize_scale'][0].numpy()
        normalize_mode = data_per_batch['normalize_mode'][0].numpy()
        self.img_save(
            phase_g_outputs,
            save_dir_prefix,
            save_img_name,
            normalize_scale,
            normalize_mode,
        )

        show_dir_prefix = f'{self.current_epoch}/show_img'
        show_img_name = data_per_batch['key'][0].split('-')[-1] + f'_{iter_idx}.png'
        self.img_show(
            data_per_batch,
            phase_g_outputs,
            show_dir_prefix,
            show_img_name,
            normalize_mode,
        )

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
        save_img_path = self.train_imgs_dir / save_dir_prefix / save_name
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
            ['ori_coarse_img_01', 'ori_coarse_img_02', 'ori_coarse_img_03'],
            ['fine_img_01', 'fine_img_02', 'fine_img_03'],
        ]
        _, c, h, w = pred_tensor.shape
        show_img = np.zeros(
            (
                (h + img_interval) * 3 + img_interval,
                (w + img_interval) * 3 + img_interval,
                3,
            ),
            dtype=np.uint8,
        )
        for row_index in range(2):
            for col_index in range(3):
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
            img_interval * 3 + h * 2 : img_interval * 3 + h * 3,
            img_interval * 2 + w * 1 : img_interval * 2 + w * 2,
            :,
        ] = show_sub_img

        show_img_path = self.train_imgs_dir / show_dir_prefix / show_name
        show_img_path.parent.mkdir(parents=True, exist_ok=True)

        io.imsave(show_img_path, show_img)

    def save_checkpoint(self):
        torch.save(
            self.model_generator.state_dict(),
            self.train_checkpoints_dir
            / f'model_generator_epoch_{self.current_epoch}.pth',
        )
        torch.save(
            self.model_dicriminator.state_dict(),
            self.train_checkpoints_dir
            / f'model_dicriminator_epoch_{self.current_epoch}.pth',
        )
