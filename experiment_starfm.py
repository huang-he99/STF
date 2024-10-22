from pathlib import Path
from logger.txt_logger import FusionLogger
import torch
import torch.nn.functional as F
import tifffile
import numpy as np
import cv2

# class Experiment:
#     def __init__(
#         self,
#         experiment_root_dir,
#         experiment_dir_prefix_list=[
#             'checkpoints',
#             'txt_logs',
#             'backend_logs',
#             'results',
#             'configs',
#         ],
#         txt_logger_name='fusion',
#         txt_logger_level='INFO',
#         train_dataloader=None,
#         val_dataloader=None,
#         test_dataloader=None,
#         model=None,
#         optimizer=None,
#         scheduler=None,
#     ):
#         self.init_experiment_dir(experiment_root_dir, experiment_dir_prefix_list)
#         self.init_logger(txt_logger_name, txt_logger_level)
#         self.init_dataloader(train_dataloader, val_dataloader, test_dataloader)
#         self.model = model.to('cuda')

#     def init_experiment_dir(self, experiment_root_dir, experiment_dir_prefix_list):
#         self.experiment_root_dir = Path(experiment_root_dir)
#         self.experiment_dir_prefix_list = experiment_dir_prefix_list
#         self.mkdir()

#     def mkdir(self):
#         for experiment_dir_prefix in self.experiment_dir_prefix_list:
#             experiment_dir = self.experiment_root_dir / experiment_dir_prefix
#             experiment_dir.mkdir(parents=True, exist_ok=True)
#             setattr(self, f'_experiment_{experiment_dir_prefix}_dir', experiment_dir)

#     def init_logger(self, txt_logger_name, txt_logger_level):
#         self.txt_logger = FusionLogger(
#             self.experiment_txt_logs_dir,
#             logger_name=txt_logger_name,
#             log_level=txt_logger_level,
#         )
#         self.backend_logger = None

#     def init_dataloader(self, train_dataloader, val_dataloader, test_dataloader):
#         if self.is_training:
#             self.train_dataloader = train_dataloader
#             self.val_dataloader = val_dataloader
#         else:
#             self.test_dataloader = test_dataloader

#     def train(self):
#         self.current_epoch = 0
#         self.max_epoch = 100
#         while self.current_epoch < self.max_epoch:
#             self.train_epoch()
#             self.validate_one_epoch()
#             self.save_checkpoint()
#             self.current_epoch += 1

#     def train_epoch(self):
#         self.model.train()
#         for iter_idx, data_per_batch in enumerate(self.train_dataloader):
#             self.train_iter(iter_idx, data_per_batch)

#     def train_iter(self, iter_idx, data_per_batch):
#         self.optimizer.zero_grad()
#         outputs = self.model(data_per_batch)
#         loss = self.criterion(outputs, data_per_batch)
#         loss.backward()
#         self.optimizer.step()
#         metrics = self.metrics(outputs, data_per_batch)
#         self.log_train_iter(iter_idx, loss, metrics)

#     def log_train_iter(self, iter_idx, loss, metrics):
#         pass


class Experiment:
    def __init__(
        self,
        experiment_root_dir,
        experiment_dir_prefix_list=[
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
        test_dataloader=None,
        model_generator=None,
        model_discriminator=None,
        optimizer_generator=None,
        optimizer_discriminator=None,
        scheduler_generator=None,
        scheduler_dicriminator=None,
        metric_list=None,
    ):
        self.is_training = True
        self.init_experiment_dir(experiment_root_dir, experiment_dir_prefix_list)
        self.init_logger(txt_logger_name, txt_logger_level)
        self.init_dataloader(train_dataloader, val_dataloader, test_dataloader)
        self.init_model(model_generator, model_discriminator)
        self.init_optimizer(optimizer_generator, optimizer_discriminator)
        self.init_scheduler(scheduler_generator, scheduler_dicriminator)
        self.init_metric(metric_list)

    def init_experiment_dir(self, experiment_root_dir, experiment_dir_prefix_list):
        self.experiment_root_dir = Path(experiment_root_dir)
        self.experiment_dir_prefix_list = experiment_dir_prefix_list
        self.mkdir()

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

    def init_scheduler(self, scheduler_generator, scheduler_dicriminator):
        self.scheduler_generator = scheduler_generator
        self.scheduler_dicriminator = scheduler_dicriminator

    def mkdir(self):
        for experiment_dir_prefix in self.experiment_dir_prefix_list:
            experiment_dir = self.experiment_root_dir / experiment_dir_prefix
            experiment_dir.mkdir(parents=True, exist_ok=True)
            setattr(self, f'experiment_{experiment_dir_prefix}_dir', experiment_dir)

    def init_logger(self, txt_logger_name, txt_logger_level):
        self.txt_logger = FusionLogger(
            logger_name=txt_logger_name,
            log_file=self.experiment_txt_logs_dir / 'log.log',
            log_level=txt_logger_level,
        )
        self.backend_logger = None

    def init_dataloader(self, train_dataloader, val_dataloader, test_dataloader):
        if self.is_training:
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
        else:
            self.test_dataloader = test_dataloader

    def init_metric(self, metric_list):
        for metric in metric_list:
            metric.to('cuda')
        self.metric_list = metric_list

    def train(self):
        self.current_epoch = 0
        self.max_epoch = 100
        self.current_warm_up_epoch = 0
        self.max_warm_up_epoch = 2
        while self.current_warm_up_epoch < self.max_warm_up_epoch:
            self.train_warm_up_epoch()
            self.val_epoch()
            self.current_warm_up_epoch += 1
        while self.current_epoch < self.max_epoch:
            self.train_epoch()
            self.val_epoch()
            self.save_checkpoint()
            self.current_epoch += 1

    def test_patch(self):
        self.model.eval()
        for iter_idx, data_per_batch in enumerate(self.test_dataloader):
            self.test_patch_iter(iter_idx, data_per_batch)

    def test_patch_iter(self, iter_idx, data_per_batch):
        patch_pos_generator = PatchGenerator()
        with torch.no_grad():
            for patch_rows, patch_cols in patch_pos_generator:
                pass

    def train_warm_up_epoch(self):
        self.model_generator.train()
        self.train_dataloader.sampler.set_epoch(self.current_warm_up_epoch)
        for iter_idx, data_per_batch in enumerate(self.train_dataloader):
            self.train_warm_up_iter(iter_idx, data_per_batch)

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
        msg = f'warm up epoch: {self.current_warm_up_epoch}, iter: {iter_idx}, pixel_loss: {pixel_loss.item()}'
        for metric in self.metric_list:
            metric_value = metric(
                phase_g_outputs, data_per_batch['fine_img_02'].to('cuda')
            )
            metric_name = metric.__name__
            msg += f', {metric_name}: {metric_value.item()}'
        self.txt_logger.info(msg)

    def train_epoch(self):
        self.model_generator.train()
        self.model_dicriminator.train()
        self.train_dataloader.sampler.set_epoch(
            self.current_epoch + self.max_warm_up_epoch
        )
        for iter_idx, data_per_batch in enumerate(self.train_dataloader):
            self.train_iter(iter_idx, data_per_batch)

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

        msg = f'epoch: {self.current_epoch}, iter: {iter_idx}, phase_d_gan_loss: {phase_d_gan_loss}, phase_g_gan_loss: {phase_g_gan_loss}, pixel_loss: {pixel_loss}, phase_g_loss: {phase_g_loss}'

        for metric in self.metric_list:
            metric_value = metric(
                (phase_g_fake_output_g + 1.0) / 2.0,
                (data_per_batch['fine_img_02'].to('cuda') + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value}'
        self.txt_logger.info(msg)

    def val_epoch(self):
        self.model_generator.eval()
        self.model_dicriminator.eval()
        for iter_idx, data_per_batch in enumerate(self.val_dataloader):
            self.val_iter(iter_idx, data_per_batch)

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
        msg = f'val epoch: {self.current_epoch}, iter: {iter_idx}'
        for metric in self.metric_list:
            metric_value = metric(
                (phase_g_outputs + 1.0) / 2.0,
                (data_per_batch['fine_img_02'].to('cuda') + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value}'
        self.txt_logger.info(msg)
        result_img = (phase_g_outputs + 1.0) / 2.0 * 10000.0
        img_name = data_per_batch['key'][0].split('-')[-1]
        self.experiment_epoch_imgs_dir = (
            self.experiment_imgs_dir / f'{self.current_epoch}'
        )
        self.experiment_epoch_imgs_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            self.experiment_epoch_imgs_dir / f'{img_name}_{iter_idx}.tif',
            result_img[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint16),
        )

        self.experiment_epoch_imgs_show_dir = (
            self.experiment_imgs_dir / f'show_{self.current_epoch}'
        )
        self.experiment_epoch_imgs_show_dir.mkdir(parents=True, exist_ok=True)
        result_img_show = (phase_g_outputs + 1.0) / 2.0 * 255.0
        result_img_show.clamp_(0, 255)
        result_img_show = (
            result_img_show[0, (3, 2, 1), :, :]
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            .astype(np.uint8)
        )
        cv2.imwrite(
            str(
                self.experiment_epoch_imgs_show_dir / f'{img_name}_{iter_idx}_show.png'
            ),
            result_img_show,
        )

    def save_checkpoint(self):
        torch.save(
            self.model_generator.state_dict(),
            self.experiment_checkpoints_dir
            / f'model_generator_epoch_{self.current_epoch}.pth',
        )
        torch.save(
            self.model_dicriminator.state_dict(),
            self.experiment_checkpoints_dir
            / f'model_dicriminator_epoch_{self.current_epoch}.pth',
        )
