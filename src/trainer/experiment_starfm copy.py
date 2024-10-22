from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage import io

from src.logger.txt_logger import FusionLogger
from src.utils.patch import PatchGenerator


class Experiment:
    def __init__(
        self,
        experiment_root_dir,
        experiment_dir_prefix_list=[
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
        model=None,
        metric_list=None,
        patch_generator: PatchGenerator = None,
    ):
        self.is_training = False
        self.is_patch_based = True
        self.init_experiment_dir(experiment_root_dir, experiment_dir_prefix_list)
        self.init_logger(txt_logger_name, txt_logger_level)
        self.init_dataloader(train_dataloader, val_dataloader, test_dataloader)
        self.init_model(model)
        self.init_metric(metric_list)
        self.patch_generator = patch_generator

    def init_experiment_dir(self, experiment_root_dir, experiment_dir_prefix_list):
        self.experiment_root_dir = Path(experiment_root_dir)
        self.experiment_dir_prefix_list = experiment_dir_prefix_list
        self.mkdir()

    def init_model(self, model):
        self.model = model.to('cuda')

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

    def test(self):
        self.model.eval()
        if self.is_patch_based:
            self.test_patch_based()
        else:
            self.test_whole_image_based()

    def test_whole_image_based(self):
        for iter_idx, data_per_batch in enumerate(self.test_dataloader):
            model_input = None
            self.test_iter(iter_idx, model_input)

    def test_patch_based(self):
        for iter_idx, data_per_batch in enumerate(self.test_dataloader):
            # prior_coarse_img = torch.cat(
            #     [
            #         data_per_batch['coarse_img_01'].unsqueeze(2),
            #         data_per_batch['coarse_img_03'].unsqueeze(2),
            #     ],
            #     dim=2,
            # )
            # prior_fine_img = torch.cat(
            #     [
            #         data_per_batch['fine_img_01'].unsqueeze(2),
            #         data_per_batch['fine_img_03'].unsqueeze(2),
            #     ],
            #     dim=1,
            # )
            # pred_coarse_img = data_per_batch['coarse_img_02'].unsqueeze(2)
            # gt_fine_img = data_per_batch['fine_img_02']
            prior_coarse_img = torch.cat(
                [
                    data_per_batch['coarse_img_01'].to('cuda').unsqueeze(2),
                    data_per_batch['coarse_img_03'].to('cuda').unsqueeze(2),
                ],
                dim=2,
            )
            prior_fine_img = torch.cat(
                [
                    data_per_batch['fine_img_01'].to('cuda').unsqueeze(2),
                    data_per_batch['fine_img_03'].to('cuda').unsqueeze(2),
                ],
                dim=2,
            )
            pred_coarse_img = data_per_batch['coarse_img_02'].to('cuda').unsqueeze(2)
            gt_fine_img = data_per_batch['fine_img_02'].to('cuda')
            img_size = gt_fine_img.shape[-2:]
            patch_generator = self.patch_generator(img_size)
            ori_img_size = data_per_batch['ori_img_size']
            pred_img = torch.zeros_like(gt_fine_img)
            cnt = torch.zeros_like(gt_fine_img)
            for top, bottom, left, right in patch_generator:
                # prior_coarse_img_patch = prior_coarse_img[..., h_range, w_range].to(
                #     'cuda'
                # )
                # prior_fine_img_patch = prior_fine_img[..., h_range, w_range].to('cuda')
                # pred_coarse_img_patch = pred_coarse_img[..., h_range, w_range].to(
                #     'cuda'
                # )
                # gt_fine_img_patch = gt_fine_img[..., h_range, w_range].to('cuda')
                prior_coarse_img_patch = prior_coarse_img[..., top:bottom, left:right]
                prior_fine_img_patch = prior_fine_img[..., top:bottom, left:right]
                pred_coarse_img_patch = pred_coarse_img[..., top:bottom, left:right]
                gt_fine_img_patch = gt_fine_img[..., top:bottom, left:right]
                model_input_list = [
                    prior_coarse_img_patch,
                    prior_fine_img_patch,
                    pred_coarse_img_patch,
                ]
                pred_img_patch = self.test_iter(iter_idx, model_input_list)
                pred_img[
                    ..., top + 25 : bottom - 25, left + 25 : right - 25
                ] += pred_img_patch
                cnt[..., top + 25 : bottom - 25, left + 25 : right - 25] += 1
                patch_idx = patch_generator.patch_index
                patch_h_idx = patch_generator.patch_h_index
                patch_w_idx = patch_generator.patch_w_index
                msg = f'Test iter: {iter_idx}, [patch_idx/(h ,w)]: [{patch_idx}/({patch_h_idx},{patch_w_idx})]'
                for metric in self.metric_list:
                    metric_value = metric(
                        pred_img_patch,
                        gt_fine_img_patch[..., 25:-25, 25:-25],
                    )
                    metric_name = metric.__name__
                    msg = msg + f', Patch_{metric_name}: {metric_value}'
                self.txt_logger.info(msg)

                ## TODO
                save_dir_prefix = f'iter_{iter_idx}/save_img_patch'
                save_img_name = f'patch_{patch_idx}_h_{patch_h_idx}_w_{patch_w_idx}.tif'
                normalize_scale = data_per_batch['normalize_scale'][0].numpy()
                normalize_mode = data_per_batch['normalize_mode'][0].numpy()
                self.img_save(
                    pred_img_patch,
                    save_dir_prefix,
                    save_img_name,
                    normalize_scale,
                    normalize_mode,
                )

                show_dir_prefix = f'iter_{iter_idx}/show_img_patch'
                show_img_name = (
                    data_per_batch['key'][0].split('-')[-1]
                    + f'_patch_{patch_idx}_h_{patch_h_idx}_w_{patch_w_idx}.png'
                )
                data_patch_per_batch = {
                    'coarse_img_01': prior_coarse_img_patch[:, :, 0, 25:-25, +25:-25],
                    'coarse_img_02': pred_coarse_img_patch[:, :, 0, +25:-25, +25:-25],
                    'coarse_img_03': prior_coarse_img_patch[:, :, 1, +25:-25, +25:-25],
                    'fine_img_01': prior_fine_img_patch[:, :, 0, +25:-25, +25:-25],
                    'fine_img_02': gt_fine_img_patch[:, :, +25:-25, +25:-25],
                    'fine_img_03': prior_fine_img_patch[:, :, 1, +25:-25, +25:-25],
                }
                self.img_show(
                    data_patch_per_batch,
                    pred_img_patch,
                    show_dir_prefix,
                    show_img_name,
                    normalize_mode,
                )
            pred_img = pred_img / cnt
            msg = f'Test iter: {iter_idx}'
            for metric in self.metric_list:
                metric_value = metric(pred_img, gt_fine_img)
                metric_name = metric.__name__
                msg = msg + f', Total_{metric_name}: {metric_value}'
            self.txt_logger.info(msg)

            ## TODO
            save_dir_prefix = f'iter_{iter_idx}/save_img_full'
            save_img_name = f'iter_{iter_idx}.tif'
            normalize_scale = data_per_batch['normalize_scale'][0].numpy()
            normalize_mode = data_per_batch['normalize_mode'][0].numpy()
            self.img_save(
                pred_img,
                save_dir_prefix,
                save_img_name,
                normalize_scale,
                normalize_mode,
            )

            show_dir_prefix = f'iter_{iter_idx}/show_img_full'
            show_img_name = data_per_batch['key'][0].split('-')[-1] + f'.png'
            self.img_show(
                data_per_batch,
                pred_img,
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
        save_img_path = self.experiment_imgs_dir / save_dir_prefix / save_name
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
            ['coarse_img_01', 'coarse_img_02', 'coarse_img_03'],
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

        show_img_path = self.experiment_imgs_dir / show_dir_prefix / show_name
        show_img_path.parent.mkdir(parents=True, exist_ok=True)

        io.imsave(show_img_path, show_img)

    def test_iter(self, iter_idx, model_input_list):
        with torch.no_grad():
            model_output = self.model(*model_input_list)
        return model_output

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
