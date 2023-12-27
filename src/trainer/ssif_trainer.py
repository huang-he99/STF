from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from skimage import io
from src.logger import get_logger, Tracker
from torch.utils.tensorboard import SummaryWriter
import shutil
import multiprocessing as mp
import scipy.io as scio


class Trainer:
    def __init__(
        self,
        congfig_path,
        train_root_dir_path,
        train_dir_prefix_list=[
            'dictionary_sparisty',
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
        #
        model=None,
    ):
        self.init_train_params()
        self.init_train_dir(train_root_dir_path, train_dir_prefix_list, congfig_path)
        # self.init_logger(txt_logger_name, txt_logger_level)
        self.init_dataloader(train_dataloader)
        self.init_model(model)
        # self.init_metric(metric_list)

        # self.init_tracker()

    def init_train_params(self):
        self.current_epoch = 0
        self.max_epoch = 1000
        self.val_interal = 50
        self.current_val_step = 0
        self.current_train_step = 0

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
        self.txt_logger = get_logger(
            logger_name=txt_logger_name,
            # log_file=self.train_txt_logs_dir / 'log.log',
            log_level=txt_logger_level,
        )
        self.backend_logger = SummaryWriter(self.train_backend_logs_dir)

    def init_dataloader(self, train_dataloader):
        self.train_dataloader = train_dataloader

    def init_model(self, model):
        # self.device = 'cuda'
        self.model = model

    # def init_metric(self, metric_list):
    #     # for metric in metric_list:
    #     #     metric.to(self.device)
    #     self.metric_list = metric_list

    # def init_tracker(self):
    #     key_list = ['mse_loss'] + [metric.__name__ for metric in self.metric_list]
    #     self.train_tracker = Tracker(*key_list)

    def train(self):
        # self.train_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.train_dataloader):
            (
                model_input_list,
                input_mean,
                input_std,
                key,
                dataset_name,
            ) = self.before_train_iter(data_per_batch)
            self.train_iter(
                iter_idx, model_input_list, input_mean, input_std, key, dataset_name
            )

    def before_train_iter(self, data_per_batch):
        model_input_list, input_mean, input_std = self.get_model_input(data_per_batch)
        key = data_per_batch['key'][0]
        dataset_name = data_per_batch['dataset_name'][0]
        return model_input_list, input_mean, input_std, key, dataset_name

    def get_model_input(self, data_per_batch):
        coarse_img_01 = data_per_batch['coarse_img_01']
        coarse_img_02 = data_per_batch['coarse_img_02']
        fine_img_01 = data_per_batch['fine_img_01']
        fine_img_02 = data_per_batch['fine_img_02']

        diff_fine_coarse_01 = fine_img_01 - coarse_img_01
        coarse_img_gradient_x, coarse_img_gradient_y = self.get_gradient(coarse_img_01)

        coarse_diff_13 = (
            data_per_batch['coarse_img_01'] - data_per_batch['coarse_img_03']
        )
        fine_diff_13 = data_per_batch['fine_img_01'] - data_per_batch['fine_img_03']

        coarse_diff_13_mean = torch.mean(coarse_diff_13, dim=(-2, -1), keepdim=True)
        fine_diff_13_mean = torch.mean(fine_diff_13, dim=(-2, -1), keepdim=True)

        coarse_diff_13_std = torch.std(coarse_diff_13, dim=(-2, -1), keepdim=True)
        fine_diff_13_std = torch.std(fine_diff_13, dim=(-2, -1), keepdim=True)

        norm_coarse_diff_13 = (coarse_diff_13 - coarse_diff_13_mean) / (
            coarse_diff_13_std + 1e-6
        )
        norm_fine_diff_13 = (fine_diff_13 - fine_diff_13_mean) / (
            fine_diff_13_std + 1e-6
        )

        model_input_list = [norm_coarse_diff_13, norm_fine_diff_13]
        input_mean = [
            coarse_diff_13_mean.squeeze(0).numpy(),
            fine_diff_13_mean.squeeze(0).numpy(),
        ]
        input_std = [
            coarse_diff_13_std.squeeze(0).numpy(),
            fine_diff_13_std.squeeze(0).numpy(),
        ]
        return model_input_list, input_mean, input_std

    def get_gradient(self, img):
        right_shift_img = F.pad(img, (1, 0, 0, 0))[:, :, :, :-1]
        gradient_x = img - right_shift_img

        down_shift_img = F.pad(img, (0, 0, 1, 0))[:, :, :-1, :]
        gradient_y = img - down_shift_img
        return gradient_x, gradient_y

    def train_iter(
        self, iter_idx, model_input_list, input_mean, input_std, key, dataset_name
    ):
        channel_num = model_input_list[0].shape[1]
        model_input_per_channel_list = [
            ([model_input[:, i].unsqueeze(1) for model_input in model_input_list],)
            for i in range(channel_num)
        ]
        pool = mp.Pool(processes=channel_num)
        result_list = pool.starmap(
            self.train_iter_per_channel, model_input_per_channel_list
        )
        # msg = f'iter_idx: {iter_idx} '
        coarse_dictionary_matrix_list = [result[0] for result in result_list]
        fine_dictionary_matrix_list = [result[1] for result in result_list]
        sparsity_matrix_list = [result[2] for result in result_list]
        coarse_dictionary_matrix = np.stack(coarse_dictionary_matrix_list, axis=0)
        fine_dictionary_matrix = np.stack(fine_dictionary_matrix_list, axis=0)
        sparsity_matrix = np.stack(sparsity_matrix_list, axis=0)
        norm_coarse_reconstruction_img = coarse_dictionary_matrix @ sparsity_matrix
        norm_fine_reconstruction_img = fine_dictionary_matrix @ sparsity_matrix

        # mse_coarse_loss = np.mean(
        #     (norm_coarse_reconstruction_img - model_input_list[0]) ** 2
        # )
        # mse_fine_loss = np.mean(
        #     (norm_fine_reconstruction_img - model_input_list[1]) ** 2
        # )
        # msg += f'mse_coarse_loss: {mse_coarse_loss}, mse_fine_loss: {mse_fine_loss} '
        # self.txt_logger.info(msg)

        state = {
            'sparisty_matrix': sparsity_matrix,
            'coarse_dictionary_matrix': coarse_dictionary_matrix,
            'fine_dictionary_matrix': fine_dictionary_matrix,
        }
        save_name = key.split('-')[-1]
        save_name = save_name[:8] + '_dictionary_sparisty_' + save_name[9:]
        save_dir = self.train_dictionary_sparisty_dir / dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = (
            self.train_dictionary_sparisty_dir / dataset_name / f'{save_name}.mat'
        )
        scio.savemat(save_path, state)

    def train_iter_per_channel(self, model_input_list):
        (
            coarse_dictionary,
            fine_dictionary,
            sparsity_matrix,
        ) = self.model.training_dictionary_pair(*model_input_list)
        return coarse_dictionary, fine_dictionary, sparsity_matrix

    def stable_train_iter(self, iter_idx, model_input_list, loss_gt, metrics_gt):
        self.optimizer.zero_grad()
        outputs = self.model(*model_input_list)
        pixel_loss = F.mse_loss(outputs, loss_gt)
        pixel_loss.backward()
        self.optimizer.step()
        msg = (
            f'epoch: {self.current_epoch}, iter: {iter_idx}, loss: {pixel_loss.item()}'
        )
        self.train_tracker.update('mse_loss', pixel_loss.item())

        for metric in self.metric_list:
            metric_value = metric(
                (outputs + 1.0) / 2.0,
                (metrics_gt + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value.item()}'
            self.backend_logger.add_scalar(
                f'train_runinng/{metric_name}',
                metric_value.item(),
                self.current_train_step,
            )
            self.train_tracker.update(metric_name, metric_value.item())
        self.txt_logger.info(msg)

    def val(self):
        self.model.eval()
        self.val_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.val_dataloader):
            model_input_list, loss_gt, metrics_gt = self.before_stable_train_iter(
                iter_idx, data_per_batch
            )
            self.val_iter(
                iter_idx, model_input_list, loss_gt, metrics_gt, data_per_batch
            )
            self.current_val_step += 1
        msg = f'val epoch: {self.current_epoch}'
        for key, value in self.val_tracker.results.items():
            msg += f', {key}: {value:.4f}'
            self.backend_logger.add_scalar(f'val/{key}', value, self.current_epoch)
        self.txt_logger.info(msg)

    def before_val_iter(self, iter_idx, data_per_batch):
        pass

    def val_iter(self, iter_idx, model_input_list, loss_gt, metrics_gt, data_per_batch):
        with torch.no_grad():
            outputs = self.model(*model_input_list)
            pixel_loss = F.mse_loss(outputs, loss_gt)
            self.val_tracker.update('mse_loss', pixel_loss.item())
        msg = f'val epoch: {self.current_epoch}, iter: {iter_idx}'
        for metric in self.metric_list:
            metric_value = metric(
                (outputs + 1.0) / 2.0,
                (metrics_gt + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value.item()}'
            self.backend_logger.add_scalar(
                f'val_runinng/{metric_name}', metric_value.item(), self.current_val_step
            )
            self.val_tracker.update(metric_name, metric_value.item())

        self.txt_logger.info(msg)

        ## TODO
        save_dir_prefix = f'{self.current_epoch}/save_img'
        save_img_name = data_per_batch['key'][0].split('-')[-1] + f'_{iter_idx}.tif'
        normalize_scale = data_per_batch['normalize_scale'][0].numpy()
        normalize_mode = data_per_batch['normalize_mode'][0].numpy()
        self.img_save(
            outputs,
            save_dir_prefix,
            save_img_name,
            normalize_scale,
            normalize_mode,
        )

        show_dir_prefix = f'{self.current_epoch}/show_img'
        show_img_name = data_per_batch['key'][0].split('-')[-1] + f'_{iter_idx}.png'
        self.img_show(
            data_per_batch,
            outputs,
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
            ['coarse_img'],
            ['fine_img'],
        ]
        _, c, h, w = pred_tensor.shape
        show_img = np.zeros(
            (
                (h + img_interval) * 3 + img_interval,
                (w + img_interval) * 1 + img_interval,
                3,
            ),
            dtype=np.uint8,
        )
        for row_index in range(2):
            for col_index in range(1):
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
            img_interval * 1 + w * 0 : img_interval * 1 + w * 1,
            :,
        ] = show_sub_img

        show_img_path = self.train_imgs_dir / show_dir_prefix / show_name
        show_img_path.parent.mkdir(parents=True, exist_ok=True)

        io.imsave(show_img_path, show_img)

    def save_checkpoint(self):
        torch.save(
            self.model.state_dict(),
            self.train_checkpoints_dir / f'model_epoch_{self.current_epoch}.pth',
        )

    def test(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        for iter_idx, data_per_batch in enumerate(self.test_dataloader):
            model_input_list, _, metrics_gt = self.before_stable_train_iter(
                iter_idx, data_per_batch
            )
            self.test_iter(
                iter_idx, model_input_list, metrics_gt, data_per_batch, checkpoint_path
            )

    def test_iter(
        self, iter_idx, model_input_list, metrics_gt, data_per_batch, checkpoint_path
    ):
        with torch.no_grad():
            outputs = self.model(*model_input_list)
        msg = f'val iter: {iter_idx}'
        for metric in self.metric_list:
            metric_value = metric(
                (outputs + 1.0) / 2.0,
                (metrics_gt + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value}'
        self.txt_logger.info(msg)

        ## TODO
        save_dir_prefix = f'test_save_img/{Path(checkpoint_path).stem}'
        save_img_name = data_per_batch['key'][0].split('-')[-1] + f'_{iter_idx}.tif'
        normalize_scale = data_per_batch['normalize_scale'][0].numpy()
        normalize_mode = data_per_batch['normalize_mode'][0].numpy()
        self.img_save(
            outputs,
            save_dir_prefix,
            save_img_name,
            normalize_scale,
            normalize_mode,
        )

        show_dir_prefix = f'show_img/{Path(checkpoint_path).stem}'
        show_img_name = data_per_batch['key'][0].split('-')[-1] + f'_{iter_idx}.png'
        self.img_show(
            data_per_batch,
            outputs,
            show_dir_prefix,
            show_img_name,
            normalize_mode,
        )
