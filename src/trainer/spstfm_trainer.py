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
        coarse_diff_13 = (
            data_per_batch['coarse_img_01'] - data_per_batch['coarse_img_03']
        )
        fine_diff_13 = data_per_batch['fine_img_01'] - data_per_batch['fine_img_03']

        coarse_diff_13_mean = torch.mean(coarse_diff_13, dim=(-2, -1), keepdim=True)
        fine_diff_13_mean = torch.mean(fine_diff_13, dim=(-2, -1), keepdim=True)

        coarse_diff_13_std = torch.std(coarse_diff_13, dim=(-2, -1), keepdim=True)
        fine_diff_13_std = torch.std(fine_diff_13, dim=(-2, -1), keepdim=True)

        norm_coarse_diff_13 = (coarse_diff_13 - coarse_diff_13_mean) / (
            8 * coarse_diff_13_std + 1e-6
        )
        norm_fine_diff_13 = (fine_diff_13 - fine_diff_13_mean) / (
            8 * fine_diff_13_std + 1e-6
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
        coarse_dictionary_matrix_list = [result[0] for result in result_list]
        fine_dictionary_matrix_list = [result[1] for result in result_list]
        sparsity_matrix_list = [result[2] for result in result_list]
        coarse_diff_dictionary = np.stack(coarse_dictionary_matrix_list, axis=0)
        fine_diff_dictionary = np.stack(fine_dictionary_matrix_list, axis=0)
        sparsity_matrix = np.stack(sparsity_matrix_list, axis=0)
        # norm_coarse_reconstruction_img = coarse_dictionary_matrix @ sparsity_matrix
        # norm_fine_reconstruction_img = fine_dictionary_matrix @ sparsity_matrix
        state = {
            'sparisty_matrix': sparsity_matrix,
            'coarse_diff_dictionary': coarse_diff_dictionary,
            'fine_diff_dictionary': fine_diff_dictionary,
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
