from pathlib import Path
from logger.txt_logger import FusionLogger
import torch
import torch.nn.functional as F
import tifffile
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter


class BaseExperiment:
    def __init__(
        self,
        experiment_root_dir,
        experiment_dir_prefix_list=[
            'checkpoints',
            'txt_logs',
            'backend_logs',
            'results',
            'configs',
        ],
        #
        txt_logger_name='fusion',
        txt_logger_level='INFO',
        #
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        #
        model=None,
        device_info=None,
        #
        optimizer=None,
        scheduler=None,
        #
        is_training=True,
    ):
        self.init_experiment_params(is_training)
        self.init_experiment_dir(experiment_root_dir, experiment_dir_prefix_list)
        self.init_logger(txt_logger_name, txt_logger_level)
        self.init_dataloader(train_dataloader, val_dataloader, test_dataloader)
        self.init_model(model)
        if self.is_training:
            self.init_optimizer(optimizer)
            self.init_scheduler(scheduler)

    def init_experiment_params(self, is_training):
        self.is_training = is_training
        self.current_epoch = 0
        self.max_epoch = 100

    def init_experiment_dir(self, experiment_root_dir, experiment_dir_prefix_list):
        self.experiment_root_dir = Path(experiment_root_dir)
        self.experiment_dir_prefix_list = experiment_dir_prefix_list
        self.mkdir()

    def mkdir(self):
        for experiment_dir_prefix in self.experiment_dir_prefix_list:
            experiment_dir = self.experiment_root_dir / experiment_dir_prefix
            experiment_dir.mkdir(parents=True, exist_ok=True)
            setattr(self, f'experiment_{experiment_dir_prefix}_dir', experiment_dir)

    def init_logger(self, txt_logger_name, txt_logger_level):
        self.txt_logger = FusionLogger(
            self.experiment_txt_logs_dir,
            logger_name=txt_logger_name,
            log_level=txt_logger_level,
        )
        self.backend_logger = SummaryWriter(self.experiment_backend_logs_dir)

    def init_dataloader(self, train_dataloader, val_dataloader, test_dataloader):
        if self.is_training:
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
        else:
            self.test_dataloader = test_dataloader

    def init_model(self, model):
        self.device = 'cuda'
        self.model = model.to(self.device)

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer(params=self.model.parameters())

    def init_scheduler(self, scheduler):
        self.scheduler = scheduler(self.optimizer) if scheduler is not None else None

    def train(self):
        self.warmup_train()
        self.stable_train()
        # while self.current_epoch < self.max_epoch:
        #     self.train_epoch()
        #     self.validate_one_epoch()
        #     self.save_checkpoint()
        #     self.current_epoch += 1

    def warmup_train(self):
        while self.current_epoch < 5:
            pass

    def stable_train(self):
        while self.current_epoch < self.max_epoch:
            self.stable_train_epoch()
            self.val()
            self.save_checkpoint()
            self.current_epoch += 1

    def stable_train_epoch(self):
        self.model.train()
        for iter_idx, data_per_batch in enumerate(self.train_dataloader):
            model_input_list, loss_gt, metrics_gt = self.before_stable_train_iter(
                iter_idx, data_per_batch
            )
            self.stable_train_iter(iter_idx, model_input_list, loss_gt, metrics_gt)

    def before_stable_train_iter(self, iter_idx, data_per_batch):
        model_input_list = self.get_model_input(data_per_batch)
        loss_gt = self.get_loss_gt(data_per_batch)
        metrics_gt = self.get_metrics_gt(data_per_batch)
        return model_input_list, loss_gt, metrics_gt

    def get_model_input(self, data_per_batch):
        model_input_key_list = []
        model_input_list = []
        for model_input_key in model_input_key_list:
            model_input_list.append(data_per_batch[model_input_key])
        return model_input_list

    def get_loss_gt(self, data_per_batch):
        loss_gt_key = ''
        return data_per_batch[loss_gt_key]

    def get_metrics_gt(self, data_per_batch):
        metrics_gt_key = ''
        return data_per_batch[metrics_gt_key]

    def stable_train_iter(self, iter_idx, model_input_list, loss_gt, metrics_gt):
        self.optimizer.zero_grad()
        outputs = self.model(model_input_list)
        loss = self.criterion(outputs, loss_gt)
        loss.backward()
        self.optimizer.step()
        # metrics = self.metrics(outputs, data_per_batch)
        # self.log_train_iter(iter_idx, loss, metrics)

    def log_train_iter(self, iter_idx, loss, metrics):
        pass
