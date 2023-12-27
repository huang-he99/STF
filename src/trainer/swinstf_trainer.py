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
import cv2
from torch import nn
from math import exp


class PixleLoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean'):
        super(PixleLoss, self).__init__()

        self.weight = weight
        self.reduction = reduction
        self.eps = 1e-3

    def forward(self, preds, target):
        return self.weight * torch.mean(
            torch.sqrt(torch.square(target - preds) + self.eps * self.eps)
        )


class MSSSIMLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(MSSSIMLoss, self).__init__()
        self.weight = weight

    def forward(self, preds, target):
        loss = self.weight * (
            1 - msssim(preds, target, val_range=2.0, normalize='relu')
        )
        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1,
    img2,
    window_size=11,
    window=None,
    size_average=True,
    full=False,
    val_range=None,
):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(
    img1, img2, window_size=11, size_average=True, val_range=None, normalize=None
):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(
            img1,
            img2,
            window_size=window_size,
            size_average=size_average,
            full=True,
            val_range=val_range,
        )
        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append((torch.relu(sim) + 1e-2))
            mcs.append((torch.relu(cs) + 1e-2))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs**weights
    pow2 = ssims**weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    # clipping
    output = (output + 0.05).clamp(max=1)
    return output


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
        model=None,
        optimizer=None,
        scheduler=None,
        metric_list=None,
    ):
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
        self.current_val_step = 0
        self.current_train_step = 0
        self.current_epoch = 0
        self.max_epoch = 1000
        self.max_warm_up_epoch = 1
        self.val_interal = 10

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

    def init_model(self, model):
        self.model = model.to('cuda')

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer(params=self.model.parameters())

    def init_scheduler(self, scheduler):
        self.scheduler = scheduler(self.optimizer)

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

        self.pixle_loss = PixleLoss().to('cuda')
        self.msssim_loss = MSSSIMLoss().to('cuda')

    def init_tracker(self):
        key_list = [
            'pixel_loss',
            'msssim_loss',
            'loss',
        ] + [metric.__name__ for metric in self.metric_list]
        self.train_tracker = Tracker(*key_list)
        self.val_tracker = Tracker(*key_list)

    def train(self):
        while self.current_epoch < self.max_epoch:
            self.train_epoch()
            self.scheduler.step()
            if (
                self.current_epoch + 1
            ) % self.val_interal == 0 or self.current_epoch == 0:
                self.val_epoch()
                self.save_checkpoint()
            self.current_epoch += 1

    def train_epoch(self):
        self.model.train()
        self.train_dataloader.sampler.set_epoch(self.current_epoch)
        self.train_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.train_dataloader):
            (model_input_list, gt) = self.before_train_iter(data_per_batch)
            self.train_iter(iter_idx, model_input_list, gt)
            self.current_train_step += 1
        lr = self.optimizer.param_groups[0]['lr']

        msg = f'epoch: {self.current_epoch}, lr: {lr:.3e}'
        for key, value in self.train_tracker.results.items():
            msg += f', {key}: {value:.4f}'
            self.backend_logger.add_scalar(f'train/{key}', value, self.current_epoch)
        self.txt_logger.info(msg)

    def before_train_iter(self, data_per_batch):
        model_input_list = self.get_model_input(data_per_batch)
        gt = data_per_batch['fine_img_02'].to('cuda')

        return (model_input_list, gt)

    def get_model_input(self, data_per_batch):
        coarse_img_01 = data_per_batch['coarse_img_01'].to('cuda')
        coarse_img_02 = data_per_batch['coarse_img_02'].to('cuda')
        fine_img_01 = data_per_batch['fine_img_01'].to('cuda')
        model_input_list = [
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
        ]
        return model_input_list

    def train_iter(self, iter_idx, model_input_list, gt):
        # Train Discriminator
        lr = self.optimizer.param_groups[0]['lr']

        msg = f'epoch: {self.current_epoch}, lr: {lr:.3e}, iter: {iter_idx}'

        self.optimizer.zero_grad()
        model_output = self.model(*model_input_list)

        pixel_loss = self.pixle_loss(model_output, gt)
        msssim_loss = self.msssim_loss(model_output, gt)

        loss = pixel_loss + msssim_loss
        loss.backward()
        self.optimizer.step()

        msg += f' pixel_loss: {pixel_loss.item():.4e}, msssim_loss: {msssim_loss.item():.4e}, loss: {loss.item():.4e}'

        self.train_tracker.update('pixel_loss', pixel_loss.item())
        self.train_tracker.update('msssim_loss', msssim_loss.item())
        self.train_tracker.update('loss', loss.item())

        self.backend_logger.add_scalar(
            f'train_running/pixel_loss',
            pixel_loss.item(),
            self.current_train_step,
        )
        self.backend_logger.add_scalar(
            f'train_running/msssim_loss', msssim_loss.item(), self.current_train_step
        )
        self.backend_logger.add_scalar(
            f'train_running/loss', loss.item(), self.current_train_step
        )

        for metric in self.metric_list:
            metric_value = metric(
                (model_output + 1.0) / 2.0,
                (gt + 1.0) / 2.0,
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
        self.model.eval()
        self.val_tracker.reset()
        for iter_idx, data_per_batch in enumerate(self.val_dataloader):
            (
                model_input_list,
                gt,
                show_img_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            ) = self.before_val_iter(data_per_batch)
            self.val_iter(
                iter_idx,
                model_input_list,
                gt,
                show_img_list,
                key,
                dataset_name,
                normalize_scale,
                normalize_mode,
            )
            self.current_val_step += 1
        msg = f'val epoch: {self.current_epoch}'
        for key, value in self.val_tracker.results.items():
            msg = msg + f', {key}: {value:.4f}'
            self.backend_logger.add_scalar(f'val/{key}', value, self.current_epoch)
        self.txt_logger.info(msg)

    def before_val_iter(self, data_per_batch):
        model_input_list = self.get_model_input(data_per_batch)
        gt = data_per_batch['fine_img_02'].to('cuda')

        show_img_list = self.get_img_show_list(data_per_batch)

        key = data_per_batch['key'][0]
        dataset_name = data_per_batch['dataset_name'][0]
        normalize_scale = data_per_batch['normalize_scale'][0].numpy()
        normalize_mode = data_per_batch['normalize_mode'][0].numpy()
        return (
            model_input_list,
            gt,
            show_img_list,
            key,
            dataset_name,
            normalize_scale,
            normalize_mode,
        )

    def get_img_show_list(self, data_per_batch):
        coarse_img_01 = data_per_batch['coarse_img_01']
        coarse_img_02 = data_per_batch['coarse_img_02']
        fine_img_01 = data_per_batch['fine_img_01']
        fine_img_02 = data_per_batch['fine_img_02']
        img_show_list = [
            coarse_img_01,
            coarse_img_02,
            fine_img_01,
            fine_img_02,
        ]

        return img_show_list

    def val_iter(
        self,
        iter_idx,
        model_input_list,
        gt,
        show_img_list,
        key,
        dataset_name,
        normalize_scale,
        normalize_mode,
    ):
        msg = f'val epoch: {self.current_epoch}, iter: {iter_idx}'
        with torch.no_grad():
            model_output = self.model(*model_input_list)
            pixel_loss = self.pixle_loss(model_output, gt)
            msssim_loss = self.msssim_loss(model_output, gt)
            loss = pixel_loss + msssim_loss

        self.val_tracker.update('pixel_loss', pixel_loss.item())
        self.val_tracker.update('msssim_loss', msssim_loss.item())
        self.val_tracker.update('loss', loss.item())

        self.backend_logger.add_scalar(
            f'val_running/pixel_loss', pixel_loss.item(), self.current_val_step
        )
        self.backend_logger.add_scalar(
            f'val_running/msssim_loss', msssim_loss.item(), self.current_val_step
        )
        self.backend_logger.add_scalar(
            f'val_running/loss', loss.item(), self.current_val_step
        )

        msg = (
            msg
            + f', pixel_loss: {pixel_loss.item():.4e}, msssim_loss: {msssim_loss.item():.4e}, loss: {loss.item():.4e}'
        )

        for metric in self.metric_list:
            metric_value = metric(
                (model_output + 1.0) / 2.0,
                (gt + 1.0) / 2.0,
            )
            metric_name = metric.__name__
            msg = msg + f', {metric_name}: {metric_value.item():.4f}'
            self.val_tracker.update(metric_name, metric_value.item())
            self.backend_logger.add_scalar(
                f'val_running/{metric_name}', metric_value.item(), self.current_val_step
            )
        self.txt_logger.info(msg)

        ## TODO

        save_dir_prefix = f'{dataset_name}/{self.current_epoch}/save_img'
        save_dir_path = self.train_imgs_dir / save_dir_prefix
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

        show_dir_prefix = f'{dataset_name}/{self.current_epoch}/show_img/fine_img'
        show_dir_path = self.train_imgs_dir / show_dir_prefix
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

    def save_checkpoint(self):
        torch.save(
            self.model.state_dict(),
            self.train_checkpoints_dir / f'model_epoch_{self.current_epoch}.pth',
        )
