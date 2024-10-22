from functools import partial

import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataloader.worker_init import worker_init_fn
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.metrics import *
from src.model.sr3 import GaussianDiffusion, UNet


dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img="Landsat_02",
        coarse_img="MODIS_02",
    ),
    data_name_tmpl_dict=dict(
        fine_img="{}_L_{}",
        coarse_img="{}_M_{}",
    ),
    is_serialize_data=True,
)

transforms_key_list = [
    "fine_img",
    "coarse_img",
]

test_transform_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000]),
    # Rotate(key_list=transforms_key_list),
    # Flip(key_list=transforms_key_list),
    Format(key_list=transforms_key_list),
]

test_dataset = dataset_cls_func(
    dataset_name="CIA",
    data_root="data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/full",
    transform_func_list=test_transform_list,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=True, seed=42),
    num_workers=0,
)


model = GaussianDiffusion(
    denoise_fn=UNet(
        in_channel=6,
        out_channel=3,
        inner_channel=64,
        channel_mults=[1, 2, 4, 8, 8],
        attn_res=(16,),
        res_blocks=2,
        dropout=0.2,
    ),
    image_size=128,
    channels=3,
    conditional=True,
    schedule_opt={
        "schedule": "linear",
        "n_timestep": 2000,
        "linear_start": 1e-6,
        "linear_end": 1e-2,
    },
)


checkpoint_path = "/home/hh/container/Pretrain_Model/sr3/I640000_E37_gen.pth"

metric_list = [
    RMSE(),
    MAE(),
    PSNR(max_value=1.0),
    SSIM(data_range=1.0),
    ERGAS(ratio=1.0 / 16.0),
    CC(),
    SAM(),
    UIQI(),
]

__all__ = [
    "test_dataloader",
    "checkpoint_path",
    "model",
    "metric_list",
]
