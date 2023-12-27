from functools import partial

from torch.utils.data import DataLoader
from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.model.starfm import STARFM
from src.metrics import *


patch_size = 120
patch_stride = 50
window_size = 51
virtual_patch_size = patch_size + window_size - 1


dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img_01='L1',
        fine_img_02='L2',
        coarse_img_01='M1',
        coarse_img_02='M2',
    ),
    data_name_tmpl_dict=dict(
        fine_img_01='{}',
        fine_img_02='{}',
        coarse_img_01='{}',
        coarse_img_02='{}',
    ),
    is_serialize_data=True,
)

test_transforms_key_list = [
    'fine_img_01',
    'fine_img_02',
    'coarse_img_01',
    'coarse_img_02',
]
test_transforms_list = [
    LoadData(key_list=test_transforms_key_list),
    Nan2Zero(key_list=test_transforms_key_list),
    RescaleToZeroOne(key_list=test_transforms_key_list, data_range=[0, 1]),
    Format(key_list=test_transforms_key_list),
]
test_dataset = dataset_cls_func(
    dataset_name='STIL',
    data_root='data/spatio_temporal_fusion/Mini_STIF_Dataset/STIL_Test',
    transform_func_list=test_transforms_list,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=1,
    sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=False, seed=42),
)

model = STARFM(window_size=window_size, patch_size=patch_size, num_classes=4)

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

patch_info_dict = dict(
    patch_size=patch_size,
    patch_stride=patch_stride,
    window_size=window_size,
)


__all__ = ['test_dataloader', 'model', 'metric_list', 'patch_info_dict']
