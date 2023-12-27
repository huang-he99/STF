from functools import partial

from torch.utils.data import DataLoader
from src.data.dataloader.data_sampler import EpochBasedSampler
from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.model.starfm import STARFM
from src.metrics import *


patch_size = 150
patch_stride = 50
window_size = 51
virtual_patch_size = patch_size + window_size - 1

test_transforms_key_list = [
    'fine_img_01',
    'fine_img_02',
    'fine_img_03',
    'coarse_img_01',
    'coarse_img_02',
    'coarse_img_03',
]
test_transforms_list = [
    LoadData(key_list=test_transforms_key_list),
    RescaleToZeroOne(key_list=test_transforms_key_list, data_range=[0, 255]),
    Format(key_list=test_transforms_key_list),
]
test_dataset = SpatioTemporalFusionDataset(
    dataset_name='CIA',
    data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-8/test/patch',
    data_prefix_tmpl_dict=dict(
        fine_img_01='Landsat_01',
        fine_img_02='Landsat_02',
        fine_img_03='Landsat_03',
        coarse_img_01='MODIS_01',
        coarse_img_02='MODIS_02',
        coarse_img_03='MODIS_03',
    ),
    data_name_tmpl_dict=dict(
        fine_img_01='{}_L_{}',
        fine_img_02='{}_L_{}',
        fine_img_03='{}_L_{}',
        coarse_img_01='{}_M_{}',
        coarse_img_02='{}_M_{}',
        coarse_img_03='{}_M_{}',
    ),
    is_serialize_data=True,
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
