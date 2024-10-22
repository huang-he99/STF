from functools import partial

from torch.utils.data import DataLoader

from src.data.dataloader.data_sampler import EpochBasedSampler

from src.data.dataset import SpatioTemporalFusionDataset
from src.data.transforms import *
from src.metrics import *
from src.model.stfdiffusion.model_2 import Unet, GaussianDiffusion


dataset_cls_func = partial(
    SpatioTemporalFusionDataset,
    data_prefix_tmpl_dict=dict(
        fine_img_01='Landsat_01',
        fine_img_02='Landsat_02',
        coarse_img_01='MODIS_01',
        coarse_img_02='MODIS_02',
    ),
    data_name_tmpl_dict=dict(
        fine_img_01='{}_L_{}',
        fine_img_02='{}_L_{}',
        coarse_img_01='{}_M_{}',
        coarse_img_02='{}_M_{}',
    ),
    is_serialize_data=True,
)

transforms_key_list = [
    'fine_img_01',
    'fine_img_02',
    'coarse_img_01',
    'coarse_img_02',
]


test_transforms_list = [
    LoadData(key_list=transforms_key_list),
    RescaleToMinusOneOne(key_list=transforms_key_list, data_range=[0, 10000]),
    Format(key_list=transforms_key_list),
]

test_dataset = dataset_cls_func(
    dataset_name='CIA',
    data_root='data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/test/patch',
    transform_func_list=test_transforms_list,
)


test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=0,
    sampler=EpochBasedSampler(dataset=test_dataset, is_shuffle=False, seed=42),
)

model = GaussianDiffusion(Unet(32), image_size=256, objective='pred_x0')

checkpoint_path = 'results/stfdiff/syy_setting-9/model_2/CIA/Unet_32~Adam_1e-4/checkpoints/model_epoch_449.pth'

metric_list = [
    RMSE(),
    MAE(),
    PSNR(max_value=1.0),
    SSIM(data_range=1.0),
    ERGAS(ratio=1.0 / 16.0),
    CC(),
    SAM(),
    UIQI(data_range=1.0),
]


__all__ = [
    'test_dataloader',
    'model',
    'checkpoint_path',
    'metric_list',
]
