from src.utils.img.process.linear_stretch import truncated_linear_stretch
import argparse
import tifffile as tiff
from pathlib import Path
from scripts.spatio_temparol_fusion.constant import *
from tqdm import tqdm
from typing import Any
from torch.nn.modules.utils import _pair


# HWC
def split_img(
    src_img,
    img_patch_size,
    stride,
):
    img_patch_size = _pair(img_patch_size)
    stride = _pair(stride)
    h, w = src_img.shape[:2]
    h_num = (
        (h - img_patch_size[0]) // stride[0] + 1
        if (h - img_patch_size[0]) % stride[0] == 0
        else (h - img_patch_size[0]) // stride[0] + 2
    )
    w_num = (
        (w - img_patch_size[1]) // stride[1] + 1
        if (w - img_patch_size[1]) % stride[1] == 0
        else (w - img_patch_size[1]) // stride[1] + 2
    )
    for h_index in range(h_num):
        for w_index in range(w_num):
            h_start = h_index * stride[0]
            w_start = w_index * stride[1]
            h_end = h_start + img_patch_size[0]
            w_end = w_start + img_patch_size[1]
            if h_end > h:
                h_start = h - img_patch_size[0]
                h_end = h
            if w_end > w:
                w_start = w - img_patch_size[1]
                w_end = w
            yield src_img[h_start:h_end, w_start:w_end, :], (
                h_start,
                h_end,
                w_start,
                w_end,
            )


def split_img_via_path(src_img_path, tar_data_dir_path, img_patch_size, stride):
    src_img_path = Path(src_img_path)
    tar_data_dir_path = Path(tar_data_dir_path)
    src_img_stem, src_img_suffix = src_img_path.stem, src_img_path.suffix
    tar_data_date_dir_path = tar_data_dir_path / src_img_stem
    tar_data_date_dir_path.mkdir(parents=True, exist_ok=True)
    data_sensor_type = src_img_stem[0]
    tar_data_name_tmpl = f'{data_sensor_type}' + r'_{}_{}_{}_{}' + f'{src_img_suffix}'
    src_img = tiff.imread(src_img_path)
    for tar_img_patch, (h_start, h_end, w_start, w_end) in split_img(
        src_img, img_patch_size, stride
    ):
        tar_data_name = tar_data_name_tmpl.format(h_start, h_end, w_start, w_end)
        tar_data_path = tar_data_date_dir_path / tar_data_name
        yield tar_data_path, tar_img_patch


# python -m scripts.spatio_temparol_fusion.process.split --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix public_processing_data/format_data/split_size_{}_stride_{} --img_patch_size 256 --stride 128

# python -m scripts.spatio_temparol_fusion.process.split --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{} --tar_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/split_size_{}_stride_{} --img_patch_size 256 --stride 128

# python -m scripts.spatio_temparol_fusion.process.split --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2 --tar_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2/split_size_{}_stride_{} --img_patch_size 256 --stride 128

# python -m scripts.spatio_temparol_fusion.process.split --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/band_4-3-2 --tar_data_prefix public_processing_data/format_data/band_4-3-2/split_size_{}_stride_{} --img_patch_size 256 --stride 128

# python -m scripts.spatio_temparol_fusion.process.split --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{} --tar_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/split_size_{}_stride_{} --img_patch_size 256 --stride 128

# python -m scripts.spatio_temparol_fusion.process.split --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/band_4-3-2 --tar_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/band_4-3-2/split_size_{}_stride_{} --img_patch_size 256 --stride 128

# python -m scripts.spatio_temparol_fusion.process.split --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2 --tar_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_4-3-2/split_size_{}_stride_{} --img_patch_size 256 --stride 256
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='format data')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--src_data_prefix', type=str, required=True)
    parser.add_argument('--tar_data_prefix', type=str, required=True)
    parser.add_argument('--img_patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    args = parser.parse_args()

    root_path = Path(args.root_path)
    src_data_prefix_tmpl = args.src_data_prefix
    tar_data_prefix_tmpl = args.tar_data_prefix
    img_patch_size = args.img_patch_size
    stride = args.stride

    print('split')

    for dataset_type in DATASET_TYPE:
        for sensor_type in SENSOR_TYPE:
            if 'crop' in src_data_prefix_tmpl:
                crop_info = CROP_INFO[dataset_type]
                crop_shift = crop_info['crop_shift']
                crop_size = crop_info['crop_size']
                crop_top = crop_shift[0]
                crop_bottom = crop_top + crop_size[0]
                crop_left = crop_shift[1]
                crop_right = crop_left + crop_size[1]
                src_data_prefix = src_data_prefix_tmpl.format(
                    crop_top, crop_bottom, crop_left, crop_right
                )
                tar_data_prefix = tar_data_prefix_tmpl.format(
                    crop_top, crop_bottom, crop_left, crop_right, img_patch_size, stride
                )
            else:
                src_data_prefix = src_data_prefix_tmpl
                tar_data_prefix = tar_data_prefix_tmpl.format(img_patch_size, stride)
            src_data_dir_path = (
                root_path / dataset_type / src_data_prefix / f'original' / sensor_type
            )
            tar_data_dir_path = (
                root_path / dataset_type / tar_data_prefix / 'original' / sensor_type
            )
            tar_data_dir_path.mkdir(parents=True, exist_ok=True)
            src_data_path_list = list(src_data_dir_path.glob('*.tif'))
            pbar = tqdm(src_data_path_list)
            for data_index, src_data_path in enumerate(pbar):
                pbar.set_description(
                    f'format {dataset_type} {sensor_type}: {src_data_path.name} {data_index + 1}/{len(src_data_path_list)}'
                )
                for tar_data_path, tar_data in split_img_via_path(
                    src_data_path,
                    tar_data_dir_path,
                    img_patch_size,
                    stride,
                ):
                    tiff.imwrite(tar_data_path, tar_data)
