from src.utils.img.process.linear_stretch import truncated_linear_stretch
import argparse
import tifffile as tiff
from pathlib import Path
from scripts.spatio_temparol_fusion.constant import *
from tqdm import tqdm
from typing import Any


def crop_img(src_img, crop_size, crop_shift):
    crop_size_h, crop_size_w = crop_size
    crop_shift_h, crop_shift_w = crop_shift
    tar_img = src_img[
        crop_shift_h : crop_shift_h + crop_size_h,
        crop_shift_w : crop_shift_w + crop_size_w,
        :,
    ]
    return tar_img


def crop_img_via_path(src_img_path, tar_data_dir_path, is_crop, crop_size, crop_shift):
    src_img_path = Path(src_img_path)
    tar_data_dir_path = Path(tar_data_dir_path)
    src_img_name = src_img_path.name
    tar_img_name = src_img_name

    src_img = tiff.imread(src_img_path)
    if is_crop:
        tar_img = crop_img(src_img, crop_size, crop_shift)
    else:
        tar_img = src_img
    tar_img_path = tar_data_dir_path / tar_img_name
    return tar_img_path, tar_img


# python -m  scripts.spatio_temparol_fusion.process.crop --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}

# python -m  scripts.spatio_temparol_fusion.process.crop --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2 --tar_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='format data')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--src_data_prefix', type=str, required=True)
    parser.add_argument('--tar_data_prefix', type=str, required=True)
    args = parser.parse_args()

    root_path = Path(args.root_path)
    src_data_prefix = args.src_data_prefix
    tar_data_prefix_tmpl = args.tar_data_prefix

    print('crop')
    for dataset_type in DATASET_TYPE:
        crop_info = CROP_INFO[dataset_type]
        crop_shift = crop_info['crop_shift']
        crop_size = crop_info['crop_size']
        is_crop = crop_info['is_crop']

        crop_top = crop_shift[0]
        crop_bottom = crop_top + crop_size[0]
        crop_left = crop_shift[1]
        crop_right = crop_left + crop_size[1]
        tar_data_prefix = tar_data_prefix_tmpl.format(
            crop_top, crop_bottom, crop_left, crop_right
        )

        for sensor_type in SENSOR_TYPE:
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
                tar_data_path, tar_data = crop_img_via_path(
                    src_data_path, tar_data_dir_path, is_crop, crop_size, crop_shift
                )
                tiff.imwrite(tar_data_path, tar_data)
