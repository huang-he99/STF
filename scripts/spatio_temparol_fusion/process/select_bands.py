from rasterio import band
from scripts.spatio_temparol_fusion.process import crop
import src
from src.utils.img.process.linear_stretch import truncated_linear_stretch
import argparse
import tifffile as tiff
from pathlib import Path
from scripts.spatio_temparol_fusion.constant import *
from tqdm import tqdm
from typing import Any
from torch.nn.modules.utils import _pair


# HWC
def select_bands(
    src_img,
    band_list,
):
    band_idx_list = [band_idx - 1 for band_idx in band_list]
    return src_img[:, :, band_idx_list]


def select_bands_via_path(
    src_img_path,
    tar_data_dir_path,
    band_list,
):
    src_img_path = Path(src_img_path)
    tar_data_dir_path = Path(tar_data_dir_path)
    src_img_name = src_img_path.name
    tar_img_name = src_img_name

    src_img = tiff.imread(src_img_path)
    tar_img = select_bands(src_img, band_list)
    tar_img_path = tar_data_dir_path / tar_img_name
    return tar_img_path, tar_img


# python -m scripts.spatio_temparol_fusion.process.select_bands --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data --tar_data_prefix public_processing_data/format_data/band_{} --band_list 4 3 2

# python -m scripts.spatio_temparol_fusion.process.select_bands --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2 --tar_data_prefix public_processing_data/format_data/linear_stretch_percent_2/band_{} --band_list 4 3 2

# python -m scripts.spatio_temparol_fusion.process.select_bands --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{} --tar_data_prefix public_processing_data/format_data/linear_stretch_percent_2/crop_{}_{}_{}_{}/band_{} --band_list 4 3 2

# python -m scripts.spatio_temparol_fusion.process.select_bands --root_path data/spatio_temporal_fusion --src_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{} --tar_data_prefix public_processing_data/format_data/crop_{}_{}_{}_{}/band_{} --band_list 4 3 2
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='format data')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--src_data_prefix', type=str, required=True)
    parser.add_argument('--tar_data_prefix', type=str, required=True)
    parser.add_argument('--band_list', type=int, nargs='+', required=True)
    args = parser.parse_args()

    root_path = Path(args.root_path)
    src_data_prefix_tmpl = args.src_data_prefix
    tar_data_prefix_tmpl = args.tar_data_prefix
    band_list = args.band_list

    print('select_bands')

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
                    crop_top,
                    crop_bottom,
                    crop_left,
                    crop_right,
                    '-'.join(list(map(str, band_list))),
                )
            else:
                src_data_prefix = src_data_prefix_tmpl
                tar_data_prefix = tar_data_prefix_tmpl.format(
                    '-'.join(list(map(str, band_list)))
                )
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
                tar_data_path, tar_data = select_bands_via_path(
                    src_data_path, tar_data_dir_path, band_list
                )
                tiff.imwrite(tar_data_path, tar_data)
