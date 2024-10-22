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
    tar_data_date_dir_path = tar_data_dir_path
    # tar_data_date_dir_path.mkdir(parents=True, exist_ok=True)
    data_sensor_type = src_img_stem
    tar_data_name_tmpl = f'{data_sensor_type}' + r'_{}_{}_{}_{}' + f'{src_img_suffix}'
    src_img = tiff.imread(src_img_path)
    for tar_img_patch, (h_start, h_end, w_start, w_end) in split_img(
        src_img, img_patch_size, stride
    ):
        tar_data_name = tar_data_name_tmpl.format(h_start, h_end, w_start, w_end)
        tar_data_path = tar_data_date_dir_path / tar_data_name
        yield tar_data_path, tar_img_patch


# python -m scripts.spatio_temparol_fusion.process.split --src_data_dir_path data/spatio_temporal_fusion/Mini_STIF_Dataset --tar_data_dir_path data/spatio_temporal_fusion/Mini_STIF_Dataset_split_size_{}_stride_{} --img_patch_size 256 --stride 256

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='format data')

    parser.add_argument('--src_data_dir_path', type=str, required=True)
    parser.add_argument('--tar_data_dir_path', type=str, required=True)
    parser.add_argument('--img_patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    args = parser.parse_args()

    src_data_dir_path = Path(args.src_data_dir_path)
    tar_dataset_dir_path = Path(
        args.tar_data_dir_path.format(args.img_patch_size, args.stride)
    )

    img_patch_size = args.img_patch_size
    stride = args.stride

    print('split')

    tar_dataset_dir_path.mkdir(parents=True, exist_ok=True)
    src_data_path_list = list(src_data_dir_path.rglob('*.tif'))

    pbar = tqdm(src_data_path_list)
    for data_index, src_data_path in enumerate(pbar):
        pbar.set_description(
            f'split: {src_data_path.name} {data_index + 1}/{len(src_data_path_list)}'
        )
        tar_data_dir_path = tar_dataset_dir_path.joinpath(*src_data_path.parts[-3:-1])
        tar_data_dir_path.mkdir(parents=True, exist_ok=True)
        for tar_data_path, tar_data in split_img_via_path(
            src_data_path,
            tar_data_dir_path,
            img_patch_size,
            stride,
        ):
            tiff.imwrite(tar_data_path, tar_data)
