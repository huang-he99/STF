import collections.abc
import multiprocessing as mp
from itertools import repeat
from turtle import st
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

import numpy as np
import torch
from numpy import pad
from sklearn import linear_model
from torch import nn
from torch.nn.modules.utils import _pair, _quadruple

from src.model.spstfm1.ksvd import KSVD
from src.model.spstfm1.omp import orthogonal_matching_pursuit_matrix
from src.utils import EPSILON


class SPSTFM(object):
    # processing all bands via multiprocessing
    def __init__(
        self,
        sample_num,
        atom_num,
        max_iter,
        init_method='data_elements',
        patch_size=7,
        stride=3,
        sparsity=3,
        temporal_weight_threshold=0.2,
    ):
        super().__init__()
        self.kernel_size = patch_size
        self.sample_dim = patch_size**2
        self.stride = stride
        self.unflod = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.sample_num = sample_num
        self.atom_num = atom_num
        self.ksvd = KSVD(
            atom_num=atom_num,
            init_method=init_method,
            sparsity=sparsity,
            max_iter=max_iter,
        )
        self.sparisty = sparsity
        self.temporal_weight_threshold = temporal_weight_threshold

    def training_dictionary_pair(
        self,
        coarse_img_01,
        coarse_img_03,
        fine_img_01,
        fine_img_03,
    ):
        r"""
        Args:
            coarse_img_01: (1,c,h,w) tensor
            coarse_img_03: (1,c,h,w) tensor
            fine_img_01: (1,c,h,w) tensor
            fine_img_03: (1,c,h,w) tensor
        """

        channel_num = coarse_img_01.shape[1]

        img_samples = self.img_preprocessing_train_phase(
            coarse_img_01,
            coarse_img_03,
            fine_img_01,
            fine_img_03,
        )
        # ctx = torch.multiprocessing.get_context("spawn")
        with mp.Pool(processes=channel_num) as pool:
            dictionary_pair_list = pool.starmap(
                self._training_dictionary_pair,
                (torch.split(img_samples, 1, dim=0)),
            )
        coarse_dictionary_list = [result[0] for result in dictionary_pair_list]
        fine_dictionary_list = [result[1] for result in dictionary_pair_list]
        sparse_matrix_list = [result[2] for result in dictionary_pair_list]
        coarse_dictionary = torch.stack(coarse_dictionary_list, axis=0)
        fine_dictionary = torch.stack(fine_dictionary_list, axis=0)
        sparse_matrix = torch.stack(sparse_matrix_list, axis=0)
        return coarse_dictionary, fine_dictionary, sparse_matrix

    def img_preprocessing_train_phase(
        self,
        coarse_img_01,
        coarse_img_03,
        fine_img_01,
        fine_img_03,
    ):
        r"""
        Args:
            coarse_img_01: (1,c,h,w) tensor
            coarse_img_03: (1,c,h,w) tensor
            fine_img_01: (1,c,h,w) tensor
            fine_img_03: (1,c,h,w) tensor
        Output:
            img_samples: (c,HW,sample_num) tensor
        """
        coarse_diff_31 = coarse_img_03 - coarse_img_01  # (1,b,h,w) tensor
        fine_diff_31 = fine_img_03 - fine_img_01  # (1,b,h,w) tensor
        (
            coarse_diff_31_mean,
            coarse_diff_31_std,
        ) = self.get_img_statistic(coarse_diff_31)
        standardized_coarse_diff_31 = (coarse_diff_31 - coarse_diff_31_mean) / (
            8 * coarse_diff_31_std + EPSILON
        )  # (1,b,h,w) tensor
        standardized_fine_diff_31 = (fine_diff_31 - coarse_diff_31_mean) / (
            8 * coarse_diff_31_std + EPSILON
        )  # (1,b,h,w) tensor
        coarse_img_patch_vectors = self.get_patch_vectors(standardized_coarse_diff_31)
        fine_img_patch_vectors = self.get_patch_vectors(standardized_fine_diff_31)
        img_samples = self.get_training_sample(
            coarse_img_patch_vectors, fine_img_patch_vectors
        )
        return img_samples

    def get_img_statistic(self, img):
        r"""
        Args:
            img: (1,c,h,w) tensor
        Output:

            img_mean: (1,c,1,1) tensor
            img_std: (1,c,1,1) tensor
        """
        img_mean = torch.mean(img, dim=(0, 2, 3), keepdim=True)
        img_std = torch.std(img, dim=(0, 2, 3), keepdim=True)
        return img_mean, img_std

    def get_patch_vectors(self, img_tensor):
        r"""
        Args:
            img_tensor: (1,c,h,w) tensor
        Output:
            img_patch_vectors: (c,HW,L) tensor
        """
        _, c, _, _ = img_tensor.shape
        img_patch_vectors = self.unflod(img_tensor).squeeze(0)
        img_patch_vectors = img_patch_vectors.reshape(
            c, -1, img_patch_vectors.shape[-1]
        )
        return img_patch_vectors

    def get_training_sample(self, coarse_img_patch_vectors, fine_img_patch_vectors):
        r"""
        Args:
            coarse_img_patch_vectors: (c,HW,L) tensor
            fine_img_patch_vectors: (c,HW,L) tensor
        Output:
            img_samples: (c,2*HW,sample_num) tensor
        """
        patch_num = coarse_img_patch_vectors.shape[-1]
        indices = torch.randperm(patch_num)[: self.sample_num]
        coarse_img_samples = coarse_img_patch_vectors[..., indices]
        fine_img_samples = fine_img_patch_vectors[..., indices]
        img_samples = torch.cat((coarse_img_samples, fine_img_samples), dim=1)
        return img_samples

    def _training_dictionary_pair(self, img_samples):
        r"""
        Args:
            img_samples: (1,2*HW,sample_num) tensor
        Output:
            coarse_dictionary: (L,atom_num) tensor
            fine_dictionary: (L,atom_num) tensor
            sparsity_matrix: (atom_num,sample_num) tensor
        """
        img_samples = img_samples.squeeze(0)  # (2*HW,sample_num) tensor
        dictionary_matrix, sparse_matrix = self.ksvd.fit(img_samples)
        coarse_dictionary = dictionary_matrix[: self.sample_dim, :]
        fine_dictionary = dictionary_matrix[self.sample_dim :, :]
        return coarse_dictionary, fine_dictionary, sparse_matrix

    def reconstruction(
        self,
        coarse_img_01,
        coarse_img_02,
        coarse_img_03,
        fine_img_01,
        fine_img_03,
        coarse_dictionary,
        fine_dictionary,
    ):
        channel_num = coarse_img_01.shape[1]

        (
            coarse_img_patch_vectors_21,
            coarse_img_patch_vectors_32,
            fine_img_patch_vectors_01,
            fine_img_patch_vectors_03,
            coarse_diff_21_mean,
            coarse_diff_21_std,
            coarse_diff_32_mean,
            coarse_diff_32_std,
        ) = self.img_preprocessing_inference_phase(
            coarse_img_01,
            coarse_img_02,
            coarse_img_03,
            fine_img_01,
            fine_img_03,
        )

        # self.HRDI_reconstruction(
        #     coarse_img_patch_vectors_21[0:1],
        #     coarse_img_patch_vectors_32[0:1],
        #     coarse_dictionary[0:1],
        #     fine_dictionary[0:1],
        # )

        with mp.Pool(processes=channel_num) as pool:
            fine_diff_img_patch_vectors_list = pool.starmap(
                self.HRDI_reconstruction,
                (
                    zip(
                        torch.split(coarse_img_patch_vectors_21[0:1], 1, dim=0),
                        torch.split(coarse_img_patch_vectors_32[0:1], 1, dim=0),
                        torch.split(coarse_dictionary[0:1], 1, dim=0),
                        torch.split(fine_dictionary[0:1], 1, dim=0),
                    )
                ),
            )

        fine_diff_img_patch_vectors_21_list = [
            result[0] for result in fine_diff_img_patch_vectors_list
        ]
        fine_diff_img_patch_vectors_32_list = [
            result[1] for result in fine_diff_img_patch_vectors_list
        ]

        fine_diff_img_patch_vectors_21 = torch.stack(
            fine_diff_img_patch_vectors_21_list, axis=0
        )  # (c,pHpW,L)
        fine_diff_img_patch_vectors_32 = torch.stack(
            fine_diff_img_patch_vectors_32_list, axis=0
        )  # (c,pHpW,L)

        fine_diff_img_patch_vectors_21 = (
            fine_diff_img_patch_vectors_21 * 8 * coarse_diff_21_std
            + coarse_diff_21_mean
        )
        fine_diff_img_patch_vectors_32 = (
            fine_diff_img_patch_vectors_32 * 8 * coarse_diff_32_std
            + coarse_diff_32_mean
        )

        temporal_weight = self.get_temporal_weight(
            coarse_img_01,
            coarse_img_02,
            coarse_img_03,
        )

        fine_img_patch_vectors_02 = temporal_weight * (
            fine_img_patch_vectors_01 + fine_diff_img_patch_vectors_21
        ) + (1 - temporal_weight) * (
            fine_img_patch_vectors_03 - fine_diff_img_patch_vectors_32
        )  # (c,pHpW,L)

        _, _, h, w = fine_img_01.shape
        fine_img_02 = self.patch_to_img(fine_img_patch_vectors_02, out_size=(h, w))

        return fine_img_02

    def img_preprocessing_inference_phase(
        self,
        coarse_img_01,
        coarse_img_02,
        coarse_img_03,
        fine_img_01,
        fine_img_03,
    ):
        r"""
        Args:
            coarse_img_01: (1,c,h,w) tensor
            coarse_img_02: (1,c,h,w) tensor
            coarse_img_03: (1,c,h,w) tensor
            fine_img_01: (1,c,h,w) tensor
            fine_img_03: (1,c,h,w) tensor
        Output:
            coarse_img_patch_vectors_21: (c,HW,L) tensor
            coarse_img_patch_vectors_32: (c,HW,L) tensor
            fine_img_patch_vectors_01: (c,HW,L) tensor
            fine_img_patch_vectors_03: (c,HW,L) tensor
        """
        coarse_diff_21 = coarse_img_02 - coarse_img_01  # (1,c,h,w) tensor
        coarse_diff_32 = coarse_img_03 - coarse_img_02  # (1,c,h,w) tensor
        (
            coarse_diff_21_mean,
            coarse_diff_21_std,
        ) = self.get_img_statistic(coarse_diff_21)
        standardized_coarse_diff_21 = (coarse_diff_21 - coarse_diff_21_mean) / (
            8 * coarse_diff_21_std + EPSILON
        )
        (
            coarse_diff_32_mean,
            coarse_diff_32_std,
        ) = self.get_img_statistic(coarse_diff_32)
        standardized_fine_diff_32 = (coarse_diff_32 - coarse_diff_32_mean) / (
            8 * coarse_diff_32_std + EPSILON
        )
        coarse_img_patch_vectors_21 = self.get_patch_vectors(
            standardized_coarse_diff_21
        )
        coarse_img_patch_vectors_32 = self.get_patch_vectors(standardized_fine_diff_32)

        fine_img_patch_vectors_01 = self.get_patch_vectors(fine_img_01)
        fine_img_patch_vectors_03 = self.get_patch_vectors(fine_img_03)

        return (
            coarse_img_patch_vectors_21,
            coarse_img_patch_vectors_32,
            fine_img_patch_vectors_01,
            fine_img_patch_vectors_03,
            coarse_diff_21_mean.squeeze(0),
            coarse_diff_21_std.squeeze(0),
            coarse_diff_32_mean.squeeze(0),
            coarse_diff_32_std.squeeze(0),
        )

    def HRDI_reconstruction(
        self,
        coarse_img_patch_vectors_21,
        coarse_img_patch_vectors_32,
        coarse_dictionary,
        fine_dictionary,
    ):
        r"""
        Args:
            coarse_img_patch_vectors_21: (1,HW,L) tensor
            coarse_img_patch_vectors_32: (1,HW,L) tensor
            fine_img_patch_vectors_01: (1,HW,L) tensor
            fine_img_patch_vectors_03: (1,HW,L) tensor
            coarse_dictionary: (1,atom_dim,atom_num) tensor
            fine_dictionary: (1,atom_dim,atom_num) tensor
        Output:
            reconstructed_fine_img: (HW,L) tensor
        """
        coarse_img_patch_vectors_21 = coarse_img_patch_vectors_21.squeeze(0)
        coarse_img_patch_vectors_32 = coarse_img_patch_vectors_32.squeeze(0)
        coarse_dictionary = coarse_dictionary.squeeze(0)
        fine_dictionary = fine_dictionary.squeeze(0)

        sparse_matrix_21 = orthogonal_matching_pursuit_matrix(
            coarse_img_patch_vectors_21,
            coarse_dictionary,
            sparsity=self.sparisty,
        )
        sparse_matrix_32 = orthogonal_matching_pursuit_matrix(
            coarse_img_patch_vectors_32,
            coarse_dictionary,
            sparsity=self.sparisty,
        )

        fine_img_patch_vectors_21 = fine_dictionary @ sparse_matrix_21
        fine_img_patch_vectors_32 = fine_dictionary @ sparse_matrix_32
        return (
            fine_img_patch_vectors_21,
            fine_img_patch_vectors_32,
        )

    def get_temporal_weight(self, coarse_img_01, coarse_img_03, coarse_img_02):
        r"""
        Args:
            coarse_img_01: (1,c,h,w) tensor
            coarse_img_03: (1,c,h,w) tensor
            coarse_img_02: (1,c,h,w) tensor
        """
        BU_01 = self.cal_BU(coarse_img_01)
        BU_02 = self.cal_BU(coarse_img_02)
        BU_03 = self.cal_BU(coarse_img_03)

        dBU_21 = torch.abs(BU_02 - BU_01)
        dBU_32 = torch.abs(BU_03 - BU_02)

        dBU_21_patch_vectors = self.get_patch_vectors(dBU_21)  # (1,1,L)
        dBU_32_patch_vectors = self.get_patch_vectors(dBU_32)

        dBU_21_patch_vectors = (
            torch.mean(dBU_21_patch_vectors, dim=1, keepdim=True) + EPSILON
        )  # (1,1,L)
        dBU_32_patch_vectors = (
            torch.mean(dBU_32_patch_vectors, dim=1, keepdim=True) + EPSILON
        )

        dBU_31 = BU_03 - BU_01
        dBU_31_patch_vectors = self.get_patch_vectors(dBU_31)
        dBU_31_patch_vectors = torch.mean(dBU_31_patch_vectors, dim=1, keepdim=True)
        # dBU_31_patch_vectors = dBU_32_patch_vectors - dBU_21_patch_vectors
        temporal_weight = (1 / dBU_21_patch_vectors) / (
            1 / dBU_21_patch_vectors + 1 / dBU_32_patch_vectors
        )
        temporal_weight[dBU_31_patch_vectors > self.temporal_weight_threshold] = 1
        temporal_weight[dBU_31_patch_vectors < -self.temporal_weight_threshold] = 0

        return temporal_weight

    def cal_BU(self, img):
        r"""
        Args:
            img: (1,c,h,w) tensor
        Output:
            BU: (1,1,h,w) tensor
        """
        _, c, _, _ = img.shape
        if c == 3:
            NDVI = (img[:, 0] - img[:, 1]) / (img[:, 0] + img[:, 1] + 1e-6)
            NDBI = (img[:, 2] - img[:, 0]) / (img[:, 2] + img[:, 0] + 1e-6)
            BU = NDVI + NDBI
            # BU = NDBI - NDVI
        if c == 6:
            NDVI = (img[:, 3] - img[:, 2]) / (img[:, 3] + img[:, 2] + 1e-6)
            NDBI = (img[:, 4] - img[:, 3]) / (img[:, 4] + img[:, 3] + 1e-6)
            BU = NDVI + NDBI
            # BU = NDBI - NDVI
        return BU.unsqueeze(1)

    def patch_to_img(self, patch_vectors, out_size):
        r"""
        Args:
            patch_vectors: (c,pHpW,L) tensor
            out_size: (h,w) tuple
        Output:
            img: (1,c,h,w) tensor
        """
        patch_vectors = patch_vectors.reshape(1, -1, patch_vectors.shape[-1])
        img = nn.Fold(out_size, kernel_size=self.kernel_size, stride=self.stride)(
            patch_vectors
        )
        cnt = torch.ones_like(patch_vectors)
        cnt = nn.Fold(out_size, kernel_size=self.kernel_size, stride=self.stride)(cnt)
        cnt[cnt == 0] = 1
        img = img / cnt
        return img


# python -m src.model.spstfm.spstfm
if __name__ == '__main__':
    import random

    import numpy as np
    import torch
    import os
    import tifffile as tiff

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    rng_seed = 42
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)

    img_h, img_w = 250, 250
    channel = 6

    coarse_img_01 = tiff.imread(
        '/home/hh/container/code/fusion/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/train/MODIS_01/Group_01_M_768_1024_768_1024.tif'
    )
    coarse_img_02 = tiff.imread(
        '/home/hh/container/code/fusion/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/train/MODIS_02/Group_01_M_768_1024_768_1024.tif'
    )
    coarse_img_03 = tiff.imread(
        '/home/hh/container/code/fusion/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/train/MODIS_03/Group_01_M_768_1024_768_1024.tif'
    )
    fine_img_01 = tiff.imread(
        '/home/hh/container/code/fusion/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/train/Landsat_01/Group_01_L_768_1024_768_1024.tif'
    )
    fine_img_02 = tiff.imread(
        '/home/hh/container/code/fusion/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/train/Landsat_02/Group_01_L_768_1024_768_1024.tif'
    )
    fine_img_03 = tiff.imread(
        '/home/hh/container/code/fusion/data/spatio_temporal_fusion/CIA/private_data/syy_setting-9/train/Landsat_03/Group_01_L_768_1024_768_1024.tif'
    )

    coarse_img_01 = (
        torch.from_numpy(coarse_img_01.transpose(2, 0, 1)).float().unsqueeze(0)
    )
    coarse_img_02 = (
        torch.from_numpy(coarse_img_02.transpose(2, 0, 1)).float().unsqueeze(0)
    )
    coarse_img_03 = (
        torch.from_numpy(coarse_img_03.transpose(2, 0, 1)).float().unsqueeze(0)
    )
    fine_img_01 = torch.from_numpy(fine_img_01.transpose(2, 0, 1)).float().unsqueeze(0)
    fine_img_02 = torch.from_numpy(fine_img_02.transpose(2, 0, 1)).float().unsqueeze(0)
    fine_img_03 = torch.from_numpy(fine_img_03.transpose(2, 0, 1)).float().unsqueeze(0)

    patch_size = 7
    sample_num = 20000
    atom_num = 512
    max_iter = 66

    spstfm = SPSTFM(
        sample_num=sample_num,
        atom_num=atom_num,
        max_iter=max_iter,
        init_method='data_elements',
        patch_size=patch_size,
        stride=3,
        sparsity=3,
    )

    coarse_dictionary, fine_dictionary, sparse_matrix = spstfm.training_dictionary_pair(
        coarse_img_01, coarse_img_03, fine_img_01, fine_img_03
    )

    fine_img_02_reconstructed = spstfm.reconstruction(
        coarse_img_01,
        coarse_img_02,
        coarse_img_03,
        fine_img_01,
        fine_img_03,
        coarse_dictionary,
        fine_dictionary,
    )

    print(fine_img_02_reconstructed.shape)

    # output = spstfm(coarse_img_01, coarse_img_03, fine_img_01, fine_img_03)
