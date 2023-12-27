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

from src.model.spstfm.ksvd import KSVD
from src.utils import EPSILON


class SPSTFM(object):
    # processing all bands via multiprocessing
    def __init__(
        self,
        sample_dim,
        sample_num,
        atom_num,
        max_iter,
        init_method='data_elements',
        patch_size=7,
        stride=3,
        sparsity=3,
    ):
        super().__init__()
        assert sample_dim == patch_size**2
        self.kernel_size = _pair(patch_size)
        self.stride = _pair(stride)
        self.unflod = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.sample_dim = sample_dim
        self.sample_num = sample_num
        self.atom_num = atom_num
        self.ksvd = KSVD(
            n_components=atom_num,
            init_method=init_method,
            sparsity=sparsity,
            max_iter=max_iter,
        )
        self.sparisty = sparsity

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

        with mp.Pool(processes=channel_num) as pool:
            dictionary_pair_list = pool.starmap(
                self.training_dictionary_pair_per_channel,
                (
                    zip(
                        torch.split(coarse_img_01, 1, dim=1),
                        torch.split(coarse_img_03, 1, dim=1),
                        torch.split(fine_img_01, 1, dim=1),
                        torch.split(fine_img_03, 1, dim=1),
                    )
                ),
            )
        coarse_diff_dictionary_list = [result[0] for result in dictionary_pair_list]
        fine_diff_dictionary_list = [result[1] for result in dictionary_pair_list]
        sparsity_matrix_list = [result[2] for result in dictionary_pair_list]
        coarse_diff_dictionary = np.stack(coarse_diff_dictionary_list, axis=0)
        fine_diff_dictionary = np.stack(fine_diff_dictionary_list, axis=0)
        sparsity_matrix = np.stack(sparsity_matrix_list, axis=0)
        return coarse_diff_dictionary, fine_diff_dictionary, sparsity_matrix

    def training_dictionary_pair_per_channel(
        self, coarse_img_01, coarse_img_03, fine_img_01, fine_img_03
    ):
        coarse_diff_13 = coarse_img_01 - coarse_img_03  # (1,1,h,w) tensor
        fine_diff_13 = fine_img_01 - fine_img_03  # (1,1,h,w) tensor

        (
            standardized_coarse_diff_13,
            coarse_diff_13_mean,
            coarse_diff_13_std,
        ) = self.standardization(coarse_diff_13)

        standardized_fine_diff_13 = (fine_diff_13 - coarse_diff_13_mean) / (
            8 * coarse_diff_13_std + EPSILON
        )

        # (
        #     standardized_fine_diff_13,
        #     fine_diff_13_mean,
        #     fine_diff_13_std,
        # ) = self.standardization(fine_diff_13)

        standardized_coarse_diff_13_patch_vectors = self.get_patch_vectors(
            standardized_coarse_diff_13, is_selected=True
        )
        standardized_fine_diff_13_patch_vectors = self.get_patch_vectors(
            standardized_fine_diff_13, is_selected=True
        )

        coarse_diff_dictionary, sparsity_matrix = self.ksvd.fit(
            standardized_coarse_diff_13_patch_vectors
        )
        fine_diff_dictionary = standardized_fine_diff_13_patch_vectors @ np.linalg.pinv(
            sparsity_matrix
        )
        return coarse_diff_dictionary, fine_diff_dictionary, sparsity_matrix

    def standardization(self, img):
        r"""
        Args:
            img: (1,1,h,w) tensor
        """
        img_mean = torch.mean(img).item()  # scalar
        img_std = torch.std(img).item()  # scalar
        standardized_img = (img - img_mean) / (
            8 * img_std + EPSILON
        )  # (1,1,h,w) tensor
        return standardized_img, img_mean, img_std

    def get_patch_vectors(self, img, is_selected=False):
        # img: b,c,h,w  tensor
        img_patch_vectors = self.unflod(img)
        img_patch_vectors = img_patch_vectors.squeeze(0)
        if is_selected:
            g = torch.Generator()
            g.manual_seed(42)
            patch_num = img_patch_vectors.shape[-1]
            indices = torch.randperm(patch_num, generator=g)[: self.sample_num]
            img_patch_vectors = img_patch_vectors[..., indices]
        return img_patch_vectors

    def reconstruction(
        self,
        coarse_img_01,
        coarse_img_02,
        coarse_img_03,
        fine_img_01,
        fine_img_03,
        coarse_diff_dictionary,
        fine_diff_dictionary,
    ):
        channel_num = coarse_img_01.shape[1]

        BU_01 = self.cal_BU(coarse_img_01)
        BU_02 = self.cal_BU(coarse_img_02)
        BU_03 = self.cal_BU(coarse_img_03)
        dBU_21 = torch.abs(BU_02 - BU_01).unsqueeze(1).repeat(1, channel_num, 1, 1)
        dBU_32 = torch.abs(BU_03 - BU_02).unsqueeze(1).repeat(1, channel_num, 1, 1)

        with mp.Pool(processes=channel_num) as pool:
            reconstructed_fine_img_list = pool.starmap(
                self.reconstruction_per_channel,
                (
                    zip(
                        torch.split(coarse_img_01, 1, dim=1),
                        torch.split(coarse_img_02, 1, dim=1),
                        torch.split(coarse_img_03, 1, dim=1),
                        torch.split(coarse_diff_dictionary, 1, dim=1),
                        torch.split(fine_diff_dictionary, 1, dim=1),
                        torch.split(fine_img_01, 1, dim=1),
                        torch.split(fine_img_03, 1, dim=1),
                        torch.split(dBU_21, 1, dim=1),
                        torch.split(dBU_32, 1, dim=1),
                    )
                ),
            )

        reconstructed_fine_img = torch.cat(reconstructed_fine_img_list, dim=1)
        return reconstructed_fine_img

    def cal_BU(self, img):
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
        return BU

    def reconstruction_per_channel(
        self,
        coarse_img_01,
        coarse_img_02,
        coarse_img_03,
        coarse_diff_dictionary,
        fine_diff_dictionary,
        fine_img_01,
        fine_img_03,
        dBU_21,
        dBU_32,
    ):
        (
            reconstruction_fine_diff_21_patch_vectors,
            reconstruction_fine_diff_32_patch_vectors,
        ) = self.HRDI_reconstruction(
            coarse_img_01,
            coarse_img_02,
            coarse_img_03,
            coarse_diff_dictionary,
            fine_diff_dictionary,
        )

        reconstructed_fine_img = self.img_reconstruction(
            fine_img_01,
            fine_img_03,
            dBU_21,
            dBU_32,
            reconstruction_fine_diff_21_patch_vectors,
            reconstruction_fine_diff_32_patch_vectors,
        )
        return reconstructed_fine_img

    def HRDI_reconstruction(
        self,
        coarse_img_01,
        coarse_img_02,
        coarse_img_03,
        coarse_diff_dictionary,
        fine_diff_dictionary,
    ):
        coarse_diff_21 = coarse_img_02 - coarse_img_01
        coarse_diff_32 = coarse_img_03 - coarse_img_02

        (
            standardized_coarse_diff_21,
            coarse_diff_21_std,
            coarse_diff_21_mean,
        ) = self.standardization(coarse_diff_21)

        (
            standardized_coarse_diff_32,
            coarse_diff_32_std,
            coarse_diff_32_mean,
        ) = self.standardization(coarse_diff_32)

        standardized_coarse_diff_21_patch_vectors = self.get_patch_vectors(
            standardized_coarse_diff_21
        )
        standardized_coarse_diff_32_patch_vectors = self.get_patch_vectors(
            standardized_coarse_diff_32
        )

        coarse_diff_dictionary = coarse_diff_dictionary.squeeze(0).squeeze(0).numpy()
        fine_diff_dictionary = fine_diff_dictionary.squeeze(0).squeeze(0).numpy()

        sparsity_matrix_21 = linear_model.orthogonal_mp(
            coarse_diff_dictionary,
            standardized_coarse_diff_21_patch_vectors,
            n_nonzero_coefs=self.sparisty,
        )
        sparisty_matrix_32 = linear_model.orthogonal_mp(
            coarse_diff_dictionary,
            standardized_coarse_diff_32_patch_vectors,
            n_nonzero_coefs=self.sparisty,
        )

        standardized_reconstruction_fine_diff_21_patch_vectors = (
            fine_diff_dictionary @ sparsity_matrix_21
        )
        standardized_reconstruction_fine_diff_32_patch_vectors = (
            fine_diff_dictionary @ sparisty_matrix_32
        )

        reconstruction_fine_diff_21_patch_vectors = (
            standardized_reconstruction_fine_diff_21_patch_vectors * coarse_diff_21_std
            + coarse_diff_21_mean
        )
        reconstruction_fine_diff_32_patch_vectors = (
            standardized_reconstruction_fine_diff_32_patch_vectors * coarse_diff_32_std
            + coarse_diff_32_mean
        )

        return (
            reconstruction_fine_diff_21_patch_vectors,
            reconstruction_fine_diff_32_patch_vectors,
        )

    def img_reconstruction(
        self,
        fine_img_01,
        fine_img_03,
        dBU_21,
        dBU_32,
        reconstruction_fine_diff_21_patch_vectors,
        reconstruction_fine_diff_32_patch_vectors,
        threshold=0.2,
    ):
        dBU_21_patch_vectors = self.get_patch_vectors(dBU_21)
        dBU_32_patch_vectors = self.get_patch_vectors(dBU_32)
        dBU_21_patch_vectors_mean = (
            np.mean(dBU_21_patch_vectors, axis=0, keepdims=True) + EPSILON
        )
        dBU_32_vectors_mean = (
            np.mean(dBU_32_patch_vectors, axis=0, keepdims=True) + EPSILON
        )
        dBU_vectors_mean_diff = dBU_32_vectors_mean - dBU_21_patch_vectors_mean
        w = (1 / dBU_21_patch_vectors_mean) / (
            1 / dBU_21_patch_vectors_mean + 1 / dBU_32_vectors_mean
        )
        w[dBU_vectors_mean_diff > threshold] = 1
        w[dBU_vectors_mean_diff < -threshold] = 0

        fine_img_01_patch_vectors = self.get_patch_vectors(fine_img_01)
        fine_img_03_patch_vectors = self.get_patch_vectors(fine_img_03)
        fine_img_02_patch_vectors = w * (
            fine_img_01_patch_vectors + reconstruction_fine_diff_21_patch_vectors
        ) + (1 - w) * (
            fine_img_03_patch_vectors - reconstruction_fine_diff_32_patch_vectors
        )
        _, _, h, w = fine_img_01.shape
        fine_img_02 = self.patch_samples_to_img(
            fine_img_02_patch_vectors, out_size=(h, w)
        )
        return fine_img_02

    def patch_samples_to_img(self, patch_vectors, out_size):
        patch_vectors = torch.from_numpy(patch_vectors).unsqueeze(0)
        img = torch.nn.Fold(out_size, kernel_size=self.kernel_size, stride=self.stride)(
            patch_vectors
        )
        cnt = torch.ones_like(patch_vectors)
        cnt = torch.nn.Fold(out_size, kernel_size=self.kernel_size, stride=self.stride)(
            cnt
        )
        cnt[cnt == 0] = 1
        img = img / cnt
        img = img
        return img.float()


# python -m src.model.spstfm.spstfm
if __name__ == '__main__':
    import sys

    sys.path.append('..')
    import multiprocessing as mp
    import os
    import random

    import numpy as np
    import torch

    from src.model.spstfm.ksvd import KSVD

    rng_seed = 42
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)

    img_h, img_w = 500, 500
    channel = 6

    coarse_img_01 = torch.randn(1, channel, img_h, img_w)
    coarse_img_02 = torch.randn(1, channel, img_h, img_w)
    coarse_img_03 = torch.randn(1, channel, img_h, img_w)
    fine_img_01 = torch.randn(1, channel, img_h, img_w)
    fine_img_02 = torch.randn(1, channel, img_h, img_w)
    fine_img_03 = torch.randn(1, channel, img_h, img_w)

    patch_size = 7
    sample_num = 2000
    atom_num = 256
    max_iter = 100

    spstfm = SPSTFM(
        sample_dim=patch_size**2,
        sample_num=sample_num,
        atom_num=atom_num,
        patch_size=patch_size,
        max_iter=max_iter,
        sparsity=3,
    )

    def work(coarse_img_01, coarse_img_03, fine_img_01, fine_img_03):
        # img_samples_per_channel = img_samples[i]
        # print(mp.current_process().name)
        (
            dictionary_matrix_per_channel,
            sparsity_matrix_per_channel,
        ) = spstfm.training_dictionary_pair(
            coarse_img_01, coarse_img_03, fine_img_01, fine_img_03
        )
        return dictionary_matrix_per_channel, sparsity_matrix_per_channel

    import time

    start = time.perf_counter()
    pool = mp.Pool(processes=1)
    results = pool.starmap(
        work,
        [
            (
                coarse_img_01[:, i],
                coarse_img_03[:, i],
                fine_img_01[:, i],
                fine_img_03[:, i],
            )
            for i in range(1)
        ],
    )
    end = time.perf_counter()
    print(end - start)

    # output = spstfm(coarse_img_01, coarse_img_03, fine_img_01, fine_img_03)
