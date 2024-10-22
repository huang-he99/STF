from numpy import pad
import torch
from torch import nn
import collections.abc
from itertools import repeat
from torch.nn.modules.utils import _pair, _quadruple
from typing import List, Tuple, Optional, overload, Dict, Any, Callable, Union
import numpy as np
from src.model.ssif.ksvd import KSVD
from sklearn import linear_model

epsilon = 1e-6


class SSIF(object):
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
        self, diff_fine_coarse_01, coarse_img_gradient_x, coarse_img_gradient_y
    ):
        diff_fine_coarse_01_patch_vector = self.get_patch_vector(
            diff_fine_coarse_01, is_selected=True
        )
        coarse_img_gradient_x_patch_vector = self.get_patch_vector(
            coarse_img_gradient_x, is_selected=True
        )
        coarse_img_gradient_y_patch_vector = self.get_patch_vector(
            coarse_img_gradient_y, is_selected=True
        )
        coarse_img_gradient_patch_vector = np.concatenate(
            coarse_img_gradient_x_samples, coarse_img_gradient_y_samples, axis=0
        )

        coarse_img_gradient_dictionary, sparsity_matrix = self.ksvd.fit(
            coarse_img_gradient_samples
        )
        diff_fine_coarse_01_dictionary = diff_fine_coarse_01_samples @ np.linalg.pinv(
            sparsity_matrix
        )
        return (
            coarse_img_gradient_dictionary,
            diff_fine_coarse_01_dictionary,
            sparsity_matrix,
        )

    def get_patch_vector(self, img, is_selected=False):
        # img: b,c,h,w  tensor
        img_patch = self.unflod(img)
        img_patch = img_patch.squeeze(0)

        if is_selected:
            g = torch.Generator()
            g.manual_seed(42)
            patch_num = img_patch.shape[-1]
            indices = torch.randperm(patch_num, generator=g)[: self.sample_num]
            patch_samples = img_patch[..., indices]
        else:
            patch_samples = img_patch
        return patch_samples.numpy()

    def predict_transition_imgs(
        self,
        coarse_img_01,
        coarse_img_02,
        fine_img_01,
        coarse_img_gradient_dictionary,
        diff_fine_coarse_01_dictionary,
    ):
        coarse_img_01_gradient = self.get_gradient(coarse_img_01)

        coarse_diff_21_samples = self.get_patch_samples(coarse_diff_21)
        coarse_diff_32_samples = self.get_patch_samples(coarse_diff_32)
        coarse_dictionary_matrix = (
            coarse_dictionary_matrix.squeeze(0).squeeze(0).numpy()
        )
        fine_dictionary_matrix = fine_dictionary_matrix.squeeze(0).squeeze(0).numpy()
        sparsity_21_matrix = linear_model.orthogonal_mp(
            coarse_dictionary_matrix,
            coarse_diff_21_samples,
            n_nonzero_coefs=self.sparisty,
        )
        sparsity_32_matrix = linear_model.orthogonal_mp(
            coarse_dictionary_matrix,
            coarse_diff_32_samples,
            n_nonzero_coefs=self.sparisty,
        )
        reconstruction_fine_21_samples = fine_dictionary_matrix @ sparsity_21_matrix
        reconstruction_fine_32_samples = fine_dictionary_matrix @ sparsity_32_matrix
        return reconstruction_fine_21_samples, reconstruction_fine_32_samples

    def img_reconstruction(
        self,
        fine_img_01,
        fine_img_03,
        dBU_21,
        dBU_32,
        coarse_diff_21_mean,
        coarse_diff_32_mean,
        coarse_diff_21_std,
        coarse_diff_32_std,
        fine_21_samples,
        fine_32_samples,
        threshold=0.2,
    ):
        dBU_21_samples = self.get_patch_samples(dBU_21)
        dBU_32_samples = self.get_patch_samples(dBU_32)
        dBU_21_samples_mean = np.mean(dBU_21_samples, axis=0, keepdims=True) + epsilon
        dBU_32_samples_mean = np.mean(dBU_32_samples, axis=0, keepdims=True) + epsilon
        dBU_samples_mean = dBU_32_samples_mean - dBU_21_samples_mean
        w = (1 / dBU_21_samples_mean) / (
            1 / dBU_21_samples_mean + 1 / dBU_32_samples_mean
        )
        w[dBU_samples_mean > threshold] = 1
        w[dBU_samples_mean < -threshold] = 0
        fine_img_01_samples = self.get_patch_samples(fine_img_01)
        fine_img_03_samples = self.get_patch_samples(fine_img_03)

        coarse_diff_21_mean = coarse_diff_21_mean.item()
        coarse_diff_32_mean = coarse_diff_32_mean.item()
        coarse_diff_21_std = coarse_diff_21_std.item()
        coarse_diff_32_std = coarse_diff_32_std.item()

        fine_21_samples = (fine_21_samples * coarse_diff_21_std) + coarse_diff_21_mean
        fine_32_samples = (fine_32_samples * coarse_diff_32_std) + coarse_diff_32_mean
        fine_img_02_samples = w * (fine_img_01_samples + fine_21_samples) + (1 - w) * (
            fine_img_03_samples - fine_32_samples
        )

        _, _, h, w = fine_img_01.shape
        fine_img_02 = self.patch_samples_to_img(fine_img_02_samples, out_size=(h, w))
        return fine_img_02

    def patch_samples_to_img(self, patch_samples, out_size):
        patch_samples = torch.from_numpy(patch_samples).unsqueeze(0)
        img = torch.nn.Fold(out_size, kernel_size=self.kernel_size, stride=self.stride)(
            patch_samples
        )
        cnt = torch.ones_like(patch_samples)
        cnt = torch.nn.Fold(out_size, kernel_size=self.kernel_size, stride=self.stride)(
            cnt
        )
        cnt[cnt == 0] = 1
        img = img / cnt
        img = img.squeeze(0)
        return img


# python -m src.model.spstfm.spstfm
if __name__ == '__main__':
    import sys

    sys.path.append('..')
    from src.model.spstfm.ksvd import KSVD
    import torch
    import os
    import numpy as np
    import random
    import multiprocessing as mp

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
