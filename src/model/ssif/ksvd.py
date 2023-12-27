from src.model.spstfm.omp import orthogonal_matching_pursuit_matrix
import torch
from torch import nn
import numpy as np
from sklearn import linear_model

# Y = DX
# class KSVD(nn.Module):
#     def __init__(
#         self,
#         n_components,
#         max_iter=10,
#         tol=1e-6,
#         sparsity=None,
#         init_method='random',
#         given_matrix=None,
#     ):
#         """
#         Parameters
#         ----------
#         n_components:
#             Number of dictionary elements

#         max_iter:
#             Maximum number of iterations

#         tol:
#             tolerance for error

#         sparsity:
#             Number of nonzero coefficients to target
#         """
#         super().__init__()
#         self.max_iter = max_iter
#         self.tol = tol
#         self.n_components = n_components
#         self.sparsity = sparsity if sparsity is not None else int(n_components * 0.1)
#         self.init_method = init_method
#         self.given_matrix = given_matrix

#     def _update_dict(self, measurment_matrix, dictionary_matrix, sparsity_matrix):
#         for j in range(self.n_components):
#             index = torch.nonzero(sparsity_matrix[j, :])[:, 0]
#             if len(index) == 0:
#                 continue
#             dictionary_matrix[:, j] = 0
#             residual = (measurment_matrix - dictionary_matrix @ sparsity_matrix)[
#                 :, index
#             ]
#             u, sigma, v = torch.linalg.svd(residual)
#             dictionary_matrix[:, j] = u[:, 0]
#             sparsity_matrix[j, index] = sigma[0] * v[0, :]
#         return dictionary_matrix, sparsity_matrix

#     def _initialize(self, measurment_matrix):
#         device = measurment_matrix.device
#         if self.init_method == 'random':
#             dictionary_matrix = torch.randn(
#                 measurment_matrix.shape[0], self.n_components, device=device
#             )
#         elif self.init_method == 'data_elements':
#             dictionary_matrix = measurment_matrix[:, : self.n_components]
#         elif self.init_method == 'svd':
#             pass
#         elif self.init_method == 'given_matrix':
#             dictionary_matrix = self.given_matrix
#         dictionary_matrix = dictionary_matrix / torch.norm(
#             dictionary_matrix, dim=0, keepdim=True
#         )
#         return dictionary_matrix

#     def fit(self, measurment_matrix):
#         """
#         Parameters
#         ----------
#         measurment_matrix: shape = [measurment_dim, measurment_num]
#         """
#         dictionary_matrix = self._initialize(measurment_matrix)
#         err_last = 0
#         for i in range(self.max_iter):
#             sparsity_matrix = orthogonal_matching_pursuit_matrix(
#                 measurment_matrix, dictionary_matrix, sparsity=self.sparsity
#             )
#             # err =
#             # gamma = self._transform(D, X)
#             # err = np.linalg.norm(X - gamma.dot(D))
#             err_now = torch.mean(
#                 (measurment_matrix - dictionary_matrix @ sparsity_matrix) ** 2
#             )
#             delta_err = torch.abs(err_now - err_last)
#             err_last = err_now
#             print(f'Iteration {i}, error: {err_now:.6f}, delta error: {delta_err:.6f}')
#             if delta_err < self.tol:
#                 break
#             dictionary_matrix, sparsity_matrix = self._update_dict(
#                 measurment_matrix, dictionary_matrix, sparsity_matrix
#             )

#         self.dictionary_matrix = dictionary_matrix
#         self.sparsity_matrix = sparsity_matrix
#         return dictionary_matrix, sparsity_matrix


class KSVD(object):
    def __init__(
        self,
        n_components,
        max_iter=10,
        tol=1e-6,
        sparsity=None,
        init_method='random',
        given_matrix=None,
    ):
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements

        max_iter:
            Maximum number of iterations

        tol:
            tolerance for error

        sparsity:
            Number of nonzero coefficients to target
        """
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.sparsity = sparsity if sparsity is not None else int(n_components * 0.1)
        self.init_method = init_method
        self.given_matrix = given_matrix

    def _update_dict(self, measurment_matrix, dictionary_matrix, sparsity_matrix):
        for j in range(self.n_components):
            index = np.nonzero(sparsity_matrix[j, :])[0]
            if len(index) == 0:
                continue
            dictionary_matrix[:, j] = 0
            residual = (measurment_matrix - dictionary_matrix @ sparsity_matrix)[
                :, index
            ]
            u, sigma, v = np.linalg.svd(residual)
            dictionary_matrix[:, j] = u[:, 0]
            sparsity_matrix[j, index] = sigma[0] * v[0, :]
        return dictionary_matrix, sparsity_matrix

    def _initialize(self, measurment_matrix):
        if self.init_method == 'random':
            dictionary_matrix = np.random.randn(
                measurment_matrix.shape[0], self.n_components
            )
        elif self.init_method == 'data_elements':
            indices = np.random.permutation(measurment_matrix.shape[1])[
                : self.n_components
            ]
            dictionary_matrix = measurment_matrix[:, indices]
        elif self.init_method == 'svd':
            pass
        elif self.init_method == 'given_matrix':
            dictionary_matrix = self.given_matrix
        dictionary_matrix = dictionary_matrix / np.linalg.norm(
            dictionary_matrix, axis=0, keepdims=True
        )
        return dictionary_matrix

    def fit(self, measurment_matrix):
        """
        Parameters
        ----------
        measurment_matrix: shape = [measurment_dim, measurment_num]
        """
        dictionary_matrix = self._initialize(measurment_matrix)
        err_last = 0
        for i in range(self.max_iter):
            sparsity_matrix = linear_model.orthogonal_mp(
                dictionary_matrix, measurment_matrix, n_nonzero_coefs=self.sparsity
            )
            # err =
            # gamma = self._transform(D, X)
            # err = np.linalg.norm(X - gamma.dot(D))
            err_now = np.mean(
                (measurment_matrix - dictionary_matrix @ sparsity_matrix) ** 2
            )
            delta_err = np.abs(err_now - err_last)
            err_last = err_now
            print(f'Iteration {i}, error: {err_now:.6f}, delta error: {delta_err:.7f}')
            if delta_err < self.tol:
                break
            dictionary_matrix, sparsity_matrix = self._update_dict(
                measurment_matrix, dictionary_matrix, sparsity_matrix
            )

        self.dictionary_matrix = dictionary_matrix
        self.sparsity_matrix = sparsity_matrix
        return dictionary_matrix, sparsity_matrix


# python -m src.model.spstfm.ksvd
if __name__ == '__main__':
    import torch
    import os
    import numpy as np
    import random

    rng_seed = 42
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    measurment_matrix = np.random.randn(200, 1000)
    ksvd = KSVD(n_components=128, init_method='random')
    # model = ksvd.fit(measurment_matrix)
    # dictionary_matrix = model.dictionary_matrix
    # sparsity_matrix = model.sparsity_matrix
    # print(dictionary_matrix)
    # import numpy as np
    # from ksvd import ApproximateKSVD

    # # X ~ gamma.dot(dictionary)
    # X = measurment_matrix.numpy().T
    # aksvd = ApproximateKSVD(n_components=128)
    # dictionary = aksvd.fit(X).components_
    # gamma = aksvd.transform(X)

    # print(dictionary.T)
