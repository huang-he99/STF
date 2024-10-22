import torch
from torch import nn
import numpy as np
from sklearn import linear_model


# Y = DX
class KSVD(nn.Module):
    def __init__(
        self,
        atom_num,
        max_iter=10,
        tol=1e-6,
        sparsity=None,
        init_method='random',
        given_matrix=None,
    ):
        r"""
        Solve the following problem:
        .. math::
            \min_{D, X} \|Y - DX\|_{2}^{2} \quad s.t. \quad \|X\|_{0} \leq s
        Args:
            atom_num: Number of dictionary elements
            max_iter: Maximum number of iterations
            tol: tolerance for error
            sparsity: Number of nonzero coefficients to target
            init_method: {'random', 'data_elements', 'svd', 'given_matrix'}
            given_matrix: Given dictionary matrix :math:`D \in \mathbb{R}^{m \times k}`,
            where :math:`m` is the dim of measurements and :math:`k` is the number of atoms.
        """
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.atom_num = atom_num
        self.sparsity = sparsity if sparsity is not None else int(atom_num * 0.1)
        self.init_method = init_method
        self.given_matrix = given_matrix

    def fit(self, observed_matrix):
        """
        Solve the following problem:
        .. math::
            \min_{D, X} \|Y - DX\|_{2}^{2} \quad s.t. \quad \|X\|_{0} \leq s
        1. Initialize dictionary matrix :math:`D \in \mathbb{R}^{m \times k}`,
        where :math:`m` is the dim of measurements and :math:`k` is the number of atoms.
        2. Solve sparse matrix :math:`X \in \mathbb{R}^{k \times n}` by OMP,
        where :math:`n` is the number of measurements.
        3. Update dictionary matrix :math:`D \in \mathbb{R}^{m \times k}` by SVD.
        4. Repeat 2 and 3 until convergence.
        Args:
            observed_matrix: Observed matrix :math:`Y \in \mathbb{R}^{m \times n}`,
            where :math:`m` is the dim of measurements and :math:`n` is the number of measurements.
        Returns:
            dictionary_matrix: Dictionary matrix :math:`D \in \mathbb{R}^{m \times k}`,
            where :math:`m` is the dim of measurements and :math:`k` is the number of atoms.
            sparsity_matrix: Sparse matrix :math:`X \in \mathbb{R}^{k \times n}`,
            where :math:`k` is the number of atoms and :math:`n` is the number of measurements.
        """
        dictionary_matrix = self._initialize(observed_matrix)
        err_last = 0
        for i in range(self.max_iter):
            sparse_matrix = linear_model.orthogonal_mp(
                dictionary_matrix, observed_matrix, n_nonzero_coefs=self.sparsity
            )
            # err =
            # gamma = self._transform(D, X)
            # err = np.linalg.norm(X - gamma.dot(D))
            err_now = np.mean(
                (observed_matrix - dictionary_matrix @ sparse_matrix) ** 2
            )
            # delta_err = torch.abs(err_now - err_last)
            # err_last = err_now
            print(f'Iteration {i}, error: {err_now:.6f}')
            if err_now < self.tol:
                break
            dictionary_matrix, sparse_matrix = self._update_dict(
                observed_matrix, dictionary_matrix, sparse_matrix
            )

        self.dictionary_matrix = dictionary_matrix
        self.sparse_matrix = sparse_matrix
        return dictionary_matrix, sparse_matrix

    def _initialize(self, observed_matrix):
        r"""
        Initialize dictionary matrix :math:`D \in \mathbb{R}^{m \times k}`,
        where :math:`m` is the dim of measurements and :math:`k` is the number of atoms.
        Args:
            observed_matrix: Observed matrix :math:`Y \in \mathbb{R}^{m \times n}`,
            where :math:`m` is the dim of measurements and :math:`n` is the number of measurements.
        Returns:
            dictionary_matrix: Dictionary matrix :math:`D \in \mathbb{R}^{m \times k}`,
            where :math:`m` is the dim of measurements and :math:`k` is the number of atoms.
        """

        if self.init_method == 'random':
            dictionary_matrix = np.random.randn(observed_matrix.shape[0], self.atom_num)
        elif self.init_method == 'data_elements':
            indices = np.random.permutation(observed_matrix.shape[1])[: self.atom_num]
            dictionary_matrix = observed_matrix[:, indices]
        elif self.init_method == 'svd':
            pass
        elif self.init_method == 'given_matrix':
            dictionary_matrix = self.given_matrix
        dictionary_matrix = dictionary_matrix / np.linalg.norm(
            dictionary_matrix, axis=0, keepdims=True
        )
        return dictionary_matrix

    def _update_dict(self, observed_matrix, dictionary_matrix, sparse_matrix):
        for j in range(self.atom_num):
            index = np.nonzero(sparse_matrix[j, :])[0]
            if len(index) == 0:
                continue
            dictionary_matrix[:, j] = 0
            residual = (observed_matrix - dictionary_matrix @ sparse_matrix)[:, index]
            u, sigma, v = np.linalg.svd(residual)
            dictionary_matrix[:, j] = u[:, 0]
            sparse_matrix[j, index] = sigma[0] * v[0, :]
        return dictionary_matrix, sparse_matrix


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
    ksvd = KSVD(atom_num=128, init_method='random')
    # model = ksvd.fit(measurment_matrix)
    # dictionary_matrix = model.dictionary_matrix
    # sparsity_matrix = model.sparsity_matrix
    # print(dictionary_matrix)
    # import numpy as np
    # from ksvd import ApproximateKSVD

    # # X ~ gamma.dot(dictionary)
    # X = measurment_matrix.numpy().T
    # aksvd = ApproximateKSVD(atom_num=128)
    # dictionary = aksvd.fit(X).components_
    # gamma = aksvd.transform(X)

    # print(dictionary.T)
