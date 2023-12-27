import torch


def orthogonal_matching_pursuit(
    observed_vector, dictionary_matrix, sparsity, err_treshold=1e-6
):
    r'''
    Solve the following problem:
    .. math::
        \min_{x} \|y - Dx\|_{2}^{2} \quad s.t. \quad \|x\|_{0} \leq s
    Args:
        observed_vector: Observed vector :math:`y \in \mathbb{R}^{m}`,
        where :math:`m` is the dim of measurements
        dictionary_matrix: Dictionary matrix :math:`D \in \mathbb{R}^{m \times k}`,
        where :math:`m` is the dim of measurements
        and :math:`k` is the number of atoms.
    Returns:
        sparse_vector: Sparse vector :math:`x \in \mathbb{R}^{k}` solved by OMP, where :math:`k` is the number of atoms.
    '''
    # initialize
    # compression_matrix_norm = torch.norm(compression_matrix, dim=0, keepdim=True)
    # unit_compression_matrix = compression_matrix / compression_matrix_norm
    _, atom_num = dictionary_matrix.shape
    subsapce_index_list = []
    subspace_coordinates = 0
    residual = observed_vector
    k = 1
    sparse_vector = torch.zeros(atom_num, device=dictionary_matrix.device)
    err = torch.sum(residual**2)
    while k <= sparsity and err > err_treshold:
        residual_projection_vector = torch.abs(dictionary_matrix.T @ residual)
        _, matching_basis_index = torch.max(residual_projection_vector, dim=0)
        subsapce_index_list.append(matching_basis_index.item())
        subspace_matrix = dictionary_matrix[:, subsapce_index_list]
        subspace_matrix_pinv = torch.pinverse(subspace_matrix)
        subspace_coordinates = subspace_matrix_pinv @ observed_vector
        subspace_representation = subspace_matrix @ subspace_coordinates
        residual = observed_vector - subspace_representation
        err = torch.sum(residual**2)
        k += 1
    sparse_vector[subsapce_index_list] = subspace_coordinates
    return sparse_vector


def orthogonal_matching_pursuit_matrix(
    observed_matrix, dictionary_matrix, sparsity, err_treshold=1e-6
):
    r"""
    Solve the following problem:
    .. math::
        \min_{X} \|Y - DX\|_{2}^{2} \quad s.t. \quad \|X\|_{0} \leq s
    Args:
        observed_matrix: Observed matrix :math:`Y \in \mathbb{R}^{m \times n}`,
        where :math:`m` is the dim of measurements
        and :math:`n` is the number of measurements.
        dictionary_matrix: Dictionary matrix :math:`D \in \mathbb{R}^{m \times k}`,
        where :math:`m` is the dim of measurements
        and :math:`k` is the number of atoms.
    Returns:
        sparse_matrix: Sparse matrix :math:`X \in \mathbb{R}^{k \times n}` solved by OMP,
        where :math:`k` is the number of atoms.
    """
    _, sample_num = observed_matrix.shape
    _, atom_num = dictionary_matrix.shape
    sparse_matrix = torch.zeros(atom_num, sample_num, device=dictionary_matrix.device)
    for sample_idx in range(sample_num):
        observed_vector = observed_matrix[:, sample_idx]
        sparse_vector = orthogonal_matching_pursuit(
            observed_vector,
            dictionary_matrix,
            sparsity=sparsity,
            err_treshold=err_treshold,
        )
        sparse_matrix[:, sample_idx] = sparse_vector

    return sparse_matrix
