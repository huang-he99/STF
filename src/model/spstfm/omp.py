import torch


def orthogonal_matching_pursuit(
    compressed_vector, compression_matrix, sparsity, err_treshold=1e-6
):
    '''
    Args:
        compressed_vector: (compression_dim, )
        compression_matrix: (compression_dim, sample_dim)
    Returns:
        sparsity_vecotr: (sample_dim, )
    '''
    # initialize
    # compression_matrix_norm = torch.norm(compression_matrix, dim=0, keepdim=True)
    unit_compression_matrix = compression_matrix
    subsapce_index_list = []
    residual = compressed_vector
    k = 1
    sparsity_vecotr = torch.zeros(
        compression_matrix.shape[1], device=compressed_vector.device
    )
    err = torch.sum(residual**2)
    while k <= sparsity and err > err_treshold:
        residual_projection_matrix = torch.abs(unit_compression_matrix.T @ residual)
        _, matching_basis_index = torch.max(residual_projection_matrix, dim=0)
        subsapce_index_list.append(matching_basis_index.item())
        subspace_matrix = compression_matrix[:, subsapce_index_list]
        subspace_matrix_pinv = torch.pinverse(subspace_matrix)
        subspace_coordinates = subspace_matrix_pinv @ compressed_vector
        subspace_representation = subspace_matrix @ subspace_coordinates
        residual = compressed_vector - subspace_representation
        err = torch.sum(residual**2)
        k += 1
    sparsity_vecotr[subsapce_index_list] = subspace_coordinates
    return sparsity_vecotr


def orthogonal_matching_pursuit_matrix(
    compressed_matrix, compression_matrix, sparsity, err_treshold=1e-6
):
    '''
    Args:
        compressed_matrix: (compression_dim, sample_num)
        compression_matrix: (compression_dim, sample_dim)
    Returns:
        sparsity_matrix: (sample_dim, sample_num)
    '''
    _, sample_num = compressed_matrix.shape
    _, sample_dim = compression_matrix.shape
    sparsity_matrix = torch.zeros(
        sample_dim, sample_num, device=compressed_matrix.device
    )
    for sample_idx in range(sample_num):
        compressed_vector = compressed_matrix[:, sample_idx]
        sparsity_vecotr = orthogonal_matching_pursuit(
            compressed_vector,
            compression_matrix,
            sparsity=sparsity,
            err_treshold=err_treshold,
        )
        sparsity_matrix[:, sample_idx] = sparsity_vecotr

    return sparsity_matrix
