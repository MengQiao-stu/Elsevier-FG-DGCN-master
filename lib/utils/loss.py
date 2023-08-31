import torch
from .sparse_utils import naive_sparse_bmm, sparse_permute


def sparse_reconstruction(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with the sparse matrix
    NOTE: this function doesn't use it in this project, because may not return correct gradients

    Args:
        assignment: torch.sparse_coo_tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    labels = labels.permute(0, 2, 1).contiguous()

    # matrix product between (n_spixels, n_pixels) and (n_pixels, channels)
    spixel_mean = naive_sparse_bmm(assignment, labels) / (torch.sparse.sum(assignment, 2).to_dense()[..., None] + 1e-16)
    if hard_assignment is None:
        # (B, n_spixels, n_pixels) -> (B, n_pixels, n_spixels)
        permuted_assignment = sparse_permute(assignment, (0, 2, 1))
        # matrix product between (n_pixels, n_spixels) and (n_spixels, channels)
        reconstructed_labels = naive_sparse_bmm(permuted_assignment, spixel_mean)
    else:
        # index sampling
        reconstructed_labels = torch.stack([sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0)
    return reconstructed_labels.permute(0, 2, 1).contiguous()


def reconstruction(assignment, labels, hard_assignment=None):
    """
    reconstruction

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    labels = labels.permute(0, 2, 1).contiguous()

    # matrix product between (n_spixels, n_pixels) and (n_pixels, channels)，将位置特征映射到超像素中
    spixel_mean = torch.bmm(assignment, labels) / (assignment.sum(2, keepdim=True) + 1e-16)
    if hard_assignment is None:
        # (B, n_spixels, n_pixels) -> (B, n_pixels, n_spixels)
        permuted_assignment = assignment.permute(0, 2, 1).contiguous()
        # matrix product between (n_pixels, n_spixels) and (n_spixels, channels)
        reconstructed_labels = torch.bmm(permuted_assignment, spixel_mean)
    else:
        # index sampling，使用硬关联矩阵映射回像素特征
        reconstructed_labels = torch.stack([sm[ha, :] for sm, ha in zip(spixel_mean, hard_assignment)], 0)
        print("将像素的位置特征映射到超像素中，并使用硬关联矩阵映射回像素")
    return reconstructed_labels.permute(0, 2, 1).contiguous()


def reconstruct_loss_with_cross_etnropy(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with cross entropy

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    reconstracted_labels = reconstruction(assignment, labels, hard_assignment) # 将位置特征映射到超像素表示中，而后使用硬关联反向映射回像素特征
    # [6,50,40000]
    reconstracted_labels = reconstracted_labels / (1e-16 + reconstracted_labels.sum(1, keepdim=True))
    mask = labels > 0
    return -(reconstracted_labels[mask] + 1e-16).log().mean()


def reconstruct_loss_with_mse(assignment, labels, hard_assignment=None):
    """
    reconstruction loss with mse 一致性损失，均方误差

    Args:
        assignment: torch.Tensor
            A Tensor of shape (B, n_spixels, n_pixels)
        labels: torch.Tensor
            A Tensor of shape (B, C, n_pixels)
        hard_assignment: torch.Tensor
            A Tensor of shape (B, n_pixels)
    """
    reconstracted_labels = reconstruction(assignment, labels, hard_assignment)
    # [6,2,40000]
    return torch.nn.functional.mse_loss(reconstracted_labels, labels)
