import torch
import numpy as np

def maximum_mean_discrepancy(k_xx, k_xy, k_yy):
    """Adapted from `KID Score`_"""
    m = k_xx.shape[0]

    diag_x = torch.diag(k_xx)
    diag_y = torch.diag(k_yy)

    kt_xx_sums = k_xx.sum(dim=-1) - diag_x
    kt_yy_sums = k_yy.sum(dim=-1) - diag_y
    k_xy_sums = k_xy.sum(dim=0)

    kt_xx_sum = kt_xx_sums.sum()
    kt_yy_sum = kt_yy_sums.sum()
    k_xy_sum = k_xy_sums.sum()

    value = (kt_xx_sum + kt_yy_sum) / (m * (m - 1))
    value -= 2 * k_xy_sum / (m**2)
    return value


def poly_kernel(f1, f2, degree = 3, gamma = None, coef = 1.0):
    """Adapted from `KID Score`_"""
    if gamma is None:
        gamma = 1.0 / f1.shape[1]
    kernel = (f1 @ f2.T * gamma + coef) ** degree
    return kernel


def poly_mmd(f_real, f_fake, degree = 3, gamma = None,  coef =1.0):
    """Adapted from `KID Score`_"""
    k_11 = poly_kernel(f_real, f_real, degree, gamma, coef)
    k_22 = poly_kernel(f_fake, f_fake, degree, gamma, coef)
    k_12 = poly_kernel(f_real, f_fake, degree, gamma, coef)
    return maximum_mean_discrepancy(k_11, k_12, k_22)

def compute_kid(fake_features, real_features,
            feature = 2048,
            subsets = 100,
            subset_size = 1000,
            degree = 3,
            gamma = None,
            coef = 1.0,
            reset_real_features = True,
            normalize = False,
            **kwargs,):
    if isinstance(fake_features,np.ndarray):
        fake_features = torch.from_numpy(fake_features)
    if isinstance(real_features,np.ndarray):
        real_features = torch.from_numpy(real_features)
    n_samples_real = real_features.shape[0]
    if n_samples_real < subset_size:
        raise ValueError("Argument `subset_size` should be smaller than the number of samples")
    n_samples_fake = fake_features.shape[0]
    if n_samples_fake < subset_size:
        raise ValueError("Argument `subset_size` should be smaller than the number of samples")
    kid_scores_ = []
    for _ in range(subsets):
        perm = torch.randperm(n_samples_real)
        f_real = real_features[perm[: subset_size]]
        perm = torch.randperm(n_samples_fake)
        f_fake = fake_features[perm[: subset_size]]

        o = poly_mmd(f_real, f_fake, degree, gamma, coef)
        kid_scores_.append(o)
    kid_scores = torch.stack(kid_scores_)
    return kid_scores.mean(), kid_scores.std(unbiased=False),kid_scores_