import torch
import numpy as np


def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e.
    subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation,
    L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally
    across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x**2)) / n_features

    x /= x_scale

    return x


def create_semisupervised_setting(labels, normal_classes, outlier_classes,
                                  known_outlier_classes, ratio_known_normal,
                                  ratio_known_outlier, ratio_pollution):
    """
    Create a semi-supervised data setting. 
    :param labels: np.array with labels of all dataset samples
    :param normal_classes: tuple with normal class labels
    :param outlier_classes: tuple with anomaly class labels
    :param known_outlier_classes: tuple with known (labeled) anomaly class labels
    :param ratio_known_normal: the desired ratio of known (labeled) normal samples
    :param ratio_known_outlier: the desired ratio of known (labeled) anomalous samples
    :param ratio_pollution: the desired pollution ratio of the unlabeled data with unknown (unlabeled) anomalies.
    :return: tuple with list of sample indices, list of original labels, and list of semi-supervised labels
    """
    idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()
    idx_outlier = np.argwhere(np.isin(labels, outlier_classes)).flatten()
    idx_known_outlier_candidates = np.argwhere(
        np.isin(labels, known_outlier_classes)).flatten()

    n_normal = len(idx_normal)

    # Solve system of linear equations to obtain respective number of samples
    a = np.array([[1, 1, 0, 0],
                  [(1 - ratio_known_normal), -ratio_known_normal,
                   -ratio_known_normal, -ratio_known_normal],
                  [
                      -ratio_known_outlier, -ratio_known_outlier,
                      -ratio_known_outlier, (1 - ratio_known_outlier)
                  ], [0, -ratio_pollution, (1 - ratio_pollution), 0]])
    b = np.array([n_normal, 0, 0, 0])
    x = np.linalg.solve(a, b)

    # Get number of samples
    n_known_normal = int(x[0])
    n_unlabeled_normal = int(x[1])
    n_unlabeled_outlier = int(x[2])
    n_known_outlier = int(x[3])

    # Sample indices
    perm_normal = np.random.permutation(n_normal)
    perm_outlier = np.random.permutation(len(idx_outlier))
    perm_known_outlier = np.random.permutation(
        len(idx_known_outlier_candidates))

    idx_known_normal = idx_normal[perm_normal[:n_known_normal]].tolist()
    idx_unlabeled_normal = idx_normal[
        perm_normal[n_known_normal:n_known_normal +
                    n_unlabeled_normal]].tolist()
    idx_unlabeled_outlier = idx_outlier[
        perm_outlier[:n_unlabeled_outlier]].tolist()
    idx_known_outlier = idx_known_outlier_candidates[
        perm_known_outlier[:n_known_outlier]].tolist()

    # Get original class labels
    labels_known_normal = labels[idx_known_normal].tolist()
    labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
    labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
    labels_known_outlier = labels[idx_known_outlier].tolist()

    # Get semi-supervised setting labels
    semi_labels_known_normal = np.ones(n_known_normal).astype(
        np.int32).tolist()
    semi_labels_unlabeled_normal = np.zeros(n_unlabeled_normal).astype(
        np.int32).tolist()
    semi_labels_unlabeled_outlier = np.zeros(n_unlabeled_outlier).astype(
        np.int32).tolist()
    semi_labels_known_outlier = (
        -np.ones(n_known_outlier).astype(np.int32)).tolist()

    # Create final lists
    list_idx = idx_known_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_known_outlier
    list_labels = labels_known_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_known_outlier
    list_semi_labels = (semi_labels_known_normal +
                        semi_labels_unlabeled_normal +
                        semi_labels_unlabeled_outlier +
                        semi_labels_known_outlier)

    return list_idx, list_labels, list_semi_labels


def one_hotify(labels, nb_classes=None):
    '''
    Converts integer labels to one-hot vectors.

    Arguments:
        labels: numpy array containing integer labels. The labels must be in
        range [0, num_labels - 1].

    Returns:
        one_hot_labels: numpy array with shape (batch_size, num_labels).
    '''
    size = len(labels)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1

    one_hot_labels = np.zeros((size, nb_classes))
    one_hot_labels[np.arange(size), labels] = 1
    return one_hot_labels


# def global_contrast_normalization(data_set, eps=1e-6):
#     '''
#     Applies global contrast normalization to the input image data.

#     Arguments:
#         data_set: numpy array of shape (batch_size, dim). If the input has
#             more than 2 dimensions (such as images), it will be flatten the
#             data.
#         eps: small constant to avoid division by very small numbers during
#             normalization. If the a divisor is smaller than eps, no division
#             will be carried out on that dimension.
#     Returns:
#         norm_data: numpy array with normalized data. Has the same shape
#             as the input.
#     '''
#     if not data_set.size:
#         # Simply return if data_set is empty
#         return data_set
#     data_shape = data_set.shape
#     # If data has more than 2 dims, normalize along all axis > 0
#     if len(data_shape) > 2:
#         size = data_shape[0]
#         norm_data = data_set.reshape((size, -1))
#     else:
#         norm_data = data_set
#     mean = norm_data.mean(axis=1)
#     norm_data -= mean[:, np.newaxis]
#     std = norm_data.std(axis=1, ddof=1)
#     std[std < eps] = 1
#     norm_data /= std[:, np.newaxis]
#     return norm_data.reshape(data_shape)


def normalization(data_set, mean=None, std=None, eps=1e-6):
    '''
    Normalizes data across each dimension by removing it's mean and dividing
    by it's standard deviation.

    Arguments:
        data_set: numpy array of shape(batch_size, ...).
        mean: numpy array with the same shape as the input, excluding the
            batch axis, that will be used as the mean. If None (Default),
            the mean will be computed from the input data.
        std: numpy array with the same shape as the input, excluding the
            batch axis, that will be used as the standard deviation. If None
            (Default), the mean will be computed from the input data.
        eps: small constant to avoid division by very small numbers during
            normalization. If the a divisor is smaller than eps, no division
            will be carried out on that dimension.
    '''
    if mean is None:
        mean = np.mean(data_set, axis=0)
    if std is None:
        std = np.std(data_set, axis=0, ddof=1)
    std[std < eps] = 1
    data_set -= mean
    data_set /= std
    return data_set, mean, std


def zca_whitening(data_set, mean=None, whitening=None):
    '''
    Applies ZCA whitening the the input data.

    Arguments:
        data_set: numpy array of shape (batch_size, dim). If the input has
            more than 2 dimensions (such as images), it will be flatten the
            data.
        mean: numpy array of shape (dim) that will be used as the mean.
            If None (Default), the mean will be computed from the input data.
        whitening: numpy array shaped (dim, dim) that will be used as the
            whitening matrix. If None (Default), the whitening matrix will be
            computed from the input data.

    Returns:
        white_data: numpy array with whitened data. Has the same shape as
            the input.
        mean: numpy array of shape (dim) that contains the mean of each input
            dimension. If mean was provided as input, this is a copy of it.
        whitening:  numpy array of shape (dim, dim) that contains the whitening
            matrix. If whitening was provided as input, this is a copy of it.
    '''
    if not data_set.size:
        # Simply return if data_set is empty
        return data_set, mean, whitening
    data_shape = data_set.shape
    size = data_shape[0]
    white_data = data_set.reshape((size, -1))

    if mean is None:
        # No mean matrix, we must compute it
        mean = white_data.mean(axis=0)
    # Remove mean
    white_data -= mean

    # If no whitening matrix, we must compute it
    if whitening is None:
        cov = np.dot(white_data.T, white_data) / size
        U, S, V = np.linalg.svd(cov)
        whitening = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 1e-6))), U.T)

    white_data = np.dot(white_data, whitening)
    return white_data.reshape(data_shape), mean, whitening


def per_channel_normalization(data_set, mean=None, std=None):
    '''
    Applies channel-wise mean and standard deviation normalization.

    Arguments:
        data_set: numpy array of shape (samples, height, width, channels).
        mean: numpy array of shape (channels,) that contains the mean values
            of the channels. If None (Default), the mean will be computed
            from the input data.
        std: numpy array of shape (channels,) that contains the standard
            deviation values of the channels. If None (Default), the mean
            will be computed from the input data.

    Returns:
        normalized_set: numpy array with normalized data. Has same shape as the
            input.
        mean: numpy array of shape (channels,) that contains the values by which
            the mean of each channel was subtracted by. If a mean was provided
            as input, this is it.
        std: numpy array of shape (channels,) that contains the values by which
            the standard deviation of each channel was divided by. If a mean was
            provided as input, this is it.
    '''
    if len(data_set.shape) < 4:
        raise Exception('Expected 4 dim tensor, found shape: %s' %
                        str(data_set.shape))
    if mean is None:
        mean = np.mean(data_set, axis=(0, 1, 2))
    if std is None:
        std = np.std(data_set, axis=(0, 1, 2))

    normalized_set = data_set - mean
    normalized_set /= std

    return normalized_set, mean, std