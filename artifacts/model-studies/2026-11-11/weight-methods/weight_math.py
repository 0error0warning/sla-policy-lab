"""CPU math for ridge weight editing and online recursive least squares."""

import numpy as np


def ridge_edit(features, residual, weight, ridge):
    weighted = features*np.sqrt(weight)[:, None]
    return weighted.T @ np.linalg.solve(weighted @ weighted.T + ridge*np.eye(len(weighted)),
                                        residual*np.sqrt(weight))


def update_fast(weights, inverse_gram, feature, target):
    direction = inverse_gram @ feature
    gain = direction/(1+feature@direction)
    return weights + gain*(target-feature@weights), inverse_gram-np.outer(gain,direction)


def kernel_features(hidden, groups):
    unit = hidden/np.linalg.norm(hidden,axis=1,keepdims=True)
    centers = np.stack([unit[rows].mean(axis=0) for rows in groups])
    center_distances = np.square(centers[:,None]-centers[None,:]).sum(axis=-1)
    bandwidth = float(np.median(center_distances[np.triu_indices(len(groups),1)]))
    assert bandwidth>0
    distance = np.square(unit[:,None]-centers[None,:]).sum(axis=-1)
    kernel = np.exp(-distance/bandwidth)
    return kernel/np.linalg.norm(kernel,axis=1,keepdims=True), bandwidth
