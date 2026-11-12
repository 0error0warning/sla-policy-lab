"""An unsupervised input compression that remains linear in the native MLP input."""

import numpy as np

from experiments.consequence_policy.policy import fit_projection


def linear_projection(values, train):
    mean, matrix = fit_projection(values, train)
    unit_mean = mean / np.dot(mean, mean)
    # At the training mean the first 32 columns vanish and the last reads one.
    # Every column is homogeneous, so the low-rank delta is a genuine delta-W.
    return np.column_stack(
        (matrix - np.outer(unit_mean, mean @ matrix), unit_mean)
    ).astype(np.float32)


def project(values, matrix):
    return np.asarray(values, dtype=np.float32) @ matrix
