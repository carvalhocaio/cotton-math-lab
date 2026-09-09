"""Mutual information via binning — captures dependence that Pearson
correlation doesn't see."""

import numpy as np


def mutual_information_binned(x: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """I(X;Y) by discretizing x and y into `bins` bins each:

    I(X;Y) = Σ p(x,y)·log(p(x,y) / (p(x)·p(y)))

    over the joint histogram. Unlike Pearson correlation, which only
    measures LINEAR association, MI captures any form of statistical
    dependence — including Y = f(X) for nonlinear f, where the
    correlation can be arbitrarily close to zero even with a
    deterministic, perfect dependence between the variables.
    """
    joint_counts, _, _ = np.histogram2d(x, y, bins=bins)
    joint_probs = joint_counts / joint_counts.sum()
    marginal_x = joint_probs.sum(axis=1)
    marginal_y = joint_probs.sum(axis=0)

    mi = 0.0
    for i in range(joint_probs.shape[0]):
        for j in range(joint_probs.shape[1]):
            if joint_probs[i, j] > 0:
                mi += joint_probs[i, j] * np.log(
                    joint_probs[i, j] / (marginal_x[i] * marginal_y[j])
                )
    return float(mi)
