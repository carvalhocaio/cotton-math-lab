"""Informação mútua via binning — captura dependência que correlação de
Pearson não vê."""

import numpy as np


def mutual_information_binned(x: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """I(X;Y) via discretização de x e y em `bins` faixas cada:

    I(X;Y) = Σ p(x,y)·log(p(x,y) / (p(x)·p(y)))

    sobre o histograma conjunto. Diferente da correlação de Pearson, que
    só mede associação LINEAR, MI captura qualquer forma de dependência
    estatística — inclusive Y = f(X) para f não-linear, onde a correlação
    pode ficar arbitrariamente perto de zero mesmo com dependência
    determinística e perfeita entre as variáveis.
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
