"""Decomposição QR: Gram-Schmidt clássico vs. reflexões de Householder."""

import numpy as np


def qr_gram_schmidt(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR via Gram-Schmidt clássico, coluna a coluna.

    Cada coluna de Q é a coluna de `matrix` menos suas projeções sobre as
    colunas anteriores, normalizada. R acumula os coeficientes das projeções.
    Numericamente instável quando colunas são quase colineares: erros de
    arredondamento em uma projeção contaminam a próxima, e Q perde
    ortogonalidade de forma acumulativa e silenciosa.
    """
    rows, cols = matrix.shape
    q = np.zeros((rows, cols))
    r = np.zeros((cols, cols))

    for j in range(cols):
        v = matrix[:, j].copy()
        for i in range(j):
            r[i, j] = q[:, i] @ matrix[:, j]
            v = v - r[i, j] * q[:, i]
        r[j, j] = np.linalg.norm(v)
        q[:, j] = v / r[j, j]

    return q, r


def qr_householder(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR via reflexões de Householder.

    Cada passo zera, com uma única reflexão ortogonal, tudo abaixo da
    diagonal na coluna atual. Reflexões são isometrias exatas — não há
    subtração de quantidades quase iguais como no Gram-Schmidt — então o
    erro não acumula: cada passo é ortogonal até a precisão de máquina,
    independentemente do condicionamento de `matrix`.
    """
    rows, cols = matrix.shape
    r = matrix.astype(np.float64).copy()
    q = np.eye(rows)

    for k in range(min(rows - 1, cols)):
        x = r[k:, k]
        sign = -1.0 if x[0] >= 0 else 1.0
        alpha = sign * np.linalg.norm(x)

        v = x.copy()
        v[0] -= alpha
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-14:
            continue
        v /= norm_v

        r[k:, :] -= 2.0 * np.outer(v, v @ r[k:, :])
        q[:, k:] -= 2.0 * np.outer(q[:, k:] @ v, v)

    return q, r
