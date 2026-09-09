"""QR decomposition: classic Gram-Schmidt vs. Householder reflections."""

import numpy as np


def qr_gram_schmidt(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR via classic Gram-Schmidt, column by column.

    Each column of Q is the column of `matrix` minus its projections
    onto the previous columns, normalized. R accumulates the projection
    coefficients. Numerically unstable when columns are nearly
    collinear: rounding errors in one projection contaminate the next,
    and Q loses orthogonality cumulatively and silently.
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
    """QR via Householder reflections.

    Each step zeroes out, with a single orthogonal reflection,
    everything below the diagonal in the current column. Reflections
    are exact isometries — there's no subtraction of near-equal
    quantities as in Gram-Schmidt — so the error doesn't accumulate:
    each step is orthogonal to machine precision, regardless of
    `matrix`'s conditioning.
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
