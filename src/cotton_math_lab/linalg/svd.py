"""One-sided Jacobi SVD (Hestenes).

Unlike decomposing the covariance (XᵀX), this method operates directly
on the columns of X, never forming XᵀX as an explicit matrix all at once.
"""

import numpy as np

from cotton_math_lab.exceptions import LinAlgError


def svd_jacobi_one_sided(
    matrix: np.ndarray,
    *,
    max_sweeps: int = 100,
    tol: float = 1e-14,
):
    """Reduced SVD (m ≥ n) via Jacobi rotations between column pairs.

    Each step picks two columns (i, j) and applies a 2×2 rotation that
    makes them orthogonal to each other — the rotation zeroes the inner
    product ⟨aᵢ, aⱼ⟩ exactly, by geometric construction, not by a
    numerical difference between close quantities. Repeating over all
    pairs (a "sweep"), and repeating sweeps, the entire matrix converges
    to mutually orthogonal columns: normalizing each column, its norm is
    the singular value and its direction is the column of U; the
    accumulated rotation is V.

    The central point: each rotation uses ⟨aᵢ, aⱼ⟩ computed on demand,
    from the current, already-refined column values — it never
    contaminates all inner products at once by forming the entire XᵀX
    before any refinement happens. That's why relative precision holds
    even when columns of `matrix` are nearly parallel (ill-conditioned).
    """
    rows, cols = matrix.shape
    if rows < cols:
        raise LinAlgError(
            f"requires at least as many rows as columns, received {matrix.shape}"
        )

    a = matrix.astype(np.float64).copy()
    v = np.eye(cols)

    for _ in range(max_sweeps):
        max_off_diagonal = 0.0

        for i in range(cols - 1):
            for j in range(i + 1, cols):
                col_i, col_j = a[:, i], a[:, j]
                alpha = col_i @ col_i
                beta = col_j @ col_j
                gamma = col_i @ col_j

                denom = np.sqrt(alpha * beta)
                if denom > 0.0:
                    max_off_diagonal = max(max_off_diagonal, abs(gamma) / denom)

                if abs(gamma) < tol * denom + 1e-300:
                    continue

                zeta = (beta - alpha) / (2.0 * gamma)
                sign = 1.0 if zeta >= 0 else -1.0
                t = sign / (abs(zeta) + np.sqrt(1.0 + zeta**2))
                c = 1.0 / np.sqrt(1.0 + t**2)
                s = c * t

                a[:, i], a[:, j] = c * col_i - s * col_j, s * col_i + c * col_j
                v[:, i], v[:, j] = c * v[:, i] - s * v[:, j], s * v[:, i] + c * v[:, j]

        if max_off_diagonal < tol:
            break

    singular_values = np.linalg.norm(a, axis=0)
    order = np.argsort(-singular_values)
    singular_values = singular_values[order]
    v = v[:, order]
    a = a[:, order]

    u = np.zeros_like(a)
    nonzero = singular_values > 1e-300
    u[:, nonzero] = a[:, nonzero] / singular_values[nonzero]

    return u, singular_values, v
