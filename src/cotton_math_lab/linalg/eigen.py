"""Eigenvalues via iterative methods."""

import numpy as np

from cotton_math_lab.exceptions import LinAlgError
from cotton_math_lab.linalg.qr import qr_householder


def power_iteration(
    matrix: np.ndarray,
    *,
    seed: int,
    max_iter: int = 1000,
    tol: float = 1e-12,
    return_iters: bool = False,
):
    """Dominant eigenvalue (largest in magnitude) and its eigenvector.

    Iterates vₖ₊₁ = A·vₖ / ‖A·vₖ‖. The dominant eigenvector's component
    geometrically dominates the sum, at the rate |λ₂/λ₁| per step —
    hence convergence being fast when there's a wide "spectral gap" and
    slow when λ₁ ≈ λ₂.

    The eigenvalue is read off via the Rayleigh quotient λ = vᵀAv / vᵀv,
    which for a unit v is simply vᵀAv.
    """
    rows, cols = matrix.shape
    if rows != cols:
        raise LinAlgError(f"matrix must be square, received {matrix.shape}")

    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise LinAlgError(
            "matrix must be symmetric — the Rayleigh quotient only "
            "guarantees a real eigenvalue and monotonic convergence for "
            "symmetric matrices"
        )

    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(rows)
    vector /= np.linalg.norm(vector)

    eigenvalue = 0.0
    for iteration in range(1, max_iter + 1):  # noqa: B007
        product = matrix @ vector
        vector = product / np.linalg.norm(product)

        previous = eigenvalue
        eigenvalue = float(vector @ matrix @ vector)  # Rayleigh quotient

        if abs(eigenvalue - previous) < tol:
            break

    if return_iters:
        return eigenvalue, vector, iteration
    return eigenvalue, vector


def eigen_spectrum(
    matrix: np.ndarray,
    *,
    seed: int,
    k: int | None = None,
    max_iter: int = 2000,
    tol: float = 1e-13,
):
    """The `k` largest eigenvalues (descending) and eigenvectors, via deflation.

    Assumes `matrix` is symmetric. After extracting (λᵢ, vᵢ) via power
    iteration, subtracts λᵢ·vᵢvᵢᵀ from the matrix — Hotelling deflation —
    so that the next iteration finds the following pair. Valid because
    the eigenvectors of a symmetric matrix are mutually orthogonal.
    """
    rows, cols = matrix.shape
    if rows != cols:
        raise LinAlgError(f"matrix must be square, received {matrix.shape}")

    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise LinAlgError(
            "matrix must be symmetric — Hotelling deflation relies on "
            "eigenvector orthogonality, which only holds in that case"
        )

    n_components = rows if k is None else k
    residual = matrix.astype(np.float64).copy()

    eigenvalues = np.empty(n_components)
    eigenvectors = np.empty((rows, n_components))

    for i in range(n_components):
        value, vector = power_iteration(residual, seed=seed, max_iter=max_iter, tol=tol)
        eigenvalues[i] = value
        eigenvectors[:, i] = vector
        residual = residual - value * np.outer(vector, vector)

    return eigenvalues, eigenvectors


def qr_algorithm(
    matrix: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-12,
    return_iters: bool = False,
):
    """Full spectrum of a symmetric matrix via unshifted QR iteration.

    At each step, factors Aₖ = QₖRₖ and recomposes in swapped order:
    Aₖ₊₁ = RₖQₖ. Since Aₖ₊₁ = Qₖᵀ Aₖ Qₖ, each step is an orthogonal
    similarity transform — the eigenvalues never change, only the basis.
    The sequence converges to a diagonal matrix whose entries are the
    eigenvalues, and the accumulated product of the Qₖ converges to the
    eigenvectors.

    Without a shift, convergence is geometric at the rate |λₖ₊₁/λₖ| —
    the same mechanism as power iteration, because the QR algorithm is,
    structurally, simultaneous subspace iteration. Narrow gaps converge
    slowly.
    """
    rows, cols = matrix.shape
    if rows != cols:
        raise LinAlgError(f"matrix must be square, received {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise LinAlgError(
            "matrix must be symmetric — without this, complex eigenvalues "
            "show up as 2×2 blocks on the diagonal and reading "
            "np.diag(current) doesn't match the matrix's real spectrum"
        )
    n = rows
    current = matrix.astype(np.float64).copy()
    accumulated_q = np.eye(n)

    for iteration in range(1, max_iter + 1):  # noqa: B007
        q, r = qr_householder(current)
        current = r @ q
        accumulated_q = accumulated_q @ q

        off_diagonal_norm = np.sqrt(np.sum(np.tril(current, k=-1) ** 2))
        if off_diagonal_norm < tol:
            break

    eigenvalues = np.diag(current)

    if return_iters:
        return eigenvalues, accumulated_q, iteration
    return eigenvalues, accumulated_q
