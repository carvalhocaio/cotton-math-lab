"""Autovalores por métodos iterativos."""

import numpy as np

from cotton_math_lab.exceptions import LinAlgError


def power_iteration(
    matrix: np.ndarray,
    *,
    seed: int,
    max_iter: int = 1000,
    tol: float = 1e-12,
    return_iters: bool = False,
):
    """Autovalor dominante (maior em módulo) e seu autovetor.

    Itera vₖ₊₁ = A·vₖ / ‖A·vₖ‖. A componente do autovetor dominante domina
    a soma geometricamente, na razão |λ₂/λ₁| por passo — daí a convergência
    ser rápida quando há um "gap espectral" largo e lenta quando λ₁ ≈ λ₂.

    O autovalor é lido pelo quociente de Rayleigh λ = vᵀAv / vᵀv, que para v
    unitário é apenas vᵀAv.
    """
    rows, cols = matrix.shape
    if rows != cols:
        raise LinAlgError(f"matriz deve ser quadrada, recebida {matrix.shape}")

    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(rows)
    vector /= np.linalg.norm(vector)

    eigenvalue = 0.0
    for iteration in range(1, max_iter + 1):
        product = matrix @ vector
        vector = product / np.linalg.norm(product)

        previous = eigenvalue
        eigenvalue = float(vector @ matrix @ vector)  # quociente de Rayleigh

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
    """Os `k` maiores autovalores (decrescente) e autovetores, por deflação.

    Assume `matrix` simétrica. Após extrair (λᵢ, vᵢ) por power iteration,
    subtrai λᵢ·vᵢvᵢᵀ da matriz — a deflação de Hotelling — de modo que a
    próxima iteração encontre o par seguinte. Válido porque autovetores de
    uma matriz simétrica são mutuamente ortogonais.
    """
    rows, cols = matrix.shape
    if rows != cols:
        raise LinAlgError(f"matriz deve ser quadrada, recebida {matrix.shape}")

    n_components = rows if k is None else k
    residual = matrix.astype(np.float64).copy()

    eigenvalues = np.empty(n_components)
    eigenvectors = np.empty((rows, n_components))

    for i in range(n_components):
        value, vector = power_iteration(
            residual, seed=seed, max_iter=max_iter, tol=tol
        )
        eigenvalues[i] = value
        eigenvectors[:, i] = vector
        residual = residual - value * np.outer(vector, vector)

    return eigenvalues, eigenvectors