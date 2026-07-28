"""Autovalores por métodos iterativos."""

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
    for iteration in range(1, max_iter + 1):  # noqa: B007
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
    """Espectro completo de uma matriz simétrica via iteração QR sem shift.

    A cada passo, fatora Aₖ = QₖRₖ e recompõe na ordem trocada:
    Aₖ₊₁ = RₖQₖ. Como Aₖ₊₁ = Qₖᵀ Aₖ Qₖ, cada passo é uma transformação de
    similaridade ortogonal — os autovalores nunca mudam, só a base. A
    sequência converge para uma matriz diagonal cujos elementos são os
    autovalores, e o produto acumulado dos Qₖ converge para os autovetores.

    Sem shift, a convergência é geométrica na razão |λₖ₊₁/λₖ| — o mesmo
    mecanismo da power iteration, porque o QR algorithm é, estruturalmente,
    iteração de subespaço simultânea. Gaps estreitos convergem devagar.
    """
    n = matrix.shape[0]
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
