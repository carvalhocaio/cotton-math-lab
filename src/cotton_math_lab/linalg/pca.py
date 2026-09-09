"""PCA via eigendecomposition of the covariance (or correlation) matrix."""

import numpy as np

from cotton_math_lab.linalg.eigen import qr_algorithm
from cotton_math_lab.linalg.svd import svd_jacobi_one_sided


def pca_via_covariance(
    matrix: np.ndarray,
    *,
    k: int | None = None,
    standardize: bool = False,
):
    """PCA by decomposing the covariance matrix with `qr_algorithm`.

    If `standardize=True`, each feature is divided by its standard
    deviation before forming the matrix — the CORRELATION matrix is
    decomposed, not the raw covariance. Without this, the feature with
    the largest numeric scale dominates the first components because of
    its unit of measurement, not because of a real correlation with the
    other variables.

    Returns (components, explained_variance, mean, scale). `scale` is a
    vector of 1s when `standardize=False`, kept to allow symmetric
    reconstruction in both cases.
    """
    n_samples, n_features = matrix.shape
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=1) if standardize else np.ones(n_features)

    standardized = (matrix - mean) / scale
    covariance = (standardized.T @ standardized) / (n_samples - 1)

    eigenvalues, eigenvectors = qr_algorithm(covariance)
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    n_components = n_features if k is None else k
    return (
        eigenvectors[:, :n_components],
        eigenvalues[:n_components],
        mean,
        scale,
    )


def pca_via_svd(
    matrix: np.ndarray,
    *,
    k: int | None = None,
    standardize: bool = False,
):
    """PCA via direct SVD on the data matrix — never forms XᵀX.

    Same interface and return meaning as `pca_via_covariance`, so both
    routes are interchangeable and comparable. The whole difference lies
    in how each one arrives at the covariance eigenvalues/eigenvectors:
    here, X's singular values already ARE the square root of the
    covariance eigenvalues — without ever forming XᵀX as an explicit
    matrix, and therefore without squaring X's condition number in the
    process.
    """
    n_samples, n_features = matrix.shape
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=1) if standardize else np.ones(n_features)

    standardized = (matrix - mean) / scale
    _, singular_values, right_vectors = svd_jacobi_one_sided(standardized)
    explained_variance = (singular_values**2) / (n_samples - 1)

    n_components = n_features if k is None else k
    return (
        right_vectors[:, :n_components],
        explained_variance[:n_components],
        mean,
        scale,
    )
