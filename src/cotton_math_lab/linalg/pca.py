"""PCA via eigendecomposição da matriz de covariância (ou correlação)."""

import numpy as np

from cotton_math_lab.linalg.eigen import qr_algorithm


def pca_via_covariance(
        matrix: np.ndarray,
        *,
        k: int | None = None,
        standardize: bool = False,
):
    """PCA decompondo a matriz de covariância com o `qr_algorithm`.

    Se `standardize=True`, cada feature é dividida pelo seu desvio-padrão
    antes de formar a matriz — decompõe-se a matriz de CORRELAÇÃO, não a
    covariância bruta. Sem isso, a feature de maior escala numérica domina
    os primeiros componentes devido à unidade de medida, não por
    correlação real com as demais variáveis.

    Retorna (components, explained_variance, mean, scale). `scale` é um
    vetor de 1s quando `standardize=False`, guardado para permitir
    reconstrução simétrica nos dois casos.
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
