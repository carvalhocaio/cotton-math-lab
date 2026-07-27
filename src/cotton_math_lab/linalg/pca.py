"""PCA via eigendecomposição da matriz de covariância (ou correlação)."""

import numpy as np

from cotton_math_lab.linalg.eigen import qr_algorithm
from cotton_math_lab.linalg.svd import svd_jacobi_one_sided


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


def pca_via_svd(
    matrix: np.ndarray,
    *,
    k: int | None = None,
    standardize: bool = False,
):
    """PCA via SVD direta na matriz de dados — nunca forma XᵀX.

    Mesma interface e mesmo significado de retorno que `pca_via_covariance`,
    para que as duas rotas sejam intercambiáveis e comparáveis. A diferença
    inteira mora em como cada uma chega aos autovalores/autovetores da
    covariância: aqui, os valores singulares de X já SÃO a raiz quadrada
    dos autovalores da covariância — sem jamais formar XᵀX como matriz
    explícita, e, portanto, sem quadrar o número de condição de X no processo.
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
