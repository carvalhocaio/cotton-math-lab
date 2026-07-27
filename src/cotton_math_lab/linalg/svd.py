"""SVD via Jacobi de um lado (Hestenes).

Ao contrário de decompor a covariância (XᵀX), este método opera direto
sobre as colunas de X, nunca formando XᵀX como uma matriz explícita de
uma vez só.
"""

import numpy as np

from cotton_math_lab.exceptions import LinAlgError


def svd_jacobi_one_sided(
    matrix: np.ndarray,
    *,
    max_sweeps: int = 100,
    tol: float = 1e-14,
):
    """SVD reduzida (m ≥ n) via rotações de Jacobi entre pares de colunas.

    Cada passo escolhe duas colunas (i, j) e aplica uma rotação 2×2 que as
    torna ortogonais entre si — a rotação zera o produto interno ⟨aᵢ, aⱼ⟩
    exatamente, por construção geométrica, não por diferença numérica entre
    quantidades próximas. Repetindo sobre todos os pares (uma "varredura"),
    e repetindo varreduras, a matriz inteira converge para colunas
    mutuamente ortogonais: normalizando cada coluna, sua norma é o valor
    singular e a direção é a coluna de U; a rotação acumulada é V.

    O ponto central: cada rotação usa ⟨aᵢ, aⱼ⟩ calculado sob demanda, a
    partir dos valores atuais e já refinados das colunas — nunca contamina
    todos os produtos internos de uma vez formando XᵀX inteira antes de
    começar a decompor. É por isso que a precisão relativa se mantém mesmo
    quando colunas de `matrix` são quase paralelas (mal-condicionadas).
    """
    rows, cols = matrix.shape
    if rows < cols:
        raise LinAlgError(
            f"requer ao menos tantas linhas quanto colunas, recebido {matrix.shape}"
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
