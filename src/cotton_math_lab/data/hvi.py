"""Synthetic HVI data generator for cotton bales.

The population parameters are *known by construction*: this lets later
modules (MLE, PCA, mutual information) be validated against the truth
we planted here, instead of against an external reference."""

from dataclasses import dataclass

import numpy as np

from cotton_math_lab.exceptions import InvalidSpecError

FEATURES: tuple[str, ...] = (
    "micronaire",  # fineness/maturity index
    "uhml",  # length (mm)
    "uniformity",  # uniformity index (%)
    "strength",  # strength (g/tex)
    "elongation",  # elongation (%)
    "rd",  # reflectance (degree of whiteness)
    "plus_b",  # yellowness
    "trash",  # trash area (%)
)


@dataclass(frozen=True, eq=False)
class HVISpec:
    """Population parameters of a multivariate normal HVI distribution."""

    features: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    correlation: np.ndarray

    def __post_init__(self) -> None:
        self._validate()
        for arr in (self.means, self.stds, self.correlation):
            arr.setflags(write=False)

    def _validate(self) -> None:
        k = len(self.features)

        if self.means.shape != (k,) or self.stds.shape != (k,):
            raise InvalidSpecError(
                f"incompatible dimensions: {k} features, "
                f"means={self.means.shape}, stds={self.stds.shape}"
            )

        if self.correlation.shape != (k, k):
            raise InvalidSpecError(
                f"incompatible dimensions: correlation {self.correlation.shape} "
                f"for {k} features"
            )

        if np.any(self.stds <= 0):
            bad = [f for f, s in zip(self.features, self.stds, strict=True) if s <= 0]
            raise InvalidSpecError(f"standard deviation must be positive: {bad}")

        if not np.allclose(self.correlation, self.correlation.T, atol=1e-12):
            raise InvalidSpecError("correlation matrix must be symmetric")

        if not np.allclose(np.diag(self.correlation), 1.0, atol=1e-12):
            raise InvalidSpecError("correlation diagonal must be 1.0")

        try:
            np.linalg.cholesky(self.correlation)
        except np.linalg.LinAlgError as exc:
            eigenvalues = np.linalg.eigvalsh(self.correlation)
            raise InvalidSpecError(
                f"correlation matrix is not positive-definite "
                f"(smallest eigenvalue: {eigenvalues.min():.4f})"
            ) from exc

    @property
    def covariance(self) -> np.ndarray:
        """Σ = D · R · D, where D = diag(σ)."""
        d = np.diag(self.stds)
        return d @ self.correlation @ d


def _build_correlation(pairs: dict[tuple[str, str], float]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(FEATURES)}
    corr = np.eye(len(FEATURES), dtype=np.float64)
    for (a, b), value in pairs.items():
        corr[idx[a], idx[b]] = value
        corr[idx[b], idx[a]] = value
    return corr


def default_spec() -> HVISpec:
    """Spec with typical values for Brazilian upland cotton."""
    means = np.array([4.30, 29.0, 82.5, 30.0, 6.5, 75.0, 8.5, 0.60])
    stds = np.array([0.40, 1.20, 1.50, 2.50, 0.80, 3.00, 1.00, 0.30])

    correlation = _build_correlation(
        {
            # fiber maturity pulls micronaire and strength together
            ("micronaire", "strength"): 0.25,
            ("micronaire", "elongation"): 0.20,
            # longer fiber tends to be finer
            ("micronaire", "uhml"): -0.15,
            # length, uniformity, and strength form the "fiber" block
            ("uhml", "uniformity"): 0.55,
            ("uhml", "strength"): 0.45,
            ("uniformity", "strength"): 0.30,
            # classic trade-off: strength × elongation
            ("strength", "elongation"): -0.20,
            # "color" block: whiter, less yellow
            ("rd", "plus_b"): -0.45,
            # trash dirties the color
            ("trash", "rd"): -0.35,
            ("trash", "plus_b"): 0.20,
        }
    )

    return HVISpec(
        features=FEATURES,
        means=means,
        stds=stds,
        correlation=correlation,
    )


def generate_bales(spec: HVISpec, n: int, seed: int) -> np.ndarray:
    """Samples `n` bales from the multivariate normal defined by `spec`.

    Uses the Cholesky decomposition Σ = L·Lᵀ: if z ~ N(0, I), then
    μ + L·z ~ N(μ, Σ). Returns a (n, k) float64 array.
    """
    rng = np.random.default_rng(seed)
    lower = np.linalg.cholesky(spec.covariance)
    standard = rng.standard_normal((n, len(spec.features)))
    return spec.means + standard @ lower.T


def generate_quality_labels(
    bales: np.ndarray,
    spec: HVISpec,
    *,
    seed: int,
    noise_std: float = 1.0,
) -> np.ndarray:
    """Synthetic binary label: "premium bale" via a linear combination of
    standardized features — more strength, uniformity, and length; less
    trash — plus Gaussian noise. A linear threshold with realistic class
    overlap (the accuracy ceiling isn't 100%), not a trivially separable
    problem.
    """
    idx = {name: i for i, name in enumerate(spec.features)}
    standardized = (bales - bales.mean(axis=0)) / bales.std(axis=0, ddof=1)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, len(bales))

    score = (
        0.6 * standardized[:, idx["strength"]]
        + 0.4 * standardized[:, idx["uniformity"]]
        + 0.3 * standardized[:, idx["uhml"]]
        - 0.5 * standardized[:, idx["trash"]]
        + noise
    )
    return (score > 0).astype(np.float64)
