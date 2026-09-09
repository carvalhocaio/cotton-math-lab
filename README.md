# cotton-math-lab

Personal math and applied ML lab, implemented from scratch in
Python/NumPy and validated against reference libraries (`scipy`,
`scikit-learn`, `torch`). The example domain is synthetic HVI (*High
Volume Instrument*) data for cotton bales — generated with population
parameters known by construction, which lets each algorithm be
objectively checked against whether it recovers the truth planted in
the data.

The goal is not to produce a general-purpose library, but to document,
module by module, the design decisions and discoveries made while
reimplementing fundamental numerical and statistical methods. Each
module has a corresponding document in [`docs/`](docs/) with the
reasoning behind the implementation choices.

## Structure

```
src/cotton_math_lab/
├── data/         # synthetic HVI data generator (docs/00-data.md)
├── linalg/       # power iteration, deflation, QR, SVD, PCA (docs/01-linalg.md)
├── autodiff/     # reverse-mode automatic differentiation engine,
│                 # finite differences, Jacobian/Hessian, optimizers
│                 # (SGD, Momentum, Nesterov, RMSProp, Adam, AdamW),
│                 # logistic regression (docs/02-autodiff.md)
├── stats/        # MLE, MAP, Beta-Binomial, intervals, bootstrap
│                 # (docs/03-stats.md)
├── models/       # logistic regression
└── infotheory/   # entropy, KL, mutual information, Jensen-Shannon/drift
                  # (docs/05-infotheory.md)
```

Second-order optimization and benchmarks (Rosenbrock, ill-conditioned
quadratics) are documented in `docs/04-optim.md`.

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
```

## Tests

Tests are organized by marker:

- `unit` — fast tests, no external dependency
- `oracle` — validates the implementation against scipy/sklearn/torch
- `slow` — convergence or statistical tests with many samples

```bash
uv run pytest                    # all tests
uv run pytest -m unit            # unit tests only
uv run pytest -m oracle          # oracle validation only
```

## Linting and formatting

```bash
make lint          # ruff check
make lint-fix       # ruff check --fix
make format         # ruff format
make format-check    # ruff format --check
make check          # lint + format-check
```

## Documentation

Each module has a document in `docs/` explaining the "why" behind the
implementation decisions — not just what the code does, but why it was
written that way:

- [`00-data.md`](docs/00-data.md) — synthetic HVI data generator
- [`01-linalg.md`](docs/01-linalg.md) — linear algebra
- [`02-autodiff.md`](docs/02-autodiff.md) — automatic differentiation
- [`03-stats.md`](docs/03-stats.md) — probability and statistics
- [`04-optim.md`](docs/04-optim.md) — optimization
- [`05-infotheory.md`](docs/05-infotheory.md) — information theory
