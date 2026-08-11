# cotton-math-lab

Laboratório pessoal de matemática e ML aplicado, implementado do zero em
Python/NumPy e validado contra bibliotecas de referência (`scipy`,
`scikit-learn`, `torch`). O domínio de exemplo é dados sintéticos de HVI
(*High Volume Instrument*) de fardos de algodão — gerados com parâmetros
populacionais conhecidos por construção, o que permite verificar
objetivamente se cada algoritmo recupera a verdade plantada nos dados.

O objetivo não é produzir uma biblioteca de uso geral, e sim documentar,
módulo a módulo, as decisões de design e as descobertas ao reimplementar
métodos numéricos e estatísticos fundamentais. Cada módulo tem um
documento correspondente em [`docs/`](docs/) com o raciocínio por trás das
escolhas de implementação.

## Estrutura

```
src/cotton_math_lab/
├── data/         # gerador sintético de dados HVI (docs/00-data.md)
├── linalg/       # power iteration, deflação, QR, SVD, PCA (docs/01-linalg.md)
├── autodiff/     # motor de diferenciação automática em modo reverso,
│                 # diferenças finitas, Jacobiano/Hessiano, otimizadores
│                 # (SGD, Momentum, Nesterov, RMSProp, Adam, AdamW),
│                 # regressão logística (docs/02-autodiff.md)
├── stats/        # MLE, MAP, Beta-Binomial, intervalos, bootstrap
│                 # (docs/03-stats.md)
├── models/       # regressão logística
└── infotheory/   # entropia, KL, informação mútua, Jensen-Shannon/drift
                  # (docs/05-infotheory.md)
```

Otimização de segunda ordem e benchmarks (Rosenbrock, quadráticas
mal-condicionadas) estão documentados em `docs/04-optim.md`.

## Setup

Requer Python 3.12+ e [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
```

## Testes

Os testes são organizados por marker:

- `unit` — testes rápidos, sem dependência externa
- `oracle` — valida a implementação contra scipy/sklearn/torch
- `slow` — testes de convergência ou estatísticos com muitas amostras

```bash
uv run pytest                    # todos os testes
uv run pytest -m unit            # apenas unitários
uv run pytest -m oracle          # apenas validação contra oráculos
```

## Lint e formatação

```bash
make lint          # ruff check
make lint-fix       # ruff check --fix
make format         # ruff format
make format-check    # ruff format --check
make check          # lint + format-check
```

## Documentação

Cada módulo tem um documento em `docs/` explicando o "porquê" das decisões
de implementação — não apenas o que o código faz, mas por que ele foi
escrito daquela forma:

- [`00-data.md`](docs/00-data.md) — gerador sintético de dados HVI
- [`01-linalg.md`](docs/01-linalg.md) — álgebra linear
- [`02-autodiff.md`](docs/02-autodiff.md) — diferenciação automática
- [`03-stats.md`](docs/03-stats.md) — probabilidade e estatística
- [`04-optim.md`](docs/04-optim.md) — otimização
- [`05-infotheory.md`](docs/05-infotheory.md) — teoria da informação
