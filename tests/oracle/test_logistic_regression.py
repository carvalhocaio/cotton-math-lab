import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from cotton_math_lab.data.hvi import (
    default_spec,
    generate_bales,
    generate_quality_labels,
)
from cotton_math_lab.models.logistic_regression import LogisticRegression

N_TOTAL = 250
N_TRAIN = 180


def _split_dataset():
    spec = default_spec()
    bales = generate_bales(spec, n=N_TOTAL, seed=2024)
    standardized = (bales - bales.mean(axis=0)) / bales.std(axis=0, ddof=1)
    labels = generate_quality_labels(bales, spec, seed=99)

    x_train, y_train = standardized[:N_TRAIN], labels[:N_TRAIN]
    x_test, y_test = standardized[N_TRAIN:], labels[N_TRAIN:]
    return x_train, y_train, x_test, y_test


@pytest.mark.slow
def test_loss_decreases_substantially_during_training():
    x_train, y_train, _, _ = _split_dataset()
    model = LogisticRegression(n_features=x_train.shape[1])

    history = model.fit(x_train, y_train)

    assert history[-1] < history[0] * 0.75


@pytest.mark.slow
def test_predictions_are_binary():
    x_train, y_train, x_test, _ = _split_dataset()
    model = LogisticRegression(n_features=x_train.shape[1])
    model.fit(x_train, y_train)

    predictions = [model.predict(row) for row in x_test]
    assert set(predictions) <= {0.0, 1.0}


@pytest.mark.slow
def test_test_accuracy_matches_sklearn():
    """O teste que fecha o módulo: nosso motor, do zero, treina um modelo
    que generaliza tão bem quanto uma biblioteca de produção no mesmo
    problema - a prova de que Tensor + SGD + as primitivas compôem um
    motor de ML que funciona de verdade, não só em exemplos de brinquedo."""
    x_train, y_train, x_test, y_test = _split_dataset()

    model = LogisticRegression(n_features=x_train.shape[1])
    model.fit(x_train, y_train)
    our_predictions = np.array([model.predict(row) for row in x_test])
    our_accuracy = (our_predictions == y_test).mean()

    reference = SklearnLogisticRegression().fit(x_train, y_train)
    reference_accuracy = reference.score(x_test, y_test)

    assert abs(our_accuracy - reference_accuracy) < 0.1
    assert our_accuracy > 0.6
