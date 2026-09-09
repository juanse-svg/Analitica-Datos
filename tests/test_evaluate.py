"""Tests del módulo evaluate.py — Lab 04."""

import pytest
from analytics.evaluate import (
    comparar_con_baseline,
    evaluar_clasificacion,
    evaluar_regresion,
)


# ── tests evaluar_regresion ──────────────────────────────────────────

def test_evaluar_regresion_retorna_dict_con_claves():
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [12.0, 18.0, 33.0, 37.0]
    resultado = evaluar_regresion(y_true, y_pred)
    assert set(resultado.keys()) == {"rmse", "mae", "r2"}


def test_evaluar_regresion_valores_correctos():
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [12.0, 18.0, 33.0, 37.0]
    resultado = evaluar_regresion(y_true, y_pred)
    assert resultado["rmse"] == pytest.approx(2.5495, abs=0.001)
    assert resultado["mae"] == pytest.approx(2.5, abs=0.001)
    assert resultado["r2"] == pytest.approx(0.948, abs=0.001)


# ── tests evaluar_clasificacion ─────────────────────────────────────

def test_evaluar_clasificacion_retorna_dict_con_claves():
    y_true = [True, True, False, False, False]
    y_pred = [True, False, False, False, True]
    y_proba = [0.9, 0.4, 0.2, 0.3, 0.6]
    resultado = evaluar_clasificacion(y_true, y_pred, y_proba)
    assert set(resultado.keys()) == {"accuracy", "f1", "roc_auc", "matriz_confusion"}


def test_evaluar_clasificacion_valores_correctos():
    y_true = [True, True, False, False, False]
    y_pred = [True, False, False, False, True]
    y_proba = [0.9, 0.4, 0.2, 0.3, 0.6]
    resultado = evaluar_clasificacion(y_true, y_pred, y_proba)
    assert resultado["accuracy"] == pytest.approx(0.6)
    assert resultado["f1"] == pytest.approx(0.5)
    assert resultado["roc_auc"] == pytest.approx(0.8333, abs=0.001)
    assert resultado["matriz_confusion"] == [[2, 1], [1, 1]]


# ── tests comparar_con_baseline ─────────────────────────────────────

def test_comparar_con_baseline_calcula_accuracy_baseline_correctamente():
    y_train = [False, False, False, True]  # mayoría: False
    y_eval = [False, False, True, True]
    y_pred_modelo = [False, True, True, True]
    
    resultado = comparar_con_baseline(y_train, y_eval, y_pred_modelo)
    assert resultado["accuracy_baseline"] == pytest.approx(0.5)


def test_comparar_con_baseline_calcula_mejora_correctamente():
    y_train = [False, False, False, True]
    y_eval = [False, False, True, True]
    y_pred_modelo = [False, True, True, True]
    
    resultado = comparar_con_baseline(y_train, y_eval, y_pred_modelo)
    assert resultado["accuracy_modelo"] == pytest.approx(0.75)
    assert resultado["mejora"] == pytest.approx(0.25)
