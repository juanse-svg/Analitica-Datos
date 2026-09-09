"""Tests del módulo train.py — Lab 04."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from analytics.train import (
    cargar_pipeline,
    entrenar_pipeline_clasificacion,
    entrenar_pipeline_regresion,
    guardar_pipeline,
)


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def datos_regresion():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "n_items": rng.integers(1, 5, size=50).astype(float),
        "precio_total": rng.uniform(10, 500, size=50),
        "ticket_promedio_historico": rng.uniform(10, 500, size=50),
    })
    X.loc[::10, "ticket_promedio_historico"] = np.nan  # nulos deliberados
    y = pd.Series(rng.uniform(1, 30, size=50))
    return X, y


@pytest.fixture
def datos_clasificacion():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "n_items": rng.integers(1, 5, size=50).astype(float),
        "precio_total": rng.uniform(10, 500, size=50),
        "tiempo_entrega_dias": rng.uniform(1, 30, size=50),
    })
    # Clases desbalanceadas: 85% False, 15% True — como review_negativa real
    y = pd.Series(rng.random(size=50) < 0.15)
    return X, y


# ── tests entrenar_pipeline_regresion ───────────────────────────────

def test_entrenar_pipeline_regresion_retorna_pipeline(datos_regresion):
    X, y = datos_regresion
    pipeline = entrenar_pipeline_regresion(X, y, n_estimators=10)
    assert isinstance(pipeline, Pipeline)


def test_pipeline_regresion_puede_predecir(datos_regresion):
    X, y = datos_regresion
    pipeline = entrenar_pipeline_regresion(X, y, n_estimators=10)
    predicciones = pipeline.predict(X)
    assert len(predicciones) == len(X)


def test_pipeline_regresion_maneja_nulos_sin_fallar(datos_regresion):
    X, y = datos_regresion
    assert X["ticket_promedio_historico"].isna().any()  # confirma que hay nulos
    pipeline = entrenar_pipeline_regresion(X, y, n_estimators=10)
    predicciones = pipeline.predict(X)  # no debe lanzar excepción
    assert not np.isnan(predicciones).any()


# ── tests entrenar_pipeline_clasificacion ───────────────────────────

def test_entrenar_pipeline_clasificacion_retorna_pipeline(datos_clasificacion):
    X, y = datos_clasificacion
    pipeline = entrenar_pipeline_clasificacion(X, y, n_estimators=10)
    assert isinstance(pipeline, Pipeline)


def test_pipeline_clasificacion_predict_proba_disponible(datos_clasificacion):
    X, y = datos_clasificacion
    pipeline = entrenar_pipeline_clasificacion(X, y, n_estimators=10)
    probabilidades = pipeline.predict_proba(X)
    assert probabilidades.shape == (len(X), 2)


def test_pipeline_clasificacion_usa_class_weight_balanced(datos_clasificacion):
    X, y = datos_clasificacion
    pipeline = entrenar_pipeline_clasificacion(X, y, n_estimators=10)
    assert pipeline.named_steps["modelo"].class_weight == "balanced"


# ── tests guardar_pipeline / cargar_pipeline ────────────────────────

def test_guardar_y_cargar_pipeline_roundtrip(datos_regresion, tmp_path):
    X, y = datos_regresion
    pipeline_original = entrenar_pipeline_regresion(X, y, n_estimators=10)

    ruta = tmp_path / "pipeline_test.joblib"
    guardar_pipeline(pipeline_original, str(ruta))
    assert ruta.exists()

    pipeline_cargado = cargar_pipeline(str(ruta))
    np.testing.assert_array_almost_equal(
        pipeline_original.predict(X), pipeline_cargado.predict(X)
    )
