"""Tests del módulo api.py — Lab 06."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from analytics.api import crear_app
from analytics.pipeline import (
    FEATURES_CLASIFICACION,
    FEATURES_REGRESION,
    entrenar_y_evaluar,
    guardar_artefactos,
)

PEDIDO_COMPRA = {
    "n_items": 2,
    "precio_total": 189.9,
    "flete_total": 24.5,
    "customer_state_freq": 0.42,
    "dia_semana_compra": 3,
    "mes_compra": 5,
    "ticket_promedio_historico": 150.0,
}

PEDIDO_ENTREGADO = {
    "n_items": 1,
    "precio_total": 89.0,
    "flete_total": 15.0,
    "customer_state_freq": 0.13,
    "tiempo_entrega_dias": 25.0,
    "entregado_tarde": True,
    "ticket_promedio_historico": None,
}


@pytest.fixture
def models_dir(tmp_path):
    """Entrena dos modelos pequeños con datos sintéticos y los guarda."""
    rng = np.random.default_rng(0)
    n = 80
    datos = pd.DataFrame(
        {col: rng.uniform(0, 1, n) for col in set(FEATURES_REGRESION + FEATURES_CLASIFICACION)}
    )
    pipe_reg, met_reg = entrenar_y_evaluar(
        datos[FEATURES_REGRESION], pd.Series(rng.uniform(2, 30, n)), "regresion"
    )
    pipe_clf, met_clf = entrenar_y_evaluar(
        datos[FEATURES_CLASIFICACION], pd.Series(rng.random(n) < 0.3), "clasificacion"
    )
    destino = tmp_path / "models"
    guardar_artefactos(
        {"pipeline_tiempo_entrega": pipe_reg, "pipeline_review_negativa": pipe_clf},
        {"pipeline_tiempo_entrega": met_reg, "pipeline_review_negativa": met_clf},
        str(destino),
    )
    return destino


@pytest.fixture
def cliente(models_dir):
    return TestClient(crear_app(str(models_dir)))


def test_salud(cliente):
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "ok"
    assert "fecha_entrenamiento" in respuesta.json()


def test_predecir_entrega_retorna_numero(cliente):
    respuesta = cliente.post("/predecir/entrega", json=PEDIDO_COMPRA)
    assert respuesta.status_code == 200
    assert isinstance(respuesta.json()["tiempo_entrega_dias"], float)


def test_predecir_entrega_acepta_ticket_historico_nulo(cliente):
    # Primer pedido del cliente: todavía no tiene ticket histórico
    pedido = {**PEDIDO_COMPRA, "ticket_promedio_historico": None}
    respuesta = cliente.post("/predecir/entrega", json=pedido)
    assert respuesta.status_code == 200


def test_predecir_resena_retorna_probabilidad(cliente):
    respuesta = cliente.post("/predecir/resena", json=PEDIDO_ENTREGADO)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert 0.0 <= cuerpo["probabilidad_resena_negativa"] <= 1.0
    assert cuerpo["resena_negativa"] == (cuerpo["probabilidad_resena_negativa"] >= 0.5)


def test_predecir_entrega_rechaza_n_items_cero(cliente):
    respuesta = cliente.post("/predecir/entrega", json={**PEDIDO_COMPRA, "n_items": 0})
    assert respuesta.status_code == 422


def test_predecir_resena_rechaza_campo_faltante(cliente):
    pedido = {k: v for k, v in PEDIDO_ENTREGADO.items() if k != "entregado_tarde"}
    respuesta = cliente.post("/predecir/resena", json=pedido)
    assert respuesta.status_code == 422


def test_crear_app_falla_sin_modelos(tmp_path):
    with pytest.raises(FileNotFoundError):
        crear_app(str(tmp_path / "vacio"))
