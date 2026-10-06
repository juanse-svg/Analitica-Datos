"""Tests del módulo segment.py — Lab 05."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from analytics.segment import (
    calcular_rfm,
    entrenar_pipeline_clustering,
    evaluar_clustering,
    perfilar_segmentos,
)


# ── tests calcular_rfm ───────────────────────────────────────────────
@pytest.fixture
def pedidos_ejemplo():
    return pd.DataFrame(
        {
            "customer_unique_id": ["A", "A", "B", "C", "C", "C"],
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2018-08-01",
                    "2018-08-15",
                    "2018-07-01",
                    "2018-06-01",
                    "2018-06-10",
                    "2018-06-20",
                ]
            ),
            "monto_pedido": [100.0, 150.0, 200.0, 50.0, 50.0, 50.0],
        }
    )


def test_calcular_rfm_retorna_columnas_correctas(pedidos_ejemplo):
    resultado = calcular_rfm(pedidos_ejemplo, fecha_referencia=pd.Timestamp("2018-09-01"))
    assert set(resultado.columns) == {"customer_unique_id", "recencia", "frecuencia", "monto"}
    assert len(resultado) == 3  # 3 clientes únicos: A, B, C


def test_calcular_rfm_valores_correctos(pedidos_ejemplo):
    resultado = calcular_rfm(pedidos_ejemplo, fecha_referencia=pd.Timestamp("2018-09-01"))
    resultado = resultado.set_index("customer_unique_id")
    assert resultado.loc["A", "recencia"] == 17
    assert resultado.loc["A", "frecuencia"] == 2
    assert resultado.loc["A", "monto"] == pytest.approx(250.0)
    assert resultado.loc["B", "recencia"] == 62
    assert resultado.loc["B", "frecuencia"] == 1
    assert resultado.loc["B", "monto"] == pytest.approx(200.0)
    assert resultado.loc["C", "recencia"] == 73
    assert resultado.loc["C", "frecuencia"] == 3
    assert resultado.loc["C", "monto"] == pytest.approx(150.0)


# ── tests entrenar_pipeline_clustering ──────────────────────────────
def test_entrenar_pipeline_clustering_retorna_pipeline():
    X = pd.DataFrame({"a": [0, 1, 20, 21], "b": [0, 1, 20, 21]})
    pipeline = entrenar_pipeline_clustering(X, n_clusters=2)
    assert isinstance(pipeline, Pipeline)


def test_entrenar_pipeline_clustering_n_clusters_correcto():
    X = pd.DataFrame({"a": [0, 1, 20, 21], "b": [0, 1, 20, 21]})
    pipeline = entrenar_pipeline_clustering(X, n_clusters=2)
    assert pipeline.named_steps["modelo"].n_clusters == 2


def test_pipeline_clustering_agrupa_puntos_separados_correctamente():
    X = pd.DataFrame({"a": [0, 1, 20, 21], "b": [0, 1, 20, 21]})
    pipeline = entrenar_pipeline_clustering(X, n_clusters=2, random_state=42)
    etiquetas = pipeline.predict(X)
    assert etiquetas[0] == etiquetas[1]  # (0,0) y (1,1): mismo cluster
    assert etiquetas[2] == etiquetas[3]  # (20,20) y (21,21): mismo cluster
    assert etiquetas[0] != etiquetas[2]  # clusters distintos entre grupos


# ── tests evaluar_clustering ────────────────────────────────────────
def test_evaluar_clustering_retorna_dict_con_claves():
    X = [[0, 0], [0, 1], [10, 0], [10, 1]]
    etiquetas = [0, 0, 1, 1]
    resultado = evaluar_clustering(X, etiquetas)
    assert set(resultado.keys()) == {"silhouette", "davies_bouldin"}


def test_evaluar_clustering_valores_correctos():
    X = [[0, 0], [0, 1], [10, 0], [10, 1]]
    etiquetas = [0, 0, 1, 1]
    resultado = evaluar_clustering(X, etiquetas)
    assert resultado["silhouette"] == pytest.approx(0.9002, abs=0.001)
    assert resultado["davies_bouldin"] == pytest.approx(0.1, abs=0.001)


# ── tests perfilar_segmentos ────────────────────────────────────────
@pytest.fixture
def df_segmentado():
    return pd.DataFrame(
        {
            "segmento": [0, 0, 1, 1],
            "recencia": [10.0, 20.0, 100.0, 120.0],
            "monto": [500.0, 300.0, 50.0, 70.0],
        }
    )


def test_perfilar_segmentos_retorna_columnas_correctas(df_segmentado):
    resultado = perfilar_segmentos(
        df_segmentado, columna_segmento="segmento", columnas_perfil=["recencia", "monto"]
    )
    assert set(resultado.columns) == {"recencia", "monto", "n_clientes"}
    assert len(resultado) == 2


def test_perfilar_segmentos_valores_correctos(df_segmentado):
    resultado = perfilar_segmentos(
        df_segmentado, columna_segmento="segmento", columnas_perfil=["recencia", "monto"]
    )
    assert resultado.loc[0, "recencia"] == pytest.approx(15.0)
    assert resultado.loc[0, "monto"] == pytest.approx(400.0)
    assert resultado.loc[0, "n_clientes"] == 2
    assert resultado.loc[1, "recencia"] == pytest.approx(110.0)
    assert resultado.loc[1, "monto"] == pytest.approx(60.0)
    assert resultado.loc[1, "n_clientes"] == 2
