"""Tests del módulo pipeline.py — Lab 06."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from analytics.pipeline import (
    ejecutar_pipeline,
    entrenar_y_evaluar,
    guardar_artefactos,
    preparar_datos,
    unir_reviews,
)

# ── tests unir_reviews ───────────────────────────────────────────────


@pytest.fixture
def pedidos_y_reviews():
    pedidos = pd.DataFrame({"order_id": ["p1", "p2", "p3"]})
    reviews = pd.DataFrame(
        {
            "order_id": ["p1", "p2", "p2"],  # p2 tiene dos reseñas; p3 ninguna
            "review_score": [5, 1, 4],
            "review_answer_timestamp": pd.to_datetime(["2018-01-05", "2018-02-01", "2018-02-10"]),
        }
    )
    return pedidos, reviews


def test_unir_reviews_no_multiplica_filas(pedidos_y_reviews):
    pedidos, reviews = pedidos_y_reviews
    resultado = unir_reviews(pedidos, reviews)
    assert len(resultado) == 3


def test_unir_reviews_conserva_la_resena_mas_reciente(pedidos_y_reviews):
    pedidos, reviews = pedidos_y_reviews
    resultado = unir_reviews(pedidos, reviews).set_index("order_id")
    assert resultado.loc["p1", "review_score"] == 5
    assert resultado.loc["p2", "review_score"] == 4  # la del 2018-02-10
    assert pd.isna(resultado.loc["p3", "review_score"])


# ── tests preparar_datos ─────────────────────────────────────────────


def test_preparar_datos_descarta_target_nulo_y_respeta_features():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "otra": [0, 0, 0, 0],
            "y": [1.5, 2.5, np.nan, 4.5],
        }
    )
    X, y = preparar_datos(df, features=["b", "a"], target="y")
    assert list(X.columns) == ["b", "a"]
    assert len(X) == len(y) == 3
    assert X["a"].isna().sum() == 1  # los nulos de las features se conservan


# ── tests entrenar_y_evaluar ─────────────────────────────────────────


@pytest.fixture
def datos_sinteticos():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(0, 10, size=80),
            "x2": rng.uniform(0, 10, size=80),
        }
    )
    y_reg = pd.Series(2 * X["x1"] + rng.normal(0, 1, size=80))
    y_clf = pd.Series(X["x1"] > 7)
    return X, y_reg, y_clf


def test_entrenar_y_evaluar_regresion(datos_sinteticos):
    X, y_reg, _ = datos_sinteticos
    pipeline, metricas = entrenar_y_evaluar(X, y_reg, "regresion")
    assert isinstance(pipeline, Pipeline)
    assert set(metricas) == {"rmse", "mae", "r2"}


def test_entrenar_y_evaluar_clasificacion_incluye_baseline(datos_sinteticos):
    X, _, y_clf = datos_sinteticos
    pipeline, metricas = entrenar_y_evaluar(X, y_clf, "clasificacion")
    assert isinstance(pipeline, Pipeline)
    assert set(metricas) == {
        "accuracy",
        "f1",
        "roc_auc",
        "matriz_confusion",
        "accuracy_baseline",
        "accuracy_modelo",
        "mejora",
    }


def test_entrenar_y_evaluar_rechaza_tipo_desconocido(datos_sinteticos):
    X, y_reg, _ = datos_sinteticos
    with pytest.raises(ValueError):
        entrenar_y_evaluar(X, y_reg, "clustering")


# ── tests guardar_artefactos ─────────────────────────────────────────


def test_guardar_artefactos_escribe_modelos_y_metricas(datos_sinteticos, tmp_path):
    X, y_reg, _ = datos_sinteticos
    pipeline, metricas = entrenar_y_evaluar(X, y_reg, "regresion")
    ruta = guardar_artefactos(
        {"modelo_x": pipeline}, {"modelo_x": metricas}, str(tmp_path / "models")
    )

    assert (tmp_path / "models" / "modelo_x.joblib").exists()
    registro = json.loads(ruta.read_text(encoding="utf-8"))
    assert {"fecha_entrenamiento", "version_sklearn", "metricas"} <= set(registro)
    assert registro["metricas"]["modelo_x"]["rmse"] == pytest.approx(metricas["rmse"])


# ── tests ejecutar_pipeline (integración con CSV sintéticos) ─────────


def _escribir_olist_sintetico(directorio, n=60, status_invalido=False):
    """Escribe 5 CSV con la misma estructura de Olist, con n pedidos."""
    rng = np.random.default_rng(7)
    ids = [f"a{i:031x}" for i in range(n)]  # 32 caracteres, con letras como en Olist
    compra = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")
    entrega = compra + pd.to_timedelta(rng.integers(2, 30, n), unit="D")
    estimada = compra + pd.to_timedelta(15, unit="D")
    clientes = [f"c{i % 40}" for i in range(n)]
    status = ["delivered"] * n
    if status_invalido:
        status[0] = "perdido"
    pd.DataFrame(
        {
            "order_id": ids,
            "customer_id": clientes,
            "order_status": status,
            "order_purchase_timestamp": compra,
            "order_approved_at": compra,
            "order_delivered_carrier_date": compra,
            "order_delivered_customer_date": entrega,
            "order_estimated_delivery_date": estimada,
        }
    ).to_csv(directorio / "olist_orders_dataset.csv", index=False)
    pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(40)],
            "customer_unique_id": [f"u{i}" for i in range(40)],
            "customer_zip_code_prefix": 10000,
            "customer_city": "sao paulo",
            "customer_state": rng.choice(["SP", "RJ", "MG"], 40),
        }
    ).to_csv(directorio / "olist_customers_dataset.csv", index=False)
    pd.DataFrame(
        {
            "order_id": ids,
            "order_item_id": 1,
            "product_id": "prod",
            "seller_id": "sel",
            "shipping_limit_date": compra,
            "price": rng.uniform(20, 500, n).round(2),
            "freight_value": rng.uniform(5, 50, n).round(2),
        }
    ).to_csv(directorio / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame(
        {
            "order_id": ids,
            "payment_sequential": 1,
            "payment_type": "credit_card",
            "payment_installments": 1,
            "payment_value": rng.uniform(20, 500, n).round(2),
        }
    ).to_csv(directorio / "olist_order_payments_dataset.csv", index=False)
    dias = (entrega - compra).days
    pd.DataFrame(
        {
            "review_id": ids,
            "order_id": ids,
            "review_score": np.where(dias > 20, 1, 5),  # lentas → negativa
            "review_comment_title": "",
            "review_comment_message": "",
            "review_creation_date": entrega,
            "review_answer_timestamp": entrega,
        }
    ).to_csv(directorio / "olist_order_reviews_dataset.csv", index=False)


def test_ejecutar_pipeline_de_punta_a_punta(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _escribir_olist_sintetico(data_dir)

    metricas = ejecutar_pipeline(str(data_dir), str(tmp_path / "models"))

    assert set(metricas) == {"pipeline_tiempo_entrega", "pipeline_review_negativa"}
    assert (tmp_path / "models" / "pipeline_tiempo_entrega.joblib").exists()
    assert (tmp_path / "models" / "pipeline_review_negativa.joblib").exists()
    assert (tmp_path / "models" / "metricas.json").exists()


def test_ejecutar_pipeline_se_detiene_con_datos_invalidos(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _escribir_olist_sintetico(data_dir, status_invalido=True)

    with pytest.raises(ValueError):
        ejecutar_pipeline(str(data_dir), str(tmp_path / "models"))
    assert not (tmp_path / "models").exists()  # no se guardó ningún modelo
