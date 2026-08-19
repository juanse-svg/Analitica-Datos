"""Tests del módulo transform.py — Lab 03."""

import pandas as pd
import pytest

from analytics.transform import (
    agregar_encoding_categoria_producto,
    agregar_encoding_estado,
    agregar_features_fecha,
    agregar_ticket_historico,
)


# ── fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def df_fechas():
    return pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2021-01-04", "2021-01-05", "2021-01-09"]  # lunes, martes, sábado
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2021-01-10", "2021-01-20", "2021-01-11"]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2021-01-15", "2021-01-15", "2021-01-10"]
            ),
        }
    )


@pytest.fixture
def df_estados():
    return pd.DataFrame(
        {
            "customer_state": ["SP", "SP", "SP", "RJ", "MG"],
        }
    )


@pytest.fixture
def df_pedidos_cliente():
    return pd.DataFrame(
        {
            "customer_id": ["x", "x", "x", "y"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2021-01-01", "2021-02-01", "2021-03-01", "2021-01-15"]
            ),
            "precio_total": [100.0, 200.0, 300.0, 50.0],
        }
    )


# ── tests agregar_features_fecha ────────────────────────────────────
def test_agrega_columnas_esperadas(df_fechas):
    resultado = agregar_features_fecha(df_fechas)
    columnas_esperadas = {
        "tiempo_entrega_dias",
        "retraso_entrega_dias",
        "entregado_tarde",
        "dia_semana_compra",
        "es_fin_de_semana",
        "mes_compra",
    }
    assert columnas_esperadas.issubset(resultado.columns)


def test_tiempo_entrega_dias_correcto(df_fechas):
    resultado = agregar_features_fecha(df_fechas)
    assert resultado.loc[0, "tiempo_entrega_dias"] == 6
    assert resultado.loc[1, "tiempo_entrega_dias"] == 15


def test_entregado_tarde_detecta_retraso(df_fechas):
    resultado = agregar_features_fecha(df_fechas)
    assert resultado.loc[1, "entregado_tarde"] == True  # entregado 5 días tarde
    assert resultado.loc[0, "entregado_tarde"] == False  # entregado a tiempo


def test_es_fin_de_semana_correcto(df_fechas):
    resultado = agregar_features_fecha(df_fechas)
    assert resultado.loc[2, "es_fin_de_semana"] == True  # 2021-01-09 es sábado
    assert resultado.loc[0, "es_fin_de_semana"] == False  # 2021-01-04 es lunes


def test_agregar_features_fecha_no_muta_original(df_fechas):
    columnas_antes = list(df_fechas.columns)
    agregar_features_fecha(df_fechas)
    assert list(df_fechas.columns) == columnas_antes


# ── tests agregar_encoding_estado ───────────────────────────────────
def test_frecuencia_estado_mas_comun(df_estados):
    resultado = agregar_encoding_estado(df_estados)
    assert resultado.loc[0, "customer_state_freq"] == pytest.approx(0.6)


def test_frecuencia_estado_menos_comun(df_estados):
    resultado = agregar_encoding_estado(df_estados)
    assert resultado.loc[3, "customer_state_freq"] == pytest.approx(0.2)


def test_encoding_no_muta_original(df_estados):
    columnas_antes = list(df_estados.columns)
    agregar_encoding_estado(df_estados)
    assert list(df_estados.columns) == columnas_antes


# ── tests agregar_ticket_historico ──────────────────────────────────
def test_primer_pedido_es_nulo(df_pedidos_cliente):
    resultado = agregar_ticket_historico(df_pedidos_cliente)
    primer_pedido_x = resultado[resultado["customer_id"] == "x"].iloc[0]
    assert pd.isna(primer_pedido_x["ticket_promedio_historico"])


def test_segundo_pedido_usa_solo_el_primero(df_pedidos_cliente):
    resultado = agregar_ticket_historico(df_pedidos_cliente)
    segundo_pedido_x = resultado[resultado["customer_id"] == "x"].iloc[1]
    assert segundo_pedido_x["ticket_promedio_historico"] == pytest.approx(100.0)


def test_tercer_pedido_promedia_los_dos_anteriores(df_pedidos_cliente):
    resultado = agregar_ticket_historico(df_pedidos_cliente)
    tercer_pedido_x = resultado[resultado["customer_id"] == "x"].iloc[2]
    assert tercer_pedido_x["ticket_promedio_historico"] == pytest.approx(150.0)


def test_no_usa_pedidos_futuros_si_llegan_desordenados():
    """La función debe ordenar internamente por fecha antes de agregar."""
    df_desordenado = pd.DataFrame(
        {
            "customer_id": ["x", "x"],
            "order_purchase_timestamp": pd.to_datetime(["2021-03-01", "2021-01-01"]),
            "precio_total": [300.0, 100.0],
        }
    )
    resultado = agregar_ticket_historico(df_desordenado)
    fila_enero = resultado[resultado["order_purchase_timestamp"] == "2021-01-01"].iloc[0]
    assert pd.isna(fila_enero["ticket_promedio_historico"])


# ── tests agregar_encoding_categoria_producto (extensión 8.1) ──────
@pytest.fixture
def df_categorias_producto():
    # 990 filas de "populares" (9 categorías al ~11% c/u) + 10 filas con
    # categorías que aparecen 1 sola vez cada una en 1000 filas (0.1%, < 1%).
    populares = [f"cat_{i}" for i in range(9) for _ in range(110)]
    raras = [f"rara_{i}" for i in range(10)]
    return pd.DataFrame({"product_category_name": populares + raras})


def test_encoding_categoria_agrupa_menos_columnas_que_get_dummies(df_categorias_producto):
    resultado = agregar_encoding_categoria_producto(df_categorias_producto)
    columnas_agrupado = [c for c in resultado.columns if c.startswith("product_category_name_")]
    columnas_sin_agrupar = pd.get_dummies(df_categorias_producto, columns=["product_category_name"])
    assert len(columnas_agrupado) < columnas_sin_agrupar.shape[1]


def test_encoding_categoria_crea_columna_otros(df_categorias_producto):
    resultado = agregar_encoding_categoria_producto(df_categorias_producto)
    assert "product_category_name_otros" in resultado.columns
    assert resultado["product_category_name_otros"].sum() == 10
