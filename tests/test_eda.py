"""Tests del módulo eda.py — Lab 03."""

import numpy as np
import pandas as pd
import pytest

from analytics.eda import (
    detectar_posible_fuga,
    distribucion_categorica,
    matriz_correlacion,
    perfil_columna,
)


# ── fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def df_mixto():
    return pd.DataFrame(
        {
            "precio": [10.0, 20.0, 30.0, np.nan, 50.0],
            "categoria": ["a", "a", "a", "b", "c"],
            "cantidad": [1, 2, 3, 4, 5],
        }
    )


# ── tests perfil_columna ─────────────────────────────────────────────
def test_perfil_columna_retorna_dict(df_mixto):
    resultado = perfil_columna(df_mixto, "precio")
    assert isinstance(resultado, dict)


def test_perfil_columna_numerica_incluye_estadisticas(df_mixto):
    resultado = perfil_columna(df_mixto, "precio")
    assert resultado["media"] == pytest.approx(27.5)
    assert resultado["min"] == 10.0
    assert resultado["max"] == 50.0


def test_perfil_columna_cuenta_nulos_correctamente(df_mixto):
    resultado = perfil_columna(df_mixto, "precio")
    assert resultado["n_nulos"] == 1
    assert resultado["pct_nulos"] == pytest.approx(20.0)


def test_perfil_columna_categorica_incluye_valor_mas_frecuente(df_mixto):
    resultado = perfil_columna(df_mixto, "categoria")
    assert resultado["valor_mas_frecuente"] == "a"
    assert resultado["freq_valor_mas_frecuente"] == 3


def test_perfil_columna_n_unicos_correcto(df_mixto):
    resultado = perfil_columna(df_mixto, "categoria")
    assert resultado["n_unicos"] == 3


# ── tests distribucion_categorica ───────────────────────────────────
def test_distribucion_categorica_conteos_correctos(df_mixto):
    resultado = distribucion_categorica(df_mixto, "categoria")
    assert resultado.loc["a", "conteo"] == 3
    assert resultado.loc["b", "conteo"] == 1


def test_distribucion_categorica_porcentajes_suman_100(df_mixto):
    resultado = distribucion_categorica(df_mixto, "categoria")
    assert resultado["porcentaje"].sum() == pytest.approx(100.0)


def test_distribucion_categorica_ordenado_descendente(df_mixto):
    resultado = distribucion_categorica(df_mixto, "categoria")
    assert resultado.index[0] == "a"  # la categoría más frecuente va primero


# ── tests matriz_correlacion ────────────────────────────────────────
def test_matriz_correlacion_diagonal_es_uno(df_mixto):
    resultado = matriz_correlacion(df_mixto, ["precio", "cantidad"])
    assert resultado.loc["precio", "precio"] == pytest.approx(1.0)
    assert resultado.loc["cantidad", "cantidad"] == pytest.approx(1.0)


def test_matriz_correlacion_es_simetrica(df_mixto):
    resultado = matriz_correlacion(df_mixto, ["precio", "cantidad"])
    assert resultado.loc["precio", "cantidad"] == pytest.approx(resultado.loc["cantidad", "precio"])


def test_matriz_correlacion_forma_correcta(df_mixto):
    resultado = matriz_correlacion(df_mixto, ["precio", "cantidad"])
    assert resultado.shape == (2, 2)


# ── tests detectar_posible_fuga (extensión 8.2) ─────────────────────
def test_detecta_fuga_con_feature_identica_al_objetivo():
    df = pd.DataFrame(
        {
            "objetivo": [1, 2, 3, 4, 5],
            "feature_filtrada": [1, 2, 3, 4, 5],  # correlación = 1.0
        }
    )
    assert detectar_posible_fuga(df, "feature_filtrada", "objetivo") is True


def test_no_detecta_fuga_con_feature_no_correlacionada():
    df = pd.DataFrame(
        {
            "objetivo": [1, 2, 3, 4, 5],
            "feature_normal": [5, 1, 4, 2, 3],  # correlación baja
        }
    )
    assert detectar_posible_fuga(df, "feature_normal", "objetivo") is False
