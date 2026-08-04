"""Tests del módulo extract.py — Lab 01."""

import pandas as pd
import pytest

from analytics.extract import cargar_csv, join_verificado


def test_cargar_csv_retorna_dataframe(tmp_path):
    """cargar_csv debe retornar un DataFrame de pandas."""
    archivo = tmp_path / "datos.csv"
    archivo.write_text("ciudad,ventas\nMedellin,100\nBogota,200\n")
    resultado = cargar_csv(str(archivo))
    assert isinstance(resultado, pd.DataFrame)


def test_cargar_csv_carga_filas_correctamente(tmp_path):
    """El DataFrame debe tener exactamente las filas del CSV."""
    archivo = tmp_path / "datos.csv"
    archivo.write_text("ciudad,ventas\nMedellin,100\nBogota,200\nCali,150\n")
    df = cargar_csv(str(archivo))
    assert len(df) == 3
    assert list(df.columns) == ["ciudad", "ventas"]


def test_cargar_csv_archivo_no_existe():
    """Si el archivo no existe, debe lanzar FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        cargar_csv("ruta/inventada/que/no/existe.csv")


def test_cargar_csv_archivo_vacio(tmp_path):
    """Si el CSV solo tiene encabezados (cero filas de datos), debe lanzar ValueError."""
    archivo = tmp_path / "vacio.csv"
    archivo.write_text("ciudad,ventas\n")  # encabezados pero sin datos
    with pytest.raises(ValueError, match="no contiene filas"):
        cargar_csv(str(archivo))


# ── tests para join_verificado ────────────────────────────────────────
def test_join_verificado_retorna_dataframe_correcto():
    """join sin duplicados en tabla derecha: resultado tiene mismas filas que izquierda."""
    left = pd.DataFrame({"id": [1, 2, 3], "valor": ["a", "b", "c"]})
    right = pd.DataFrame({"id": [1, 2, 3], "extra": ["x", "y", "z"]})
    resultado = join_verificado(left, right, on="id")
    assert isinstance(resultado, pd.DataFrame)
    assert len(resultado) == 3
    assert "extra" in resultado.columns


def test_join_verificado_falla_con_duplicados():
    """Si la tabla derecha tiene duplicados en la clave, debe lanzar AssertionError."""
    left = pd.DataFrame({"id": [1, 2], "valor": ["a", "b"]})
    right = pd.DataFrame({"id": [1, 1, 2], "extra": ["x", "y", "z"]})  # id=1 duplicado
    with pytest.raises(AssertionError, match="filas"):
        join_verificado(left, right, on="id", nombre="test_dup")


def test_join_verificado_left_join_no_pierde_filas():
    """Con how='left', filas sin match en la derecha se mantienen con NaN."""
    left = pd.DataFrame({"id": [1, 2, 3], "valor": ["a", "b", "c"]})
    right = pd.DataFrame({"id": [1, 2], "extra": ["x", "y"]})  # id=3 no está
    resultado = join_verificado(left, right, on="id")
    assert len(resultado) == 3
    assert pd.isna(resultado.loc[resultado["id"] == 3, "extra"].values[0])
