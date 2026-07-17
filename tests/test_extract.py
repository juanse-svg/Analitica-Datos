"""Tests del módulo extract.py — Lab 01."""

import pandas as pd
import pytest

from analytics.extract import cargar_csv


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
