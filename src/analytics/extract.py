"""Módulo de ingesta de datos — carga de fuentes externas hacia el pipeline."""

import pandas as pd


def cargar_csv(ruta: str) -> pd.DataFrame:
    """Carga un archivo CSV y verifica que contenga al menos una fila de datos.

    Args:
        ruta: ruta al archivo CSV a cargar.

    Returns:
        DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: si el archivo no existe en la ruta indicada.
        ValueError: si el archivo existe pero no contiene filas de datos.
    """
    df = pd.read_csv(ruta)
    if df.empty:
        raise ValueError("El archivo no contiene filas de datos")
    return df
