"""Módulo de calidad de datos — contratos pandera y trazabilidad de filtrado."""

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

SCHEMA_ORDERS = DataFrameSchema(
    columns={
        "order_id": Column(
            str,
            Check.str_length(32, 32),
            nullable=False,
            unique=True,
        ),
        "order_status": Column(
            str,
            Check.isin(
                [
                    "delivered",
                    "shipped",
                    "canceled",
                    "unavailable",
                    "invoiced",
                    "processing",
                    "created",
                    "approved",
                ]
            ),
        ),
        "order_purchase_timestamp": Column(pa.DateTime, nullable=False),
    },
    coerce=True,
)

SCHEMA_ITEMS = DataFrameSchema(
    columns={
        "order_id": Column(str, nullable=False),
        "order_item_id": Column(int, Check.greater_than_or_equal_to(1)),
        "product_id": Column(str, nullable=False),
        "seller_id": Column(str, nullable=False),
        "price": Column(
            float,
            [Check.greater_than(0), Check.less_than_or_equal_to(7_000)],
            nullable=False,
        ),
        "freight_value": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
    },
    coerce=True,
)


def validar_schema(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    nombre: str,
) -> list[str]:
    """Valida df contra schema y retorna la lista de errores encontrados.

    No lanza excepción — captura SchemaErrors y retorna los mensajes como strings
    para que el pipeline pueda continuar y registrar todos los problemas.

    Args:
        df: DataFrame a validar.
        schema: schema pandera contra el cual validar.
        nombre: nombre descriptivo del DataFrame para los mensajes de error.

    Returns:
        Lista de strings con los errores encontrados. Lista vacía si el schema pasa.
    """
    try:
        schema.validate(df, lazy=True)
        return []
    except pa.errors.SchemaErrors as exc:
        casos = exc.failure_cases
        errores = []
        for _, fila in casos.iterrows():
            errores.append(
                f"{nombre}: columna '{fila['column']}' — {fila['check']} "
                f"(valor inválido: {fila['failure_case']})"
            )
        return errores


def generar_reporte_calidad(
    df_entrada: pd.DataFrame,
    df_salida: pd.DataFrame,
    errores: list[str],
    nombre: str = "dataset",
) -> dict:
    """Genera un reporte de la operación de calidad con trazabilidad completa.

    Imprime el reporte con print() en el formato:
    [CALIDAD] nombre: N entrada → M salida (K descartados, X.XX%)

    Args:
        df_entrada: DataFrame antes del filtrado/validación.
        df_salida: DataFrame después del filtrado/validación.
        errores: lista de strings con los errores detectados.
        nombre: nombre descriptivo del dataset para el log.

    Returns:
        Dict con keys:
        - 'nombre': str
        - 'filas_entrada': int
        - 'filas_salida': int
        - 'filas_descartadas': int
        - 'tasa_rechazo': str (formato "X.XX%")
        - 'errores': list[str]
    """
    filas_entrada = len(df_entrada)
    filas_salida = len(df_salida)
    filas_descartadas = filas_entrada - filas_salida
    tasa = (filas_descartadas / filas_entrada * 100) if filas_entrada > 0 else 0.0
    tasa_rechazo = f"{tasa:.2f}%"

    print(
        f"[CALIDAD] {nombre}: {filas_entrada} entrada → {filas_salida} salida "
        f"({filas_descartadas} descartados, {tasa_rechazo})"
    )

    return {
        "nombre": nombre,
        "filas_entrada": filas_entrada,
        "filas_salida": filas_salida,
        "filas_descartadas": filas_descartadas,
        "tasa_rechazo": tasa_rechazo,
        "errores": errores,
    }
