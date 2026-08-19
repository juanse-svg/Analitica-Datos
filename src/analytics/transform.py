"""Módulo de transformación — features reproducibles sin fuga de información."""

import pandas as pd


def agregar_features_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas derivadas de las fechas de pedido y entrega.

    No muta el DataFrame recibido — retorna una copia con las columnas nuevas.

    Columnas nuevas:
        tiempo_entrega_dias: días entre compra y entrega real (float, NaN si no se ha entregado).
        retraso_entrega_dias: días entre la entrega real y la fecha estimada.
            Positivo significa que llegó tarde; negativo, que llegó antes.
        entregado_tarde: True si retraso_entrega_dias > 0.
        dia_semana_compra: día de la semana de la compra (0=lunes, 6=domingo).
        es_fin_de_semana: True si dia_semana_compra es sábado o domingo.
        mes_compra: mes de la compra (1-12).

    Args:
        df: DataFrame con columnas order_purchase_timestamp,
            order_delivered_customer_date, order_estimated_delivery_date
            (todas datetime64).

    Returns:
        Copia de df con las 6 columnas nuevas agregadas.
    """
    df = df.copy()

    compra = df["order_purchase_timestamp"]
    entrega_real = df["order_delivered_customer_date"]
    entrega_estimada = df["order_estimated_delivery_date"]

    df["tiempo_entrega_dias"] = (entrega_real - compra).dt.days.astype(float)
    df["retraso_entrega_dias"] = (entrega_real - entrega_estimada).dt.days.astype(float)
    df["entregado_tarde"] = df["retraso_entrega_dias"] > 0
    df["dia_semana_compra"] = compra.dt.weekday
    df["es_fin_de_semana"] = df["dia_semana_compra"].isin([5, 6])
    df["mes_compra"] = compra.dt.month

    return df


def agregar_encoding_estado(
    df: pd.DataFrame,
    columna: str = "customer_state",
) -> pd.DataFrame:
    """Agrega una columna de frequency encoding para una columna categórica.

    No muta el DataFrame recibido — retorna una copia con la columna nueva.

    Args:
        df: DataFrame que contiene la columna a codificar.
        columna: nombre de la columna categórica a codificar.

    Returns:
        Copia de df con una columna nueva llamada f"{columna}_freq", donde
        cada fila tiene la frecuencia relativa (0.0 a 1.0) de su categoría
        en el DataFrame completo.
    """
    df = df.copy()
    frecuencias = df[columna].value_counts(normalize=True)
    df[f"{columna}_freq"] = df[columna].map(frecuencias)
    return df


def agregar_ticket_historico(
    df: pd.DataFrame,
    columna_cliente: str = "customer_id",
    columna_valor: str = "precio_total",
    columna_fecha: str = "order_purchase_timestamp",
) -> pd.DataFrame:
    """Agrega el promedio histórico de compra por cliente, sin fuga temporal.

    Para cada fila, calcula el promedio de columna_valor usando SOLO los
    pedidos anteriores del mismo cliente (ordenados por columna_fecha).
    El primer pedido de cada cliente queda con NaN — no tiene historial previo.

    No muta el DataFrame recibido. El DataFrame retornado puede estar en
    distinto orden de filas que el original (se ordena internamente por
    columna_fecha para calcular la agregación correctamente).

    Args:
        df: DataFrame con las columnas columna_cliente, columna_valor y
            columna_fecha.
        columna_cliente: columna de agrupación (identificador del cliente).
        columna_valor: columna numérica sobre la que se calcula el promedio.
        columna_fecha: columna datetime usada para ordenar cronológicamente.

    Returns:
        Copia de df, ordenada por columna_fecha, con una columna nueva
        llamada "ticket_promedio_historico".
    """
    df = df.copy().sort_values(columna_fecha)

    df["ticket_promedio_historico"] = (
        df.groupby(columna_cliente)[columna_valor]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    return df


def agregar_encoding_categoria_producto(
    df: pd.DataFrame,
    columna: str = "product_category_name",
    umbral_frecuencia: float = 0.01,
) -> pd.DataFrame:
    """Aplica one-hot encoding a una columna categórica agrupando categorías raras.

    Toda categoría con frecuencia relativa menor a umbral_frecuencia se agrupa
    en una categoría "otros" antes de codificar. Esto evita generar una columna
    dummy por cada categoría poco frecuente, lo cual infla la dimensionalidad
    sin aportar señal.

    No muta el DataFrame recibido — retorna una copia con las columnas dummy
    nuevas y sin la columna categórica original.

    Args:
        df: DataFrame que contiene la columna a codificar.
        columna: nombre de la columna categórica a codificar.
        umbral_frecuencia: frecuencia relativa mínima (0.0 a 1.0) para que una
            categoría conserve su propia columna dummy.

    Returns:
        Copia de df con columnas dummy f"{columna}_<valor>" agregadas y la
        columna original eliminada.
    """
    df = df.copy()

    frecuencias = df[columna].value_counts(normalize=True)
    categorias_raras = frecuencias[frecuencias < umbral_frecuencia].index
    columna_agrupada = df[columna].where(~df[columna].isin(categorias_raras), "otros")

    dummies = pd.get_dummies(columna_agrupada, prefix=columna)
    df = pd.concat([df.drop(columns=[columna]), dummies], axis=1)

    return df
