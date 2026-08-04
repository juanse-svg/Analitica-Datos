"""Módulo de ingesta de datos — carga de fuentes externas hacia el pipeline."""

from pathlib import Path

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


# Columnas de fecha por tabla — se parsean como datetime64 al cargar.
_COLUMNAS_FECHA_ORDERS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
_COLUMNAS_FECHA_REVIEWS = ["review_creation_date", "review_answer_timestamp"]

# Mapa de nombre lógico -> (nombre de archivo, columnas de fecha a parsear).
_TABLAS_OLIST = {
    "orders": ("olist_orders_dataset.csv", _COLUMNAS_FECHA_ORDERS),
    "items": ("olist_order_items_dataset.csv", None),
    "customers": ("olist_customers_dataset.csv", None),
    "payments": ("olist_order_payments_dataset.csv", None),
    "reviews": ("olist_order_reviews_dataset.csv", _COLUMNAS_FECHA_REVIEWS),
}


def cargar_olist(data_dir: str) -> dict[str, pd.DataFrame]:
    """Carga las 5 tablas principales de Olist desde data_dir.

    Registra con print() la forma (filas × columnas) de cada tabla al cargarla.
    Las columnas de fecha se cargan como datetime64, no como object.

    Args:
        data_dir: ruta al directorio que contiene los archivos CSV de Olist.

    Returns:
        Dict con keys 'orders', 'items', 'customers', 'payments', 'reviews',
        cada uno mapeado a su DataFrame correspondiente.

    Raises:
        FileNotFoundError: si alguno de los 5 archivos no existe en data_dir.
    """
    base = Path(data_dir)
    tablas: dict[str, pd.DataFrame] = {}
    for nombre, (archivo, columnas_fecha) in _TABLAS_OLIST.items():
        ruta = base / archivo
        df = pd.read_csv(ruta, parse_dates=columnas_fecha)
        print(f"{nombre}: {df.shape}")
        tablas[nombre] = df
    return tablas


def join_verificado(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    on: str | list[str],
    how: str = "left",
    nombre: str = "join",
) -> pd.DataFrame:
    """Realiza un merge y verifica que el resultado no multiplique filas.

    Un join que multiplica filas indica que la tabla derecha tiene duplicados
    en la columna clave — error silencioso sin esta verificación.

    Args:
        df_left: DataFrame izquierdo (el que define el número de filas esperado).
        df_right: DataFrame derecho.
        on: columna(s) clave del join.
        how: tipo de join ('left', 'inner', 'outer', 'right').
        nombre: nombre descriptivo para el mensaje de error.

    Returns:
        DataFrame resultante del merge.

    Raises:
        AssertionError: si el resultado tiene más filas que df_left.
    """
    resultado = df_left.merge(df_right, on=on, how=how)
    if len(resultado) > len(df_left):
        raise AssertionError(
            f"{nombre}: el resultado tiene {len(resultado)} filas, "
            f"más que las {len(df_left)} filas de entrada — "
            "la tabla derecha probablemente tiene claves duplicadas"
        )
    return resultado


def construir_dataset_base(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combina las tablas de Olist en un único DataFrame analítico.

    Estrategia de joins:
    - orders × customers → left join on customer_id
    - + payments_agg → left join on order_id (payments debe agregarse primero)
    - + items_agg → left join on order_id (items debe agregarse primero)

    Agrega payments antes del join: total_pago (sum) y n_cuotas (max) por order_id.
    Agrega items antes del join: n_items (count) y ticket_total (sum de price) por
    order_id.

    Args:
        tablas: dict retornado por cargar_olist().

    Returns:
        DataFrame con una fila por pedido (99,441 filas si los datos son completos).
    """
    orders = tablas["orders"]
    customers = tablas["customers"]
    payments = tablas["payments"]
    items = tablas["items"]

    payments_agg = (
        payments.groupby("order_id")
        .agg(total_pago=("payment_value", "sum"), n_cuotas=("payment_installments", "max"))
        .reset_index()
    )

    items_agg = (
        items.groupby("order_id")
        .agg(n_items=("order_item_id", "count"), ticket_total=("price", "sum"))
        .reset_index()
    )

    df = join_verificado(orders, customers, on="customer_id", nombre="orders_customers")
    df = join_verificado(df, payments_agg, on="order_id", nombre="orders_payments_agg")
    df = join_verificado(df, items_agg, on="order_id", nombre="orders_items_agg")
    return df