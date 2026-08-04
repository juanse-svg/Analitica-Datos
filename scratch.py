from analytics.extract import cargar_olist, construir_dataset_base

tablas = cargar_olist("data/raw")
df = construir_dataset_base(tablas)

print("\nShape final:", df.shape)
print("\nTipos de datos:")
print(df.dtypes)

from analytics.quality import validar_schema, SCHEMA_ORDERS

errores = validar_schema(tablas["orders"], SCHEMA_ORDERS, "orders")
print(f"\nTotal errores: {len(errores)}")
for e in errores[:10]:
    print(e)

import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema

# Copia de SCHEMA_ORDERS pero con coerce=False
SCHEMA_ORDERS_SIN_COERCE = DataFrameSchema(
    columns={
        "order_id": Column(str, Check.str_length(32, 32), nullable=False, unique=True),
        "order_status": Column(str, Check.isin([
            "delivered", "shipped", "canceled", "unavailable",
            "invoiced", "processing", "created", "approved",
        ])),
        "order_purchase_timestamp": Column(pa.DateTime, nullable=False),
    },
    coerce=False,
)

# Simula el caso en que el CSV se cargó SIN parse_dates,
# es decir, la columna de fecha llega como texto (object/str)
orders_sin_parsear = tablas["orders"].copy()
orders_sin_parsear["order_purchase_timestamp"] = orders_sin_parsear["order_purchase_timestamp"].astype(str)

errores = validar_schema(orders_sin_parsear, SCHEMA_ORDERS_SIN_COERCE, "orders_sin_parsear")
print(f"Total errores: {len(errores)}")
for e in errores[:5]:
    print(e)