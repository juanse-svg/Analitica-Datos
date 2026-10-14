"""Orquestación del pipeline completo del semestre."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from analytics import evaluate, extract, quality, train, transform

FEATURES_REGRESION = [
    "n_items",
    "precio_total",
    "flete_total",
    "customer_state_freq",
    "dia_semana_compra",
    "mes_compra",
    "ticket_promedio_historico",
]

FEATURES_CLASIFICACION = [
    "n_items",
    "precio_total",
    "flete_total",
    "customer_state_freq",
    "tiempo_entrega_dias",
    "entregado_tarde",
    "ticket_promedio_historico",
]

NOMBRE_MODELO_ENTREGA = "pipeline_tiempo_entrega"
NOMBRE_MODELO_RESENA = "pipeline_review_negativa"


def unir_reviews(df: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Une review_score al DataFrame de pedidos, sin multiplicar filas.

    Un pedido puede tener más de una reseña: se conserva la de
    review_answer_timestamp más reciente (mismo criterio del Tema 4, sección 6.4).

    Args:
        df: DataFrame con una fila por pedido y la columna 'order_id'.
        reviews: tabla de reseñas con 'order_id', 'review_score' y
            'review_answer_timestamp'.

    Returns:
        Copia de df con la columna 'review_score' agregada (NaN si el pedido
        no tiene reseña). Mismo número de filas que df.
    """
    reviews_unicas = (
        reviews.sort_values("review_answer_timestamp")
        .drop_duplicates(subset="order_id", keep="last")
        .loc[:, ["order_id", "review_score"]]
    )
    # join_verificado (Lab 02) lanza AssertionError si el merge multiplica filas.
    return extract.join_verificado(
        df, reviews_unicas, on="order_id", how="left", nombre="pedidos_reviews"
    )


def preparar_datos(
    df: pd.DataFrame, features: list[str], target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Selecciona features y target, descartando filas sin target.

    Las features pueden tener nulos (los imputa el Pipeline); el target no.

    Args:
        df: DataFrame con las columnas de features y target.
        features: columnas a usar como X, en ese orden.
        target: nombre de la columna objetivo.

    Returns:
        Tupla (X, y) sin filas con target nulo.
    """
    con_target = df[df[target].notna()]
    X = con_target[features].copy()
    y = con_target[target].copy()
    return X, y


def entrenar_y_evaluar(
    X: pd.DataFrame,
    y: pd.Series,
    tipo: Literal["regresion", "clasificacion"],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    """Divide en train/test, entrena con train.py y evalúa con evaluate.py.

    Para clasificación, la división es estratificada y las métricas
    incluyen también la comparación contra el baseline.

    Args:
        X: features.
        y: target.
        tipo: "regresion" o "clasificacion".
        test_size: proporción del conjunto de test.
        random_state: semilla para reproducibilidad.

    Returns:
        Tupla (pipeline entrenado, dict de métricas sobre test).

    Raises:
        ValueError: si tipo no es "regresion" ni "clasificacion".
    """
    if tipo == "regresion":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        pipeline = train.entrenar_pipeline_regresion(X_train, y_train, random_state=random_state)
        metricas = evaluate.evaluar_regresion(y_test, pipeline.predict(X_test))
        return pipeline, metricas

    if tipo == "clasificacion":
        # stratify=y mantiene la proporción de positivos (~15 % en review_negativa)
        # igual en train y test; sin esto, el test podría quedar casi sin positivos.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        pipeline = train.entrenar_pipeline_clasificacion(
            X_train, y_train, random_state=random_state
        )
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metricas = evaluate.evaluar_clasificacion(
            y_test, y_pred, y_proba
        ) | evaluate.comparar_con_baseline(y_train, y_test, y_pred)
        return pipeline, metricas

    raise ValueError(f"tipo desconocido: {tipo!r} (use 'regresion' o 'clasificacion')")


def guardar_artefactos(modelos: dict[str, Pipeline], metricas: dict, models_dir: str) -> Path:
    """Guarda los pipelines y un archivo metricas.json con trazabilidad.

    Cada pipeline se guarda como f"{nombre}.joblib". metricas.json contiene
    las métricas recibidas más 'fecha_entrenamiento' (ISO 8601, UTC) y
    'version_sklearn', para saber después con qué se entrenó cada modelo.

    Args:
        modelos: dict nombre → pipeline entrenado.
        metricas: dict nombre → dict de métricas de ese modelo.
        models_dir: directorio de salida (se crea si no existe).

    Returns:
        Ruta del archivo metricas.json escrito.
    """
    destino = Path(models_dir)
    destino.mkdir(parents=True, exist_ok=True)

    for nombre, pipeline in modelos.items():
        train.guardar_pipeline(pipeline, str(destino / f"{nombre}.joblib"))

    registro = {
        "fecha_entrenamiento": datetime.now(UTC).isoformat(),
        "version_sklearn": sklearn.__version__,
        "metricas": metricas,
    }
    ruta = destino / "metricas.json"
    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(registro, archivo, indent=2, ensure_ascii=False)
    return ruta


def ejecutar_pipeline(data_dir: str, models_dir: str) -> dict:
    """Ejecuta el pipeline completo: extract → quality → transform → train → evaluate.

    Pasos:
        1. Cargar las tablas con extract.cargar_olist.
        2. Validar orders e items con quality.validar_schema (SCHEMA_ORDERS,
           SCHEMA_ITEMS). Si hay errores, detenerse con ValueError: no se
           entrena con datos que violan el contrato.
        3. Construir el dataset base y aplicar las tres transformaciones de
           transform.py (fecha, encoding de estado, ticket histórico).
        4. Unir review_score con unir_reviews y crear
           review_negativa = review_score <= 2 (solo donde hay reseña).
        5. Entrenar y evaluar los dos modelos con entrenar_y_evaluar.
        6. Guardar ambos pipelines y las métricas con guardar_artefactos.

    Args:
        data_dir: directorio con los CSV de Olist.
        models_dir: directorio donde se guardan los artefactos.

    Returns:
        Dict nombre del modelo → dict de métricas.

    Raises:
        ValueError: si orders o items no cumplen su schema.
    """
    print("[1/6] Cargando datos")
    tablas = extract.cargar_olist(data_dir)

    print("[2/6] Validando calidad")
    errores = quality.validar_schema(
        tablas["orders"], quality.SCHEMA_ORDERS, "orders"
    ) + quality.validar_schema(tablas["items"], quality.SCHEMA_ITEMS, "items")
    if errores:
        detalle = "\n".join(errores[:10])
        raise ValueError(
            f"Los datos violan el contrato de calidad ({len(errores)} errores); "
            f"no se entrena ningún modelo.\n{detalle}"
        )

    print("[3/6] Transformando")
    df = extract.construir_dataset_base(tablas)
    df = transform.agregar_features_fecha(df)
    df = transform.agregar_encoding_estado(df, columna="customer_state")
    # En Olist, customer_id es distinto en cada pedido; el cliente real es
    # customer_unique_id. Con customer_id todo el ticket histórico sería NaN.
    df = transform.agregar_ticket_historico(df, columna_cliente="customer_unique_id")

    print("[4/6] Uniendo reseñas")
    df = unir_reviews(df, tablas["reviews"])
    # .where conserva NaN donde no hay reseña; sin ella, NaN <= 2 daría False y el
    # pedido entraría al entrenamiento como "reseña positiva" sin tenerla.
    df["review_negativa"] = (df["review_score"] <= 2).where(df["review_score"].notna())

    print("[5/6] Entrenando y evaluando")
    X_reg, y_reg = preparar_datos(df, FEATURES_REGRESION, "tiempo_entrega_dias")
    pipe_reg, met_reg = entrenar_y_evaluar(X_reg, y_reg, "regresion")

    X_clf, y_clf = preparar_datos(df, FEATURES_CLASIFICACION, "review_negativa")
    pipe_clf, met_clf = entrenar_y_evaluar(X_clf, y_clf.astype(bool), "clasificacion")

    metricas = {NOMBRE_MODELO_ENTREGA: met_reg, NOMBRE_MODELO_RESENA: met_clf}

    print("[6/6] Guardando artefactos")
    ruta = guardar_artefactos(
        {NOMBRE_MODELO_ENTREGA: pipe_reg, NOMBRE_MODELO_RESENA: pipe_clf},
        metricas,
        models_dir,
    )
    print(f"Listo: {ruta}")
    return metricas


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(description="Pipeline completo Olist — AD")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args(argv)
    metricas = ejecutar_pipeline(args.data_dir, args.models_dir)
    print(json.dumps(metricas, indent=2))


if __name__ == "__main__":
    main()
