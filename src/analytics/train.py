import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

def entrenar_pipeline_regresion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 10,
    random_state: int = 42,
) -> Pipeline:
    """Construye y entrena un pipeline de regresión.

    El pipeline tiene tres pasos: imputación de nulos con la mediana,
    escalado estándar, y un RandomForestRegressor.

    Args:
        X_train: features de entrenamiento (pueden tener valores nulos).
        y_train: target de entrenamiento (tiempo_entrega_dias).
        n_estimators: número de árboles del RandomForestRegressor.
        max_depth: profundidad máxima de cada árbol.
        random_state: semilla para reproducibilidad.

    Returns:
        Pipeline ya entrenado (fit ya ejecutado), listo para .predict().
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('modelo', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def entrenar_pipeline_clasificacion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 8,
    random_state: int = 42,
) -> Pipeline:
    """Construye y entrena un pipeline de clasificación.

    El pipeline tiene tres pasos: imputación de nulos con la mediana,
    escalado estándar, y un RandomForestClassifier con
    class_weight="balanced" (para compensar clases desbalanceadas).

    Args:
        X_train: features de entrenamiento (pueden tener valores nulos).
        y_train: target binario de entrenamiento (review_negativa).
        n_estimators: número de árboles del RandomForestClassifier.
        max_depth: profundidad máxima de cada árbol.
        random_state: semilla para reproducibilidad.

    Returns:
        Pipeline ya entrenado (fit ya ejecutado), con predict_proba disponible.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('modelo', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def guardar_pipeline(pipeline: Pipeline, ruta: str) -> None:
    """Serializa un pipeline entrenado con joblib.

    Args:
        pipeline: pipeline ya entrenado (con .fit() ejecutado).
        ruta: ruta del archivo de salida (ej. "models/pipeline_entrega.joblib").
    """
    joblib.dump(pipeline, ruta)


def cargar_pipeline(ruta: str) -> Pipeline:
    """Carga un pipeline previamente serializado con joblib.

    Args:
        ruta: ruta del archivo .joblib a cargar.

    Returns:
        El pipeline deserializado, listo para .predict().
    """
    return joblib.load(ruta)
