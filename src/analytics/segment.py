from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


def calcular_rfm(pedidos, fecha_referencia):
    agregado = (
        pedidos.groupby("customer_unique_id")
        .agg(
            ultima_compra=("order_purchase_timestamp", "max"),
            frecuencia=("order_purchase_timestamp", "count"),
            monto=("monto_pedido", "sum"),
        )
        .reset_index()
    )
    agregado["recencia"] = (fecha_referencia - agregado["ultima_compra"]).dt.days
    return agregado[["customer_unique_id", "recencia", "frecuencia", "monto"]]


def entrenar_pipeline_clustering(X, n_clusters, random_state=42):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("modelo", KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)),
        ]
    )
    pipeline.fit(X)
    return pipeline


def evaluar_clustering(X, labels):
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def perfilar_segmentos(df, columna_segmento, columnas_perfil):
    resultado = df.groupby(columna_segmento)[columnas_perfil].agg("mean")
    resultado["n_clientes"] = df.groupby(columna_segmento).size()
    return resultado
