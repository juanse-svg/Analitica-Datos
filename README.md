# Analitica-Datos

Proyecto del semestre — Analítica de Datos (UPB). Pipeline sobre el dataset público de Olist.

## Módulos

| Lab | Módulo | Qué hace |
|-----|--------|----------|
| 02 | `extract.py`, `quality.py` | Carga de las 5 tablas de Olist, joins verificados, contratos pandera |
| 03 | `transform.py`, `eda.py` | Features de fecha, encoding de estado, ticket histórico sin fuga |
| 04 | `train.py`, `evaluate.py` | Pipelines de regresión y clasificación, métricas y baseline |
| 05 | `segment.py` | Segmentación de clientes (RFM + KMeans) |
| 06 | `pipeline.py`, `api.py` | Orquestación en un comando + API REST con FastAPI |

## Uso

```bash
uv sync --all-groups
uv run pytest -v

# Entrenar los dos modelos con los datos reales (CSV de Olist en data/raw/)
uv run python -m analytics.pipeline --data-dir data/raw --models-dir models

# Levantar la API (documentación interactiva en http://localhost:8000/docs)
uv run uvicorn analytics.api:crear_app --factory --port 8000
```

### Endpoints

| Método | Ruta | Respuesta |
|--------|------|-----------|
| GET | `/salud` | `{"estado": "ok", "fecha_entrenamiento": ...}` |
| POST | `/predecir/entrega` | `{"tiempo_entrega_dias": float}` — al momento de la compra |
| POST | `/predecir/resena` | `{"probabilidad_resena_negativa": float, "resena_negativa": bool}` — pedido ya entregado |

### Docker (opcional)

```bash
docker build -t olist-api .
docker run -p 8000:8000 olist-api
```
