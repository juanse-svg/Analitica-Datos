# Lab 06 — Análisis y reflexión

> Para pegar en el cuerpo del pull request (sección 9 del enunciado).

## 9.1 Un solo endpoint encadenando los dos modelos

La propuesta introduce *training-serving skew*. El modelo de reseñas se entrenó con
`tiempo_entrega_dias` **real**: la diferencia observada entre la compra y la entrega, con toda
su variabilidad (pedidos que tardaron 40 días, retrasos por transportadora, etc.). En la
propuesta se le pasaría, en cambio, una **predicción** del modelo de regresión: un valor
suavizado, cercano a la media y que casi nunca toma los extremos. Justo los extremos (entregas
muy lentas) son la señal más fuerte de una reseña negativa, así que el modelo vería en
producción una distribución distinta a la del entrenamiento y subestimaría el riesgo de forma
sistemática y silenciosa. Además, `entregado_tarde` no se puede conocer al momento de la
compra. Es el mismo problema del caso Uber Michelangelo: una feature que se calcula de una
manera al entrenar y de otra al servir. Por eso cada modelo tiene su propio endpoint, ligado al
momento de la vida del pedido en que sus features existen de verdad.

## 9.2 Detenerse vs. filtrar y continuar

**Detenerse** es mejor cuando el error indica un problema estructural o sistemático: cambió el
formato de la fuente, apareció un `order_status` nuevo, una columna llegó vacía o con otro tipo,
o los errores afectan a muchas filas. Entrenar así produciría un modelo que *parece* válido
pero aprendió de datos corruptos, y además lo reemplazaría en `models/`.

**Filtrar y continuar** es razonable cuando son unas pocas filas aisladas (un precio fuera de
rango, un `order_id` mal formado), el patrón es conocido y descartarlas no sesga la muestra.

`generar_reporte_calidad` da justo la información para decidir: `filas_descartadas` y
`tasa_rechazo` (¿0.01 % o 30 %?) y la lista de `errores` (¿qué columnas y qué checks
fallan?). Una política práctica sería filtrar y continuar por debajo de un umbral (por ejemplo,
< 1 % de rechazo y sin columnas nuevas o faltantes), y detenerse por encima de él.

## 9.3 Para qué sirve `version_sklearn`

Un `.joblib` es un objeto de Python serializado con pickle: solo es confiable cargarlo con la
misma versión de scikit-learn con la que se guardó. Con otra versión puede fallar al cargar,
emitir un `InconsistentVersionWarning` o, peor, cargar y predecir distinto sin avisar (cambian
defaults, atributos internos de los árboles o del imputer). Guardar la versión permite
reproducir el entorno y auditar qué produjo cada predicción.

Si la versión instalada no coincide, `crear_app` no debería servir en silencio. La opción más
segura es fallar al arrancar con un error claro (`RuntimeError`: "modelo entrenado con
sklearn X, instalado Y — reentrene con `python -m analytics.pipeline` o fije la versión en
`uv.lock`"). Como mínimo, debería registrar una advertencia y exponer ambas versiones en
`/salud` para que el monitoreo lo detecte. Como reentrenar cuesta un solo comando, fallar
rápido es preferible a predecir con un modelo de comportamiento incierto.

## 9.4 Señales para reentrenar (datos 2016-2018 desplegados hoy)

**Data drift** (se ve de inmediato, sin esperar etiquetas): comparar la distribución de las
features que llegan a la API contra la del entrenamiento. Por ejemplo: `precio_total` y
`flete_total` desplazados por inflación, otra mezcla de estados (`customer_state_freq`), más
pedidos con `ticket_promedio_historico` nulo (clientes nuevos), otra estacionalidad en
`mes_compra`. Se mide con PSI o la prueba KS por feature, y también con la distribución de las
propias predicciones (por ejemplo, si la probabilidad media de reseña negativa sube de 15 % a
35 % sin causa aparente). El registro de predicciones (extensión 8.2) es la materia prima.

**Performance drift** (requiere la verdad, que llega con retraso): cuando el pedido se entrega
se conoce el `tiempo_entrega_dias` real y se puede calcular el RMSE/MAE en producción; cuando
el cliente escribe la reseña se conoce `review_negativa` y se puede calcular F1/ROC-AUC. Si
esas métricas, en ventanas móviles, se degradan respecto a las de `metricas.json` (por ejemplo,
MAE 20 % peor, o el modelo ya no supera al baseline), hay que reentrenar.

El data drift es una alerta temprana pero no prueba que el modelo esté peor. El performance
drift sí lo prueba, pero llega días o semanas tarde. Por eso conviene monitorear ambos.
