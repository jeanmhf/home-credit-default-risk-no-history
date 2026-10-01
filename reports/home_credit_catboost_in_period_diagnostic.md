# CatBoost: Logloss de entrenamiento y validación

Se seleccionaron 100,000 casos de las semanas 0–70 y se dividieron aleatoriamente, estratificando por `target`: 80,000 para entrenamiento y 20,000 para validación. Las tasas positivas fueron 3.38% y 3.38%.

El ajuste usa los hiperparámetros seleccionados por Optuna: depth=4, learning_rate=0.10972661298889345, l2_leaf_reg=8.800251535369485, random_strength=0.05494594621885618, bagging_temperature=1.795232264200432, con 282 árboles. En el menor Logloss de validación (árbol 274), el Logloss fue 0.13609 en entrenamiento y 0.14203 en validación; la diferencia fue 0.00594.

Las dos líneas se calculan sobre particiones distintas de la misma muestra original de 100,000 casos (80,000 train y 20,000 validation), no sobre las mismas filas. Esta gráfica revisa sobreajuste dentro de las semanas 0–70; no usa la evaluación temporal de semanas 81–91 ni reemplaza sus métricas.

![Logloss de entrenamiento y validación](figures/home_credit_catboost_in_period_curves.png)

Esta comprobación vuelve a entrenar el modelo solo para registrar la curva; no modifica el artefacto del CatBoost final ni su calibrador.
