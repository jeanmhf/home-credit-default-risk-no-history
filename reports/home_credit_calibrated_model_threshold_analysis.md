# Comparación de umbrales: modelos calibrados

Los escenarios simulan aprobar cuando la probabilidad calibrada de default es menor o igual al umbral. Ambos modelos se evaluaron sobre los mismos 131,548 casos de las semanas 81–91, con tasa observada de default de 2.12% (2,788 casos). `Defaults rechazados` muestra el número y porcentaje de todos los defaults observados capturados por la regla; `Buenos clientes rechazados` son casos sin default observado que se rechazarían. Son simulaciones retrospectivas: las semanas 81–91 ya se inspeccionaron durante el desarrollo y no constituyen un test final independiente.

## CatBoost

| Umbral | Aprobadas | % aprobadas | Default entre aprobadas | Defaults rechazados | Buenos clientes rechazados |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 1,923 | 1.5% | 0.52% | 2,778 (99.6%) | 126,847 |
| 1% | 21,107 | 16.0% | 0.68% | 2,645 (94.9%) | 107,796 |
| 1.5% | 51,381 | 39.1% | 0.90% | 2,324 (83.4%) | 77,843 |
| 2% | 76,110 | 57.9% | 1.11% | 1,945 (69.8%) | 53,493 |
| 2.25% | 85,352 | 64.9% | 1.19% | 1,769 (63.5%) | 44,427 |
| 2.5% | 93,085 | 70.8% | 1.26% | 1,613 (57.9%) | 36,850 |
| 3% | 104,749 | 79.6% | 1.40% | 1,317 (47.2%) | 25,482 |
| 4% | 118,088 | 89.8% | 1.63% | 863 (31.0%) | 12,597 |
| 5% | 124,208 | 94.4% | 1.75% | 611 (21.9%) | 6,729 |
| 7.5% | 129,244 | 98.2% | 1.94% | 285 (10.2%) | 2,019 |
| 10% | 130,409 | 99.1% | 2.01% | 169 (6.1%) | 970 |
| 15% | 131,062 | 99.6% | 2.06% | 89 (3.2%) | 397 |
| 20% | 131,281 | 99.8% | 2.08% | 56 (2.0%) | 211 |

## XGBoost

| Umbral | Aprobadas | % aprobadas | Default entre aprobadas | Defaults rechazados | Buenos clientes rechazados |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 1,149 | 0.9% | 0.26% | 2,785 (99.9%) | 127,614 |
| 1% | 21,964 | 16.7% | 0.63% | 2,650 (95.1%) | 106,934 |
| 1.5% | 52,267 | 39.7% | 0.93% | 2,304 (82.6%) | 76,977 |
| 2% | 77,980 | 59.3% | 1.14% | 1,901 (68.2%) | 51,667 |
| 2.25% | 87,526 | 66.5% | 1.23% | 1,714 (61.5%) | 42,308 |
| 2.5% | 95,165 | 72.3% | 1.28% | 1,567 (56.2%) | 34,816 |
| 3% | 106,557 | 81.0% | 1.44% | 1,257 (45.1%) | 23,734 |
| 4% | 119,072 | 90.5% | 1.67% | 800 (28.7%) | 11,676 |
| 5% | 125,027 | 95.0% | 1.79% | 551 (19.8%) | 5,970 |
| 7.5% | 129,673 | 98.6% | 1.95% | 253 (9.1%) | 1,622 |
| 10% | 130,667 | 99.3% | 2.03% | 138 (4.9%) | 743 |
| 15% | 131,196 | 99.7% | 2.07% | 72 (2.6%) | 280 |
| 20% | 131,373 | 99.9% | 2.09% | 43 (1.5%) | 132 |

No hay un umbral óptimo sin definir el balance deseado entre aprobaciones, defaults y buenos pagadores rechazados.
