# Home Credit - Credit Risk Model Stability (Kaggle 2024): auditoría inicial

- Filas en train_base: 1,526,659
- Unidad del target: caso de crédito (`case_id`), no necesariamente persona única.
- case_id únicos: 1,526,659
- Columnas de train_base: 5 (case_id, date_decision, MONTH, WEEK_NUM, target)
- Fechas de decisión: 2019-01-01 a 2020-10-05
- Aunque la competencia es de 2024, los casos de entrenamiento observados son de 2019–2020.
- Semanas: 0 a 91
- Filas del diccionario: 465
- Variables candidatas por descripción: 242
- Tasa media de target: 0.0314

## Tablas Parquet seleccionadas

                   table    rows  columns  feature_columns  unique_case_id  duplicate_case_id_rows  base_case_coverage_pct  rows_per_case  applicant_rows_num_group1_0  applicant_cases_num_group1_0  missing_feature_cells_pct  features_with_dictionary_definition  candidate_features_by_description
      train_base.parquet 1526659        5                3         1526659                       0                  100.00           1.00                            0                             0                       0.00                                    0                                  0
train_static_0_0.parquet 1003757      168              167         1003757                       0                   65.75           1.00                            0                             0                      33.12                                  167                                 77
train_static_0_1.parquet  522902      168              167          522902                       0                   34.25           1.00                            0                             0                      26.32                                  167                                 77
  train_person_1.parquet 2973991       37               35         1526659                 1447332                  100.00           1.95                      1526659                       1526659                      49.05                                   35                                 25
  train_person_2.parquet 1643410       11                8         1435105                  208305                   94.00           1.15                      1463928                       1435041                      36.72                                    8                                  7

Validación de static_0: mismo esquema = True; case_id repetidos entre partes = 0; casos únicos al concatenar = 1,526,659 (100.00% de base). Se deben concatenar las dos partes por filas antes de unirlas a train_base.
En person_1, num_group1 = 0 identifica al solicitante: la tabla confirma una fila del solicitante por cada case_id. person_2 contiene registros anidados de personas y relaciones, por lo que queda como extensión posterior. Las tablas externas de bureau también quedan para una comparación posterior.

## Conteo del target

target
0    1478665
1      47994

Esta auditoría inspecciona tablas y claves con pandas; todavía no entrena modelos ni descarga las tablas externas de bureau.
