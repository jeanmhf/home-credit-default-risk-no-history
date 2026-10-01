# Auditoría de la tabla Home Credit para modelado

- Filas: 1,526,659; columnas: 207.
- `case_id` únicos: 1,526,659.
- Target: `target` (0: 1,478,665; 1: 47,994; tasa positiva: 3.14%).
- Periodo de solicitud: 2019-01-01 a 2020-10-05.
- Variables candidatas: 202; `case_id`, `target` y fechas/calendario se mantienen fuera de esa lista.
- Cobertura de los merges: `static_0` 100.00%; solicitante de `person_1` 100.00%.
- Parquet generado: `C:\Users\Usuario\Documents\ChatGPT\ML Project 1\data\processed\home_credit_train_joined.parquet` (208.32 MB).

## Variables con más valores faltantes

- `relationshiptoclient_642T`: 100.00%
- `relationshiptoclient_415T`: 100.00%
- `maritalst_703L`: 100.00%
- `housingtype_772L`: 100.00%
- `childnum_185L`: 100.00%
- `remitter_829L`: 100.00%
- `clientscnt_136L`: 99.97%
- `lastrepayingdate_696D`: 99.84%
- `lastotherinc_902A`: 99.80%
- `lastotherlnsexpense_631A`: 99.80%
- `payvacationpostpone_4187118D`: 99.39%
- `birthdate_87D`: 99.18%

La tabla conserva campos candidatos; todavía falta revisar valores faltantes, leakage y variables sensibles antes de definir los X finales. No se entrenó ningún modelo.
