# Instrucciones para este proyecto

Este es un proyecto individual. El código debe ser fácil de leer, ejecutar y modificar por una sola persona.

## Estilo de implementación

- Evita la sobreingeniería. Resuelve cada tarea con el enfoque más directo que sea claro y correcto.
- Prefiere `pandas` y operaciones reconocibles como `DataFrame`, `merge`, `concat`, `groupby`, `read_parquet` y `to_parquet`.
- No uses la API de `pyarrow` directamente para leer o escribir Parquet; usa pandas.
- Evita clases, capas, fábricas, configuraciones genéricas y abstracciones que no aporten a una necesidad concreta del proyecto.
- No agregues condicionales defensivos para escenarios hipotéticos o de producción. Conserva validaciones simples que comprueben claves, cardinalidad de merges, target y leakage.
- Mantén las dependencias al mínimo; añade una biblioteca solo si resuelve una necesidad real.
- Prefiere scripts `.py` claros para el flujo de trabajo. No crees notebooks salvo que el usuario los pida expresamente.
- Usa nombres descriptivos y comentarios breves para explicar decisiones que no sean obvias; evita documentar línea por línea.

## Datos y modelado

- Mantén una fila por caso de crédito (`case_id`) en la tabla de entrenamiento y conserva `target` separado de los predictores.
- Valida antes de modelar la unicidad de `case_id`, las cardinalidades y la cobertura de los merges.
- No incluyas identificadores ni `target` entre las variables independientes. Revisa leakage y variables sensibles antes de entrenar.
- No presentes el conjunto Home Credit como representativo de Perú: sus casos de entrenamiento observados son de 2019–2020 y pertenecen al contexto de Home Credit.
- No avances al siguiente hito del proyecto sin informar al usuario qué se completó y qué resultado produjo.

## Credenciales y archivos locales

- Lee credenciales desde variables de entorno o `.env`; nunca las imprimas, incluyas en reportes o guardes en archivos versionados.
- No edites ni reemplaces datos crudos salvo que el usuario lo solicite. Escribe los resultados procesados bajo `data/processed/` y los reportes bajo `reports/`.
