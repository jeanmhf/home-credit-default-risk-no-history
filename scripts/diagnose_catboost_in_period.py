from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "home_credit_train_no_history_features.parquet"
FIGURE_PATH = ROOT / "reports" / "figures" / "home_credit_catboost_in_period_curves.png"
HISTORY_PATH = ROOT / "reports" / "metrics" / "home_credit_catboost_in_period_history.csv"
REPORT_PATH = ROOT / "reports" / "home_credit_catboost_in_period_diagnostic.md"

NUMERIC_FEATURES = [
    "maininc_215A",
    "mainoccupationinc_384A",
    "credamount_770A",
    "annuity_780A",
    "numinstls_657L",
    "eir_270L",
    "downpmt_116A",
    "price_1097A",
]
CATEGORICAL_FEATURES = [
    "incometype_1044T",
    "empl_employedtotal_800L",
    "empl_industry_691L",
    "education_927M",
    "familystate_447L",
    "credtype_322L",
    "disbursementtype_67L",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["age_years"]
SAMPLE_SIZE = 100_000
ITERATIONS = 282
PARAMETERS = {
    "depth": 4,
    "learning_rate": 0.10972661298889345,
    "l2_leaf_reg": 8.800251535369485,
    "random_strength": 0.05494594621885618,
    "bagging_temperature": 1.795232264200432,
}


def prepare_features(rows):
    features = rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("Missing").astype(str)

    decision_date = pd.to_datetime(rows["date_decision"])
    birth_date = pd.to_datetime(rows["birth_259D"])
    features["age_years"] = ((decision_date - birth_date).dt.days / 365.25).round()
    return features[FEATURES]


df = pd.read_parquet(DATA_PATH)
assert df["case_id"].is_unique, "case_id debe ser único."
assert set(df["target"].dropna().unique()).issubset({0, 1}), "target debe ser binario."

# Diagnóstico interno con casos del mismo periodo; no es el test temporal final.
sample = df.loc[df["WEEK_NUM"] <= 70].sample(n=SAMPLE_SIZE, random_state=42)
train_rows, valid_rows = train_test_split(
    sample,
    test_size=0.20,
    random_state=42,
    stratify=sample["target"],
)

X_train = prepare_features(train_rows)
X_valid = prepare_features(valid_rows)
model = CatBoostClassifier(
    iterations=ITERATIONS,
    loss_function="Logloss",
    eval_metric="Logloss",
    random_seed=42,
    thread_count=-1,
    allow_writing_files=False,
    verbose=False,
    **PARAMETERS,
)
model.fit(
    X_train,
    train_rows["target"],
    cat_features=CATEGORICAL_FEATURES,
    eval_set=(X_valid, valid_rows["target"]),
    use_best_model=False,
    verbose=False,
)

results = model.get_evals_result()
history = pd.DataFrame(
    {
        "iteration": range(1, ITERATIONS + 1),
        "training_log_loss": results["learn"]["Logloss"],
        "validation_log_loss": results["validation"]["Logloss"],
    }
)
best_iteration = int(history["validation_log_loss"].idxmin()) + 1
training_loss_at_best = float(
    history.loc[best_iteration - 1, "training_log_loss"]
)
validation_loss_at_best = float(
    history.loc[best_iteration - 1, "validation_log_loss"]
)
loss_gap = validation_loss_at_best - training_loss_at_best

FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
history.to_csv(HISTORY_PATH, index=False)

figure, axis = plt.subplots(figsize=(9, 5))
axis.plot(
    history["iteration"], history["training_log_loss"], label="Training (80,000 casos)"
)
axis.plot(
    history["iteration"], history["validation_log_loss"], label="Validation (20,000 casos)"
)
axis.axvline(
    ITERATIONS,
    color="gray",
    linestyle="--",
    label=f"CatBoost elegido: {ITERATIONS} árboles",
)
axis.set_title("CatBoost: Logloss de entrenamiento vs. validación")
axis.set_xlabel("Número de árboles")
axis.set_ylabel("Logloss (menor es mejor)")
axis.legend()
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
plt.close(figure)

parameters_text = ", ".join(
    f"{name}={value}" for name, value in PARAMETERS.items()
)

report = f"""# CatBoost: Logloss de entrenamiento y validación

Se seleccionaron {SAMPLE_SIZE:,} casos de las semanas 0–70 y se dividieron aleatoriamente, estratificando por `target`: {len(train_rows):,} para entrenamiento y {len(valid_rows):,} para validación. Las tasas positivas fueron {train_rows['target'].mean():.2%} y {valid_rows['target'].mean():.2%}.

El ajuste usa los hiperparámetros seleccionados por Optuna: {parameters_text}, con {ITERATIONS} árboles. En el menor Logloss de validación (árbol {best_iteration}), el Logloss fue {training_loss_at_best:.5f} en entrenamiento y {validation_loss_at_best:.5f} en validación; la diferencia fue {loss_gap:.5f}.

Las dos líneas se calculan sobre particiones distintas de la misma muestra original de 100,000 casos (80,000 train y 20,000 validation), no sobre las mismas filas. Esta gráfica revisa sobreajuste dentro de las semanas 0–70; no usa la evaluación temporal de semanas 81–91 ni reemplaza sus métricas.

![Logloss de entrenamiento y validación](figures/home_credit_catboost_in_period_curves.png)

Esta comprobación vuelve a entrenar el modelo solo para registrar la curva; no modifica el artefacto del CatBoost final ni su calibrador.
"""
REPORT_PATH.write_text(report, encoding="utf-8")

print(f"Casos: {SAMPLE_SIZE:,}; train: {len(train_rows):,}; validation: {len(valid_rows):,}")
print(f"Tasa positiva train: {train_rows['target'].mean():.2%}")
print(f"Tasa positiva validation: {valid_rows['target'].mean():.2%}")
print(
    f"Menor validation Logloss: {validation_loss_at_best:.5f} "
    f"en el árbol {best_iteration}; training Logloss allí: {training_loss_at_best:.5f}; "
    f"diferencia: {loss_gap:.5f}"
)
print(f"Gráfica: {FIGURE_PATH}")
