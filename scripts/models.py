from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "home_credit_train_no_history_features.parquet"
BASELINE_MODEL_PATH = ROOT / "models" / "home_credit_logistic_core.joblib"
TUNED_MODEL_PATH = ROOT / "models" / "home_credit_logistic_core_tuned.joblib"
METRICS_PATH = ROOT / "reports" / "metrics" / "home_credit_logistic_core.csv"
TUNING_PATH = ROOT / "reports" / "metrics" / "home_credit_logistic_tuning_search.csv"

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
TUNING_TRAIN_MAX_WEEK = 70
TUNING_VALID_MIN_WEEK = 71
TUNING_VALID_MAX_WEEK = 80
FINAL_TRAIN_MAX_WEEK = 80
TUNING_SAMPLE_SIZE = 300_000
C_VALUES = [0.001, 0.01, 0.1, 1.0]
L1_RATIOS = [0.0, 1.0]
CLASS_WEIGHTS = [None, "balanced"]
BASELINE_C = 1.0
BASELINE_L1_RATIO = 0.0
MAX_ITER = 500


def build_model(c, l1_ratio=BASELINE_L1_RATIO, class_weight=None):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES + ["age_years"]),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            # l1_ratio=0 equivale a L2 y l1_ratio=1 a L1.
            # class_weight=None conserva la prevalencia natural de las probabilidades.
            (
                "logistic_regression",
                LogisticRegression(
                    C=c,
                    l1_ratio=l1_ratio,
                    class_weight=class_weight,
                    max_iter=MAX_ITER,
                    tol=1e-4 if l1_ratio == 0.0 else 1e-3,
                    solver="saga" if l1_ratio == 1.0 else "lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


df = pd.read_parquet(DATA_PATH)

assert df["case_id"].is_unique, "case_id debe ser único."
assert set(df["target"].dropna().unique()).issubset({0, 1}), "target debe ser binario."
assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["birth_259D"]).issubset(df.columns)

# Edad al momento de la solicitud; se crea en memoria y no se guarda en el Parquet.
decision_date = pd.to_datetime(df["date_decision"])
birth_date = pd.to_datetime(df["birth_259D"])
df["age_years"] = ((decision_date - birth_date).dt.days / 365.25).round()

# El tuning usa semanas 0-70 y 71-80. Las semanas 81-91 quedan para la evaluación final.
tuning_train_mask = df["WEEK_NUM"] <= TUNING_TRAIN_MAX_WEEK
tuning_valid_mask = df["WEEK_NUM"].between(TUNING_VALID_MIN_WEEK, TUNING_VALID_MAX_WEEK)

# Una muestra fija acelera la búsqueda; la evaluación final y el reentrenamiento usan todos los casos.
X_tuning_train = df.loc[tuning_train_mask, FEATURES].sample(
    n=TUNING_SAMPLE_SIZE, random_state=42
)
y_tuning_train = df.loc[X_tuning_train.index, "target"]
X_tuning_valid = df.loc[tuning_valid_mask, FEATURES]
y_tuning_valid = df.loc[tuning_valid_mask, "target"]

tuning_rows = []
for c in C_VALUES:
    for l1_ratio in L1_RATIOS:
        for class_weight in CLASS_WEIGHTS:
            candidate_model = build_model(c, l1_ratio, class_weight)
            candidate_model.fit(X_tuning_train, y_tuning_train)
            probability = candidate_model.predict_proba(X_tuning_valid)[:, 1]
            iterations = int(candidate_model.named_steps["logistic_regression"].n_iter_[0])
            tuning_rows.append(
                {
                    "C": c,
                    "l1_ratio": l1_ratio,
                    "class_weight": "none" if class_weight is None else class_weight,
                    "tuning_average_precision": float(
                        average_precision_score(y_tuning_valid, probability)
                    ),
                    "tuning_train_rows": len(X_tuning_train),
                    "tuning_validation_rows": int(tuning_valid_mask.sum()),
                    "iterations": iterations,
                    "converged": iterations < MAX_ITER,
                }
            )

tuning_results = pd.DataFrame(tuning_rows).sort_values(
    "tuning_average_precision", ascending=False
)
best_parameters = tuning_results.loc[tuning_results["converged"]].iloc[0]
best_c = float(best_parameters["C"])
best_l1_ratio = float(best_parameters["l1_ratio"])
best_class_weight = (
    None if best_parameters["class_weight"] == "none" else str(best_parameters["class_weight"])
)
best_tuning_average_precision = float(best_parameters["tuning_average_precision"])
baseline_tuning_average_precision = float(
    tuning_results.loc[
        (tuning_results["C"] == BASELINE_C)
        & (tuning_results["l1_ratio"] == BASELINE_L1_RATIO)
        & (tuning_results["class_weight"] == "none"),
        "tuning_average_precision",
    ].iloc[0]
)

final_train_mask = df["WEEK_NUM"] <= FINAL_TRAIN_MAX_WEEK
final_valid_mask = df["WEEK_NUM"] > FINAL_TRAIN_MAX_WEEK
X_train = df.loc[final_train_mask, FEATURES]
y_train = df.loc[final_train_mask, "target"]
X_valid = df.loc[final_valid_mask, FEATURES]
y_valid = df.loc[final_valid_mask, "target"]

baseline_model = build_model(BASELINE_C, BASELINE_L1_RATIO, None)
baseline_model.fit(X_train, y_train)

tuned_model = build_model(best_c, best_l1_ratio, best_class_weight)
tuned_model.fit(X_train, y_train)

validation_prevalence = float(y_valid.mean())
baseline_probability = [validation_prevalence] * len(y_valid)

comparison_rows = []
for name, c, l1_ratio, class_weight, fitted_model, tuning_average_precision in [
    (
        "baseline",
        BASELINE_C,
        BASELINE_L1_RATIO,
        None,
        baseline_model,
        baseline_tuning_average_precision,
    ),
    (
        "tuned",
        best_c,
        best_l1_ratio,
        best_class_weight,
        tuned_model,
        best_tuning_average_precision,
    ),
]:
    probability = fitted_model.predict_proba(X_valid)[:, 1]
    comparison_rows.append(
        {
            "model": name,
            "C": c,
            "l1_ratio": l1_ratio,
            "class_weight": "none" if class_weight is None else class_weight,
            "tuning_average_precision": tuning_average_precision,
            "validation_week_min": int(df.loc[final_valid_mask, "WEEK_NUM"].min()),
            "validation_week_max": int(df.loc[final_valid_mask, "WEEK_NUM"].max()),
            "train_rows": int(final_train_mask.sum()),
            "validation_rows": int(final_valid_mask.sum()),
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": validation_prevalence,
            "average_precision": float(average_precision_score(y_valid, probability)),
            "average_precision_baseline": validation_prevalence,
            "roc_auc": float(roc_auc_score(y_valid, probability)),
            "brier_score": float(brier_score_loss(y_valid, probability)),
            "brier_baseline": float(brier_score_loss(y_valid, baseline_probability)),
            "log_loss": float(log_loss(y_valid, probability)),
            "log_loss_baseline": float(log_loss(y_valid, baseline_probability)),
            "feature_count": len(FEATURES),
        }
    )

metrics = pd.DataFrame(comparison_rows)

BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(baseline_model, BASELINE_MODEL_PATH)
joblib.dump(tuned_model, TUNED_MODEL_PATH)
metrics.to_csv(METRICS_PATH, index=False)
tuning_results.to_csv(TUNING_PATH, index=False)

print("Búsqueda de C, l1_ratio y class_weight (mayor Average Precision gana):")
print(tuning_results.to_string(index=False))
print(
    f"\nParámetros elegidos: C={best_c}, l1_ratio={best_l1_ratio}, "
    f"class_weight={best_class_weight}"
)
print("\nComparación final en semanas 81-91:")
print(
    metrics[
        [
            "model",
            "C",
            "l1_ratio",
            "class_weight",
            "average_precision",
            "roc_auc",
            "brier_score",
            "log_loss",
        ]
    ].to_string(index=False)
)
print(f"\nBaseline guardado en: {BASELINE_MODEL_PATH}")
print(f"Modelo ajustado guardado en: {TUNED_MODEL_PATH}")
print(f"Métricas guardadas en: {METRICS_PATH}")
print(f"Resultados del tuning guardados en: {TUNING_PATH}")
