from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "home_credit_train_no_history_features.parquet"
BASE_MODEL_PATH = ROOT / "models" / "home_credit_catboost_calibrated_base.cbm"
CALIBRATOR_PATH = ROOT / "models" / "home_credit_catboost_platt_calibrator.joblib"
TUNING_PATH = ROOT / "reports" / "metrics" / "home_credit_catboost_calibration_tuning.csv"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "home_credit_catboost_calibrated_oot_predictions.parquet"
METRICS_PATH = ROOT / "reports" / "metrics" / "home_credit_catboost_calibrated_oot_metrics.csv"
DECILES_PATH = ROOT / "reports" / "metrics" / "home_credit_catboost_calibrated_oot_deciles.csv"
WEEKLY_PATH = ROOT / "reports" / "metrics" / "home_credit_catboost_calibrated_oot_weekly.csv"

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
TUNING_TRAIN_MAX_WEEK = 50
TUNING_VALID_MIN_WEEK = 51
TUNING_VALID_MAX_WEEK = 60
MODEL_TRAIN_MAX_WEEK = 70
CALIBRATION_MIN_WEEK = 71
CALIBRATION_MAX_WEEK = 80
TUNING_SAMPLE_SIZE = 100_000
OPTUNA_TRIALS = 30
MAX_ITERATIONS = 1_200
EARLY_STOPPING_ROUNDS = 50


def prepare_features(rows):
    features = rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("Missing").astype(str)

    decision_date = pd.to_datetime(rows["date_decision"])
    birth_date = pd.to_datetime(rows["birth_259D"])
    features["age_years"] = ((decision_date - birth_date).dt.days / 365.25).round()
    return features[FEATURES]


def build_catboost(parameters, iterations):
    return CatBoostClassifier(
        iterations=iterations,
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
        **parameters,
    )


def calculate_metrics(name, target, raw_probability, calibrated_probability):
    rows = []
    for prediction_type, probability in [
        ("raw", raw_probability),
        ("platt_calibrated", calibrated_probability),
    ]:
        rows.append(
            {
                "model": name,
                "probability_type": prediction_type,
                "cases": len(target),
                "target_positives": int(target.sum()),
                "target_rate": float(target.mean()),
                "mean_predicted_probability": float(probability.mean()),
                "average_precision": float(average_precision_score(target, probability)),
                "roc_auc": float(roc_auc_score(target, probability)),
                "brier_score": float(brier_score_loss(target, probability)),
                "log_loss": float(log_loss(target, probability, labels=[0, 1])),
            }
        )
    return rows


df = pd.read_parquet(DATA_PATH)

assert df["case_id"].is_unique, "case_id debe ser único."
assert set(df["target"].dropna().unique()).issubset({0, 1}), "target debe ser binario."
assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["birth_259D", "date_decision"]).issubset(df.columns)

tuning_train_rows = df.loc[df["WEEK_NUM"] <= TUNING_TRAIN_MAX_WEEK]
tuning_valid_rows = df.loc[
    df["WEEK_NUM"].between(TUNING_VALID_MIN_WEEK, TUNING_VALID_MAX_WEEK)
]
X_tuning_train = prepare_features(
    tuning_train_rows.sample(n=TUNING_SAMPLE_SIZE, random_state=42)
)
y_tuning_train = tuning_train_rows.loc[X_tuning_train.index, "target"]
X_tuning_valid = prepare_features(tuning_valid_rows)
y_tuning_valid = tuning_valid_rows["target"]

def objective(trial):
    candidate = {
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        "random_strength": trial.suggest_float("random_strength", 0, 5),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
    }
    candidate_model = build_catboost(candidate, MAX_ITERATIONS)
    candidate_model.fit(
        X_tuning_train,
        y_tuning_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(X_tuning_valid, y_tuning_valid),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=False,
    )
    probability = candidate_model.predict_proba(X_tuning_valid)[:, 1]
    trial.set_user_attr("best_iteration", candidate_model.get_best_iteration())
    trial.set_user_attr("tuning_train_rows", len(X_tuning_train))
    trial.set_user_attr("tuning_validation_rows", len(X_tuning_valid))
    return float(average_precision_score(y_tuning_valid, probability))


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=OPTUNA_TRIALS)

tuning_results = study.trials_dataframe()
tuning_results = tuning_results.sort_values("value", ascending=False)
best_trial = study.best_trial
best_candidate = best_trial.params
best_iterations = int(best_trial.user_attrs["best_iteration"]) + 1

train_rows = df.loc[df["WEEK_NUM"] <= MODEL_TRAIN_MAX_WEEK]
calibration_rows = df.loc[
    df["WEEK_NUM"].between(CALIBRATION_MIN_WEEK, CALIBRATION_MAX_WEEK)
]
oot_rows = df.loc[df["WEEK_NUM"] > CALIBRATION_MAX_WEEK]

assert train_rows["WEEK_NUM"].max() < calibration_rows["WEEK_NUM"].min()
assert calibration_rows["WEEK_NUM"].max() < oot_rows["WEEK_NUM"].min()

model = build_catboost(best_candidate, best_iterations)
model.fit(
    prepare_features(train_rows),
    train_rows["target"],
    cat_features=CATEGORICAL_FEATURES,
    verbose=False,
)

calibration_margin = model.predict(
    prepare_features(calibration_rows), prediction_type="RawFormulaVal"
).reshape(-1, 1)
calibrator = LogisticRegression(C=1_000_000, max_iter=1_000)
calibrator.fit(calibration_margin, calibration_rows["target"])
assert calibrator.coef_[0, 0] > 0, "La calibración invirtió el orden del riesgo."

oot_features = prepare_features(oot_rows)
oot_margin = model.predict(oot_features, prediction_type="RawFormulaVal").reshape(-1, 1)
raw_probability = model.predict_proba(oot_features)[:, 1]
calibrated_probability = calibrator.predict_proba(oot_margin)[:, 1]

predictions = oot_rows[["case_id", "WEEK_NUM", "target"]].copy()
predictions["probability_target_1_raw"] = raw_probability
predictions["probability_target_1_calibrated"] = calibrated_probability
risk_rank = predictions["probability_target_1_calibrated"].rank(
    method="first", ascending=False
)
predictions["risk_decile"] = pd.qcut(
    risk_rank, q=10, labels=range(1, 11)
).astype(int)

metric_rows = []
metric_rows.extend(
    calculate_metrics(
        "catboost_temporal_fit",
        predictions["target"],
        predictions["probability_target_1_raw"],
        predictions["probability_target_1_calibrated"],
    )
)
overall_metrics = pd.DataFrame(metric_rows)
overall_metrics["week_min"] = int(predictions["WEEK_NUM"].min())
overall_metrics["week_max"] = int(predictions["WEEK_NUM"].max())
overall_metrics["evaluation_status"] = (
    "Development out-of-time evaluation; weeks 81-91 were inspected in earlier comparisons."
)
overall_metrics["tuning_weeks"] = "51-60"
overall_metrics["model_training_weeks"] = "0-70"
overall_metrics["calibration_weeks"] = "71-80"
overall_metrics["calibration_rows"] = len(calibration_rows)
overall_metrics["calibration_positives"] = int(calibration_rows["target"].sum())
overall_metrics["depth"] = best_candidate["depth"]
overall_metrics["l2_leaf_reg"] = best_candidate["l2_leaf_reg"]
overall_metrics["iterations"] = best_iterations

decile_metrics = (
    predictions.groupby("risk_decile", as_index=False)
    .agg(
        cases=("target", "size"),
        target_positives=("target", "sum"),
        observed_target_rate=("target", "mean"),
        mean_raw_probability=("probability_target_1_raw", "mean"),
        mean_calibrated_probability=("probability_target_1_calibrated", "mean"),
    )
    .sort_values("risk_decile")
)
base_rate = float(predictions["target"].mean())
decile_metrics["lift_over_base_rate"] = (
    decile_metrics["observed_target_rate"] / base_rate
)
decile_metrics["raw_calibration_gap"] = (
    decile_metrics["observed_target_rate"] - decile_metrics["mean_raw_probability"]
)
decile_metrics["calibrated_calibration_gap"] = (
    decile_metrics["observed_target_rate"]
    - decile_metrics["mean_calibrated_probability"]
)
decile_metrics["cumulative_recall"] = (
    decile_metrics["target_positives"].cumsum() / predictions["target"].sum()
)

weekly_rows = []
for week, week_predictions in predictions.groupby("WEEK_NUM", sort=True):
    week_target = week_predictions["target"]
    for probability_type, column in [
        ("raw", "probability_target_1_raw"),
        ("platt_calibrated", "probability_target_1_calibrated"),
    ]:
        probability = week_predictions[column]
        weekly_rows.append(
            {
                "WEEK_NUM": int(week),
                "probability_type": probability_type,
                "cases": len(week_predictions),
                "target_positives": int(week_target.sum()),
                "target_rate": float(week_target.mean()),
                "mean_predicted_probability": float(probability.mean()),
                "average_precision": float(
                    average_precision_score(week_target, probability)
                ),
                "roc_auc": float(roc_auc_score(week_target, probability)),
                "brier_score": float(brier_score_loss(week_target, probability)),
                "log_loss": float(log_loss(week_target, probability, labels=[0, 1])),
            }
        )
weekly_metrics = pd.DataFrame(weekly_rows)

BASE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)
PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save_model(BASE_MODEL_PATH)
joblib.dump(calibrator, CALIBRATOR_PATH)
tuning_results.to_csv(TUNING_PATH, index=False)
predictions.to_parquet(PREDICTIONS_PATH, index=False)
overall_metrics.to_csv(METRICS_PATH, index=False)
decile_metrics.to_csv(DECILES_PATH, index=False)
weekly_metrics.to_csv(WEEKLY_PATH, index=False)

print(f"Optuna CatBoost: {OPTUNA_TRIALS} trials; semanas 0-50 train y 51-60 valid")
print(tuning_results.to_string(index=False))
print(
    f"\nCatBoost final: {best_candidate}, iterations={best_iterations}; "
    f"entrenamiento 0-70, calibración 71-80"
)
print("\nEvaluación semanas 81-91:")
print(
    overall_metrics[
        [
            "probability_type",
            "target_rate",
            "mean_predicted_probability",
            "average_precision",
            "roc_auc",
            "brier_score",
            "log_loss",
        ]
    ].to_string(index=False)
)
print(f"\nModelo base guardado en: {BASE_MODEL_PATH}")
print(f"Calibrador guardado en: {CALIBRATOR_PATH}")
print(f"Predicciones guardadas en: {PREDICTIONS_PATH}")
