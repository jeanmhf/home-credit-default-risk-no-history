from pathlib import Path

import joblib
import optuna
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
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "home_credit_train_no_history_features.parquet"
BASE_MODEL_PATH = ROOT / "models" / "home_credit_xgboost_calibrated_base.joblib"
CALIBRATOR_PATH = ROOT / "models" / "home_credit_xgboost_platt_calibrator.joblib"
TUNING_PATH = ROOT / "reports" / "metrics" / "home_credit_xgboost_calibration_tuning.csv"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "home_credit_xgboost_calibrated_oot_predictions.parquet"
METRICS_PATH = ROOT / "reports" / "metrics" / "home_credit_xgboost_calibrated_oot_metrics.csv"
DECILES_PATH = ROOT / "reports" / "metrics" / "home_credit_xgboost_calibrated_oot_deciles.csv"
WEEKLY_PATH = ROOT / "reports" / "metrics" / "home_credit_xgboost_calibrated_oot_weekly.csv"
COMPARISON_PATH = ROOT / "reports" / "metrics" / "home_credit_calibrated_model_comparison.csv"
CATBOOST_METRICS_PATH = ROOT / "reports" / "metrics" / "home_credit_catboost_calibrated_oot_metrics.csv"

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
MAX_ESTIMATORS = 1_200
EARLY_STOPPING_ROUNDS = 50


def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES + ["age_years"]),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_xgboost(parameters, iterations, early_stopping_rounds=None):
    model_parameters = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": iterations,
        "random_state": 42,
        "n_jobs": 4,
        "verbosity": 0,
        **parameters,
    }
    if early_stopping_rounds is not None:
        model_parameters["early_stopping_rounds"] = early_stopping_rounds
    return XGBClassifier(**model_parameters)


def prepare_features(rows):
    features = rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("Missing").astype(str)

    decision_date = pd.to_datetime(rows["date_decision"])
    birth_date = pd.to_datetime(rows["birth_259D"])
    features["age_years"] = ((decision_date - birth_date).dt.days / 365.25).round()
    return features[FEATURES]


def calculate_metrics(name, target, raw_probability, calibrated_probability):
    rows = []
    for probability_type, probability in [
        ("raw", raw_probability),
        ("platt_calibrated", calibrated_probability),
    ]:
        rows.append(
            {
                "model": name,
                "probability_type": probability_type,
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
assert set(
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["birth_259D", "date_decision"]
).issubset(df.columns)

df["age_years"] = (
    (pd.to_datetime(df["date_decision"]) - pd.to_datetime(df["birth_259D"])).dt.days
    / 365.25
).round()

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

tuning_preprocessor = build_preprocessor()
X_tuning_train_encoded = tuning_preprocessor.fit_transform(X_tuning_train)
X_tuning_valid_encoded = tuning_preprocessor.transform(X_tuning_valid)


def objective(trial):
    parameters = {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 12, log=True),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }
    candidate_model = build_xgboost(
        parameters,
        MAX_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    candidate_model.fit(
        X_tuning_train_encoded,
        y_tuning_train,
        eval_set=[(X_tuning_valid_encoded, y_tuning_valid)],
        verbose=False,
    )
    probability = candidate_model.predict_proba(X_tuning_valid_encoded)[:, 1]
    trial.set_user_attr("best_iteration", int(candidate_model.best_iteration))
    trial.set_user_attr("tuning_train_rows", len(X_tuning_train))
    trial.set_user_attr("tuning_validation_rows", len(X_tuning_valid))
    return float(average_precision_score(y_tuning_valid, probability))


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

tuning_results = study.trials_dataframe().sort_values("value", ascending=False)
best_trial = study.best_trial
best_parameters = best_trial.params
best_iterations = int(best_trial.user_attrs["best_iteration"]) + 1

train_rows = df.loc[df["WEEK_NUM"] <= MODEL_TRAIN_MAX_WEEK]
calibration_rows = df.loc[
    df["WEEK_NUM"].between(CALIBRATION_MIN_WEEK, CALIBRATION_MAX_WEEK)
]
oot_rows = df.loc[df["WEEK_NUM"] > CALIBRATION_MAX_WEEK]

assert train_rows["WEEK_NUM"].max() < calibration_rows["WEEK_NUM"].min()
assert calibration_rows["WEEK_NUM"].max() < oot_rows["WEEK_NUM"].min()

preprocessor = build_preprocessor()
X_train = preprocessor.fit_transform(prepare_features(train_rows))
X_calibration = preprocessor.transform(prepare_features(calibration_rows))
X_oot = preprocessor.transform(prepare_features(oot_rows))

model = build_xgboost(best_parameters, best_iterations)
model.fit(X_train, train_rows["target"], verbose=False)

calibration_margin = model.predict(X_calibration, output_margin=True).reshape(-1, 1)
calibrator = LogisticRegression(C=1_000_000, max_iter=1_000)
calibrator.fit(calibration_margin, calibration_rows["target"])
assert calibrator.coef_[0, 0] > 0, "La calibración invirtió el orden del riesgo."

raw_probability = model.predict_proba(X_oot)[:, 1]
oot_margin = model.predict(X_oot, output_margin=True).reshape(-1, 1)
calibrated_probability = calibrator.predict_proba(oot_margin)[:, 1]

predictions = oot_rows[["case_id", "WEEK_NUM", "target"]].copy()
predictions["probability_target_1_raw"] = raw_probability
predictions["probability_target_1_calibrated"] = calibrated_probability
risk_rank = predictions["probability_target_1_calibrated"].rank(
    method="first", ascending=False
)
predictions["risk_decile"] = pd.qcut(risk_rank, q=10, labels=range(1, 11)).astype(int)

overall_metrics = pd.DataFrame(
    calculate_metrics(
        "xgboost_temporal_fit",
        predictions["target"],
        raw_probability,
        calibrated_probability,
    )
)
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
overall_metrics["optuna_trials"] = OPTUNA_TRIALS
overall_metrics["best_average_precision_validation"] = float(best_trial.value)
overall_metrics["max_depth"] = best_parameters["max_depth"]
overall_metrics["min_child_weight"] = best_parameters["min_child_weight"]
overall_metrics["learning_rate"] = best_parameters["learning_rate"]
overall_metrics["subsample"] = best_parameters["subsample"]
overall_metrics["colsample_bytree"] = best_parameters["colsample_bytree"]
overall_metrics["reg_alpha"] = best_parameters["reg_alpha"]
overall_metrics["reg_lambda"] = best_parameters["reg_lambda"]
overall_metrics["scale_pos_weight"] = best_parameters["scale_pos_weight"]
overall_metrics["gamma"] = best_parameters["gamma"]
overall_metrics["n_estimators"] = best_iterations

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
decile_metrics["lift_over_base_rate"] = (
    decile_metrics["observed_target_rate"] / predictions["target"].mean()
)
decile_metrics["calibrated_gap"] = (
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
                "average_precision": float(average_precision_score(week_target, probability)),
                "roc_auc": float(roc_auc_score(week_target, probability)),
                "brier_score": float(brier_score_loss(week_target, probability)),
                "log_loss": float(log_loss(week_target, probability, labels=[0, 1])),
            }
        )
weekly_metrics = pd.DataFrame(weekly_rows)

catboost_metrics = pd.read_csv(CATBOOST_METRICS_PATH)
catboost_calibrated = catboost_metrics.loc[
    catboost_metrics["probability_type"] == "platt_calibrated"
].iloc[0]
xgboost_calibrated = overall_metrics.loc[
    overall_metrics["probability_type"] == "platt_calibrated"
].iloc[0]
comparison = pd.DataFrame(
    [
        {
            "model": "CatBoost Optuna",
            "training_weeks": catboost_calibrated["model_training_weeks"],
            "calibration_weeks": catboost_calibrated["calibration_weeks"],
            "average_precision": catboost_calibrated["average_precision"],
            "roc_auc": catboost_calibrated["roc_auc"],
            "brier_score": catboost_calibrated["brier_score"],
            "log_loss": catboost_calibrated["log_loss"],
        },
        {
            "model": "XGBoost Optuna",
            "training_weeks": xgboost_calibrated["model_training_weeks"],
            "calibration_weeks": xgboost_calibrated["calibration_weeks"],
            "average_precision": xgboost_calibrated["average_precision"],
            "roc_auc": xgboost_calibrated["roc_auc"],
            "brier_score": xgboost_calibrated["brier_score"],
            "log_loss": xgboost_calibrated["log_loss"],
        },
    ]
)

BASE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)
PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
pipeline = Pipeline([("preprocessor", preprocessor), ("xgboost", model)])
joblib.dump(pipeline, BASE_MODEL_PATH)
joblib.dump(calibrator, CALIBRATOR_PATH)
tuning_results.to_csv(TUNING_PATH, index=False)
predictions.to_parquet(PREDICTIONS_PATH, index=False)
overall_metrics.to_csv(METRICS_PATH, index=False)
decile_metrics.to_csv(DECILES_PATH, index=False)
weekly_metrics.to_csv(WEEKLY_PATH, index=False)
comparison.to_csv(COMPARISON_PATH, index=False)

print(
    f"Optuna XGBoost con GPU: {OPTUNA_TRIALS} trials; "
    "train semanas 0-50, validación 51-60"
)
print(tuning_results.head(5).to_string(index=False))
print(
    f"\nMejores parámetros: {best_parameters}, "
    f"n_estimators={best_iterations}; entrenamiento 0-70, calibración 71-80"
)
print("\nComparación calibrada en semanas 81-91:")
print(comparison.to_string(index=False))
print(f"\nPredicciones: {PREDICTIONS_PATH}")
print(f"Métricas: {METRICS_PATH}")
print(f"Comparación: {COMPARISON_PATH}")
