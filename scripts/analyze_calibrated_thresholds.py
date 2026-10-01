from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_PATHS = {
    "CatBoost": ROOT / "data" / "processed" / "home_credit_catboost_calibrated_oot_predictions.parquet",
    "XGBoost": ROOT / "data" / "processed" / "home_credit_xgboost_calibrated_oot_predictions.parquet",
}
METRICS_PATH = ROOT / "reports" / "metrics" / "home_credit_calibrated_model_threshold_scenarios.csv"
REPORT_PATH = ROOT / "reports" / "home_credit_calibrated_model_threshold_analysis.md"
CATBOOST_METRICS_PATH = ROOT / "reports" / "metrics" / "home_credit_catboost_threshold_scenarios.csv"
CATBOOST_REPORT_PATH = ROOT / "reports" / "home_credit_catboost_threshold_analysis.md"

# Riesgo máximo estimado para aprobar; son escenarios, no umbrales elegidos.
THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.0225, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20]


def calculate_scenarios(model_name, df):
    probability = df["probability_target_1_calibrated"]
    target = df["target"]
    total_defaults = int(target.sum())
    total_good = int((target == 0).sum())
    rows = []

    for threshold in THRESHOLDS:
        approved = probability <= threshold
        rejected = ~approved
        approved_cases = int(approved.sum())
        rejected_cases = int(rejected.sum())
        defaults_approved = int(target[approved].sum())
        defaults_rejected = int(target[rejected].sum())
        good_rejected = int(((target == 0) & rejected).sum())

        rows.append(
            {
                "model": model_name,
                "threshold_pct": threshold * 100,
                "approved_cases": approved_cases,
                "approval_rate_pct": approved_cases / len(df) * 100,
                "defaults_among_approved": defaults_approved,
                "observed_default_rate_approved_pct": defaults_approved / approved_cases * 100,
                "rejected_cases": rejected_cases,
                "defaults_rejected": defaults_rejected,
                "default_capture_rate_pct": defaults_rejected / total_defaults * 100,
                "good_cases_rejected": good_rejected,
                "good_rejection_rate_pct": good_rejected / total_good * 100,
            }
        )

    scenarios = pd.DataFrame(rows)
    assert (scenarios["approved_cases"] + scenarios["rejected_cases"] == len(df)).all()
    assert (
        scenarios["defaults_among_approved"] + scenarios["defaults_rejected"]
        == total_defaults
    ).all()
    assert scenarios["approved_cases"].is_monotonic_increasing
    return scenarios


predictions = {
    model_name: pd.read_parquet(path)
    for model_name, path in PREDICTION_PATHS.items()
}

for model_name, df in predictions.items():
    assert df["case_id"].is_unique, f"case_id debe ser único en {model_name}."
    assert set(df["target"].unique()).issubset({0, 1}), "target debe ser binario."
    assert df["probability_target_1_calibrated"].between(0, 1).all()

catboost = predictions["CatBoost"].sort_values("case_id").reset_index(drop=True)
xgboost = predictions["XGBoost"].sort_values("case_id").reset_index(drop=True)
assert catboost[["case_id", "WEEK_NUM", "target"]].equals(
    xgboost[["case_id", "WEEK_NUM", "target"]]
), "Los modelos deben evaluarse sobre los mismos casos y targets."

scenarios = pd.concat(
    [
        calculate_scenarios(model_name, df)
        for model_name, df in predictions.items()
    ],
    ignore_index=True,
)
catboost_scenarios = scenarios.loc[scenarios["model"] == "CatBoost"].drop(
    columns="model"
)

METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
scenarios.to_csv(METRICS_PATH, index=False)
catboost_scenarios.to_csv(CATBOOST_METRICS_PATH, index=False)


def format_table(model_name):
    rows = []
    model_scenarios = scenarios.loc[scenarios["model"] == model_name]
    for row in model_scenarios.itertuples(index=False):
        rows.append(
            f"| {row.threshold_pct:g}% | {row.approved_cases:,} | "
            f"{row.approval_rate_pct:.1f}% | {row.observed_default_rate_approved_pct:.2f}% | "
            f"{row.defaults_rejected:,} ({row.default_capture_rate_pct:.1f}%) | "
            f"{row.good_cases_rejected:,} |"
        )
    return "\n".join(rows)


sample = predictions["CatBoost"]
target = sample["target"]
header = "| Umbral | Aprobadas | % aprobadas | Default entre aprobadas | Defaults rechazados | Buenos clientes rechazados |\n|---:|---:|---:|---:|---:|---:|"
disclaimer = (
    "Los escenarios simulan aprobar cuando la probabilidad calibrada de default es menor o igual al umbral. "
    f"Ambos modelos se evaluaron sobre los mismos {len(sample):,} casos de las semanas "
    f"{sample['WEEK_NUM'].min()}–{sample['WEEK_NUM'].max()}, con tasa observada de default de {target.mean():.2%} "
    f"({int(target.sum()):,} casos). `Defaults rechazados` muestra el número y porcentaje de todos los defaults "
    "observados capturados por la regla; `Buenos clientes rechazados` son casos sin default observado que se rechazarían. "
    "Son simulaciones retrospectivas: las semanas 81–91 ya se inspeccionaron durante el desarrollo y no constituyen un test final independiente."
)

report = f"""# Comparación de umbrales: modelos calibrados

{disclaimer}

## CatBoost

{header}
{format_table('CatBoost')}

## XGBoost

{header}
{format_table('XGBoost')}

No hay un umbral óptimo sin definir el balance deseado entre aprobaciones, defaults y buenos pagadores rechazados.
"""
REPORT_PATH.write_text(report, encoding="utf-8")

catboost_report = f"""# Escenarios de umbral: CatBoost calibrado

{disclaimer}

{header}
{format_table('CatBoost')}
"""
CATBOOST_REPORT_PATH.write_text(catboost_report, encoding="utf-8")

print(f"Casos comparados: {len(sample):,}; tasa de default: {target.mean():.2%}")
print(scenarios.loc[scenarios["threshold_pct"] == 2.5].to_string(index=False))
print(f"\nReporte: {REPORT_PATH}")
print(f"CSV completo: {METRICS_PATH}")
