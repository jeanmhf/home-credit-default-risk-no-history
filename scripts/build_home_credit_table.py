"""Build and validate a one-row-per-credit-case Home Credit training table."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "home_credit_2024"
OUTPUT = ROOT / "data" / "processed" / "home_credit_train_joined.parquet"
REPORTS = ROOT / "reports" / "metrics"


def main():
    base = pd.read_parquet(RAW / "train_base.parquet")
    static_0 = pd.concat(
        [
            pd.read_parquet(RAW / "train_static_0_0.parquet"),
            pd.read_parquet(RAW / "train_static_0_1.parquet"),
        ],
        ignore_index=True,
    )
    applicant = pd.read_parquet(
        RAW / "train_person_1.parquet",
        filters=[("num_group1", "==", 0)],
    ).drop(columns="num_group1")

    if not base["case_id"].is_unique:
        raise ValueError("train_base tiene case_id repetidos.")
    if not static_0["case_id"].is_unique:
        raise ValueError("static_0 tiene case_id repetidos después de concatenar.")
    if not applicant["case_id"].is_unique:
        raise ValueError("person_1 tiene más de una fila del solicitante por case_id.")
    if set(base.columns) & set(static_0.columns) != {"case_id"}:
        raise ValueError("Hay columnas duplicadas entre base y static_0.")
    if set(base.columns) & set(applicant.columns) != {"case_id"}:
        raise ValueError("Hay columnas duplicadas entre base y person_1.")
    if set(static_0.columns) & set(applicant.columns) != {"case_id"}:
        raise ValueError("Hay columnas duplicadas entre static_0 y person_1.")
    if base["target"].isna().any() or not set(base["target"].unique()).issubset({0, 1}):
        raise ValueError("target debe estar completo y contener solo 0/1.")

    table = base.merge(
        static_0,
        on="case_id",
        how="left",
        validate="one_to_one",
        indicator="static_merge",
    )
    static_coverage = round(100 * table["static_merge"].eq("both").mean(), 2)
    if not table["static_merge"].eq("both").all():
        raise ValueError("No todos los casos de base encontraron fila en static_0.")
    table = table.drop(columns="static_merge")

    table = table.merge(
        applicant,
        on="case_id",
        how="left",
        validate="one_to_one",
        indicator="applicant_merge",
    )
    applicant_coverage = round(100 * table["applicant_merge"].eq("both").mean(), 2)
    if not table["applicant_merge"].eq("both").all():
        raise ValueError("No todos los casos de base encontraron fila del solicitante.")
    table = table.drop(columns="applicant_merge")

    if len(table) != len(base) or not table["case_id"].is_unique:
        raise ValueError("La tabla final cambió el número de casos o repite case_id.")

    metadata = {"case_id", "target", "date_decision", "MONTH", "WEEK_NUM"}
    candidate_features = [column for column in table.columns if column not in metadata]
    if "target" in candidate_features:
        raise ValueError("target no puede formar parte de las variables independientes.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT, index=False, compression="snappy")

    missing = pd.Series({
        column: round(float(table[column].isna().mean() * 100), 2)
        for column in candidate_features
    })
    missing.rename_axis("variable").rename("missing_pct").sort_values(
        ascending=False
    ).to_csv(REPORTS / "home_credit_missing_by_variable.csv")
    target_counts = table["target"].value_counts().sort_index()
    report = {
        "rows": len(table),
        "columns": len(table.columns),
        "case_id_unique": bool(table["case_id"].is_unique),
        "target_column": "target",
        "target_counts": {str(key): int(value) for key, value in target_counts.items()},
        "target_rate": round(float(table["target"].mean()), 5),
        "candidate_feature_count": len(candidate_features),
        "candidate_features": candidate_features,
        "metadata_columns_not_predictors": sorted(metadata),
        "static_0_merge_coverage_pct": static_coverage,
        "applicant_merge_coverage_pct": applicant_coverage,
        "decision_date_min": str(table["date_decision"].min()),
        "decision_date_max": str(table["date_decision"].max()),
        "output_file": str(OUTPUT),
        "output_size_mb": round(OUTPUT.stat().st_size / 1_000_000, 2),
    }
    (REPORTS / "home_credit_training_table_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    missing_lines = [
        f"- `{variable}`: {percentage:.2f}%"
        for variable, percentage in missing.sort_values(ascending=False).head(12).items()
    ]
    markdown = [
        "# Auditoría de la tabla Home Credit para modelado",
        "",
        f"- Filas: {len(table):,}; columnas: {len(table.columns)}.",
        f"- `case_id` únicos: {table['case_id'].nunique():,}.",
        f"- Target: `target` (0: {target_counts.get(0, 0):,}; 1: {target_counts.get(1, 0):,}; tasa positiva: {report['target_rate']:.2%}).",
        f"- Periodo de solicitud: {report['decision_date_min']} a {report['decision_date_max']}.",
        f"- Variables candidatas: {len(candidate_features)}; `case_id`, `target` y fechas/calendario se mantienen fuera de esa lista.",
        f"- Cobertura de los merges: `static_0` {static_coverage:.2f}%; solicitante de `person_1` {applicant_coverage:.2f}%.",
        f"- Parquet generado: `{OUTPUT}` ({report['output_size_mb']:.2f} MB).",
        "",
        "## Variables con más valores faltantes",
        "",
        *missing_lines,
        "",
        "La tabla conserva campos candidatos; todavía falta revisar valores faltantes, leakage y variables sensibles antes de definir los X finales. No se entrenó ningún modelo.",
    ]
    (REPORTS / "home_credit_training_table_audit.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    print(f"Tabla creada: {OUTPUT}")
    print(f"Filas y columnas: {table.shape}")
    print(f"case_id únicos: {table['case_id'].nunique()}")
    print(f"Target: {target_counts.to_dict()} (tasa positiva: {report['target_rate']:.2%})")
    print(f"Variables candidatas: {len(candidate_features)}")
    print(f"Cobertura de merges: static_0={static_coverage}%, solicitante={applicant_coverage}%")
    print(f"Tamaño Parquet: {report['output_size_mb']} MB")
    print(f"Reporte: {REPORTS / 'home_credit_training_table_audit.md'}")


if __name__ == "__main__":
    main()
