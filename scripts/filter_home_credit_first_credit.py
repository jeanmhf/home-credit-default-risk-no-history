"""Create a Home Credit table without prior-credit-history features."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "home_credit_train_filtered.parquet"
OUTPUT = ROOT / "data" / "processed" / "home_credit_train_no_history_features.parquet"
DEFINITIONS = ROOT / "data" / "raw" / "home_credit_2024" / "feature_definitions.csv"
REPORT = ROOT / "reports" / "metrics" / "home_credit_first_credit_filter.md"
AUDIT = ROOT / "reports" / "metrics" / "home_credit_first_credit_feature_audit.csv"


HISTORY_PATTERN = (
    r"previous|last .*(?:payment|instalment|installment|loan|credit|application|contract|"
    r"delinquency|unpaid|rejected|approved)|last [0-9]+ months|in the last [0-9]+ months|"
    r"in last [0-9]+ months|past due|days past due|overdue|unpaid|outstanding|debt|"
    r"active credits?|active loans?|active revolving credits?|credit card|payments? made|"
    r"paid instalments?|paid installments?|instalments? .*paid|installments? .*paid|"
    r"number of incoming payments|number of paid|number of unpaid|delinquency|\bDPD\b|"
    r"days before due|credit applications .*rejected|non-activated credits|"
    r"master contract|communications indicating low income|first due date"
)

# These fields are ambiguous or describe a prior rejection/payment delay.
HISTORY_FEATURES_EXTRA = [
    "daysoverduetolerancedd_3976961L",
    "lastrejectreason_759M",
    "monthsannuity_845L",
]

# These are not the applicant's own repayment history.
KEEP_HISTORY_EXCEPTIONS = [
    "credamount_770A",
    "posfstqpd30lastmonth_3976962P",
    "posfpd30lastmonth_3976960P",
    "posfpd10lastmonth_333P",
    "clientscnt3m_3712950L",
    "clientscnt6m_3712949L",
]

# Their availability at application is unclear from the short descriptions.
TIMING_UNCLEAR_FEATURES = [
    "annuitynextmonth_57A",
    "disbursedcredamount_1113A",
    "firstdatedue_489D",
]

METADATA = ["case_id", "date_decision", "MONTH", "WEEK_NUM", "target"]


def main():
    table = pd.read_parquet(INPUT)
    definitions = pd.read_csv(DEFINITIONS)[["Variable", "Description"]]
    features = [column for column in table.columns if column not in METADATA]
    audit = pd.DataFrame({"Variable": features}).merge(
        definitions, on="Variable", how="left", validate="one_to_one"
    )

    audit["date_gt_decision_rows"] = 0
    decision_date = pd.to_datetime(table["date_decision"], errors="coerce")
    date_features = [column for column in features if column.endswith("D")]
    for column in date_features:
        feature_date = pd.to_datetime(table[column], errors="coerce")
        date_gt_decision_rows = int((feature_date.notna() & (feature_date > decision_date)).sum())
        audit.loc[audit["Variable"] == column, "date_gt_decision_rows"] = date_gt_decision_rows

    audit["reason"] = ""
    history_mask = audit["Description"].str.contains(
        HISTORY_PATTERN, case=False, regex=True, na=False
    ) & ~audit["Variable"].isin(KEEP_HISTORY_EXCEPTIONS)
    history_mask |= audit["Variable"].isin(HISTORY_FEATURES_EXTRA)
    audit.loc[history_mask, "reason"] = "Historial de crédito o pagos previos"

    timing_unclear_mask = audit["Variable"].isin(TIMING_UNCLEAR_FEATURES)
    audit.loc[timing_unclear_mask & ~history_mask, "reason"] = "Disponibilidad temporal no confirmada"
    audit.loc[timing_unclear_mask & history_mask, "reason"] += "; disponibilidad temporal no confirmada"

    date_gt_decision_mask = audit["date_gt_decision_rows"] > 0
    audit.loc[date_gt_decision_mask & audit["reason"].eq(""), "reason"] = (
        "Valor de fecha > date_decision; apartado preventivamente"
    )
    audit.loc[date_gt_decision_mask & audit["reason"].ne(""), "reason"] += (
        "; valor de fecha > date_decision, apartado preventivamente"
    )
    audit["action"] = audit["reason"].eq("").map({True: "keep", False: "remove"})

    removed = audit.loc[audit["action"] == "remove"]
    kept = audit.loc[audit["action"] == "keep"]
    filtered = table.drop(columns=removed["Variable"].tolist())

    if not table["case_id"].is_unique or not filtered["case_id"].is_unique:
        raise ValueError("case_id debe ser único antes y después del filtro.")
    if len(filtered) != len(table) or not filtered["target"].equals(table["target"]):
        raise ValueError("El filtro debe conservar filas y target sin cambios.")

    remaining_date_features = [column for column in kept["Variable"] if column.endswith("D")]
    remaining_dates_after_decision = {
        column: int(
            (
                pd.to_datetime(filtered[column], errors="coerce").notna()
                & (pd.to_datetime(filtered[column], errors="coerce") > decision_date)
            ).sum()
        )
        for column in remaining_date_features
    }
    if any(remaining_dates_after_decision.values()):
        raise ValueError("Quedó una variable de fecha con valores posteriores a date_decision.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(OUTPUT, index=False, compression="snappy")
    audit.sort_values(["action", "Variable"]).to_csv(AUDIT, index=False)

    date_columns_after_decision = audit.loc[
        date_gt_decision_mask, ["Variable", "date_gt_decision_rows"]
    ]
    history_count = int(history_mask.sum())
    timing_unclear_count = int(timing_unclear_mask.sum())
    report = [
        "# Home Credit sin variables de historial crediticio previo",
        "",
        f"- Origen: `{INPUT.name}`; se conserva intacto.",
        f"- Casos: {len(filtered):,}; `case_id` únicos: {filtered['case_id'].nunique():,}.",
        f"- Predictores candidatos iniciales: {len(features)}.",
        "- Se filtraron variables, no filas: la tabla no identifica una cohorte verificada de solicitantes primerizos.",
        f"- Variables retiradas en total: {len(removed)}; las razones se superponen.",
        f"- Variables marcadas por historial previo: {history_count}.",
        f"- Variables excluidas por disponibilidad temporal no confirmada: {timing_unclear_count}.",
        f"- Variables de fecha con algún valor mayor que `date_decision`, apartadas preventivamente: {len(date_columns_after_decision)}.",
        f"- Predictores conservados: {len(kept)}; columnas finales, incluidos metadatos y target: {len(filtered.columns)}.",
        f"- Dataset generado: `{OUTPUT}`.",
        "- `target`, `case_id` y fechas/calendario se conservan como target o metadatos; no son predictores.",
        "",
        "## Revisión temporal",
        "",
        "Kaggle define `D` como una transformación de fecha; no significa que la variable use información futura. El anfitrión indica que la predicción ocurre en `date_decision` y que los datos están disponibles en ese momento ([convención de sufijos](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data), [aclaración del anfitrión](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/472866)).",
        "Algunas columnas contienen valores de fecha mayores que `date_decision`. Eso por sí solo no demuestra cuándo se conoció el dato: puede ser una fecha de evento futura conocida al solicitar el crédito. Se apartaron preventivamente para revisarlas por variable; no se etiquetan automáticamente como leakage.",
        "También se apartaron campos de desembolso, cuota del mes siguiente y primera fecha de vencimiento porque sus descripciones no confirman que estén disponibles antes de decidir.",
        "`target` se conserva como etiqueta: representa el resultado del crédito observado después de la decisión y nunca entra entre los predictores.",
        "",
        "## Variables con valores de fecha mayores que `date_decision`, apartadas para revisar",
        "",
        *[
            f"- `{row.Variable}`: {row.date_gt_decision_rows:,} filas con valor de fecha mayor."
            for row in date_columns_after_decision.itertuples()
        ],
        "",
        "## Otras variables retiradas",
        "",
        *[
            f"- `{row.Variable}` — {row.reason}. {row.Description}"
            for row in removed.loc[~date_gt_decision_mask].itertuples()
        ],
        "",
        "Se conservaron las tasas agregadas del punto de venta (`pos*`) porque describen el canal y no los pagos previos de la persona. Los conteos de coincidencias de teléfonos tampoco representan su historial crediticio individual.",
        "",
        f"Auditoría completa de las {len(features)} variables: `{AUDIT}`.",
        "No se entrenó un modelo. Las variables sensibles, incluida `sex_738L`, aún requieren revisión antes de definir los predictores finales.",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Dataset: {OUTPUT}")
    print(f"Filas/columnas: {filtered.shape}")
    print(f"Variables retiradas: {len(removed)}; conservadas: {len(kept)}")
    print(f"Variables de fecha apartadas preventivamente: {len(date_columns_after_decision)}")
    print(f"Reporte: {REPORT}")


if __name__ == "__main__":
    main()
