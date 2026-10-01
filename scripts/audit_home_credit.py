"""List and inspect the smallest useful slice of Home Credit 2024.

Run with the project environment after installing ``.[kaggle]``. The script
reads ``KAGGLE_API_TOKEN`` from the ignored local ``.env`` file, downloads only
the competition file list, feature dictionary, and training base table, then
uses pandas to summarize the target and identify candidate feature variables.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "home-credit-credit-risk-model-stability"
RAW_DIR = ROOT / "data" / "raw" / "home_credit_2024"
TEMP_DIR = RAW_DIR / "_download"
REPORT_DIR = ROOT / "reports" / "metrics"
FILES_TO_DOWNLOAD = [
    "feature_definitions.csv",
    "parquet_files/train/train_base.parquet",
    "parquet_files/train/train_static_0_0.parquet",
    "parquet_files/train/train_static_0_1.parquet",
    "parquet_files/train/train_person_1.parquet",
    "parquet_files/train/train_person_2.parquet",
]
FEATURE_WORDS = (
    r"income|salary|wage|employ|occupation|age|housing|credit|loan|debt|"
    r"payment|annuity|person|family|birth|education|work"
)


def load_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def kaggle_cli() -> str:
    local_cli = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    cli = str(local_cli) if local_cli.exists() else shutil.which("kaggle")
    if not cli:
        raise SystemExit("Falta Kaggle CLI. Instálalo con: pip install -e '.[kaggle]'")
    return cli


def run_kaggle(*args: str) -> str:
    command = [kaggle_cli(), *args]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout or "Falló un comando de Kaggle.")
    return result.stdout


def download_file(remote_file: str) -> Path:
    destination = RAW_DIR / Path(remote_file).name
    downloaded = TEMP_DIR / destination.name
    archive_path = TEMP_DIR / f"{destination.name}.zip"
    if destination.exists():
        return destination

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    if downloaded.exists():
        shutil.move(str(downloaded), str(destination))
        return destination

    if not archive_path.exists():
        run_kaggle(
            "competitions", "download", COMPETITION,
            "--file", remote_file, "--path", str(TEMP_DIR), "--quiet",
        )
    if not downloaded.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extract(destination.name, TEMP_DIR)
        archive_path.unlink()
    shutil.move(str(downloaded), str(destination))
    return destination


def list_competition_files() -> list[dict]:
    files = []
    page_token = None
    while True:
        command = ["competitions", "files", COMPETITION, "--page-size", "200", "--format", "json"]
        if page_token:
            command.extend(["--page-token", page_token])
        output = run_kaggle(*command)
        if output.lstrip().startswith("["):
            files.extend(json.loads(output))
            break
        header, _, body = output.partition("\n[")
        if not body:
            raise RuntimeError("No pude leer la lista JSON de archivos que devolvió Kaggle.")
        files.extend(json.loads("[" + body))
        page_token = header.removeprefix("Next Page Token = ").strip() or None
        if not header.startswith("Next Page Token = "):
            break
    return files


def check_competition_entry() -> bool:
    output = run_kaggle(
        "competitions", "list", "--search", COMPETITION, "--format", "json"
    )
    competitions = json.loads(output)
    competition = next(
        (item for item in competitions if COMPETITION in item.get("ref", "")),
        None,
    )
    if competition is None:
        raise RuntimeError("Kaggle no devolvió la competencia solicitada.")
    return bool(competition.get("userHasEntered", False))


def main() -> None:
    load_env()
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("Agrega KAGGLE_API_TOKEN al archivo .env antes de ejecutar el script.")

    if not check_competition_entry():
        raise SystemExit(
            "El token de .env corresponde a una cuenta que Kaggle no muestra como inscrita. "
            "Usa un token generado por la cuenta que se unió a la competencia."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Consultando archivos disponibles en Kaggle...")
    files = list_competition_files()
    inventory = pd.DataFrame(files)
    inventory.to_csv(REPORT_DIR / "home_credit_file_inventory.csv", index=False)
    print(f"Archivos del concurso: {len(inventory)}")
    inventory["format"] = inventory["name"].str.extract(r"\.(csv|parquet)$", expand=False)
    print("Tamaño total por formato (GB):")
    print((inventory.groupby("format")["size"].sum() / 1_000_000_000).round(2).to_string())
    key_files = inventory[inventory["name"].str.contains(
        r"feature_definitions|train_base|train_static_0|train_person_1|train_person_2",
        case=False,
        regex=True,
    )]
    print(key_files.to_string(index=False))

    try:
        for file in FILES_TO_DOWNLOAD:
            print(f"Descargando {file} ...")
            download_file(file)
    except RuntimeError as error:
        if "403 Client Error" in str(error):
            raise SystemExit(
                "Kaggle rechazó la descarga (403). Comprueba que el token de .env "
                "pertenezca a una cuenta inscrita y vuelve a ejecutar el script."
            ) from None
        raise

    definitions = pd.read_csv(RAW_DIR / "feature_definitions.csv")
    base = pd.read_parquet(RAW_DIR / "train_base.parquet")
    definition_text = definitions["Description"].fillna("").astype(str)
    candidates = definitions.loc[
        definition_text.str.contains(FEATURE_WORDS, case=False, regex=True, na=False)
    ]
    candidates.to_csv(REPORT_DIR / "home_credit_candidate_variables.csv", index=False)

    base_cases = base["case_id"].nunique()
    table_summary = []
    available_candidates = []
    static_columns = {}
    static_case_ids = {}
    for filename in [Path(file).name for file in FILES_TO_DOWNLOAD if file.endswith(".parquet")]:
        table = pd.read_parquet(RAW_DIR / filename)
        if filename.startswith("train_static_0_"):
            static_columns[filename] = list(table.columns)
            static_case_ids[filename] = table["case_id"].copy()
        feature_columns = [
            column for column in table.columns
            if column not in {"case_id", "num_group1", "num_group2", "target"}
        ]
        matched_definitions = definitions[definitions["Variable"].isin(feature_columns)]
        matched_text = matched_definitions["Description"].fillna("").astype(str)
        table_candidates = matched_definitions.loc[
            matched_text.str.contains(FEATURE_WORDS, case=False, regex=True, na=False)
        ].copy()
        table_label = "train_static_0 (particiones)" if filename.startswith("train_static_0_") else filename
        table_candidates.insert(0, "table", table_label)
        available_candidates.append(table_candidates)
        unique_cases = table["case_id"].nunique()
        covered_cases = table.loc[
            table["case_id"].isin(base["case_id"]), "case_id"
        ].nunique()
        applicant_cases = 0
        applicant_rows = 0
        if "num_group1" in table.columns:
            applicant_rows_table = table.loc[table["num_group1"].eq(0)]
            applicant_rows = len(applicant_rows_table)
            applicant_cases = applicant_rows_table["case_id"].nunique()
        missing_cells = sum(table[column].isna().sum() for column in feature_columns)
        total_feature_cells = len(table) * len(feature_columns)
        table_summary.append({
            "table": filename,
            "rows": len(table),
            "columns": len(table.columns),
            "feature_columns": len(feature_columns),
            "unique_case_id": unique_cases,
            "duplicate_case_id_rows": int(table.duplicated("case_id").sum()),
            "base_case_coverage_pct": round(100 * covered_cases / base_cases, 2),
            "rows_per_case": round(len(table) / max(unique_cases, 1), 2),
            "applicant_rows_num_group1_0": applicant_rows,
            "applicant_cases_num_group1_0": applicant_cases,
            "missing_feature_cells_pct": round(
                100 * missing_cells / total_feature_cells, 2
            ) if total_feature_cells else 0,
            "features_with_dictionary_definition": len(matched_definitions),
            "candidate_features_by_description": len(table_candidates),
        })

    table_summary = pd.DataFrame(table_summary)
    static_parts = ["train_static_0_0.parquet", "train_static_0_1.parquet"]
    static_case_audit = {
        "same_columns": static_columns[static_parts[0]] == static_columns[static_parts[1]],
        "overlap_case_ids": int(
            static_case_ids[static_parts[0]].isin(static_case_ids[static_parts[1]]).sum()
        ),
        "combined_unique_case_ids": int(
            pd.concat([static_case_ids[name] for name in static_parts]).nunique()
        ),
        "base_unique_case_ids": int(base_cases),
    }
    static_case_audit["base_coverage_pct"] = round(
        100 * static_case_audit["combined_unique_case_ids"] / base_cases, 2
    )
    (REPORT_DIR / "home_credit_static_partition_audit.json").write_text(
        json.dumps(static_case_audit, indent=2), encoding="utf-8"
    )
    table_summary.to_csv(REPORT_DIR / "home_credit_table_audit.csv", index=False)
    pd.concat(available_candidates, ignore_index=True).drop_duplicates(
        subset=["table", "Variable"]
    ).to_csv(
        REPORT_DIR / "home_credit_available_candidate_variables.csv", index=False
    )

    target_counts = base["target"].value_counts(dropna=False).sort_index()
    target_rate = base["target"].mean()
    report = [
        "# Home Credit - Credit Risk Model Stability (Kaggle 2024): auditoría inicial",
        "",
        f"- Filas en train_base: {len(base):,}",
        "- Unidad del target: caso de crédito (`case_id`), no necesariamente persona única.",
        f"- case_id únicos: {base['case_id'].nunique():,}",
        f"- Columnas de train_base: {len(base.columns)} ({', '.join(base.columns)})",
        f"- Fechas de decisión: {base['date_decision'].min()} a {base['date_decision'].max()}",
        "- Aunque la competencia es de 2024, los casos de entrenamiento observados son de 2019–2020.",
        f"- Semanas: {base['WEEK_NUM'].min()} a {base['WEEK_NUM'].max()}",
        f"- Filas del diccionario: {len(definitions):,}",
        f"- Variables candidatas por descripción: {len(candidates):,}",
        f"- Tasa media de target: {target_rate:.4f}",
        "",
        "## Tablas Parquet seleccionadas",
        "",
        table_summary.to_string(index=False),
        "",
        f"Validación de static_0: mismo esquema = {static_case_audit['same_columns']}; case_id repetidos entre partes = {static_case_audit['overlap_case_ids']:,}; casos únicos al concatenar = {static_case_audit['combined_unique_case_ids']:,} ({static_case_audit['base_coverage_pct']:.2f}% de base). Se deben concatenar las dos partes por filas antes de unirlas a train_base.",
        "En person_1, num_group1 = 0 identifica al solicitante: la tabla confirma una fila del solicitante por cada case_id. person_2 contiene registros anidados de personas y relaciones, por lo que queda como extensión posterior. Las tablas externas de bureau también quedan para una comparación posterior.",
        "",
        "## Conteo del target",
        "",
        target_counts.to_string(),
        "",
        "Esta auditoría inspecciona tablas y claves con pandas; todavía no entrena modelos ni descarga las tablas externas de bureau.",
    ]
    report_path = REPORT_DIR / "home_credit_initial_audit.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nResumen train_base:")
    print("Filas y columnas:", base.shape)
    print("Columnas:", base.columns.tolist())
    print("case_id únicos:", base["case_id"].nunique())
    print("Tasa target=1:", round(float(target_rate), 4))
    print("Diccionario:", definitions.shape)
    print("Variables candidatas guardadas:", len(candidates))
    print("Reporte:", report_path)


if __name__ == "__main__":
    main()
