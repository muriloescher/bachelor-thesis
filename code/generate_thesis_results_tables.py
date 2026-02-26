"""Generate LaTeX result tables for the thesis from a curated results CSV.

This script is intentionally minimal: it only generates the exact table
fragments used in the thesis (one file per table):

- thesis/tables/results_inflection.tex
- thesis/tables/results_analysis_um_lemma.tex
- thesis/tables/results_analysis_um_msd.tex
- thesis/tables/results_analysis_ud_lemma.tex
- thesis/tables/results_analysis_ud_msd.tex

By default, this script uses the hand-curated snapshot at:
    code/results/end_results.csv

If you instead want to derive the latest rows automatically from the full log
CSV, you can pass --csv code/results/all_results.csv.

When the input CSV contains multiple rows for the same
(model_type, model_name, language, direction, split), the script selects the
most recent by timestamp.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "code" / "results" / "end_results.csv"
FALLBACK_CSV_PATH = PROJECT_ROOT / "code" / "results" / "all_results.csv"
THESIS_TABLES_DIR = PROJECT_ROOT / "thesis" / "tables"


LANG_ORDER = ["amh", "azg", "dsb", "eng", "grc", "ita", "kat", "por"]

# UD-based analysis: lemma metrics are available for more languages than MSD metrics
UD_LANGS_LEMMA = {"amh", "eng", "grc", "ita", "kat", "por"}
UD_LANGS_MSD = {"eng", "grc", "ita", "por"}


def _parse_float(value: str) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value == "--":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _fmt_num(value: float | None, *, ndigits: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "--"
    return f"{value:.{ndigits}f}"


def _fmt_pct(value: float | None, *, ndigits: int = 1) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "--"
    return f"{(value * 100):.{ndigits}f}"


def _latex_bold(text: str, bold: bool) -> str:
    return f"\\textbf{{{text}}}" if bold else text


def _parse_timestamp(value: str) -> datetime:
    # CSV uses ISO 8601 without timezone, e.g. 2026-02-22T17:34:14.688709
    return datetime.fromisoformat(value.strip())


@dataclass(frozen=True)
class RowKey:
    model_type: str
    model_name: str
    language: str
    direction: str
    split: str


@dataclass
class ResultRow:
    model_type: str
    model_name: str
    language: str
    direction: str
    split: str
    timestamp: datetime

    lemma_accuracy: float | None
    lemma_mean_levenshtein: float | None
    msd_accuracy: float | None
    msd_f1: float | None

    use_context: str


def load_latest_rows(csv_path: Path) -> dict[RowKey, ResultRow]:
    latest: dict[RowKey, ResultRow] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            model_type = (raw.get("model_type") or "").strip()
            model_name = (raw.get("model_name") or "").strip()
            language = (raw.get("language") or "").strip()
            direction = (raw.get("direction") or "").strip()
            split = (raw.get("split") or "").strip()
            timestamp = _parse_timestamp(raw["timestamp"])

            key = RowKey(
                model_type=model_type,
                model_name=model_name,
                language=language,
                direction=direction,
                split=split,
            )

            row = ResultRow(
                model_type=model_type,
                model_name=model_name,
                language=language,
                direction=direction,
                split=split,
                timestamp=timestamp,
                lemma_accuracy=_parse_float(raw.get("lemma_accuracy") or ""),
                lemma_mean_levenshtein=_parse_float(raw.get("lemma_mean_levenshtein") or ""),
                msd_accuracy=_parse_float(raw.get("msd_accuracy") or ""),
                msd_f1=_parse_float(raw.get("msd_f1") or ""),
                use_context=(raw.get("use_context") or "").strip(),
            )

            prev = latest.get(key)
            if prev is None or row.timestamp > prev.timestamp:
                latest[key] = row

    return latest


def _resolve_csv_path(arg: str | None) -> Path:
    if arg:
        return (PROJECT_ROOT / arg).resolve() if not Path(arg).is_absolute() else Path(arg)
    if DEFAULT_CSV_PATH.exists():
        return DEFAULT_CSV_PATH
    return FALLBACK_CSV_PATH


def _inflection_model_label(row: ResultRow) -> str | None:
    if row.model_type == "nonneural":
        return "Non-neural baseline"
    if row.model_type == "neural":
        return "Neural baseline"
    if row.model_type == "byt5" and row.direction == "forward":
        if "/byt5-forward-" in row.model_name:
            return "ByT5"
        if "/byt5-bidir-" in row.model_name:
            return "ByT5 (bidirectional)"
    return None


def _analysis_model_label(row: ResultRow) -> str | None:
    if row.model_type == "byt5" and row.direction == "inverse":
        if "/byt5-inverse-" in row.model_name:
            return "ByT5 (inverse, UniMorph)"
        if "/byt5-bidir-" in row.model_name and "-context" not in row.model_name:
            return "ByT5 (bidirectional, inverse, UniMorph)"
        if "-context" in row.model_name:
            return "ByT5 (bidirectional + context, UD)"

    if row.model_type == "llm":
        if row.model_name == "meta-llama/llama-3.1-8b-instruct":
            return "LLM (Llama-3.1-8B, UD)"
        if row.model_name == "qwen/qwen3-8b":
            return "LLM (Qwen3-8B, UD)"
        safe = row.model_name.replace("_", "\\_")
        return f"LLM ({safe}, UD)"

    return None


def _analysis_model_label_short(full_label: str) -> str:
    short = {
        "ByT5 (inverse, UniMorph)": "ByT5 inv",
        "ByT5 (bidirectional, inverse, UniMorph)": "ByT5 bi-inv",
        "ByT5 (bidirectional + context, UD)": "ByT5 ctx",
        "LLM (Llama-3.1-8B, UD)": "LLM Llama",
        "LLM (Qwen3-8B, UD)": "LLM Qwen",
    }
    return short.get(full_label, full_label)


def _inflection_model_label_short(full_label: str) -> str:
    short = {
        "Non-neural baseline": "Non-neural",
        "Neural baseline": "Neural",
        "ByT5": "ByT5",
        "ByT5 (bidirectional)": "ByT5 bi",
    }
    return short.get(full_label, full_label)


def _collect_inflection(rows: Iterable[ResultRow]) -> tuple[dict[str, dict[str, ResultRow]], list[str]]:
    data: dict[str, dict[str, ResultRow]] = {lang: {} for lang in LANG_ORDER}
    for row in rows:
        if row.split != "test" or row.direction != "forward":
            continue
        if row.language not in data:
            continue
        label = _inflection_model_label(row)
        if label is None:
            continue
        data[row.language][label] = row

    models = ["Non-neural baseline", "Neural baseline", "ByT5", "ByT5 (bidirectional)"]
    return data, models


def _collect_analysis(rows: Iterable[ResultRow]) -> dict[str, dict[str, ResultRow]]:
    data: dict[str, dict[str, ResultRow]] = {lang: {} for lang in LANG_ORDER}
    for row in rows:
        if row.split != "test":
            continue
        if row.language not in data:
            continue
        label = _analysis_model_label(row)
        if label is None:
            continue
        data[row.language][label] = row
    return data


def _write_inflection_table(rows: Iterable[ResultRow], out_path: Path) -> None:
    data, models = _collect_inflection(rows)
    headers = [l.lower() for l in LANG_ORDER]

    def _cell_acc(model: str, lang: str) -> float | None:
        row = data[lang].get(model)
        return None if row is None else row.lemma_accuracy

    def _cell_lev(model: str, lang: str) -> float | None:
        row = data[lang].get(model)
        return None if row is None else row.lemma_mean_levenshtein

    best_acc: dict[str, float | None] = {}
    best_lev: dict[str, float | None] = {}
    for lang in LANG_ORDER:
        acc_vals = [v for m in models if (v := _cell_acc(m, lang)) is not None]
        lev_vals = [v for m in models if (v := _cell_lev(m, lang)) is not None]
        best_acc[lang] = max(acc_vals) if acc_vals else None
        best_lev[lang] = min(lev_vals) if lev_vals else None

    ncols = 1 + len(LANG_ORDER)

    lines: list[str] = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\normalsize")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.10}")
    lines.append("\\begin{tabular}{l" + "r" * len(LANG_ORDER) + "}")
    lines.append("\\toprule")
    lines.append("Model & " + " & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Accuracy (\\%, $\\uparrow$)}}}} \\\\")
    for model in models:
        row_cells: list[str] = []
        for lang in LANG_ORDER:
            value = _cell_acc(model, lang)
            s = _fmt_pct(value)
            s = _latex_bold(s, best_acc[lang] is not None and value == best_acc[lang])
            row_cells.append(s)
        lines.append(f"{_inflection_model_label_short(model)} & " + " & ".join(row_cells) + " \\\\")

    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{Mean Levenshtein distance ($\\downarrow$)}}}} \\\\")
    for model in models:
        row_cells = []
        for lang in LANG_ORDER:
            value = _cell_lev(model, lang)
            s = _fmt_num(value)
            s = _latex_bold(s, best_lev[lang] is not None and value == best_lev[lang])
            row_cells.append(s)
        lines.append(f"{_inflection_model_label_short(model)} & " + " & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Inflection results (test set): accuracy and mean Levenshtein distance.}")
    lines.append("\\label{tab:results-inflection-matrix}")
    lines.append("\\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_analysis_tables(
    *,
    rows: Iterable[ResultRow],
    out_path_um_lemma: Path,
    out_path_um_msd: Path,
    out_path_ud_lemma: Path,
    out_path_ud_msd: Path,
) -> None:
    data = _collect_analysis(rows)

    um_models = [
        "ByT5 (inverse, UniMorph)",
        "ByT5 (bidirectional, inverse, UniMorph)",
    ]
    ud_models = [
        "ByT5 (bidirectional + context, UD)",
        "LLM (Llama-3.1-8B, UD)",
        "LLM (Qwen3-8B, UD)",
    ]

    um_langs = list(LANG_ORDER)
    ud_lemma_langs = [l for l in LANG_ORDER if l in UD_LANGS_LEMMA]
    ud_msd_langs = [l for l in LANG_ORDER if l in UD_LANGS_MSD]

    def _metric(lang: str, model: str, attr: str) -> float | None:
        row = data[lang].get(model)
        return None if row is None else getattr(row, attr)

    def _best_per_lang(
        attr: str,
        *,
        higher_is_better: bool,
        lang_order: list[str],
        model_list: list[str],
    ) -> dict[str, float | None]:
        best: dict[str, float | None] = {}
        for lang in lang_order:
            values = [v for m in model_list if (v := _metric(lang, m, attr)) is not None]
            if not values:
                best[lang] = None
            else:
                best[lang] = max(values) if higher_is_better else min(values)
        return best

    def _write_stacked(
        *,
        lang_order: list[str],
        model_list: list[str],
        title: str,
        label: str,
        metric_a_attr: str,
        metric_a_title: str,
        metric_a_higher: bool,
        metric_b_attr: str,
        metric_b_title: str,
        metric_b_higher: bool,
    ) -> list[str]:
        headers = [l.lower() for l in lang_order]
        ncols = 1 + len(lang_order)

        best_a = _best_per_lang(metric_a_attr, higher_is_better=metric_a_higher, lang_order=lang_order, model_list=model_list)
        best_b = _best_per_lang(metric_b_attr, higher_is_better=metric_b_higher, lang_order=lang_order, model_list=model_list)

        out: list[str] = []
        out.append("\\begin{table}[h]")
        out.append("\\centering")
        out.append("\\small")
        out.append("\\setlength{\\tabcolsep}{5pt}")
        out.append("\\renewcommand{\\arraystretch}{1.10}")
        out.append("\\begin{tabular}{l" + "r" * len(lang_order) + "}")
        out.append("\\toprule")
        out.append("Model & " + " & ".join(headers) + " \\\\")
        out.append("\\midrule")

        out.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{metric_a_title}}}}} \\\\")
        for model in model_list:
            row_cells: list[str] = []
            for lang in lang_order:
                value = _metric(lang, model, metric_a_attr)
                s = _fmt_pct(value) if metric_a_attr in {"lemma_accuracy", "msd_accuracy", "msd_f1"} else _fmt_num(value)
                s = _latex_bold(s, best_a[lang] is not None and value == best_a[lang])
                row_cells.append(s)
            out.append(f"{_analysis_model_label_short(model)} & " + " & ".join(row_cells) + " \\\\")

        out.append("\\midrule")
        out.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{metric_b_title}}}}} \\\\")
        for model in model_list:
            row_cells = []
            for lang in lang_order:
                value = _metric(lang, model, metric_b_attr)
                s = _fmt_pct(value) if metric_b_attr in {"lemma_accuracy", "msd_accuracy", "msd_f1"} else _fmt_num(value)
                s = _latex_bold(s, best_b[lang] is not None and value == best_b[lang])
                row_cells.append(s)
            out.append(f"{_analysis_model_label_short(model)} & " + " & ".join(row_cells) + " \\\\")

        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append(f"\\caption{{{title}}}")
        out.append(f"\\label{{{label}}}")
        out.append("\\end{table}")
        out.append("")
        return out

    um_lemma_lines = _write_stacked(
        lang_order=um_langs,
        model_list=um_models,
        title="Morphological analysis (UniMorph, test set): lemma metrics.",
        label="tab:results-analysis-um-lemma-matrix",
        metric_a_attr="lemma_accuracy",
        metric_a_title="Lemma accuracy (\\%, $\\uparrow$)",
        metric_a_higher=True,
        metric_b_attr="lemma_mean_levenshtein",
        metric_b_title="Lemma mean Levenshtein ($\\downarrow$)",
        metric_b_higher=False,
    )
    um_msd_lines = _write_stacked(
        lang_order=um_langs,
        model_list=um_models,
        title="Morphological analysis (UniMorph, test set): \\acs{MSD} metrics.",
        label="tab:results-analysis-um-msd-matrix",
        metric_a_attr="msd_accuracy",
        metric_a_title="\\acs{MSD} accuracy (\\%, $\\uparrow$)",
        metric_a_higher=True,
        metric_b_attr="msd_f1",
        metric_b_title="\\acs{MSD} F1 (\\%, $\\uparrow$)",
        metric_b_higher=True,
    )
    ud_lemma_lines = _write_stacked(
        lang_order=ud_lemma_langs,
        model_list=ud_models,
        title="Morphological analysis (UD, test set): lemma metrics.",
        label="tab:results-analysis-ud-lemma-matrix",
        metric_a_attr="lemma_accuracy",
        metric_a_title="Lemma accuracy (\\%, $\\uparrow$)",
        metric_a_higher=True,
        metric_b_attr="lemma_mean_levenshtein",
        metric_b_title="Lemma mean Levenshtein ($\\downarrow$)",
        metric_b_higher=False,
    )
    ud_msd_lines = _write_stacked(
        lang_order=ud_msd_langs,
        model_list=ud_models,
        title="Morphological analysis (UD, test set): \\acs{MSD} metrics.",
        label="tab:results-analysis-ud-msd-matrix",
        metric_a_attr="msd_accuracy",
        metric_a_title="\\acs{MSD} accuracy (\\%, $\\uparrow$)",
        metric_a_higher=True,
        metric_b_attr="msd_f1",
        metric_b_title="\\acs{MSD} F1 (\\%, $\\uparrow$)",
        metric_b_higher=True,
    )

    out_path_um_lemma.parent.mkdir(parents=True, exist_ok=True)
    out_path_um_lemma.write_text("\n".join(um_lemma_lines), encoding="utf-8")
    out_path_um_msd.parent.mkdir(parents=True, exist_ok=True)
    out_path_um_msd.write_text("\n".join(um_msd_lines), encoding="utf-8")
    out_path_ud_lemma.parent.mkdir(parents=True, exist_ok=True)
    out_path_ud_lemma.write_text("\n".join(ud_lemma_lines), encoding="utf-8")
    out_path_ud_msd.parent.mkdir(parents=True, exist_ok=True)
    out_path_ud_msd.write_text("\n".join(ud_msd_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis LaTeX result tables from a results CSV")
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "CSV path to read results from. Defaults to code/results/end_results.csv if it exists, "
            "otherwise falls back to code/results/all_results.csv."
        ),
    )
    args = parser.parse_args()

    csv_path = _resolve_csv_path(args.csv)
    latest = load_latest_rows(csv_path)
    rows = list(latest.values())

    inflection = THESIS_TABLES_DIR / "results_inflection.tex"
    analysis_um_lemma = THESIS_TABLES_DIR / "results_analysis_um_lemma.tex"
    analysis_um_msd = THESIS_TABLES_DIR / "results_analysis_um_msd.tex"
    analysis_ud_lemma = THESIS_TABLES_DIR / "results_analysis_ud_lemma.tex"
    analysis_ud_msd = THESIS_TABLES_DIR / "results_analysis_ud_msd.tex"

    _write_inflection_table(rows, inflection)
    _write_analysis_tables(
        rows=rows,
        out_path_um_lemma=analysis_um_lemma,
        out_path_um_msd=analysis_um_msd,
        out_path_ud_lemma=analysis_ud_lemma,
        out_path_ud_msd=analysis_ud_msd,
    )

    print("Wrote:")
    print(f"- {inflection}")
    print(f"- {analysis_um_lemma}")
    print(f"- {analysis_um_msd}")
    print(f"- {analysis_ud_lemma}")
    print(f"- {analysis_ud_msd}")


if __name__ == "__main__":
    main()
