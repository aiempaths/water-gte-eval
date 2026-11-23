"""Schema/data-quality validator.

Checks: timestamp monotonicity, duplicate timestamps, gap report (against an
expected step, inferred when not given), flatline report per tag (runs of
identical consecutive values), missing values per tag, and unit documentation
from tags.yaml (unit-less tags are reported, or errors when require_units).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .config import IngestConfigError


def load_tags_yaml(path: str | Path | None) -> dict:
    if not path:
        return {"tags": {}, "settings": {}}
    p = Path(path)
    if not p.is_file():
        raise IngestConfigError(
            f"tags_yaml not found: {p}. Create it (see configs/tags.yaml template): "
            "map each detected tag to {'unit': ...}."
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise IngestConfigError(f"tags_yaml {p} must be a mapping with a 'tags' key")
    tags = raw.get("tags") or {}
    if not isinstance(tags, dict):
        raise IngestConfigError(f"tags_yaml {p}: 'tags' must be a mapping of tag -> {{unit: ...}}")
    return {"tags": tags, "settings": raw.get("settings") or {}}


def _flatline(s: pd.Series, min_run: int, max_examples: int) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return {"longest_run": 0, "n_runs_ge_min": 0, "examples": []}
    run_id = (s != s.shift()).cumsum()
    sizes = s.groupby(run_id).size()
    long_runs = sizes[sizes >= min_run]
    examples = []
    for rid in long_runs.index[:max_examples]:
        pos = run_id[run_id == rid].index
        examples.append(
            {"start": pos[0].isoformat(), "length": int(long_runs[rid]), "value": float(s.loc[pos[0]])}
        )
    return {
        "longest_run": int(sizes.max()),
        "n_runs_ge_min": int(len(long_runs)),
        "examples": examples,
    }


def validate_quality(
    df: pd.DataFrame,
    tag_columns: list[str],
    tags_cfg: dict | None = None,
    expected_step: pd.Timedelta | None = None,
    flatline_min_run: int = 10,
    require_units: bool = False,
    label_columns: list[str] | None = None,
    max_examples: int = 20,
) -> dict:
    tags_cfg = tags_cfg or {"tags": {}, "settings": {}}
    idx = df.index
    report: dict = {
        "n_rows": int(len(df)),
        "time_min": idx.min().isoformat() if len(idx) else None,
        "time_max": idx.max().isoformat() if len(idx) else None,
        "duration": str(idx.max() - idx.min()) if len(idx) else None,
        "monotonic": bool(idx.is_monotonic_increasing) if len(idx) else True,
        "n_duplicate_rows": 0,
        "duplicate_examples": [],
        "expected_step": None,
        "n_gaps": 0,
        "missing_periods": 0,
        "largest_gap": None,
        "gap_examples": [],
        "flatline_min_run": flatline_min_run,
        "flatline": {},
        "missing_values": {},
        "unitless_tags": [],
        "range_violations": {},
        "errors": [],
        "warnings": [],
    }
    label_columns = label_columns or []

    if len(idx) == 0:
        report["errors"].append("empty dataframe")
        report["ok"] = False
        return report

    if not idx.is_monotonic_increasing:
        report["errors"].append("timestamps are not monotonically increasing")
    dup_mask = idx.duplicated(keep=False)
    report["n_duplicate_rows"] = int(dup_mask.sum())
    if report["n_duplicate_rows"]:
        dup_ts = idx[dup_mask].unique()[:max_examples]
        report["duplicate_examples"] = [str(t) for t in dup_ts]
        report["errors"].append(f"{report['n_duplicate_rows']} rows share duplicate timestamps")
    report["strictly_increasing"] = bool(idx.is_monotonic_increasing and not dup_mask.any())

    if len(idx) >= 2:
        diffs = pd.Series(idx).diff().dropna()
        if expected_step is None:
            expected_step = diffs.median()
        report["expected_step"] = str(expected_step)
        gap_mask = diffs > expected_step * 1.0001
        gaps = diffs[gap_mask]
        report["n_gaps"] = int(len(gaps))
        if len(gaps):
            report["missing_periods"] = int((gaps / expected_step).round().sum() - len(gaps))
            report["largest_gap"] = str(gaps.max())
            positions = gaps.index.to_numpy()
            for pos in positions[:max_examples]:
                report["gap_examples"].append(
                    {"after": idx[pos - 1].isoformat(), "before": idx[pos].isoformat(), "size": str(gaps.loc[pos])}
                )
        if report["n_gaps"]:
            report["warnings"].append(
                f"{report['n_gaps']} gap(s) larger than expected step {expected_step} "
                f"({report['missing_periods']} missing periods, largest {report['largest_gap']})"
            )

    for c in tag_columns:
        report["flatline"][c] = _flatline(df[c], flatline_min_run, max_examples)
        report["missing_values"][c] = int(df[c].isna().sum())
        fl = report["flatline"][c]
        if fl["n_runs_ge_min"]:
            report["warnings"].append(
                f"tag {c!r}: {fl['n_runs_ge_min']} flatline run(s) >= {flatline_min_run} "
                f"(longest {fl['longest_run']})"
            )
        if report["missing_values"][c]:
            report["warnings"].append(f"tag {c!r}: {report['missing_values'][c]} missing values")

    documented = tags_cfg.get("tags", {})
    unitless = []
    for c in tag_columns:
        entry = documented.get(c)
        if not isinstance(entry, dict) or not entry.get("unit"):
            unitless.append(c)
        else:
            lo, hi = entry.get("min"), entry.get("max")
            if lo is not None or hi is not None:
                s = pd.to_numeric(df[c], errors="coerce")
                mask = pd.Series(False, index=s.index)
                if lo is not None:
                    mask = mask | (s < lo)
                if hi is not None:
                    mask = mask | (s > hi)
                n_viol = int(mask.sum())
                if n_viol:
                    report["range_violations"][c] = n_viol
    report["unitless_tags"] = unitless
    if unitless:
        msg = f"{len(unitless)} tag(s) without a documented unit in tags.yaml: {unitless}"
        if require_units:
            report["errors"].append(msg)
        else:
            report["warnings"].append(msg)
    if report["range_violations"]:
        report["warnings"].append(f"range violations vs tags.yaml: {report['range_violations']}")

    report["ok"] = not report["errors"]
    return report


def format_quality_report(name: str, report: dict) -> str:
    lines = [
        f"data-quality report: {name}",
        f"  rows: {report['n_rows']}   range: {report['time_min']} .. {report['time_max']}  ({report['duration']})",
        f"  monotonic time: {report['monotonic']} (strictly increasing: {report.get('strictly_increasing')})",
        f"  duplicate timestamps: {report['n_duplicate_rows']} rows"
        + (f" e.g. {report['duplicate_examples'][:5]}" if report["duplicate_examples"] else ""),
        f"  gaps (expected step {report['expected_step']}): {report['n_gaps']} gap(s), "
        f"{report['missing_periods']} missing period(s), largest {report['largest_gap']}",
    ]
    fl = report["flatline"]
    active = {k: v for k, v in fl.items() if v["n_runs_ge_min"]}
    lines.append(
        f"  flatlines (>= {report['flatline_min_run']} identical consecutive values): "
        + (", ".join(f"{k}: {v['n_runs_ge_min']} run(s), longest {v['longest_run']}" for k, v in active.items())
           if active else "none")
    )
    mv = {k: v for k, v in report["missing_values"].items() if v}
    lines.append(f"  missing values per tag: {mv if mv else 'none'}")
    lines.append(
        f"  unit-less tags: {report['unitless_tags'] if report['unitless_tags'] else 'none (all documented in tags.yaml)'}"
    )
    if report["range_violations"]:
        lines.append(f"  range violations: {report['range_violations']}")
    lines.append(f"  errors: {report['errors'] if report['errors'] else 'none'}")
    lines.append(f"  warnings: {len(report['warnings'])}")
    for w in report["warnings"]:
        lines.append(f"    - {w}")
    lines.append(f"  ok: {report['ok']}")
    return "\n".join(lines)
