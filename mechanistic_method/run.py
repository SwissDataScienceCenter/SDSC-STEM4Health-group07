#!/usr/bin/env python3
"""Generate the challenge submissions, or refit the original model recipe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
DATA_FILES = ("bioreaction_remaining.csv", "bioreaction_template.csv")
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
SUBMISSIONS = (
    "submission_best.csv",
    "submission_nosearch.csv",
    "submission_mechanistic_blend.csv",
)
ALTERNATIVES = (
    "b10_nosearch_bayesresid_smoothcells",
    "b11_mechanistic_only_bayesresid",
)
TRAINING_STEPS = (
    ("Fit biological models", ["variants.py", "m01_base", "m02_cov7", "m11_consHill", "m14_re6", "m21_re8"]),
    ("Fit statistical models", ["statm.py", "s25_search_all_plus_base25", "s26_evidence_all_plus_base25"]),
    ("Fit the hybrid corrections", ["hybrid3.py", "m11_consHill"]),
    ("Fit the particle model used by comparison blends", ["p.py", "m21_re8", "kernel", "1.0", "cal", "flags", "_b25"]),
    ("Fit the final particle model with 1000 scenarios", ["-c", "import p2; p2.particle2('m21_re8', N=1000, resid='evidence', tag='_all_n1000', nosub=False)"]),
    ("Combine the model predictions", ["final.py"]),
    ("Write submission files and experiment report", ["report.py"]),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_records(path: Path) -> dict:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["name"]] = record
    if not records:
        raise ValueError(f"No recorded model results in {path}")
    return records


def check_dependencies() -> dict:
    if sys.version_info < (3, 11):
        raise RuntimeError("Use Python 3.11 or newer. The supplied package versions were tested with Python 3.12.")
    versions = {"python": sys.version.split()[0]}
    for distribution, module in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("scikit-learn", "sklearn")):
        try:
            importlib.import_module(module)
            versions[distribution] = importlib.metadata.version(distribution)
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            raise RuntimeError(
                "A required Python package is missing or cannot be imported. "
                "Run: python -m pip install -r requirements.txt\n"
                f"Details for {distribution}: {exc}"
            ) from exc
    return versions


def prepare_run(retrain: bool) -> Path:
    for name in DATA_FILES:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Missing {name}. Extract the whole ZIP and keep its folder structure.")

    if not retrain:
        manifest = json.loads((ROOT / "cache_manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(
                    f"{name} has changed. The saved predictions belong to the original bundled data. "
                    "Restore the original file, or use --retrain with data in the required challenge format."
                )

    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    mode = "retrain" if retrain else "regenerate"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    work = Path(tempfile.mkdtemp(prefix=f"{mode}_{stamp}_", dir=runs))
    lab = work / "lab"
    (lab / "ode").mkdir(parents=True)
    (lab / "oof").mkdir()
    for name in DATA_FILES:
        shutil.copyfile(ROOT / name, work / name)
    for source in (ROOT / "lab").glob("*.py"):
        shutil.copyfile(source, lab / source.name)

    if not retrain:
        # Only export mode imports the original recorded results and predictions.
        # Training starts with empty parameter, prediction, and result directories.
        records = experiment_records(ROOT / "lab" / "results.jsonl")
        best = min(records.values(), key=lambda record: record["mean"])["name"]
        for name in {best, *ALTERNATIVES}:
            source = ROOT / "lab" / "oof" / f"{name}.npz"
            shutil.copyfile(source, lab / "oof" / source.name)
        for name in ("results.jsonl", "results_partition2.jsonl"):
            source = ROOT / "lab" / name
            if source.exists():
                shutil.copyfile(source, lab / name)
    return work


def run_step(arguments: list[str], work: Path, env: dict, log) -> None:
    command = [sys.executable, "-u", *arguments]
    with subprocess.Popen(
        command,
        cwd=work / "lab",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def validate_submissions(work: Path) -> dict:
    import numpy as np
    import pandas as pd

    template = pd.read_csv(work / DATA_FILES[1], dtype={"batch_id": str})
    data = pd.read_csv(work / DATA_FILES[0], dtype={"batch_id": str})
    flags = data["to_predict"].astype(str).str.strip().str.lower()
    if not flags.isin(["true", "false", "yes", "no", "1", "0"]).all():
        raise ValueError("Unrecognized to_predict values in the data.")
    requested = data.loc[flags.isin(["true", "yes", "1"]), ["batch_id", "elapsed_time"]].copy()
    requested = requested.sort_values(["batch_id", "elapsed_time"], kind="stable")
    requested["sample"] = requested.groupby("batch_id").cumcount()
    requested_template = template[["batch_id", "elapsed_time"]].copy()
    requested_template["sample"] = requested_template.groupby("batch_id").cumcount()
    if len(requested) != len(template):
        raise ValueError("The template and to_predict rows have different lengths.")
    alignment = requested_template.merge(
        requested, on=["batch_id", "sample"], how="outer", sort=False,
        validate="one_to_one", indicator=True, suffixes=("_template", "_data")
    )
    if not (alignment["_merge"] == "both").all() or not np.allclose(
        alignment["elapsed_time_template"], alignment["elapsed_time_data"], atol=1e-6, rtol=0
    ):
        raise ValueError("The template does not match the requested rows in the data.")

    result = {"status": "passed", "files": {}}
    expected_columns = ["batch_id", "elapsed_time", *TARGETS]
    for name in SUBMISSIONS:
        output = pd.read_csv(work / name, dtype={"batch_id": str})
        if list(output.columns) != expected_columns or list(output.columns) != list(template.columns):
            raise ValueError(f"Wrong columns in {name}")
        if len(output) != len(template) or output["batch_id"].tolist() != template["batch_id"].tolist():
            raise ValueError(f"Wrong number or order of rows in {name}")
        if not np.allclose(output["elapsed_time"], template["elapsed_time"], atol=1e-12, rtol=0):
            raise ValueError(f"Sampling times do not match the template in {name}")
        if output.duplicated(["batch_id", "elapsed_time"]).any():
            raise ValueError(f"Duplicate requested rows in {name}")
        values = output[TARGETS].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not (values > 0).all() or output.isna().any().any():
            raise ValueError(f"Missing, infinite, or nonpositive predictions in {name}")
        result["files"][name] = {
            "rows": len(output),
            "batches": int(output["batch_id"].nunique()),
            "predicted_columns": len(TARGETS),
            "predicted_values": int(values.size),
            "missing_values": 0,
            "all_predictions_finite_and_positive": True,
            "template_order_and_times_match": True,
            "matches_to_predict_rows": True,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true", help="Fit the original model recipe in a fresh folder. This can take a long time.")
    parser.add_argument("--threads", type=int, default=1, help="Numerical library threads per process (default: 1).")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    work = None
    try:
        versions = check_dependencies()
        work = prepare_run(args.retrain)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
            env[name] = str(args.threads)
        mode = "retrain" if args.retrain else "regenerate"
        steps = TRAINING_STEPS if args.retrain else (("Export saved predictions", ["report.py"]),)
        with (work / "run_output.txt").open("w", encoding="utf-8") as log:
            def announce(message: str) -> None:
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()

            announce(f"Mode: {mode}")
            announce(f"Working folder: {work}")
            announce("Fitting the original recipe from the bundled data." if args.retrain else "Using the bundled saved predictions and recorded scores.")
            for index, (title, arguments) in enumerate(steps, start=1):
                announce(f"\nStep {index} of {len(steps)}: {title}")
                run_step(arguments, work, env, log)
            validation = validate_submissions(work)
            announce("\nAll three CSV files passed the row, column, time, and prediction checks.")

        records = experiment_records(work / "lab" / "results.jsonl")
        best = min(records.values(), key=lambda record: record["mean"])
        info = {
            "method": "Hybrid mechanistic and machine learning ensemble",
            "mode": mode,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "working_folder": str(work.relative_to(ROOT)),
            "versions": versions,
            "input_sha256": {name: sha256(work / name) for name in DATA_FILES},
            "recorded_experiment_count": len(records),
            "selected_model": best["name"],
            "cv_mean_mape_percent": best["mean"],
            "score_source": "computed in this training run" if args.retrain else "original saved experiment results",
            "validation_note": "The original ODE folds start from a fit to all labelled batches. Reported CV may be optimistic. This launcher preserves that modelling procedure.",
            "submission_performed": False,
        }
        (work / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        (work / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        output = ROOT / "outputs"
        output.mkdir(exist_ok=True)
        for name in (*SUBMISSIONS, "experiments_log.md", "run_output.txt", "validation.json", "run_info.json"):
            shutil.copyfile(work / name, output / name)
        print(f"\nResults are ready in: {output}")
        print(f"Main submission: {output / SUBMISSIONS[0]}")
        print("No leaderboard submission was made.")
        return 0
    except KeyboardInterrupt:
        print("\nRun interrupted. This run did not finish publishing a confirmed result set.", file=sys.stderr)
        if work:
            print(f"Partial work: {work}", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nRun stopped: {exc}", file=sys.stderr)
        print("This run did not finish publishing a confirmed result set. Check the log before using files in outputs.", file=sys.stderr)
        if work:
            print(f"Check the log and partial work in: {work}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
