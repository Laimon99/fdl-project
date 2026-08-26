from __future__ import annotations

import io
import json
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table
from tensorflow import keras

from .constants import LABELS, MANIFEST_PATH, PROJECT_ROOT
from .data import validate_manifest
from .datasets import read_manifest
from .utils import ProjectError, ensure_directory, read_json, sha256_file, write_json

console = Console()


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    evidence: str


FINAL_REQUIRED_FILES = (
    "data/processed/dataset_source.json",
    "data/processed/raw_inventory.csv",
    "data/processed/manifest.csv",
    "data/processed/manifest.metadata.json",
    "artifacts/tables/eda_summary.json",
    "artifacts/tables/experiment_leaderboard.csv",
    "artifacts/tables/seed_stability_validation.summary.json",
    "artifacts/models/speech_commands_best.keras",
    "artifacts/models/speech_commands_best.yaml",
    "artifacts/models/selection.json",
    "docs/final_report.md",
    "docs/model_card.md",
    "presentation/FDL_Speech_Commands.pptx",
    "presentation/FDL_Speech_Commands.pdf",
    "presentation/presentation_script.md",
    "presentation/q_and_a.md",
)


def _check_required_files() -> AuditCheck:
    missing = [path for path in FINAL_REQUIRED_FILES if not (PROJECT_ROOT / path).is_file()]
    return AuditCheck(
        "required deliverables",
        not missing,
        "all present" if not missing else "missing: " + ", ".join(missing),
    )


def _check_placeholders() -> AuditCheck:
    markers = ("TBD", "TODO", "PLACEHOLDER", "INSERT NAME")
    offenders: list[str] = []
    paths = [PROJECT_ROOT / "README.md"]
    paths.extend((PROJECT_ROOT / "docs").glob("*.md"))
    paths.extend((PROJECT_ROOT / "presentation").glob("*.md"))
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").upper()
        if any(marker in text for marker in markers):
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())
    return AuditCheck(
        "no unresolved placeholders",
        not offenders,
        "none found" if not offenders else ", ".join(offenders),
    )


def _check_git_clean() -> AuditCheck:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    return AuditCheck("clean Git worktree", not status, "clean" if not status else status[:500])


def _check_manifest() -> AuditCheck:
    if not MANIFEST_PATH.exists():
        return AuditCheck("data leakage controls", False, "manifest missing")
    try:
        manifest = read_manifest(MANIFEST_PATH)
        validate_manifest(manifest)
    except Exception as error:
        return AuditCheck("data leakage controls", False, str(error))
    return AuditCheck(
        "data leakage controls",
        True,
        f"{len(manifest):,} examples; speakers and silence regions disjoint",
    )


def _check_model_and_metrics() -> AuditCheck:
    model_path = PROJECT_ROOT / "artifacts" / "models" / "speech_commands_best.keras"
    selection_path = PROJECT_ROOT / "artifacts" / "models" / "selection.json"
    if not model_path.exists() or not selection_path.exists():
        return AuditCheck("loadable final model and metrics", False, "model or selection missing")
    try:
        selection = read_json(selection_path)
        evaluation = (
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / selection["selected_experiment"]
            / "evaluation_testing"
            / "metrics.json"
        )
        metrics = read_json(evaluation)
        model = keras.models.load_model(model_path)
        output_classes = int(model.output_shape[-1])
        valid_metrics = all(
            np.isfinite(float(metrics[key])) for key in ("accuracy", "macro_f1", "balanced_accuracy")
        )
        bootstrap_count = metrics["bootstrap_confidence_intervals"]["resamples"]
        passed = output_classes == len(LABELS) and valid_metrics and bootstrap_count == 10_000
        evidence = (
            f"{output_classes} outputs; accuracy={metrics['accuracy']:.4f}; "
            f"macro-F1={metrics['macro_f1']:.4f}; bootstrap={bootstrap_count:,}"
        )
        return AuditCheck("loadable final model and metrics", passed, evidence)
    except Exception as error:
        return AuditCheck("loadable final model and metrics", False, str(error))


def _check_presentation_signatures() -> AuditCheck:
    pptx = PROJECT_ROOT / "presentation" / "FDL_Speech_Commands.pptx"
    pdf = PROJECT_ROOT / "presentation" / "FDL_Speech_Commands.pdf"
    if not pptx.exists() or not pdf.exists():
        return AuditCheck("presentation files", False, "PPTX or PDF missing")
    with pptx.open("rb") as stream:
        pptx_ok = stream.read(2) == b"PK"
    with pdf.open("rb") as stream:
        pdf_ok = stream.read(5) == b"%PDF-"
    return AuditCheck(
        "presentation files",
        pptx_ok and pdf_ok,
        f"PPTX={pptx.stat().st_size / 2**20:.2f} MiB; PDF={pdf.stat().st_size / 2**20:.2f} MiB",
    )


def audit_project() -> list[AuditCheck]:
    return [
        _check_required_files(),
        _check_placeholders(),
        _check_git_clean(),
        _check_manifest(),
        _check_model_and_metrics(),
        _check_presentation_signatures(),
    ]


def print_audit(checks: list[AuditCheck]) -> None:
    table = Table(title="Final submission audit")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Evidence")
    for check in checks:
        table.add_row(
            check.name,
            "[green]PASS[/]" if check.passed else "[red]FAIL[/]",
            check.evidence,
        )
    console.print(table)


def package_submission(
    output: str | Path = PROJECT_ROOT / "submission" / "fdl_speech_commands_elearning.zip",
) -> Path:
    checks = audit_project()
    print_audit(checks)
    failures = [check for check in checks if not check.passed]
    if failures:
        raise ProjectError("Submission package refused: final audit contains failures")

    tracked_raw = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=PROJECT_ROOT
    )
    tracked = [Path(value.decode("utf-8")) for value in tracked_raw.split(b"\0") if value]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    output = Path(output)
    ensure_directory(output.parent)
    entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in tracked:
            source = PROJECT_ROOT / relative
            if not source.is_file():
                continue
            archive.write(source, relative.as_posix())
            entries.append(
                {
                    "path": relative.as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )
        manifest = {
            "git_commit": commit,
            "files": entries,
            "audit": [asdict(check) for check in checks],
        }
        buffer = io.StringIO()
        json.dump(manifest, buffer, indent=2, sort_keys=True)
        archive.writestr("SUBMISSION_MANIFEST.json", buffer.getvalue() + "\n")

    write_json(output.parent / "audit.json", [asdict(check) for check in checks])
    digest = sha256_file(output)
    console.print(f"[green]Submission package:[/] {output}")
    console.print(f"SHA-256: {digest}")
    return output
