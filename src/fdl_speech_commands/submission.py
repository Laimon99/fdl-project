from __future__ import annotations

import io
import json
import re
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
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "uv.lock",
    "configs/e01_mfcc_mlp.yaml",
    "configs/e02_logmel_cnn.yaml",
    "configs/e03_logmel_dscnn.yaml",
    "configs/e04_logmel_dscnn_aug.yaml",
    "configs/e05_logmel_crnn_aug.yaml",
    "data/processed/dataset_source.json",
    "data/processed/raw_inventory.csv",
    "data/processed/manifest.csv",
    "data/processed/manifest.metadata.json",
    "artifacts/tables/eda_summary.json",
    "artifacts/tables/experiment_leaderboard.csv",
    "artifacts/tables/robustness_ablation_validation.csv",
    "artifacts/tables/seed_stability_validation.summary.json",
    "artifacts/figures/eda/class_spectrogram_examples.png",
    "artifacts/figures/eda/duration_and_energy.png",
    "artifacts/figures/experiment_leaderboard.png",
    "artifacts/figures/robustness_ablation_validation.png",
    "artifacts/models/speech_commands_best.keras",
    "artifacts/models/speech_commands_best.yaml",
    "artifacts/models/selection.json",
    "artifacts/runs/e01_mfcc_mlp/best_model.keras",
    "artifacts/runs/e02_logmel_cnn/best_model.keras",
    "artifacts/runs/e03_logmel_dscnn/best_model.keras",
    "artifacts/runs/e04_logmel_dscnn_aug/best_model.keras",
    "artifacts/runs/e05_logmel_crnn_aug/best_model.keras",
    "artifacts/runs/e05_logmel_crnn_aug_seed7/best_model.keras",
    "artifacts/runs/e05_logmel_crnn_aug_seed21/best_model.keras",
    "artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/metrics.json",
    "artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/predictions.csv",
    "artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/classification_report.csv",
    "artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/confusion_matrices.png",
    "artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/reliability_diagram.png",
    "artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/robustness.csv",
    "artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/qualitative_samples.csv",
    "artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/qualitative_gallery.png",
    "artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/qualitative_audio/high_confidence_error_scv1-026775.wav",
    "artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/qualitative_audio/uncertain_correct_scv1-027703.wav",
    "docs/data_card.md",
    "docs/experiment_protocol.md",
    "docs/final_report.md",
    "docs/model_card.md",
    "docs/reproducibility.md",
    "docs/demo_guide.md",
    "docs/submission_checklist.md",
    "docs/references.bib",
    "scripts/reproduce.ps1",
    "scripts/reproduce.sh",
    "presentation/FDL_Speech_Commands.pptx",
    "presentation/FDL_Speech_Commands.pdf",
    "presentation/README.md",
    "presentation/presentation_script.md",
    "presentation/q_and_a.md",
)

PLACEHOLDER_MARKERS = (
    "TBD",
    "TODO",
    "PLACEHOLDER",
    "INSERT ",
    "THIRD GROUP MEMBER",
    "FOUNDATIONS OF DEEP LEARNING PROJECT GROUP",
)

AUTHOR_PLACEHOLDER_FILES = {
    "CITATION.cff",
    "pyproject.toml",
    "presentation/FDL_Speech_Commands.pptx",
    "presentation/README.md",
    "presentation/presentation_script.md",
}


def _contains_placeholder(text: str) -> bool:
    normalized = text.upper()
    return any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def _pptx_contains_placeholder(path: Path) -> bool:
    """Inspect visible slide and speaker-note text without requiring PowerPoint."""

    with zipfile.ZipFile(path) as archive:
        relevant = re.compile(
            r"ppt/(?:slides/slide|notesSlides/notesSlide)\d+\.xml$"
        )
        for name in archive.namelist():
            if not relevant.fullmatch(name):
                continue
            xml = archive.read(name).decode("utf-8", errors="replace")
            # Text can be split across multiple DrawingML runs, so remove tags before scanning.
            visible_text = re.sub(r"<[^>]+>", " ", xml)
            if _contains_placeholder(visible_text):
                return True
    return False


def _check_required_files() -> AuditCheck:
    missing = [path for path in FINAL_REQUIRED_FILES if not (PROJECT_ROOT / path).is_file()]
    return AuditCheck(
        "required deliverables",
        not missing,
        "all present" if not missing else "missing: " + ", ".join(missing),
    )


def _check_placeholders() -> AuditCheck:
    offenders: list[str] = []
    paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "CITATION.cff",
        PROJECT_ROOT / "pyproject.toml",
    ]
    paths.extend((PROJECT_ROOT / "docs").glob("*.md"))
    paths.extend((PROJECT_ROOT / "presentation").glob("*.md"))
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if _contains_placeholder(text):
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())

    pptx = PROJECT_ROOT / "presentation" / "FDL_Speech_Commands.pptx"
    if pptx.is_file():
        try:
            if _pptx_contains_placeholder(pptx):
                offenders.append(pptx.relative_to(PROJECT_ROOT).as_posix())
        except (OSError, zipfile.BadZipFile):
            offenders.append(pptx.relative_to(PROJECT_ROOT).as_posix())
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
    try:
        with zipfile.ZipFile(pptx) as archive:
            slide_count = sum(
                bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", name))
                for name in archive.namelist()
            )
        pdf_bytes = pdf.read_bytes()
        pdf_ok = pdf_bytes.startswith(b"%PDF-")
        pdf_page_count = len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))
        passed = slide_count == 15 and pdf_ok and pdf_page_count == slide_count
        evidence = (
            f"{slide_count} PPTX slides / {pdf_page_count} PDF pages; "
            f"PPTX={pptx.stat().st_size / 2**20:.2f} MiB; "
            f"PDF={pdf.stat().st_size / 2**20:.2f} MiB"
        )
        return AuditCheck("presentation files", passed, evidence)
    except (OSError, zipfile.BadZipFile) as error:
        return AuditCheck("presentation files", False, str(error))


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


def _blocking_failures(
    checks: list[AuditCheck], *, review_mode: bool = False
) -> list[AuditCheck]:
    failures = [check for check in checks if not check.passed]
    if not review_mode:
        return failures

    blocking: list[AuditCheck] = []
    for check in failures:
        if check.name != "no unresolved placeholders":
            blocking.append(check)
            continue
        offenders = {value.strip() for value in check.evidence.split(",") if value.strip()}
        if not offenders or not offenders.issubset(AUTHOR_PLACEHOLDER_FILES):
            blocking.append(check)
    return blocking


def package_submission(
    output: str | Path = PROJECT_ROOT / "submission" / "fdl_speech_commands_elearning.zip",
    *,
    review_mode: bool = False,
) -> Path:
    checks = audit_project()
    print_audit(checks)
    failures = _blocking_failures(checks, review_mode=review_mode)
    if failures:
        raise ProjectError("Submission package refused: final audit contains failures")

    if review_mode:
        console.print(
            "[yellow]Review mode:[/] only the known author-identity fields may remain; "
            "this archive must not be uploaded before personalization and a strict audit."
        )

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
            "package_mode": "review" if review_mode else "final",
            "remaining_action": (
                "Replace the three author identities, regenerate PPTX/PDF, and pass strict audit."
                if review_mode
                else None
            ),
            "files": entries,
            "audit": [asdict(check) for check in checks],
        }
        buffer = io.StringIO()
        json.dump(manifest, buffer, indent=2, sort_keys=True)
        archive.writestr("SUBMISSION_MANIFEST.json", buffer.getvalue() + "\n")

    audit_name = "review_audit.json" if review_mode else "audit.json"
    write_json(output.parent / audit_name, [asdict(check) for check in checks])
    digest = sha256_file(output)
    console.print(f"[green]Submission package:[/] {output}")
    console.print(f"SHA-256: {digest}")
    return output
