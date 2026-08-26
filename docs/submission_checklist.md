# Submission and requirements traceability

This checklist maps every operative requirement in `project_notes.md` to concrete evidence
in the repository. It also separates automated checks from actions that the group must carry
out manually on eLearning.

## Requirement traceability

| Requirement | Evidence | Status |
|---|---|---|
| Three-person group and group introduction | Slide 1 and `presentation/presentation_script.md` | Requires final full names |
| English slides and oral presentation | 15-slide English deck and English timed script | Ready |
| Approximately 15 minutes / five minutes each | Slides 1–5, 6–10, and 11–15 assigned to the three speakers | Ready; rehearse |
| Problem introduction | Slides 1–2; `docs/final_report.md` | Ready |
| Dataset analysis before modeling | Slides 3–5; `docs/data_card.md`; EDA figures and tables | Ready |
| Deep-learning solution and training choices | Slides 6–9; configs; reusable package code | Ready |
| Multiple and alternative experiments | Five-model matrix on slide 7 and `artifacts/tables/experiment_leaderboard.csv` | Ready |
| Failed/negative experiments and trade-offs | Augmentation ablation on slide 11 and report discussion | Ready |
| Positive and negative quantitative results | Slides 10–13; bootstrap CIs, calibration, robustness, latency, per-class results | Ready |
| Positive and negative qualitative results | Slide 14; error gallery; traceable playable WAV files | Ready |
| Original models trained by the group | MLP, CNN, DS-CNN, and CRNN trained from scratch; zero pretrained components | Ready |
| Python code | `src/`, `configs/`, `tests/`, `pyproject.toml`, and locked `uv.lock` | Ready |
| Trained model / saved weights | `artifacts/models/speech_commands_best.keras` plus resolved YAML | Ready |
| PDF slide submission | `presentation/FDL_Speech_Commands.pdf` | Ready after names |
| Reproducibility | deterministic manifest, checksums, seeds, CLI, tests, CI, and guide | Ready |
| Academic integrity / explainability | original modular implementation, model card, report, script, and Q&A | Group review required |

Speech Commands is a professor-proposed dataset and the chosen task is audio classification,
so no external-dataset approval or orange-topic exception is required. The project does not
redistribute the original dataset.

## Automated preflight

Run from a clean clone or the repository root:

```bash
uv sync --extra dev
uv run pytest
uv run fdl-speech doctor
uv run fdl-speech audit
uv run fdl-speech package
```

The audit checks the manifest, final model, presentation file signatures, mandatory files,
unresolved identity tokens, and Git cleanliness. The packager writes a ZIP containing the
versioned submission material and a SHA-256 manifest.

When the technical work is ready but the three full author identities have not yet been
provided, `uv run fdl-speech package --review` creates an explicitly labelled review bundle.
Review mode waives only the known identity fields; missing artifacts, unrelated unfinished markers, dirty
Git state, invalid models, leakage, or presentation problems still block packaging. The review
ZIP records the failed identity audit in its manifest and **must not be uploaded to eLearning**.
After personalization, run the strict `audit` and `package` commands without `--review`.

## Manual eLearning procedure

1. Fill in every group member's full name and rehearse the final script with all three members.
2. Let every member inspect the PDF, run the demo, and answer the Q&A without reading notes.
3. Run the automated preflight and open the generated ZIP on a second machine if possible.
4. One group representative uploads the material directly to eLearning before **Tuesday,
   8 September 2026 at 23:59 Europe/Rome**.
5. Confirm that eLearning contains the PDF slides, Python code, and `.keras` model. Do not
   submit code or slides by email or Google Drive.
6. Download the uploaded files once and verify their sizes or hashes against
   `submission/audit.json`; retain the eLearning receipt or confirmation screenshot.
7. Bring an offline copy of the PDF, repository, model, and demo WAV files to the presentation
   on **Tuesday, 15 September 2026**.

Small changes during the final week are permitted by the supplied notes, but any replacement
upload must remain complete and within the official deadline.
