from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fdl_speech_commands.submission import (
    AuditCheck,
    _blocking_failures,
    _contains_placeholder,
    _pptx_contains_placeholder,
)


def _write_minimal_pptx(path: Path, text: str) -> None:
    slide_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f"<a:t>{text}</a:t></p:sld>"
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("ppt/slides/slide1.xml", slide_xml)


def test_placeholder_detection_is_case_insensitive() -> None:
    assert _contains_placeholder("insert full name")
    assert _contains_placeholder("TODO: final review")
    assert not _contains_placeholder("All authors are named")


def test_pptx_placeholder_detection_reads_slide_xml(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.pptx"
    complete = tmp_path / "complete.pptx"
    _write_minimal_pptx(incomplete, "Simone (INSERT SURNAME)")
    _write_minimal_pptx(complete, "Simone Rossi")

    assert _pptx_contains_placeholder(incomplete)
    assert not _pptx_contains_placeholder(complete)


def test_review_mode_only_waives_known_author_files() -> None:
    identity_only = AuditCheck(
        "no unresolved placeholders",
        False,
        "CITATION.cff, presentation/presentation_script.md",
    )
    unrelated_todo = AuditCheck("no unresolved placeholders", False, "README.md")
    dirty_git = AuditCheck("clean Git worktree", False, "M README.md")

    assert _blocking_failures([identity_only], review_mode=True) == []
    assert _blocking_failures([identity_only]) == [identity_only]
    assert _blocking_failures([unrelated_todo], review_mode=True) == [unrelated_todo]
    assert _blocking_failures([dirty_git], review_mode=True) == [dirty_git]
