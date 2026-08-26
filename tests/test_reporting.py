import pandas as pd

from fdl_speech_commands.reporting import _markdown_table


def test_markdown_table_formats_metrics() -> None:
    frame = pd.DataFrame([{"experiment": "e01", "accuracy": 0.812345}])
    table = _markdown_table(frame, ["experiment", "accuracy"])
    assert "| experiment | accuracy |" in table
    assert "| e01 | 0.8123 |" in table

