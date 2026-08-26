from fdl_speech_commands.constants import LABELS, TARGET_WORDS
from fdl_speech_commands.data import _interval_sets_overlap, speaker_id_from_filename


def test_locked_label_space() -> None:
    assert len(LABELS) == 12
    assert LABELS[:2] == ("_silence_", "_unknown_")
    assert len(TARGET_WORDS) == 10


def test_speaker_id_removes_recording_suffix() -> None:
    assert speaker_id_from_filename("speaker123_nohash_4.wav") == "speaker123"
    assert speaker_id_from_filename("speaker123.wav") == "speaker123"


def test_interval_overlap_detection() -> None:
    assert _interval_sets_overlap([(0, 100), (200, 300)], [(99, 150)])
    assert not _interval_sets_overlap([(0, 100)], [(100, 200)])
    assert not _interval_sets_overlap([], [(0, 100)])

