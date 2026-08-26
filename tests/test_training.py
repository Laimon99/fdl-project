from pathlib import Path

from fdl_speech_commands.training import tensorboard_log_dir


def test_tensorboard_path_is_unchanged_on_non_windows() -> None:
    run_dir = Path("/tmp/project/artifacts/runs/e02")
    assert tensorboard_log_dir(run_dir, platform_name="linux") == run_dir / "tensorboard"


def test_tensorboard_path_falls_back_for_unicode_windows_workspace() -> None:
    run_dir = Path(r"C:\Users\student\Università\fdl\artifacts\runs\e02")
    temporary_root = Path(r"C:\Users\student\AppData\Local\Temp")
    actual = tensorboard_log_dir(
        run_dir,
        platform_name="win32",
        temporary_root=temporary_root,
    )
    assert actual == temporary_root / "fdl_speech_commands_tensorboard" / "e02"


def test_tensorboard_path_stays_local_for_ascii_windows_workspace() -> None:
    run_dir = Path(r"C:\fdl\artifacts\runs\e02")
    assert tensorboard_log_dir(run_dir, platform_name="win32") == run_dir / "tensorboard"
