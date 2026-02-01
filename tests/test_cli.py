"""Tests for the CLI module."""

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCliArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_parse_args_with_video_path_only(self) -> None:
        """CLI accepts video path as positional argument."""
        from scripts.cli import parse_args

        args = parse_args(["video.mp4"])

        assert args.video == "video.mp4"

    def test_parse_args_with_output_flag(self) -> None:
        """CLI accepts --output flag for custom output path."""
        from scripts.cli import parse_args

        args = parse_args(["video.mp4", "--output", "subs.srt"])

        assert args.video == "video.mp4"
        assert args.output == "subs.srt"

    def test_parse_args_with_model_flag(self) -> None:
        """CLI accepts --model flag for whisper model size."""
        from scripts.cli import parse_args

        args = parse_args(["video.mp4", "--model", "small"])

        assert args.model == "small"

    def test_parse_args_with_language_flag(self) -> None:
        """CLI accepts --language flag for language code."""
        from scripts.cli import parse_args

        args = parse_args(["video.mp4", "--language", "en"])

        assert args.language == "en"

    def test_parse_args_with_all_flags(self) -> None:
        """CLI accepts all flags together."""
        from scripts.cli import parse_args

        args = parse_args([
            "video.mp4",
            "--output", "subs.srt",
            "--model", "large-v2",
            "--language", "fr",
        ])

        assert args.video == "video.mp4"
        assert args.output == "subs.srt"
        assert args.model == "large-v2"
        assert args.language == "fr"

    def test_parse_args_default_values(self) -> None:
        """CLI has correct default values for optional arguments."""
        from scripts.cli import parse_args

        args = parse_args(["video.mp4"])

        assert args.output is None
        assert args.model == "base"
        assert args.language == "is"

    def test_parse_args_missing_video_path_raises_error(self) -> None:
        """CLI raises error when video path is not provided."""
        from scripts.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args([])


class TestCliModelChoices:
    """Tests for CLI model choices validation."""

    @pytest.mark.parametrize("model", ["tiny", "base", "small", "medium", "large-v2"])
    def test_parse_args_accepts_valid_model_choices(self, model: str) -> None:
        """CLI accepts all valid model choices."""
        from scripts.cli import parse_args

        args = parse_args(["video.mp4", "--model", model])

        assert args.model == model

    def test_parse_args_rejects_invalid_model_choice(self) -> None:
        """CLI rejects invalid model choice."""
        from scripts.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["video.mp4", "--model", "invalid-model"])


class TestCliMainFunction:
    """Tests for CLI main function."""

    def test_main_calls_process_video_with_correct_arguments(
        self, tmp_path: Path
    ) -> None:
        """main() calls process_video with parsed arguments."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.return_value = str(tmp_path / "test.srt")

            exit_code = main([str(video_path)])

        mock_process.assert_called_once_with(
            str(video_path),
            output_path=None,
            model_size="base",
            language="is",
        )
        assert exit_code == 0

    def test_main_passes_output_path_to_process_video(
        self, tmp_path: Path
    ) -> None:
        """main() passes output path to process_video."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        output_path = str(tmp_path / "custom.srt")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.return_value = output_path

            main([str(video_path), "--output", output_path])

        call_kwargs = mock_process.call_args
        assert call_kwargs[1]["output_path"] == output_path

    def test_main_passes_model_size_to_process_video(
        self, tmp_path: Path
    ) -> None:
        """main() passes model size to process_video."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.return_value = str(tmp_path / "test.srt")

            main([str(video_path), "--model", "large-v2"])

        call_kwargs = mock_process.call_args
        assert call_kwargs[1]["model_size"] == "large-v2"

    def test_main_passes_language_to_process_video(
        self, tmp_path: Path
    ) -> None:
        """main() passes language to process_video."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.return_value = str(tmp_path / "test.srt")

            main([str(video_path), "--language", "en"])

        call_kwargs = mock_process.call_args
        assert call_kwargs[1]["language"] == "en"


class TestCliProgressMessages:
    """Tests for CLI progress messages."""

    def test_main_prints_processing_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() prints processing message before starting."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.return_value = str(tmp_path / "test.srt")

            main([str(video_path)])

        captured = capsys.readouterr()
        assert "Processing video" in captured.out

    def test_main_prints_success_message_with_output_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() prints success message with output path."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        output_path = str(tmp_path / "test.srt")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.return_value = output_path

            main([str(video_path)])

        captured = capsys.readouterr()
        assert "Subtitles saved to:" in captured.out
        assert output_path in captured.out


class TestCliErrorHandling:
    """Tests for CLI error handling."""

    def test_main_returns_exit_code_1_on_file_not_found(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() returns exit code 1 when video file not found."""
        from scripts.cli import main

        exit_code = main(["/nonexistent/video.mp4"])

        assert exit_code == 1

    def test_main_prints_user_friendly_error_for_file_not_found(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() prints user-friendly error for file not found."""
        from scripts.cli import main

        main(["/nonexistent/video.mp4"])

        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "not found" in captured.err.lower() or "does not exist" in captured.err.lower()

    def test_main_returns_exit_code_1_on_audio_extraction_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() returns exit code 1 on audio extraction error."""
        from scripts.cli import main
        from scripts.exceptions import AudioExtractionError

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.side_effect = AudioExtractionError("FFmpeg failed")

            exit_code = main([str(video_path)])

        assert exit_code == 1

    def test_main_prints_user_friendly_error_for_audio_extraction_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() prints user-friendly error for audio extraction failure."""
        from scripts.cli import main
        from scripts.exceptions import AudioExtractionError

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.side_effect = AudioExtractionError("FFmpeg failed")

            main([str(video_path)])

        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_main_returns_exit_code_1_on_transcription_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() returns exit code 1 on transcription error."""
        from scripts.cli import main
        from scripts.exceptions import TranscriptionError

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.side_effect = TranscriptionError("Whisper failed")

            exit_code = main([str(video_path)])

        assert exit_code == 1

    def test_main_prints_user_friendly_error_for_transcription_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() prints user-friendly error for transcription failure."""
        from scripts.cli import main
        from scripts.exceptions import TranscriptionError

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.side_effect = TranscriptionError("Whisper failed")

            main([str(video_path)])

        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_main_returns_exit_code_1_on_value_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() returns exit code 1 on ValueError (empty transcription)."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.side_effect = ValueError("Empty transcription")

            exit_code = main([str(video_path)])

        assert exit_code == 1

    def test_main_returns_exit_code_0_on_success(
        self, tmp_path: Path
    ) -> None:
        """main() returns exit code 0 on successful processing."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.return_value = str(tmp_path / "test.srt")

            exit_code = main([str(video_path)])

        assert exit_code == 0


class TestCliHelpMessage:
    """Tests for CLI help message."""

    def test_parse_args_shows_help_with_h_flag(self) -> None:
        """CLI shows help message with -h flag."""
        from scripts.cli import parse_args

        with pytest.raises(SystemExit) as exc_info:
            parse_args(["-h"])

        assert exc_info.value.code == 0

    def test_parse_args_shows_help_with_help_flag(self) -> None:
        """CLI shows help message with --help flag."""
        from scripts.cli import parse_args

        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])

        assert exc_info.value.code == 0


class TestCliModuleExecution:
    """Tests for running CLI as a module."""

    def test_module_main_imports_and_calls_main(self) -> None:
        """__main__.py imports and calls main function."""
        import scripts.__main__

        # The module should have imported main from cli
        assert hasattr(scripts.__main__, "main")

    def test_module_execution_via_subprocess_shows_help(self) -> None:
        """python -m scripts --help shows help message."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "scripts", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "video" in result.stdout.lower()
        assert "--output" in result.stdout
        assert "--model" in result.stdout
        assert "--language" in result.stdout

    def test_module_execution_via_subprocess_with_nonexistent_file(self) -> None:
        """python -m scripts with nonexistent file returns exit code 1."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "scripts", "/nonexistent/video.mp4"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error:" in result.stderr


class TestCliShortFlags:
    """Tests for CLI short flag aliases."""

    def test_parse_args_accepts_short_output_flag(self) -> None:
        """CLI accepts -o as short form of --output."""
        from scripts.cli import parse_args

        args = parse_args(["video.mp4", "-o", "subs.srt"])

        assert args.output == "subs.srt"

    def test_parse_args_accepts_short_model_flag(self) -> None:
        """CLI accepts -m as short form of --model."""
        from scripts.cli import parse_args

        args = parse_args(["video.mp4", "-m", "small"])

        assert args.model == "small"

    def test_parse_args_accepts_short_language_flag(self) -> None:
        """CLI accepts -l as short form of --language."""
        from scripts.cli import parse_args

        args = parse_args(["video.mp4", "-l", "en"])

        assert args.language == "en"
