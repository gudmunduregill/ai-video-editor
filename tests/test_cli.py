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
        assert args.model == "large-v2"
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
            model_size="large-v2",
            language="is",
            subtitle_format="srt",
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
        """python -m scripts --help shows help message with subcommands."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "scripts", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Main help now shows subcommands, not individual flags
        assert "subtitle" in result.stdout
        assert "edit" in result.stdout
        assert "apply-edl" in result.stdout

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


class TestCliSubcommandStructure:
    """Tests for CLI subcommand argument parsing."""

    def test_parse_args_subtitle_subcommand_accepts_video_path(self) -> None:
        """CLI subtitle subcommand accepts video path as positional argument."""
        from scripts.cli import parse_args

        args = parse_args(["subtitle", "video.mp4"])

        assert args.command == "subtitle"
        assert args.video == "video.mp4"

    def test_parse_args_subtitle_subcommand_accepts_all_flags(self) -> None:
        """CLI subtitle subcommand accepts all existing flags."""
        from scripts.cli import parse_args

        args = parse_args([
            "subtitle", "video.mp4",
            "--output", "subs.srt",
            "--model", "large-v2",
            "--language", "is",
            "--format", "vtt",
        ])

        assert args.command == "subtitle"
        assert args.video == "video.mp4"
        assert args.output == "subs.srt"
        assert args.model == "large-v2"
        assert args.language == "is"
        assert args.format == "vtt"

    def test_parse_args_edit_subcommand_accepts_video_path(self) -> None:
        """CLI edit subcommand accepts video path as positional argument."""
        from scripts.cli import parse_args

        args = parse_args(["edit", "video.mp4"])

        assert args.command == "edit"
        assert args.video == "video.mp4"

    def test_parse_args_edit_subcommand_accepts_output_flag(self) -> None:
        """CLI edit subcommand accepts --output flag for EDL path."""
        from scripts.cli import parse_args

        args = parse_args(["edit", "video.mp4", "--output", "cuts.edl.json"])

        assert args.command == "edit"
        assert args.output == "cuts.edl.json"

    def test_parse_args_edit_subcommand_accepts_short_output_flag(self) -> None:
        """CLI edit subcommand accepts -o as short form of --output."""
        from scripts.cli import parse_args

        args = parse_args(["edit", "video.mp4", "-o", "cuts.edl.json"])

        assert args.output == "cuts.edl.json"

    def test_parse_args_edit_subcommand_accepts_transcript_flag(self) -> None:
        """CLI edit subcommand accepts --transcript flag for existing transcript."""
        from scripts.cli import parse_args

        args = parse_args(["edit", "video.mp4", "--transcript", "video.srt"])

        assert args.command == "edit"
        assert args.transcript == "video.srt"

    def test_parse_args_edit_subcommand_accepts_short_transcript_flag(self) -> None:
        """CLI edit subcommand accepts -t as short form of --transcript."""
        from scripts.cli import parse_args

        args = parse_args(["edit", "video.mp4", "-t", "video.srt"])

        assert args.transcript == "video.srt"

    def test_parse_args_edit_subcommand_accepts_auto_flag(self) -> None:
        """CLI edit subcommand accepts --auto flag for auto-apply."""
        from scripts.cli import parse_args

        args = parse_args(["edit", "video.mp4", "--auto"])

        assert args.command == "edit"
        assert args.auto is True

    def test_parse_args_edit_subcommand_auto_default_is_false(self) -> None:
        """CLI edit subcommand defaults --auto to False."""
        from scripts.cli import parse_args

        args = parse_args(["edit", "video.mp4"])

        assert args.auto is False

    def test_parse_args_edit_subcommand_default_values(self) -> None:
        """CLI edit subcommand has correct default values."""
        from scripts.cli import parse_args

        args = parse_args(["edit", "video.mp4"])

        assert args.output is None
        assert args.transcript is None
        assert args.auto is False

    def test_parse_args_apply_edl_subcommand_accepts_video_and_edl(self) -> None:
        """CLI apply-edl subcommand accepts video path and EDL path."""
        from scripts.cli import parse_args

        args = parse_args(["apply-edl", "video.mp4", "video.edl.json"])

        assert args.command == "apply-edl"
        assert args.video == "video.mp4"
        assert args.edl == "video.edl.json"

    def test_parse_args_apply_edl_subcommand_accepts_output_flag(self) -> None:
        """CLI apply-edl subcommand accepts --output flag for output video."""
        from scripts.cli import parse_args

        args = parse_args([
            "apply-edl", "video.mp4", "video.edl.json",
            "--output", "video_edited.mp4",
        ])

        assert args.command == "apply-edl"
        assert args.output == "video_edited.mp4"

    def test_parse_args_apply_edl_subcommand_accepts_short_output_flag(self) -> None:
        """CLI apply-edl subcommand accepts -o as short form of --output."""
        from scripts.cli import parse_args

        args = parse_args([
            "apply-edl", "video.mp4", "video.edl.json",
            "-o", "video_edited.mp4",
        ])

        assert args.output == "video_edited.mp4"

    def test_parse_args_apply_edl_subcommand_default_output_is_none(self) -> None:
        """CLI apply-edl subcommand defaults output to None."""
        from scripts.cli import parse_args

        args = parse_args(["apply-edl", "video.mp4", "video.edl.json"])

        assert args.output is None

    def test_parse_args_apply_edl_missing_edl_path_raises_error(self) -> None:
        """CLI apply-edl raises error when EDL path is not provided."""
        from scripts.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["apply-edl", "video.mp4"])


class TestCliBackwardCompatibility:
    """Tests for CLI backward compatibility."""

    def test_parse_args_bare_video_path_defaults_to_subtitle(self) -> None:
        """CLI treats bare video path as subtitle subcommand."""
        from scripts.cli import parse_args

        args = parse_args(["video.mp4"])

        assert args.command == "subtitle"
        assert args.video == "video.mp4"

    def test_parse_args_bare_video_path_with_all_flags(self) -> None:
        """CLI accepts all flags with bare video path (backward compatibility)."""
        from scripts.cli import parse_args

        args = parse_args([
            "video.mp4",
            "--output", "subs.srt",
            "--model", "large-v2",
            "--language", "is",
            "--format", "vtt",
        ])

        assert args.command == "subtitle"
        assert args.video == "video.mp4"
        assert args.output == "subs.srt"
        assert args.model == "large-v2"
        assert args.language == "is"
        assert args.format == "vtt"

    def test_parse_args_bare_video_path_with_short_flags(self) -> None:
        """CLI accepts short flags with bare video path."""
        from scripts.cli import parse_args

        args = parse_args(["video.mp4", "-o", "subs.srt", "-m", "small", "-l", "en"])

        assert args.command == "subtitle"
        assert args.output == "subs.srt"
        assert args.model == "small"
        assert args.language == "en"


class TestCliSubcommandHelp:
    """Tests for CLI subcommand help messages."""

    def test_module_execution_shows_subcommands_in_help(self) -> None:
        """python -m scripts --help shows available subcommands."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "scripts", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "subtitle" in result.stdout
        assert "edit" in result.stdout
        assert "apply-edl" in result.stdout

    def test_module_execution_subtitle_subcommand_help(self) -> None:
        """python -m scripts subtitle --help shows subtitle options."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "scripts", "subtitle", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--output" in result.stdout
        assert "--model" in result.stdout
        assert "--language" in result.stdout
        assert "--format" in result.stdout

    def test_module_execution_edit_subcommand_help(self) -> None:
        """python -m scripts edit --help shows edit options."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "scripts", "edit", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--output" in result.stdout
        assert "--transcript" in result.stdout
        assert "--auto" in result.stdout

    def test_module_execution_apply_edl_subcommand_help(self) -> None:
        """python -m scripts apply-edl --help shows apply-edl options."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "scripts", "apply-edl", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--output" in result.stdout
        assert "edl" in result.stdout.lower()


class TestCliFormatFlag:
    """Tests for CLI --format flag."""

    def test_parse_args_with_format_flag(self) -> None:
        """CLI accepts --format flag for subtitle format."""
        from scripts.cli import parse_args

        args = parse_args(["subtitle", "video.mp4", "--format", "vtt"])

        assert args.format == "vtt"

    def test_parse_args_with_short_format_flag(self) -> None:
        """CLI accepts -f as short form of --format."""
        from scripts.cli import parse_args

        args = parse_args(["subtitle", "video.mp4", "-f", "vtt"])

        assert args.format == "vtt"

    def test_parse_args_format_default_is_srt(self) -> None:
        """CLI defaults to srt format."""
        from scripts.cli import parse_args

        args = parse_args(["subtitle", "video.mp4"])

        assert args.format == "srt"

    @pytest.mark.parametrize("format_value", ["srt", "vtt"])
    def test_parse_args_accepts_valid_format_choices(self, format_value: str) -> None:
        """CLI accepts all valid format choices."""
        from scripts.cli import parse_args

        args = parse_args(["subtitle", "video.mp4", "--format", format_value])

        assert args.format == format_value

    def test_parse_args_rejects_invalid_format_choice(self) -> None:
        """CLI rejects invalid format choice."""
        from scripts.cli import parse_args

        with pytest.raises(SystemExit):
            parse_args(["subtitle", "video.mp4", "--format", "invalid"])

    def test_main_passes_format_to_process_video(
        self, tmp_path: Path
    ) -> None:
        """main() passes format to process_video."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.return_value = str(tmp_path / "test.vtt")

            main([str(video_path), "--format", "vtt"])

        call_kwargs = mock_process.call_args
        assert call_kwargs[1]["subtitle_format"] == "vtt"

    def test_main_passes_default_format_to_process_video(
        self, tmp_path: Path
    ) -> None:
        """main() passes default format (srt) to process_video."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.process_video") as mock_process:
            mock_process.return_value = str(tmp_path / "test.srt")

            main([str(video_path)])

        call_kwargs = mock_process.call_args
        assert call_kwargs[1]["subtitle_format"] == "srt"

    def test_module_execution_shows_format_flag_in_subtitle_help(self) -> None:
        """python -m scripts subtitle --help shows --format flag."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "scripts", "subtitle", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--format" in result.stdout or "-f" in result.stdout


class TestCliEditSubcommandExecution:
    """Tests for CLI edit subcommand execution."""

    def test_main_edit_calls_edit_video_with_correct_arguments(
        self, tmp_path: Path
    ) -> None:
        """main() edit subcommand calls edit_video with parsed arguments."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.edit_video") as mock_edit:
            mock_edit.return_value = {
                "edl_path": str(tmp_path / "test.edl.json"),
                "transcript_for_review": "transcript text",
                "video_duration": 120.0,
                "segment_count": 10,
            }

            exit_code = main(["edit", str(video_path)])

        mock_edit.assert_called_once_with(
            str(video_path),
            output_path=None,
            transcript_path=None,
            edl_path=None,
            auto_apply=False,
            use_ai=False,
        )
        assert exit_code == 0

    def test_main_edit_passes_output_path(
        self, tmp_path: Path
    ) -> None:
        """main() edit subcommand passes output path to edit_video."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        edl_path = str(tmp_path / "custom.edl.json")

        with patch("scripts.cli.edit_video") as mock_edit:
            mock_edit.return_value = {
                "edl_path": edl_path,
                "transcript_for_review": "transcript text",
                "video_duration": 120.0,
                "segment_count": 10,
            }

            main(["edit", str(video_path), "--output", edl_path])

        call_kwargs = mock_edit.call_args
        assert call_kwargs[1]["edl_path"] == edl_path

    def test_main_edit_passes_transcript_path(
        self, tmp_path: Path
    ) -> None:
        """main() edit subcommand passes transcript path to edit_video."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        transcript_path = str(tmp_path / "existing.srt")

        with patch("scripts.cli.edit_video") as mock_edit:
            mock_edit.return_value = {
                "edl_path": str(tmp_path / "test.edl.json"),
                "transcript_for_review": "transcript text",
                "video_duration": 120.0,
                "segment_count": 10,
            }

            main(["edit", str(video_path), "--transcript", transcript_path])

        call_kwargs = mock_edit.call_args
        assert call_kwargs[1]["transcript_path"] == transcript_path

    def test_main_edit_passes_auto_flag(
        self, tmp_path: Path
    ) -> None:
        """main() edit subcommand passes auto flag to edit_video."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.edit_video") as mock_edit:
            mock_edit.return_value = {
                "edl_path": str(tmp_path / "test.edl.json"),
                "transcript_for_review": "transcript text",
                "video_duration": 120.0,
                "segment_count": 10,
                "edited_video_path": str(tmp_path / "test_edited.mp4"),
            }

            main(["edit", str(video_path), "--auto"])

        call_kwargs = mock_edit.call_args
        assert call_kwargs[1]["auto_apply"] is True

    def test_main_edit_passes_ai_flag(
        self, tmp_path: Path
    ) -> None:
        """main() edit subcommand passes --ai flag to edit_video."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.edit_video") as mock_edit:
            mock_edit.return_value = {
                "edl_path": str(tmp_path / "test.edl.json"),
                "transcript_for_review": "transcript text",
                "video_duration": 120.0,
                "segment_count": 10,
                "ai_used": True,
            }

            main(["edit", str(video_path), "--ai"])

        call_kwargs = mock_edit.call_args
        assert call_kwargs[1]["use_ai"] is True

    def test_main_edit_handles_llm_client_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() edit subcommand handles LLMClientError gracefully."""
        from scripts.cli import main
        from scripts.llm_client import LLMClientError

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        with patch("scripts.cli.edit_video") as mock_edit:
            mock_edit.side_effect = LLMClientError("ANTHROPIC_API_KEY not set")

            exit_code = main(["edit", str(video_path), "--ai"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "AI analysis failed" in captured.err

    def test_main_edit_prints_edl_path_on_success(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() edit subcommand prints EDL path on success."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        edl_path = str(tmp_path / "test.edl.json")

        with patch("scripts.cli.edit_video") as mock_edit:
            mock_edit.return_value = {
                "edl_path": edl_path,
                "transcript_for_review": "transcript text",
                "video_duration": 120.0,
                "segment_count": 10,
            }

            main(["edit", str(video_path)])

        captured = capsys.readouterr()
        assert edl_path in captured.out

    def test_main_edit_returns_exit_code_1_on_file_not_found(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() edit subcommand returns exit code 1 when video not found."""
        from scripts.cli import main

        exit_code = main(["edit", "/nonexistent/video.mp4"])

        assert exit_code == 1

    def test_main_edit_prints_error_on_file_not_found(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() edit subcommand prints error when video not found."""
        from scripts.cli import main

        main(["edit", "/nonexistent/video.mp4"])

        captured = capsys.readouterr()
        assert "Error:" in captured.err


class TestCliApplyEdlSubcommandExecution:
    """Tests for CLI apply-edl subcommand execution."""

    def test_main_apply_edl_calls_apply_edl_to_video_with_correct_arguments(
        self, tmp_path: Path
    ) -> None:
        """main() apply-edl subcommand calls apply_edl_to_video with parsed arguments."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        edl_path = tmp_path / "test.edl.json"
        edl_path.write_text('{"source_video": "test.mp4", "segments": [], "total_duration": 120.0}')

        with patch("scripts.cli.apply_edl_to_video") as mock_apply:
            mock_apply.return_value = str(tmp_path / "test_edited.mp4")

            exit_code = main(["apply-edl", str(video_path), str(edl_path)])

        mock_apply.assert_called_once_with(
            str(video_path),
            str(edl_path),
            None,
            srt_path=None,
        )
        assert exit_code == 0

    def test_main_apply_edl_passes_output_path(
        self, tmp_path: Path
    ) -> None:
        """main() apply-edl subcommand passes output path to apply_edl_to_video."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        edl_path = tmp_path / "test.edl.json"
        edl_path.write_text('{}')
        output_path = str(tmp_path / "custom_output.mp4")

        with patch("scripts.cli.apply_edl_to_video") as mock_apply:
            mock_apply.return_value = output_path

            main(["apply-edl", str(video_path), str(edl_path), "--output", output_path])

        call_args = mock_apply.call_args
        assert call_args[0][2] == output_path

    def test_main_apply_edl_prints_output_path_on_success(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() apply-edl subcommand prints output video path on success."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        edl_path = tmp_path / "test.edl.json"
        edl_path.write_text('{}')
        output_path = str(tmp_path / "test_edited.mp4")

        with patch("scripts.cli.apply_edl_to_video") as mock_apply:
            mock_apply.return_value = output_path

            main(["apply-edl", str(video_path), str(edl_path)])

        captured = capsys.readouterr()
        assert output_path in captured.out

    def test_main_apply_edl_returns_exit_code_1_on_video_not_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() apply-edl subcommand returns exit code 1 when video not found."""
        from scripts.cli import main

        edl_path = tmp_path / "test.edl.json"
        edl_path.write_text('{}')

        exit_code = main(["apply-edl", "/nonexistent/video.mp4", str(edl_path)])

        assert exit_code == 1

    def test_main_apply_edl_returns_exit_code_1_on_edl_not_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() apply-edl subcommand returns exit code 1 when EDL not found."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")

        exit_code = main(["apply-edl", str(video_path), "/nonexistent/edl.json"])

        assert exit_code == 1

    def test_main_apply_edl_prints_error_on_file_not_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() apply-edl subcommand prints error when file not found."""
        from scripts.cli import main

        edl_path = tmp_path / "test.edl.json"
        edl_path.write_text('{}')

        main(["apply-edl", "/nonexistent/video.mp4", str(edl_path)])

        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_main_apply_edl_returns_exit_code_1_on_video_cutting_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() apply-edl subcommand returns exit code 1 on video cutting error."""
        from scripts.cli import main
        from scripts.exceptions import VideoCuttingError

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        edl_path = tmp_path / "test.edl.json"
        edl_path.write_text('{}')

        with patch("scripts.cli.apply_edl_to_video") as mock_apply:
            mock_apply.side_effect = VideoCuttingError("FFmpeg failed")

            exit_code = main(["apply-edl", str(video_path), str(edl_path)])

        assert exit_code == 1

    def test_main_apply_edl_prints_error_on_video_cutting_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() apply-edl subcommand prints error on video cutting error."""
        from scripts.cli import main
        from scripts.exceptions import VideoCuttingError

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        edl_path = tmp_path / "test.edl.json"
        edl_path.write_text('{}')

        with patch("scripts.cli.apply_edl_to_video") as mock_apply:
            mock_apply.side_effect = VideoCuttingError("FFmpeg failed")

            main(["apply-edl", str(video_path), str(edl_path)])

        captured = capsys.readouterr()
        assert "Error:" in captured.err


class TestCliApplyEdlSrtFlag:
    """Tests for CLI apply-edl --srt flag."""

    def test_parse_args_apply_edl_accepts_srt_flag(self) -> None:
        """CLI apply-edl subcommand accepts --srt flag for input SRT file."""
        from scripts.cli import parse_args

        args = parse_args([
            "apply-edl", "video.mp4", "video.edl.json",
            "--srt", "video.srt",
        ])

        assert args.command == "apply-edl"
        assert args.srt == "video.srt"

    def test_parse_args_apply_edl_srt_default_is_none(self) -> None:
        """CLI apply-edl subcommand defaults --srt to None."""
        from scripts.cli import parse_args

        args = parse_args(["apply-edl", "video.mp4", "video.edl.json"])

        assert args.srt is None

    def test_main_apply_edl_passes_srt_path(
        self, tmp_path: Path
    ) -> None:
        """main() apply-edl subcommand passes srt_path to apply_edl_to_video."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        edl_path = tmp_path / "test.edl.json"
        edl_path.write_text('{}')
        srt_path = tmp_path / "test.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:05,000\nHello")

        with patch("scripts.cli.apply_edl_to_video") as mock_apply:
            mock_apply.return_value = {
                "video_path": str(tmp_path / "test_edited.mp4"),
                "srt_path": str(tmp_path / "test_edited.srt"),
            }

            main(["apply-edl", str(video_path), str(edl_path), "--srt", str(srt_path)])

        call_args = mock_apply.call_args
        assert call_args[1]["srt_path"] == str(srt_path)

    def test_main_apply_edl_prints_srt_output_path_on_success(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() apply-edl subcommand prints SRT output path when --srt is provided."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        edl_path = tmp_path / "test.edl.json"
        edl_path.write_text('{}')
        srt_path = tmp_path / "test.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:05,000\nHello")
        output_srt_path = str(tmp_path / "test_edited.srt")

        with patch("scripts.cli.apply_edl_to_video") as mock_apply:
            mock_apply.return_value = {
                "video_path": str(tmp_path / "test_edited.mp4"),
                "srt_path": output_srt_path,
            }

            main(["apply-edl", str(video_path), str(edl_path), "--srt", str(srt_path)])

        captured = capsys.readouterr()
        assert output_srt_path in captured.out

    def test_main_apply_edl_returns_exit_code_1_on_srt_not_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """main() apply-edl subcommand returns exit code 1 when SRT file not found."""
        from scripts.cli import main

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"dummy")
        edl_path = tmp_path / "test.edl.json"
        edl_path.write_text('{}')

        with patch("scripts.cli.apply_edl_to_video") as mock_apply:
            mock_apply.side_effect = FileNotFoundError("SRT file not found")

            exit_code = main([
                "apply-edl", str(video_path), str(edl_path),
                "--srt", "/nonexistent/file.srt"
            ])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
