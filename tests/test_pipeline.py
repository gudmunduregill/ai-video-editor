"""Tests for the pipeline module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.exceptions import AudioExtractionError, TranscriptionError
from scripts.transcription import TranscriptSegment


class TestProcessVideoBasic:
    """Basic unit tests for process_video function."""

    def test_process_video_file_not_found(self) -> None:
        """process_video raises FileNotFoundError when video file doesn't exist."""
        from scripts.pipeline import process_video

        non_existent_path = "/path/to/nonexistent/video.mp4"

        with pytest.raises(FileNotFoundError) as exc_info:
            process_video(non_existent_path)

        assert "video.mp4" in str(exc_info.value) or "not found" in str(
            exc_info.value
        ).lower()

    def test_process_video_returns_srt_path_with_same_basename(
        self, tmp_path: Path
    ) -> None:
        """process_video returns SRT path with same name as video but .srt extension."""
        from scripts.pipeline import process_video

        # Create a dummy video file
        video_path = tmp_path / "my_video.mp4"
        video_path.write_bytes(b"dummy video content")

        mock_segments = [
            TranscriptSegment(start=0.0, end=2.5, text="Hello, world!"),
        ]

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                with patch("scripts.pipeline.write_srt") as mock_write:
                    # Setup mocks
                    mock_extract.return_value = str(tmp_path / "temp_audio.wav")
                    mock_transcribe.return_value = mock_segments

                    result = process_video(str(video_path))

        # Should return SRT path with same basename
        expected_srt_path = str(tmp_path / "my_video.srt")
        assert result == expected_srt_path

    def test_process_video_extracts_audio_to_temp_file(
        self, tmp_path: Path
    ) -> None:
        """process_video extracts audio to a temporary WAV file."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")

        mock_segments = [
            TranscriptSegment(start=0.0, end=2.5, text="Test"),
        ]

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                with patch("scripts.pipeline.write_srt"):
                    mock_extract.return_value = str(tmp_path / "temp.wav")
                    mock_transcribe.return_value = mock_segments

                    process_video(str(video_path))

        # Verify extract_audio was called with the video path
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args[0][0] == str(video_path)

    def test_process_video_transcribes_extracted_audio(
        self, tmp_path: Path
    ) -> None:
        """process_video transcribes the extracted audio file."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")
        temp_audio_path = str(tmp_path / "temp_audio.wav")

        mock_segments = [
            TranscriptSegment(start=0.0, end=2.5, text="Test"),
        ]

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                with patch("scripts.pipeline.write_srt"):
                    mock_extract.return_value = temp_audio_path
                    mock_transcribe.return_value = mock_segments

                    process_video(str(video_path))

        # Verify transcribe was called with the audio path
        mock_transcribe.assert_called_once()
        call_args = mock_transcribe.call_args
        assert call_args[0][0] == temp_audio_path

    def test_process_video_writes_srt_with_segments(
        self, tmp_path: Path
    ) -> None:
        """process_video writes transcript segments to SRT file."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")

        mock_segments = [
            TranscriptSegment(start=0.0, end=2.5, text="Hello"),
            TranscriptSegment(start=2.6, end=5.0, text="World"),
        ]

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                with patch("scripts.pipeline.write_srt") as mock_write:
                    mock_extract.return_value = str(tmp_path / "temp.wav")
                    mock_transcribe.return_value = mock_segments

                    result = process_video(str(video_path))

        # Verify write_srt was called with segments and correct output path
        mock_write.assert_called_once()
        call_args = mock_write.call_args
        assert call_args[0][0] == mock_segments
        assert call_args[0][1] == result

    def test_process_video_passes_model_size_parameter(
        self, tmp_path: Path
    ) -> None:
        """process_video passes model_size parameter to transcribe."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")

        mock_segments = [TranscriptSegment(start=0.0, end=1.0, text="Test")]

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                with patch("scripts.pipeline.write_srt"):
                    mock_extract.return_value = str(tmp_path / "temp.wav")
                    mock_transcribe.return_value = mock_segments

                    process_video(str(video_path), model_size="large-v2")

        # Verify model_size was passed to transcribe
        call_kwargs = mock_transcribe.call_args[1]
        assert call_kwargs.get("model_size") == "large-v2"

    def test_process_video_passes_language_parameter(
        self, tmp_path: Path
    ) -> None:
        """process_video passes language parameter to transcribe."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")

        mock_segments = [TranscriptSegment(start=0.0, end=1.0, text="Test")]

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                with patch("scripts.pipeline.write_srt"):
                    mock_extract.return_value = str(tmp_path / "temp.wav")
                    mock_transcribe.return_value = mock_segments

                    process_video(str(video_path), language="en")

        # Verify language was passed to transcribe
        call_kwargs = mock_transcribe.call_args[1]
        assert call_kwargs.get("language") == "en"

    def test_process_video_uses_default_model_size(
        self, tmp_path: Path
    ) -> None:
        """process_video uses 'base' as default model_size."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")

        mock_segments = [TranscriptSegment(start=0.0, end=1.0, text="Test")]

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                with patch("scripts.pipeline.write_srt"):
                    mock_extract.return_value = str(tmp_path / "temp.wav")
                    mock_transcribe.return_value = mock_segments

                    process_video(str(video_path))

        # Verify default model_size is "base"
        call_kwargs = mock_transcribe.call_args[1]
        assert call_kwargs.get("model_size") == "base"


class TestProcessVideoCleanup:
    """Tests for temp file cleanup in process_video."""

    def test_process_video_cleans_up_temp_audio_on_success(
        self, tmp_path: Path
    ) -> None:
        """process_video removes temp audio file after successful processing."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")

        # Create an actual temp file to test cleanup
        temp_audio_path = tmp_path / "temp_audio.wav"
        temp_audio_path.write_bytes(b"dummy audio")

        mock_segments = [TranscriptSegment(start=0.0, end=1.0, text="Test")]

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                with patch("scripts.pipeline.write_srt"):
                    mock_extract.return_value = str(temp_audio_path)
                    mock_transcribe.return_value = mock_segments

                    process_video(str(video_path))

        # Temp audio file should be cleaned up
        assert not temp_audio_path.exists()

    def test_process_video_cleans_up_temp_audio_on_transcription_failure(
        self, tmp_path: Path
    ) -> None:
        """process_video removes temp audio file even when transcription fails."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")

        # Create an actual temp file to test cleanup
        temp_audio_path = tmp_path / "temp_audio.wav"
        temp_audio_path.write_bytes(b"dummy audio")

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                mock_extract.return_value = str(temp_audio_path)
                mock_transcribe.side_effect = TranscriptionError("Test error")

                with pytest.raises(TranscriptionError):
                    process_video(str(video_path))

        # Temp audio file should still be cleaned up
        assert not temp_audio_path.exists()

    def test_process_video_cleans_up_temp_audio_on_write_failure(
        self, tmp_path: Path
    ) -> None:
        """process_video removes temp audio file even when SRT write fails."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")

        # Create an actual temp file to test cleanup
        temp_audio_path = tmp_path / "temp_audio.wav"
        temp_audio_path.write_bytes(b"dummy audio")

        mock_segments = [TranscriptSegment(start=0.0, end=1.0, text="Test")]

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                with patch("scripts.pipeline.write_srt") as mock_write:
                    mock_extract.return_value = str(temp_audio_path)
                    mock_transcribe.return_value = mock_segments
                    mock_write.side_effect = IOError("Write failed")

                    with pytest.raises(IOError):
                        process_video(str(video_path))

        # Temp audio file should still be cleaned up
        assert not temp_audio_path.exists()


class TestProcessVideoErrorHandling:
    """Tests for error handling in process_video."""

    def test_process_video_propagates_audio_extraction_error(
        self, tmp_path: Path
    ) -> None:
        """process_video propagates AudioExtractionError from extract_audio."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            mock_extract.side_effect = AudioExtractionError("FFmpeg failed")

            with pytest.raises(AudioExtractionError) as exc_info:
                process_video(str(video_path))

            assert "FFmpeg failed" in str(exc_info.value)

    def test_process_video_propagates_transcription_error(
        self, tmp_path: Path
    ) -> None:
        """process_video propagates TranscriptionError from transcribe."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                mock_extract.return_value = str(tmp_path / "temp.wav")
                mock_transcribe.side_effect = TranscriptionError("Model failed")

                with pytest.raises(TranscriptionError) as exc_info:
                    process_video(str(video_path))

                assert "Model failed" in str(exc_info.value)

    def test_process_video_handles_empty_transcription(
        self, tmp_path: Path
    ) -> None:
        """process_video raises ValueError when transcription returns empty segments."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")

        # Create temp audio file for cleanup test
        temp_audio_path = tmp_path / "temp_audio.wav"
        temp_audio_path.write_bytes(b"dummy audio")

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                mock_extract.return_value = str(temp_audio_path)
                mock_transcribe.return_value = []  # Empty segments

                with pytest.raises(ValueError) as exc_info:
                    process_video(str(video_path))

                assert "empty" in str(exc_info.value).lower()

        # Temp file should still be cleaned up
        assert not temp_audio_path.exists()


class TestProcessVideoOutputPath:
    """Tests for output path handling in process_video."""

    def test_process_video_supports_custom_output_path(
        self, tmp_path: Path
    ) -> None:
        """process_video allows specifying custom output SRT path."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "test_video.mp4"
        video_path.write_bytes(b"dummy video content")
        custom_output = tmp_path / "custom_output.srt"

        mock_segments = [TranscriptSegment(start=0.0, end=1.0, text="Test")]

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                with patch("scripts.pipeline.write_srt") as mock_write:
                    mock_extract.return_value = str(tmp_path / "temp.wav")
                    mock_transcribe.return_value = mock_segments

                    result = process_video(
                        str(video_path), output_path=str(custom_output)
                    )

        assert result == str(custom_output)
        mock_write.assert_called_once()
        assert mock_write.call_args[0][1] == str(custom_output)

    def test_process_video_handles_video_with_multiple_extensions(
        self, tmp_path: Path
    ) -> None:
        """process_video correctly handles video files with multiple dots in name."""
        from scripts.pipeline import process_video

        video_path = tmp_path / "my.video.file.mp4"
        video_path.write_bytes(b"dummy video content")

        mock_segments = [TranscriptSegment(start=0.0, end=1.0, text="Test")]

        with patch("scripts.pipeline.extract_audio") as mock_extract:
            with patch("scripts.pipeline.transcribe") as mock_transcribe:
                with patch("scripts.pipeline.write_srt"):
                    mock_extract.return_value = str(tmp_path / "temp.wav")
                    mock_transcribe.return_value = mock_segments

                    result = process_video(str(video_path))

        # Should replace only the last extension
        expected_srt_path = str(tmp_path / "my.video.file.srt")
        assert result == expected_srt_path

    def test_process_video_handles_various_video_extensions(
        self, tmp_path: Path
    ) -> None:
        """process_video handles different video file extensions."""
        from scripts.pipeline import process_video

        mock_segments = [TranscriptSegment(start=0.0, end=1.0, text="Test")]

        for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            video_path = tmp_path / f"video{ext}"
            video_path.write_bytes(b"dummy video content")

            with patch("scripts.pipeline.extract_audio") as mock_extract:
                with patch("scripts.pipeline.transcribe") as mock_transcribe:
                    with patch("scripts.pipeline.write_srt"):
                        mock_extract.return_value = str(tmp_path / "temp.wav")
                        mock_transcribe.return_value = mock_segments

                        result = process_video(str(video_path))

            expected_srt_path = str(tmp_path / "video.srt")
            assert result == expected_srt_path


# Path to the test video file
TEST_VIDEO_PATH = "/home/gudmundur/ai-youtube/input/test_video.mov"


@pytest.mark.slow
class TestProcessVideoIntegration:
    """Integration tests using real video file."""

    @pytest.fixture(autouse=True)
    def skip_if_no_ffmpeg(self) -> None:
        """Skip tests if ffmpeg is not available."""
        import subprocess

        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ffmpeg not available")

    @pytest.fixture(autouse=True)
    def skip_if_no_test_video(self) -> None:
        """Skip tests if test video is not available."""
        if not os.path.exists(TEST_VIDEO_PATH):
            pytest.skip(f"Test video not found: {TEST_VIDEO_PATH}")

    def test_process_video_creates_srt_file(self, tmp_path: Path) -> None:
        """process_video creates an SRT file from the video."""
        from scripts.pipeline import process_video

        output_path = tmp_path / "output.srt"

        result = process_video(
            TEST_VIDEO_PATH,
            output_path=str(output_path),
            model_size="tiny",  # Use tiny model for faster tests
        )

        assert os.path.exists(result)
        assert result == str(output_path)

    def test_process_video_srt_has_valid_content(self, tmp_path: Path) -> None:
        """process_video creates SRT file with valid subtitle format."""
        from scripts.pipeline import process_video

        output_path = tmp_path / "output.srt"

        result = process_video(
            TEST_VIDEO_PATH,
            output_path=str(output_path),
            model_size="tiny",
        )

        # Read and verify SRT content
        with open(result, "r", encoding="utf-8") as f:
            content = f.read()

        # SRT should have at least one subtitle entry
        assert len(content) > 0

        # SRT format: starts with "1" (first subtitle number)
        lines = content.strip().split("\n")
        assert lines[0] == "1"

        # Should have timestamp line with arrow
        assert "-->" in lines[1]

        # Should have some text content
        assert len(lines[2].strip()) > 0

    def test_process_video_cleans_up_temp_files(self, tmp_path: Path) -> None:
        """process_video cleans up temporary audio files."""
        from scripts.pipeline import process_video

        output_path = tmp_path / "output.srt"

        # Get list of files in temp directory before
        temp_dir = tempfile.gettempdir()
        wav_files_before = set(
            f for f in os.listdir(temp_dir) if f.endswith(".wav")
        )

        process_video(
            TEST_VIDEO_PATH,
            output_path=str(output_path),
            model_size="tiny",
        )

        # Get list of WAV files after
        wav_files_after = set(
            f for f in os.listdir(temp_dir) if f.endswith(".wav")
        )

        # No new WAV files should remain in temp directory
        new_wav_files = wav_files_after - wav_files_before
        assert len(new_wav_files) == 0, f"Temp WAV files not cleaned up: {new_wav_files}"

    def test_process_video_default_output_path(self, tmp_path: Path) -> None:
        """process_video creates SRT in same directory as video when output_path is None."""
        from scripts.pipeline import process_video
        import shutil

        # Copy test video to tmp_path to avoid creating SRT in input directory
        video_copy = tmp_path / "test_video.mov"
        shutil.copy(TEST_VIDEO_PATH, video_copy)

        result = process_video(str(video_copy), model_size="tiny")

        expected_srt = str(tmp_path / "test_video.srt")
        assert result == expected_srt
        assert os.path.exists(result)

    def test_process_video_with_language_parameter(self, tmp_path: Path) -> None:
        """process_video works with explicit language parameter."""
        from scripts.pipeline import process_video

        output_path = tmp_path / "output.srt"

        # Should not raise an error when language is specified
        result = process_video(
            TEST_VIDEO_PATH,
            output_path=str(output_path),
            model_size="tiny",
            language="en",
        )

        assert os.path.exists(result)
        # Should have some content
        with open(result, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 0
