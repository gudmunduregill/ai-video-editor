"""Tests for the audio_extractor module."""

import os
import tempfile
import wave
from pathlib import Path

import pytest

from scripts.audio_extractor import extract_audio
from scripts.exceptions import AudioExtractionError


# Path to the test video file
TEST_VIDEO_PATH = "/home/gudmundur/ai-youtube/input/test_video.mov"


class TestExtractAudioBasic:
    """Basic unit tests for extract_audio function."""

    def test_extract_audio_video_not_found(self) -> None:
        """extract_audio raises FileNotFoundError when video file doesn't exist."""
        non_existent_path = "/path/to/nonexistent/video.mp4"

        with pytest.raises(FileNotFoundError) as exc_info:
            extract_audio(non_existent_path)

        assert "video.mp4" in str(exc_info.value) or "not found" in str(exc_info.value).lower()

    def test_extract_audio_invalid_file_raises_error(self, tmp_path: Path) -> None:
        """extract_audio raises AudioExtractionError for invalid video file."""
        # Create an invalid video file (just a text file with .mp4 extension)
        invalid_video = tmp_path / "invalid.mp4"
        invalid_video.write_text("This is not a video file")

        with pytest.raises(AudioExtractionError):
            extract_audio(str(invalid_video))


@pytest.mark.slow
class TestExtractAudioIntegration:
    """Integration tests using real video file."""

    @pytest.fixture(autouse=True)
    def skip_if_no_ffmpeg(self) -> None:
        """Skip tests if ffmpeg is not available."""
        import subprocess
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ffmpeg not available")

    @pytest.fixture(autouse=True)
    def skip_if_no_test_video(self) -> None:
        """Skip tests if test video is not available."""
        if not os.path.exists(TEST_VIDEO_PATH):
            pytest.skip(f"Test video not found: {TEST_VIDEO_PATH}")

    def test_extract_audio_creates_file(self, tmp_path: Path) -> None:
        """extract_audio creates an output audio file."""
        output_path = tmp_path / "output.wav"

        result = extract_audio(TEST_VIDEO_PATH, str(output_path))

        assert os.path.exists(result)
        assert result == str(output_path)

    def test_extract_audio_default_output_path(self) -> None:
        """extract_audio creates a temp file when output_path is None."""
        result = extract_audio(TEST_VIDEO_PATH, None)

        try:
            assert os.path.exists(result)
            assert result.endswith(".wav")
            # Should be in a temp directory
            assert "tmp" in result.lower() or "temp" in result.lower()
        finally:
            # Clean up the temp file
            if os.path.exists(result):
                os.remove(result)

    def test_extract_audio_custom_output_path(self, tmp_path: Path) -> None:
        """extract_audio respects the provided output path."""
        custom_output = tmp_path / "custom_audio.wav"

        result = extract_audio(TEST_VIDEO_PATH, str(custom_output))

        assert result == str(custom_output)
        assert os.path.exists(str(custom_output))

    def test_extract_audio_output_is_valid_wav(self, tmp_path: Path) -> None:
        """extract_audio produces a valid WAV file with correct parameters."""
        output_path = tmp_path / "output.wav"

        result = extract_audio(TEST_VIDEO_PATH, str(output_path))

        # Verify the output is a valid WAV file by reading it
        with wave.open(result, "rb") as wav_file:
            # Check sample rate is 16kHz (optimal for Whisper)
            assert wav_file.getframerate() == 16000
            # Check mono channel
            assert wav_file.getnchannels() == 1
            # Check there's actual audio data
            assert wav_file.getnframes() > 0

    def test_extract_audio_output_has_audio_content(self, tmp_path: Path) -> None:
        """extract_audio produces a file with non-trivial size."""
        output_path = tmp_path / "output.wav"

        result = extract_audio(TEST_VIDEO_PATH, str(output_path))

        # The output file should have meaningful size (at least 1KB)
        file_size = os.path.getsize(result)
        assert file_size > 1024, f"Audio file too small: {file_size} bytes"


class TestExtractAudioEdgeCases:
    """Edge case tests for extract_audio function."""

    @pytest.fixture(autouse=True)
    def skip_if_no_ffmpeg(self) -> None:
        """Skip tests if ffmpeg is not available."""
        import subprocess
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ffmpeg not available")

    def test_extract_audio_overwrites_existing_file(self, tmp_path: Path) -> None:
        """extract_audio overwrites existing output file."""
        output_path = tmp_path / "output.wav"
        # Create an existing file
        output_path.write_text("existing content")

        if not os.path.exists(TEST_VIDEO_PATH):
            pytest.skip(f"Test video not found: {TEST_VIDEO_PATH}")

        result = extract_audio(TEST_VIDEO_PATH, str(output_path))

        assert os.path.exists(result)
        # The content should no longer be the original text
        with open(result, "rb") as f:
            content = f.read()
            assert content != b"existing content"
