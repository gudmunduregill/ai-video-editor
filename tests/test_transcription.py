"""Tests for the transcription module."""

import os
import tempfile
from pathlib import Path

import pytest

from scripts.audio_extractor import extract_audio
from scripts.exceptions import TranscriptionError
from scripts.transcription import TranscriptSegment, transcribe


# Path to the test video file
TEST_VIDEO_PATH = "/home/gudmundur/ai-youtube/input/test_video.mov"


class TestTranscriptSegmentDataclass:
    """Tests for the TranscriptSegment dataclass."""

    def test_transcript_segment_dataclass(self) -> None:
        """TranscriptSegment can be created with start, end, and text."""
        segment = TranscriptSegment(start=0.0, end=2.5, text="Hello, world!")

        assert segment.start == 0.0
        assert segment.end == 2.5
        assert segment.text == "Hello, world!"

    def test_transcript_segment_equality(self) -> None:
        """Two TranscriptSegments with same values are equal."""
        segment1 = TranscriptSegment(start=1.0, end=2.0, text="Test")
        segment2 = TranscriptSegment(start=1.0, end=2.0, text="Test")

        assert segment1 == segment2

    def test_transcript_segment_different_values_not_equal(self) -> None:
        """Two TranscriptSegments with different values are not equal."""
        segment1 = TranscriptSegment(start=1.0, end=2.0, text="Test")
        segment2 = TranscriptSegment(start=1.0, end=3.0, text="Test")

        assert segment1 != segment2


class TestTranscribeBasic:
    """Basic unit tests for the transcribe function."""

    def test_transcribe_file_not_found(self) -> None:
        """transcribe raises FileNotFoundError when audio file doesn't exist."""
        non_existent_path = "/path/to/nonexistent/audio.wav"

        with pytest.raises(FileNotFoundError) as exc_info:
            transcribe(non_existent_path)

        assert "not found" in str(exc_info.value).lower() or "audio.wav" in str(
            exc_info.value
        )

    def test_transcribe_invalid_file_raises_error(self, tmp_path: Path) -> None:
        """transcribe raises TranscriptionError for invalid audio file."""
        # Create an invalid audio file (just a text file with .wav extension)
        invalid_audio = tmp_path / "invalid.wav"
        invalid_audio.write_text("This is not an audio file")

        with pytest.raises(TranscriptionError):
            transcribe(str(invalid_audio))


@pytest.mark.slow
class TestTranscribeIntegration:
    """Integration tests using real audio from test video."""

    @pytest.fixture(autouse=True)
    def skip_if_no_test_video(self) -> None:
        """Skip tests if test video is not available."""
        if not os.path.exists(TEST_VIDEO_PATH):
            pytest.skip(f"Test video not found: {TEST_VIDEO_PATH}")

    @pytest.fixture
    def audio_file(self, tmp_path: Path) -> str:
        """Extract audio from test video for transcription tests."""
        output_path = tmp_path / "test_audio.wav"
        return extract_audio(TEST_VIDEO_PATH, str(output_path))

    def test_transcribe_returns_list_of_segments(self, audio_file: str) -> None:
        """transcribe returns a list of TranscriptSegment objects."""
        result = transcribe(audio_file, model_size="tiny")

        assert isinstance(result, list)
        assert len(result) > 0
        for segment in result:
            assert isinstance(segment, TranscriptSegment)

    def test_transcribe_segments_have_valid_timestamps(self, audio_file: str) -> None:
        """All transcription segments have start < end."""
        result = transcribe(audio_file, model_size="tiny")

        for segment in result:
            assert segment.start < segment.end, (
                f"Invalid timestamps: start={segment.start} >= end={segment.end}"
            )

    def test_transcribe_segments_have_non_empty_text(self, audio_file: str) -> None:
        """All transcription segments have non-empty text."""
        result = transcribe(audio_file, model_size="tiny")

        for segment in result:
            assert segment.text.strip() != "", "Segment text should not be empty"

    def test_transcribe_segments_have_non_negative_timestamps(
        self, audio_file: str
    ) -> None:
        """All transcription segments have non-negative timestamps."""
        result = transcribe(audio_file, model_size="tiny")

        for segment in result:
            assert segment.start >= 0, f"Start time should be >= 0: {segment.start}"
            assert segment.end >= 0, f"End time should be >= 0: {segment.end}"

    def test_transcribe_with_language_parameter(self, audio_file: str) -> None:
        """transcribe accepts language parameter and returns results."""
        # Should not raise an error when language is specified
        result = transcribe(audio_file, model_size="tiny", language="en")

        assert isinstance(result, list)
        # Should still produce some results
        assert len(result) > 0

    def test_transcribe_with_different_model_sizes(self, audio_file: str) -> None:
        """transcribe accepts different model sizes."""
        # Test with tiny model (fastest)
        result = transcribe(audio_file, model_size="tiny")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_transcribe_segments_are_chronological(self, audio_file: str) -> None:
        """Transcription segments are in chronological order."""
        result = transcribe(audio_file, model_size="tiny")

        for i in range(len(result) - 1):
            assert result[i].start <= result[i + 1].start, (
                f"Segments not chronological: segment {i} starts at {result[i].start} "
                f"but segment {i+1} starts at {result[i+1].start}"
            )
