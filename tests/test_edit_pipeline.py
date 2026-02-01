"""Tests for the edit_pipeline module."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.edit_decision import EditAction, EditDecisionList, EditSegment
from scripts.transcription import TranscriptSegment


# Test fixtures
@pytest.fixture
def sample_transcript_segments() -> list[TranscriptSegment]:
    """Sample transcript segments for testing."""
    return [
        TranscriptSegment(start=0.0, end=5.0, text="Hello everyone"),
        TranscriptSegment(start=5.0, end=10.0, text="Umm, let me think"),
        TranscriptSegment(start=10.0, end=15.0, text="So the answer is yes"),
    ]


@pytest.fixture
def sample_srt_content() -> str:
    """Sample SRT file content."""
    return """1
00:00:00,000 --> 00:00:05,000
Hello everyone

2
00:00:05,000 --> 00:00:10,000
Umm, let me think

3
00:00:10,000 --> 00:00:15,000
So the answer is yes"""


@pytest.fixture
def sample_edl() -> EditDecisionList:
    """Sample EDL for testing."""
    return EditDecisionList(
        source_video="/path/to/video.mp4",
        segments=[
            EditSegment(
                start=0.0,
                end=5.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[0],
            ),
            EditSegment(
                start=5.0,
                end=10.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[1],
            ),
            EditSegment(
                start=10.0,
                end=15.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[2],
            ),
        ],
        total_duration=15.0,
    )


class TestLoadTranscript:
    """Tests for _load_transcript function."""

    def test_load_transcript_parses_srt_file(
        self, tmp_path: Path, sample_srt_content: str
    ) -> None:
        """_load_transcript correctly parses SRT file into TranscriptSegments."""
        from scripts.edit_pipeline import _load_transcript

        srt_path = tmp_path / "test.srt"
        srt_path.write_text(sample_srt_content)

        segments = _load_transcript(str(srt_path))

        assert len(segments) == 3
        assert segments[0].text == "Hello everyone"
        assert segments[0].start == 0.0
        assert segments[0].end == 5.0
        assert segments[1].text == "Umm, let me think"
        assert segments[2].text == "So the answer is yes"

    def test_load_transcript_handles_multiline_text(self, tmp_path: Path) -> None:
        """_load_transcript handles multiline subtitle text."""
        from scripts.edit_pipeline import _load_transcript

        srt_content = """1
00:00:00,000 --> 00:00:05,000
This is line one
And this is line two"""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content)

        segments = _load_transcript(str(srt_path))

        assert len(segments) == 1
        assert "This is line one" in segments[0].text
        assert "And this is line two" in segments[0].text

    def test_load_transcript_file_not_found(self) -> None:
        """_load_transcript raises FileNotFoundError for missing file."""
        from scripts.edit_pipeline import _load_transcript

        with pytest.raises(FileNotFoundError):
            _load_transcript("/path/to/nonexistent.srt")

    def test_load_transcript_empty_file(self, tmp_path: Path) -> None:
        """_load_transcript handles empty SRT file."""
        from scripts.edit_pipeline import _load_transcript

        srt_path = tmp_path / "empty.srt"
        srt_path.write_text("")

        segments = _load_transcript(str(srt_path))

        assert segments == []

    def test_load_transcript_parses_timestamps_correctly(self, tmp_path: Path) -> None:
        """_load_transcript correctly parses various timestamp formats."""
        from scripts.edit_pipeline import _load_transcript

        srt_content = """1
01:23:45,678 --> 02:34:56,789
Long video segment"""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content)

        segments = _load_transcript(str(srt_path))

        # 1*3600 + 23*60 + 45.678 = 5025.678
        assert abs(segments[0].start - 5025.678) < 0.001
        # 2*3600 + 34*60 + 56.789 = 9296.789
        assert abs(segments[0].end - 9296.789) < 0.001


class TestCreateInitialEdl:
    """Tests for _create_initial_edl function."""

    def test_create_initial_edl_all_keep(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_create_initial_edl creates EDL with all segments as KEEP."""
        from scripts.edit_pipeline import _create_initial_edl

        edl = _create_initial_edl(
            segments=sample_transcript_segments,
            video_path="/path/to/video.mp4",
            duration=15.0,
        )

        assert edl.source_video == "/path/to/video.mp4"
        assert edl.total_duration == 15.0
        assert len(edl.segments) == 3
        for segment in edl.segments:
            assert segment.action == EditAction.KEEP

    def test_create_initial_edl_correct_timestamps(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_create_initial_edl preserves segment timestamps."""
        from scripts.edit_pipeline import _create_initial_edl

        edl = _create_initial_edl(
            segments=sample_transcript_segments,
            video_path="/path/to/video.mp4",
            duration=15.0,
        )

        assert edl.segments[0].start == 0.0
        assert edl.segments[0].end == 5.0
        assert edl.segments[1].start == 5.0
        assert edl.segments[1].end == 10.0
        assert edl.segments[2].start == 10.0
        assert edl.segments[2].end == 15.0

    def test_create_initial_edl_transcript_indices(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_create_initial_edl sets correct transcript indices."""
        from scripts.edit_pipeline import _create_initial_edl

        edl = _create_initial_edl(
            segments=sample_transcript_segments,
            video_path="/path/to/video.mp4",
            duration=15.0,
        )

        assert edl.segments[0].transcript_indices == [0]
        assert edl.segments[1].transcript_indices == [1]
        assert edl.segments[2].transcript_indices == [2]

    def test_create_initial_edl_empty_segments(self) -> None:
        """_create_initial_edl handles empty segment list."""
        from scripts.edit_pipeline import _create_initial_edl

        edl = _create_initial_edl(
            segments=[],
            video_path="/path/to/video.mp4",
            duration=10.0,
        )

        assert edl.segments == []
        assert edl.total_duration == 10.0


class TestEditVideo:
    """Tests for edit_video function."""

    def test_edit_video_generates_transcript_when_not_provided(
        self, tmp_path: Path
    ) -> None:
        """edit_video calls process_video when no transcript_path is provided."""
        from scripts.edit_pipeline import edit_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello"""
        srt_path = tmp_path / "video.srt"
        srt_path.write_text(srt_content)

        with patch("scripts.edit_pipeline.process_video") as mock_process:
            with patch("scripts.edit_pipeline.get_video_duration", return_value=10.0):
                mock_process.return_value = str(srt_path)

                result = edit_video(str(video_path))

        mock_process.assert_called_once()
        assert "edl_path" in result
        assert "transcript_for_review" in result

    def test_edit_video_uses_provided_transcript(self, tmp_path: Path) -> None:
        """edit_video uses existing transcript when provided."""
        from scripts.edit_pipeline import edit_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello everyone"""
        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(srt_content)

        with patch("scripts.edit_pipeline.get_video_duration", return_value=10.0):
            result = edit_video(str(video_path), transcript_path=str(srt_path))

        assert "edl_path" in result
        assert "transcript_for_review" in result
        assert "Hello everyone" in result["transcript_for_review"]

    def test_edit_video_creates_edl_file(self, tmp_path: Path) -> None:
        """edit_video creates EDL JSON file."""
        from scripts.edit_pipeline import edit_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello"""
        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(srt_content)

        with patch("scripts.edit_pipeline.get_video_duration", return_value=10.0):
            result = edit_video(str(video_path), transcript_path=str(srt_path))

        edl_path = result["edl_path"]
        assert os.path.exists(edl_path)

        # Verify EDL file is valid JSON
        with open(edl_path) as f:
            edl_data = json.load(f)
        assert "source_video" in edl_data
        assert "segments" in edl_data

    def test_edit_video_returns_formatted_transcript(self, tmp_path: Path) -> None:
        """edit_video returns transcript formatted for AI review."""
        from scripts.edit_pipeline import edit_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello everyone

2
00:00:05,000 --> 00:00:10,000
Second segment"""
        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(srt_content)

        with patch("scripts.edit_pipeline.get_video_duration", return_value=10.0):
            result = edit_video(str(video_path), transcript_path=str(srt_path))

        formatted = result["transcript_for_review"]
        assert "[0]" in formatted
        assert "[1]" in formatted
        assert "Hello everyone" in formatted
        assert "Second segment" in formatted

    def test_edit_video_file_not_found(self) -> None:
        """edit_video raises FileNotFoundError for non-existent video."""
        from scripts.edit_pipeline import edit_video

        with pytest.raises(FileNotFoundError):
            edit_video("/path/to/nonexistent/video.mp4")

    def test_edit_video_returns_video_duration(self, tmp_path: Path) -> None:
        """edit_video includes video duration in result."""
        from scripts.edit_pipeline import edit_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello"""
        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(srt_content)

        with patch("scripts.edit_pipeline.get_video_duration", return_value=15.5):
            result = edit_video(str(video_path), transcript_path=str(srt_path))

        assert result["video_duration"] == 15.5

    def test_edit_video_custom_edl_path(self, tmp_path: Path) -> None:
        """edit_video uses custom edl_path when provided."""
        from scripts.edit_pipeline import edit_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello"""
        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(srt_content)

        custom_edl_path = tmp_path / "custom_edl.json"

        with patch("scripts.edit_pipeline.get_video_duration", return_value=10.0):
            result = edit_video(
                str(video_path),
                transcript_path=str(srt_path),
                edl_path=str(custom_edl_path),
            )

        assert result["edl_path"] == str(custom_edl_path)
        assert os.path.exists(custom_edl_path)

    def test_edit_video_returns_segment_count(self, tmp_path: Path) -> None:
        """edit_video includes segment count in result."""
        from scripts.edit_pipeline import edit_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
First

2
00:00:05,000 --> 00:00:10,000
Second

3
00:00:10,000 --> 00:00:15,000
Third"""
        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(srt_content)

        with patch("scripts.edit_pipeline.get_video_duration", return_value=15.0):
            result = edit_video(str(video_path), transcript_path=str(srt_path))

        assert result["segment_count"] == 3


class TestApplyEdlToVideo:
    """Tests for apply_edl_to_video function."""

    def test_apply_edl_loads_and_applies(
        self, tmp_path: Path, sample_edl: EditDecisionList
    ) -> None:
        """apply_edl_to_video loads EDL and calls cut_video."""
        from scripts.edit_decision import edl_to_json
        from scripts.edit_pipeline import apply_edl_to_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        edl_path = tmp_path / "edl.json"
        edl_path.write_text(edl_to_json(sample_edl))

        with patch("scripts.edit_pipeline.cut_video") as mock_cut:
            mock_cut.return_value = str(tmp_path / "output.mp4")

            result = apply_edl_to_video(str(video_path), str(edl_path))

        mock_cut.assert_called_once()
        assert result == str(tmp_path / "output.mp4")

    def test_apply_edl_passes_output_path(
        self, tmp_path: Path, sample_edl: EditDecisionList
    ) -> None:
        """apply_edl_to_video passes output_path to cut_video."""
        from scripts.edit_decision import edl_to_json
        from scripts.edit_pipeline import apply_edl_to_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        edl_path = tmp_path / "edl.json"
        edl_path.write_text(edl_to_json(sample_edl))

        output_path = tmp_path / "custom_output.mp4"

        with patch("scripts.edit_pipeline.cut_video") as mock_cut:
            mock_cut.return_value = str(output_path)

            result = apply_edl_to_video(
                str(video_path), str(edl_path), str(output_path)
            )

        # Verify cut_video was called with output_path
        call_args = mock_cut.call_args
        assert call_args[1].get("output_path") == str(output_path) or call_args[0][2] == str(output_path)

    def test_apply_edl_file_not_found_video(
        self, tmp_path: Path, sample_edl: EditDecisionList
    ) -> None:
        """apply_edl_to_video raises FileNotFoundError for missing video."""
        from scripts.edit_decision import edl_to_json
        from scripts.edit_pipeline import apply_edl_to_video

        edl_path = tmp_path / "edl.json"
        edl_path.write_text(edl_to_json(sample_edl))

        with pytest.raises(FileNotFoundError):
            apply_edl_to_video("/path/to/nonexistent/video.mp4", str(edl_path))

    def test_apply_edl_file_not_found_edl(self, tmp_path: Path) -> None:
        """apply_edl_to_video raises FileNotFoundError for missing EDL."""
        from scripts.edit_pipeline import apply_edl_to_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        with pytest.raises(FileNotFoundError):
            apply_edl_to_video(str(video_path), "/path/to/nonexistent/edl.json")

    def test_apply_edl_invalid_json(self, tmp_path: Path) -> None:
        """apply_edl_to_video raises error for invalid JSON."""
        from scripts.edit_pipeline import apply_edl_to_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        edl_path = tmp_path / "invalid.json"
        edl_path.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            apply_edl_to_video(str(video_path), str(edl_path))


class TestParseSrtTimestamp:
    """Tests for _parse_srt_timestamp helper function."""

    def test_parse_simple_timestamp(self) -> None:
        """_parse_srt_timestamp parses HH:MM:SS,mmm format."""
        from scripts.edit_pipeline import _parse_srt_timestamp

        result = _parse_srt_timestamp("00:00:05,000")
        assert result == 5.0

    def test_parse_with_milliseconds(self) -> None:
        """_parse_srt_timestamp correctly handles milliseconds."""
        from scripts.edit_pipeline import _parse_srt_timestamp

        result = _parse_srt_timestamp("00:00:05,500")
        assert result == 5.5

    def test_parse_with_hours_and_minutes(self) -> None:
        """_parse_srt_timestamp handles full timestamp."""
        from scripts.edit_pipeline import _parse_srt_timestamp

        result = _parse_srt_timestamp("01:30:45,123")
        # 1*3600 + 30*60 + 45.123 = 5445.123
        assert abs(result - 5445.123) < 0.001

    def test_parse_invalid_timestamp(self) -> None:
        """_parse_srt_timestamp raises ValueError for invalid format."""
        from scripts.edit_pipeline import _parse_srt_timestamp

        with pytest.raises(ValueError):
            _parse_srt_timestamp("invalid")
