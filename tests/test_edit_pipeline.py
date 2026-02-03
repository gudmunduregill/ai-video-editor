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
        assert result["video_path"] == str(tmp_path / "output.mp4")
        assert "srt_path" not in result

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


class TestIterSrtSegments:
    """Tests for _iter_srt_segments streaming parser."""

    def test_iter_srt_segments_yields_segments(
        self, tmp_path: Path, sample_srt_content: str
    ) -> None:
        """_iter_srt_segments yields TranscriptSegment objects."""
        from scripts.edit_pipeline import _iter_srt_segments

        srt_path = tmp_path / "test.srt"
        srt_path.write_text(sample_srt_content)

        segments = list(_iter_srt_segments(str(srt_path)))

        assert len(segments) == 3
        assert segments[0].text == "Hello everyone"
        assert segments[1].text == "Umm, let me think"
        assert segments[2].text == "So the answer is yes"

    def test_iter_srt_segments_is_generator(
        self, tmp_path: Path, sample_srt_content: str
    ) -> None:
        """_iter_srt_segments returns a generator, not a list."""
        from scripts.edit_pipeline import _iter_srt_segments
        from types import GeneratorType

        srt_path = tmp_path / "test.srt"
        srt_path.write_text(sample_srt_content)

        result = _iter_srt_segments(str(srt_path))

        assert isinstance(result, GeneratorType)

    def test_iter_srt_segments_handles_no_trailing_newline(
        self, tmp_path: Path
    ) -> None:
        """_iter_srt_segments handles file without trailing blank line."""
        from scripts.edit_pipeline import _iter_srt_segments

        # SRT content without trailing blank line
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello everyone"""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content)

        segments = list(_iter_srt_segments(str(srt_path)))

        assert len(segments) == 1
        assert segments[0].text == "Hello everyone"

    def test_iter_srt_segments_file_not_found(self) -> None:
        """_iter_srt_segments raises FileNotFoundError for missing file."""
        from scripts.edit_pipeline import _iter_srt_segments

        with pytest.raises(FileNotFoundError):
            # Need to consume the generator to trigger the error
            list(_iter_srt_segments("/path/to/nonexistent.srt"))

    def test_iter_srt_segments_empty_file(self, tmp_path: Path) -> None:
        """_iter_srt_segments handles empty SRT file."""
        from scripts.edit_pipeline import _iter_srt_segments

        srt_path = tmp_path / "empty.srt"
        srt_path.write_text("")

        segments = list(_iter_srt_segments(str(srt_path)))

        assert segments == []

    def test_iter_srt_segments_handles_extra_blank_lines(
        self, tmp_path: Path
    ) -> None:
        """_iter_srt_segments handles SRT with extra blank lines."""
        from scripts.edit_pipeline import _iter_srt_segments

        srt_content = """

1
00:00:00,000 --> 00:00:05,000
First segment


2
00:00:05,000 --> 00:00:10,000
Second segment

"""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content)

        segments = list(_iter_srt_segments(str(srt_path)))

        assert len(segments) == 2
        assert segments[0].text == "First segment"
        assert segments[1].text == "Second segment"


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


class TestParseAiResponse:
    """Tests for _parse_ai_response function."""

    def test_parse_ai_response_basic(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_parse_ai_response parses basic KEEP/REMOVE format."""
        from scripts.edit_pipeline import _parse_ai_response

        ai_response = """
[KEEP] 0: Introduction
[REMOVE] 1: "Umm, let me think" - Filler
[KEEP] 2: Main content
"""
        result = _parse_ai_response(ai_response, sample_transcript_segments)

        assert len(result) == 3
        assert result[0].action == EditAction.KEEP
        assert result[0].transcript_indices == [0]
        assert result[1].action == EditAction.REMOVE
        assert result[1].transcript_indices == [1]
        assert result[1].reason == '"Umm, let me think" - Filler'
        assert result[2].action == EditAction.KEEP
        assert result[2].transcript_indices == [2]

    def test_parse_ai_response_range_indices(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_parse_ai_response handles index ranges like 0-2."""
        from scripts.edit_pipeline import _parse_ai_response

        ai_response = "[KEEP] 0-2: All content"

        result = _parse_ai_response(ai_response, sample_transcript_segments)

        assert len(result) == 1
        assert result[0].transcript_indices == [0, 1, 2]
        assert result[0].start == 0.0
        assert result[0].end == 15.0  # End of segment 2

    def test_parse_ai_response_treats_review_as_keep(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_parse_ai_response treats [REVIEW] as [KEEP]."""
        from scripts.edit_pipeline import _parse_ai_response

        ai_response = "[REVIEW] 0: Borderline content"

        result = _parse_ai_response(ai_response, sample_transcript_segments)

        assert len(result) == 1
        assert result[0].action == EditAction.KEEP
        assert result[0].reason is None  # KEEP/REVIEW don't have reasons

    def test_parse_ai_response_handles_empty_response(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_parse_ai_response returns empty list for empty response."""
        from scripts.edit_pipeline import _parse_ai_response

        result = _parse_ai_response("", sample_transcript_segments)
        assert result == []

        result = _parse_ai_response("   ", sample_transcript_segments)
        assert result == []

    def test_parse_ai_response_skips_invalid_indices(
        self, sample_transcript_segments: list[TranscriptSegment], capsys: pytest.CaptureFixture
    ) -> None:
        """_parse_ai_response skips and warns about invalid indices."""
        from scripts.edit_pipeline import _parse_ai_response

        ai_response = """
[KEEP] 0: Valid
[KEEP] 99: Invalid index
[KEEP] 2: Also valid
"""
        result = _parse_ai_response(ai_response, sample_transcript_segments)

        assert len(result) == 2  # Skipped invalid index 99
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "99" in captured.err

    def test_parse_ai_response_ignores_non_matching_lines(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_parse_ai_response ignores lines that don't match the pattern."""
        from scripts.edit_pipeline import _parse_ai_response

        ai_response = """
## Video Context
This is a test video

## Edit Summary
2 segments to keep, 1 to remove

[KEEP] 0: Intro
Random text that should be ignored
[REMOVE] 1: Filler - retake
[KEEP] 2: Content
"""
        result = _parse_ai_response(ai_response, sample_transcript_segments)

        assert len(result) == 3

    def test_parse_ai_response_case_insensitive(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_parse_ai_response handles case-insensitive action keywords."""
        from scripts.edit_pipeline import _parse_ai_response

        ai_response = """
[keep] 0: Lowercase
[KEEP] 1: Uppercase
[Keep] 2: Mixed case
"""
        result = _parse_ai_response(ai_response, sample_transcript_segments)

        assert len(result) == 3
        for seg in result:
            assert seg.action == EditAction.KEEP


class TestAnalyzeWithAi:
    """Tests for _analyze_with_ai function."""

    def test_analyze_with_ai_returns_empty_when_not_using_ai(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_analyze_with_ai returns empty list and None response when use_ai=False."""
        from scripts.edit_pipeline import _analyze_with_ai

        segments, raw_response = _analyze_with_ai(
            transcript="[0] 0-5: Hello",
            segments=sample_transcript_segments,
            use_ai=False,
        )

        assert segments == []
        assert raw_response is None

    def test_analyze_with_ai_calls_llm(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_analyze_with_ai calls LLM and parses response when use_ai=True."""
        from scripts.edit_pipeline import _analyze_with_ai

        mock_response = "[KEEP] 0-2: All content"

        with patch("scripts.edit_pipeline.load_agent_prompt") as mock_load:
            with patch("scripts.edit_pipeline.analyze_transcript") as mock_analyze:
                mock_load.return_value = "Test prompt"
                mock_analyze.return_value = mock_response

                segments, raw_response = _analyze_with_ai(
                    transcript="[0] 0-5: Hello",
                    segments=sample_transcript_segments,
                    use_ai=True,
                )

        mock_load.assert_called_once_with("video-editor")
        mock_analyze.assert_called_once()
        assert len(segments) == 1
        assert segments[0].transcript_indices == [0, 1, 2]
        assert raw_response == mock_response

    def test_analyze_with_ai_propagates_llm_error(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_analyze_with_ai propagates LLMClientError."""
        from scripts.edit_pipeline import _analyze_with_ai
        from scripts.llm_client import LLMClientError

        with patch("scripts.edit_pipeline.load_agent_prompt") as mock_load:
            mock_load.side_effect = LLMClientError("API key not set")

            with pytest.raises(LLMClientError):
                _analyze_with_ai(
                    transcript="[0] 0-5: Hello",
                    segments=sample_transcript_segments,
                    use_ai=True,
                )


class TestCreateEdlFromAiSegments:
    """Tests for _create_edl_from_ai_segments function."""

    def test_create_edl_fills_gaps_with_keep(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_create_edl_from_ai_segments fills uncovered indices with KEEP."""
        from scripts.edit_pipeline import _create_edl_from_ai_segments

        # AI only covers segment 1 with REMOVE
        ai_segments = [
            EditSegment(
                start=5.0,
                end=10.0,
                action=EditAction.REMOVE,
                reason="Filler",
                transcript_indices=[1],
            )
        ]

        edl = _create_edl_from_ai_segments(
            ai_segments=ai_segments,
            all_segments=sample_transcript_segments,
            video_path="/path/to/video.mp4",
            duration=15.0,
        )

        # Should have 3 segments: KEEP[0], REMOVE[1], KEEP[2]
        assert len(edl.segments) == 3
        assert edl.segments[0].action == EditAction.KEEP
        assert edl.segments[0].transcript_indices == [0]
        assert edl.segments[1].action == EditAction.REMOVE
        assert edl.segments[1].transcript_indices == [1]
        assert edl.segments[2].action == EditAction.KEEP
        assert edl.segments[2].transcript_indices == [2]

    def test_create_edl_sorts_by_start_time(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_create_edl_from_ai_segments sorts segments by start time."""
        from scripts.edit_pipeline import _create_edl_from_ai_segments

        # AI segments in reverse order
        ai_segments = [
            EditSegment(
                start=10.0,
                end=15.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[2],
            ),
            EditSegment(
                start=0.0,
                end=5.0,
                action=EditAction.REMOVE,
                reason="Filler",
                transcript_indices=[0],
            ),
        ]

        edl = _create_edl_from_ai_segments(
            ai_segments=ai_segments,
            all_segments=sample_transcript_segments,
            video_path="/path/to/video.mp4",
            duration=15.0,
        )

        # Should be sorted by start time
        assert edl.segments[0].start == 0.0
        assert edl.segments[1].start == 5.0  # Gap filled
        assert edl.segments[2].start == 10.0

    def test_create_edl_merges_consecutive_gaps(
        self, sample_transcript_segments: list[TranscriptSegment]
    ) -> None:
        """_create_edl_from_ai_segments handles consecutive gap indices."""
        from scripts.edit_pipeline import _create_edl_from_ai_segments

        # No AI segments - all gaps
        ai_segments: list[EditSegment] = []

        edl = _create_edl_from_ai_segments(
            ai_segments=ai_segments,
            all_segments=sample_transcript_segments,
            video_path="/path/to/video.mp4",
            duration=15.0,
        )

        # Should have one KEEP segment covering all
        assert len(edl.segments) == 1
        assert edl.segments[0].action == EditAction.KEEP
        assert edl.segments[0].transcript_indices == [0, 1, 2]


class TestEditVideoWithAi:
    """Tests for edit_video function with use_ai parameter."""

    def test_edit_video_with_ai_false_creates_all_keep(self, tmp_path: Path) -> None:
        """edit_video with use_ai=False creates all-KEEP EDL (existing behavior)."""
        from scripts.edit_pipeline import edit_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello

2
00:00:05,000 --> 00:00:10,000
World"""
        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(srt_content)

        with patch("scripts.edit_pipeline.get_video_duration", return_value=10.0):
            result = edit_video(
                str(video_path),
                transcript_path=str(srt_path),
                use_ai=False,
            )

        assert result["ai_used"] is False

        # Load EDL and verify all KEEP
        with open(result["edl_path"]) as f:
            edl_data = json.load(f)

        for seg in edl_data["segments"]:
            assert seg["action"] == "keep"

    def test_edit_video_with_ai_true_calls_llm(self, tmp_path: Path) -> None:
        """edit_video with use_ai=True uses AI analysis."""
        from scripts.edit_pipeline import edit_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello

2
00:00:05,000 --> 00:00:10,000
Um let me try again"""
        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(srt_content)

        mock_ai_response = """
[KEEP] 0: Intro
[REMOVE] 1: Retake
"""

        with patch("scripts.edit_pipeline.get_video_duration", return_value=10.0):
            with patch("scripts.edit_pipeline.load_agent_prompt", return_value="Test"):
                with patch("scripts.edit_pipeline.analyze_transcript", return_value=mock_ai_response):
                    result = edit_video(
                        str(video_path),
                        transcript_path=str(srt_path),
                        use_ai=True,
                    )

        assert result["ai_used"] is True

        # Load EDL and verify AI decisions
        with open(result["edl_path"]) as f:
            edl_data = json.load(f)

        assert len(edl_data["segments"]) == 2
        assert edl_data["segments"][0]["action"] == "keep"
        assert edl_data["segments"][1]["action"] == "remove"

    def test_edit_video_falls_back_on_parse_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """edit_video falls back to all-KEEP when AI response parsing fails."""
        from scripts.edit_pipeline import edit_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello"""
        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(srt_content)

        # AI returns unparseable response
        mock_ai_response = "I don't understand the request"

        with patch("scripts.edit_pipeline.get_video_duration", return_value=5.0):
            with patch("scripts.edit_pipeline.load_agent_prompt", return_value="Test"):
                with patch("scripts.edit_pipeline.analyze_transcript", return_value=mock_ai_response):
                    result = edit_video(
                        str(video_path),
                        transcript_path=str(srt_path),
                        use_ai=True,
                    )

        # Should fall back to all-KEEP
        with open(result["edl_path"]) as f:
            edl_data = json.load(f)

        assert edl_data["segments"][0]["action"] == "keep"

        # Should warn about fallback and show response preview
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "Falling back" in captured.err
        assert "AI response preview" in captured.err
        assert "I don't understand" in captured.err

    def test_edit_video_warns_when_all_remove(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """edit_video warns when AI suggests removing all segments."""
        from scripts.edit_pipeline import edit_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello"""
        srt_path = tmp_path / "transcript.srt"
        srt_path.write_text(srt_content)

        # AI suggests removing everything
        mock_ai_response = "[REMOVE] 0: Bad take"

        with patch("scripts.edit_pipeline.get_video_duration", return_value=5.0):
            with patch("scripts.edit_pipeline.load_agent_prompt", return_value="Test"):
                with patch("scripts.edit_pipeline.analyze_transcript", return_value=mock_ai_response):
                    edit_video(
                        str(video_path),
                        transcript_path=str(srt_path),
                        use_ai=True,
                    )

        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "removing all segments" in captured.err


class TestApplyEdlToVideoWithSrt:
    """Tests for apply_edl_to_video with SRT file support."""

    def test_apply_edl_with_srt_generates_adjusted_srt(
        self, tmp_path: Path, sample_edl: EditDecisionList
    ) -> None:
        """apply_edl_to_video with srt_path generates adjusted SRT file."""
        from scripts.edit_decision import edl_to_json
        from scripts.edit_pipeline import apply_edl_to_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        edl_path = tmp_path / "edl.json"
        edl_path.write_text(edl_to_json(sample_edl))

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello"""
        srt_path = tmp_path / "input.srt"
        srt_path.write_text(srt_content)

        with patch("scripts.edit_pipeline.cut_video") as mock_cut:
            with patch("scripts.edit_pipeline.adjust_srt_for_edl") as mock_adjust_srt:
                mock_cut.return_value = str(tmp_path / "output.mp4")
                mock_adjust_srt.return_value = str(tmp_path / "output.srt")

                result = apply_edl_to_video(
                    str(video_path), str(edl_path), srt_path=str(srt_path)
                )

        # Should have called both cut_video and adjust_srt_for_edl
        mock_cut.assert_called_once()
        mock_adjust_srt.assert_called_once()

        # Result should be a dict with both paths
        assert isinstance(result, dict)
        assert "video_path" in result
        assert "srt_path" in result
        assert result["video_path"] == str(tmp_path / "output.mp4")
        assert result["srt_path"] == str(tmp_path / "output.srt")

    def test_apply_edl_without_srt_returns_dict_with_video_path_only(
        self, tmp_path: Path, sample_edl: EditDecisionList
    ) -> None:
        """apply_edl_to_video without srt_path returns dict with video_path only."""
        from scripts.edit_decision import edl_to_json
        from scripts.edit_pipeline import apply_edl_to_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        edl_path = tmp_path / "edl.json"
        edl_path.write_text(edl_to_json(sample_edl))

        with patch("scripts.edit_pipeline.cut_video") as mock_cut:
            mock_cut.return_value = str(tmp_path / "output.mp4")

            result = apply_edl_to_video(str(video_path), str(edl_path))

        # Should return dict with video_path, no srt_path
        assert isinstance(result, dict)
        assert result["video_path"] == str(tmp_path / "output.mp4")
        assert "srt_path" not in result

    def test_apply_edl_with_srt_generates_correct_output_path(
        self, tmp_path: Path, sample_edl: EditDecisionList
    ) -> None:
        """apply_edl_to_video generates SRT output path based on video output path."""
        from scripts.edit_decision import edl_to_json
        from scripts.edit_pipeline import apply_edl_to_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        edl_path = tmp_path / "edl.json"
        edl_path.write_text(edl_to_json(sample_edl))

        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello"""
        srt_path = tmp_path / "input.srt"
        srt_path.write_text(srt_content)

        output_video_path = str(tmp_path / "custom_output.mp4")

        with patch("scripts.edit_pipeline.cut_video") as mock_cut:
            with patch("scripts.edit_pipeline.adjust_srt_for_edl") as mock_adjust_srt:
                mock_cut.return_value = output_video_path
                # Capture the output_path argument passed to adjust_srt_for_edl
                def capture_call(srt_input, edl, output_path):
                    return output_path
                mock_adjust_srt.side_effect = capture_call

                result = apply_edl_to_video(
                    str(video_path), str(edl_path),
                    output_path=output_video_path,
                    srt_path=str(srt_path)
                )

        # The SRT output path should match the video output path but with .srt extension
        expected_srt_path = str(tmp_path / "custom_output.srt")
        assert result["srt_path"] == expected_srt_path

    def test_apply_edl_with_srt_file_not_found(
        self, tmp_path: Path, sample_edl: EditDecisionList
    ) -> None:
        """apply_edl_to_video raises FileNotFoundError for missing SRT file."""
        from scripts.edit_decision import edl_to_json
        from scripts.edit_pipeline import apply_edl_to_video

        video_path = tmp_path / "video.mp4"
        video_path.touch()

        edl_path = tmp_path / "edl.json"
        edl_path.write_text(edl_to_json(sample_edl))

        with patch("scripts.edit_pipeline.cut_video") as mock_cut:
            mock_cut.return_value = str(tmp_path / "output.mp4")

            with pytest.raises(FileNotFoundError):
                apply_edl_to_video(
                    str(video_path), str(edl_path), srt_path="/nonexistent/file.srt"
                )
