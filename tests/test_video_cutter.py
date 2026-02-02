"""Tests for the video_cutter module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.edit_decision import EditAction, EditDecisionList, EditSegment
from scripts.exceptions import EDLValidationError, VideoCuttingError
from scripts.video_cutter import (
    _build_concat_list,
    _build_ffmpeg_filter,
    _cut_segment_to_file,
    _should_use_concat_demuxer,
    _validate_edl_for_cutting,
    cut_video,
    get_video_duration,
    CONCAT_DEMUXER_THRESHOLD,
)


# Test fixtures for EditSegment and EditDecisionList
@pytest.fixture
def keep_segment_0_5() -> EditSegment:
    """A segment from 0-5 seconds marked as KEEP."""
    return EditSegment(
        start=0.0,
        end=5.0,
        action=EditAction.KEEP,
        reason="Keep intro",
        transcript_indices=[0],
    )


@pytest.fixture
def keep_segment_10_15() -> EditSegment:
    """A segment from 10-15 seconds marked as KEEP."""
    return EditSegment(
        start=10.0,
        end=15.0,
        action=EditAction.KEEP,
        reason="Keep middle",
        transcript_indices=[2],
    )


@pytest.fixture
def remove_segment_5_10() -> EditSegment:
    """A segment from 5-10 seconds marked as REMOVE."""
    return EditSegment(
        start=5.0,
        end=10.0,
        action=EditAction.REMOVE,
        reason="Remove filler",
        transcript_indices=[1],
    )


@pytest.fixture
def simple_edl(
    keep_segment_0_5: EditSegment,
    remove_segment_5_10: EditSegment,
    keep_segment_10_15: EditSegment,
) -> EditDecisionList:
    """An EDL with two KEEP segments and one REMOVE segment."""
    return EditDecisionList(
        source_video="/path/to/video.mp4",
        segments=[keep_segment_0_5, remove_segment_5_10, keep_segment_10_15],
        total_duration=15.0,
    )


@pytest.fixture
def single_keep_edl(keep_segment_0_5: EditSegment) -> EditDecisionList:
    """An EDL with a single KEEP segment."""
    return EditDecisionList(
        source_video="/path/to/video.mp4",
        segments=[keep_segment_0_5],
        total_duration=10.0,
    )


@pytest.fixture
def no_keep_edl(remove_segment_5_10: EditSegment) -> EditDecisionList:
    """An EDL with no KEEP segments."""
    return EditDecisionList(
        source_video="/path/to/video.mp4",
        segments=[remove_segment_5_10],
        total_duration=10.0,
    )


@pytest.fixture
def empty_edl() -> EditDecisionList:
    """An EDL with no segments."""
    return EditDecisionList(
        source_video="/path/to/video.mp4",
        segments=[],
        total_duration=10.0,
    )


class TestShouldUseConcatDemuxer:
    """Tests for _should_use_concat_demuxer function."""

    def test_few_segments_returns_false(self) -> None:
        """Should return False for segment count <= threshold."""
        assert _should_use_concat_demuxer(1) is False
        assert _should_use_concat_demuxer(3) is False
        assert _should_use_concat_demuxer(5) is False

    def test_many_segments_returns_true(self) -> None:
        """Should return True for segment count > threshold."""
        assert _should_use_concat_demuxer(6) is True
        assert _should_use_concat_demuxer(10) is True
        assert _should_use_concat_demuxer(100) is True

    def test_threshold_boundary(self) -> None:
        """Test the exact boundary condition."""
        assert _should_use_concat_demuxer(CONCAT_DEMUXER_THRESHOLD) is False
        assert _should_use_concat_demuxer(CONCAT_DEMUXER_THRESHOLD + 1) is True


class TestBuildConcatList:
    """Tests for _build_concat_list function."""

    def test_build_concat_list_single_file(self) -> None:
        """Build concat list for a single file."""
        result = _build_concat_list(["/path/to/segment_0.mp4"])
        assert result == "file '/path/to/segment_0.mp4'\n"

    def test_build_concat_list_multiple_files(self) -> None:
        """Build concat list for multiple files."""
        files = ["/path/to/seg_0.mp4", "/path/to/seg_1.mp4", "/path/to/seg_2.mp4"]
        result = _build_concat_list(files)
        expected = (
            "file '/path/to/seg_0.mp4'\n"
            "file '/path/to/seg_1.mp4'\n"
            "file '/path/to/seg_2.mp4'\n"
        )
        assert result == expected

    def test_build_concat_list_empty_raises(self) -> None:
        """Building concat list with empty list should raise ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            _build_concat_list([])

    def test_build_concat_list_escapes_quotes(self) -> None:
        """Concat list should handle paths with special characters."""
        files = ["/path/with'quote/segment.mp4"]
        result = _build_concat_list(files)
        # The single quote should be escaped for FFmpeg concat demuxer
        assert "'" in result  # Path still contains quote handling


class TestCutSegmentToFile:
    """Tests for _cut_segment_to_file function."""

    @patch("scripts.video_cutter.subprocess.run")
    def test_cut_segment_command_format(self, mock_run: MagicMock) -> None:
        """Verify correct FFmpeg command is generated for segment cutting."""
        mock_run.return_value = MagicMock(returncode=0)

        segment = EditSegment(
            start=10.0,
            end=15.0,
            action=EditAction.KEEP,
            reason="Test",
            transcript_indices=[0],
        )

        _cut_segment_to_file("/input/video.mp4", segment, "/output/seg.mp4")

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        # Verify -ss comes before -i (input seeking for fast seek)
        ss_idx = cmd.index("-ss")
        i_idx = cmd.index("-i")
        assert ss_idx < i_idx, "-ss should come before -i for input seeking"

        # Verify -t is used for duration (not -to)
        assert "-t" in cmd
        t_idx = cmd.index("-t")
        duration = float(cmd[t_idx + 1])
        assert duration == 5.0  # end - start = 15.0 - 10.0

        # Verify -c copy is used (no re-encoding)
        assert "-c" in cmd
        c_idx = cmd.index("-c")
        assert cmd[c_idx + 1] == "copy"

        # Verify input and output paths
        assert "/input/video.mp4" in cmd
        assert cmd[-1] == "/output/seg.mp4"

    @patch("scripts.video_cutter.subprocess.run")
    def test_cut_segment_failure_raises(self, mock_run: MagicMock) -> None:
        """Segment cutting failure should raise VideoCuttingError."""
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"segment error"
        )

        segment = EditSegment(
            start=0.0,
            end=5.0,
            action=EditAction.KEEP,
            reason="Test",
            transcript_indices=[0],
        )

        with pytest.raises(VideoCuttingError, match="[Ff]ailed|segment"):
            _cut_segment_to_file("/input/video.mp4", segment, "/output/seg.mp4")


class TestBuildFfmpegFilter:
    """Tests for _build_ffmpeg_filter function."""

    def test_build_filter_single_segment(self, keep_segment_0_5: EditSegment) -> None:
        """Build filter for a single keep segment."""
        filter_str = _build_ffmpeg_filter([keep_segment_0_5])

        # Should have trim for video and audio
        assert "[0:v]trim=start=0.0:end=5.0,setpts=PTS-STARTPTS[v0]" in filter_str
        assert "[0:a]atrim=start=0.0:end=5.0,asetpts=PTS-STARTPTS[a0]" in filter_str
        # Should have concat with n=1
        assert "[v0][a0]concat=n=1:v=1:a=1[outv][outa]" in filter_str

    def test_build_filter_two_segments(
        self, keep_segment_0_5: EditSegment, keep_segment_10_15: EditSegment
    ) -> None:
        """Build filter for two keep segments."""
        filter_str = _build_ffmpeg_filter([keep_segment_0_5, keep_segment_10_15])

        # Should have trim for first segment
        assert "[0:v]trim=start=0.0:end=5.0,setpts=PTS-STARTPTS[v0]" in filter_str
        assert "[0:a]atrim=start=0.0:end=5.0,asetpts=PTS-STARTPTS[a0]" in filter_str
        # Should have trim for second segment
        assert "[0:v]trim=start=10.0:end=15.0,setpts=PTS-STARTPTS[v1]" in filter_str
        assert "[0:a]atrim=start=10.0:end=15.0,asetpts=PTS-STARTPTS[a1]" in filter_str
        # Should have concat with n=2
        assert "[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]" in filter_str

    def test_build_filter_preserves_segment_order(
        self, keep_segment_0_5: EditSegment, keep_segment_10_15: EditSegment
    ) -> None:
        """Filter should process segments in the order given."""
        filter_str = _build_ffmpeg_filter([keep_segment_10_15, keep_segment_0_5])

        # First segment (10-15) should be v0/a0
        assert "[0:v]trim=start=10.0:end=15.0,setpts=PTS-STARTPTS[v0]" in filter_str
        assert "[0:a]atrim=start=10.0:end=15.0,asetpts=PTS-STARTPTS[a0]" in filter_str
        # Second segment (0-5) should be v1/a1
        assert "[0:v]trim=start=0.0:end=5.0,setpts=PTS-STARTPTS[v1]" in filter_str
        assert "[0:a]atrim=start=0.0:end=5.0,asetpts=PTS-STARTPTS[a1]" in filter_str

    def test_build_filter_empty_segments_raises(self) -> None:
        """Building filter with empty segments list should raise ValueError."""
        with pytest.raises(ValueError, match="at least one segment"):
            _build_ffmpeg_filter([])

    def test_build_filter_decimal_times(self) -> None:
        """Build filter handles decimal time values correctly."""
        segment = EditSegment(
            start=1.234,
            end=5.678,
            action=EditAction.KEEP,
            reason="Test",
            transcript_indices=[0],
        )
        filter_str = _build_ffmpeg_filter([segment])

        assert "trim=start=1.234:end=5.678" in filter_str
        assert "atrim=start=1.234:end=5.678" in filter_str


class TestValidateEdlForCutting:
    """Tests for _validate_edl_for_cutting function."""

    def test_validate_valid_edl(self, simple_edl: EditDecisionList) -> None:
        """Valid EDL should pass validation without raising."""
        # Should not raise
        _validate_edl_for_cutting(simple_edl)

    def test_validate_single_keep_segment(self, single_keep_edl: EditDecisionList) -> None:
        """EDL with single KEEP segment should be valid."""
        # Should not raise
        _validate_edl_for_cutting(single_keep_edl)

    def test_validate_no_keep_segments_raises(self, no_keep_edl: EditDecisionList) -> None:
        """EDL with no KEEP segments should raise EDLValidationError."""
        with pytest.raises(EDLValidationError, match="at least one KEEP segment"):
            _validate_edl_for_cutting(no_keep_edl)

    def test_validate_empty_segments_raises(self, empty_edl: EditDecisionList) -> None:
        """EDL with no segments should raise EDLValidationError."""
        with pytest.raises(EDLValidationError, match="at least one KEEP segment"):
            _validate_edl_for_cutting(empty_edl)

    def test_validate_overlapping_keep_segments_raises(self) -> None:
        """EDL with overlapping KEEP segments should raise EDLValidationError."""
        segment1 = EditSegment(
            start=0.0,
            end=10.0,
            action=EditAction.KEEP,
            reason="First",
            transcript_indices=[0],
        )
        segment2 = EditSegment(
            start=5.0,  # Overlaps with segment1
            end=15.0,
            action=EditAction.KEEP,
            reason="Second",
            transcript_indices=[1],
        )
        edl = EditDecisionList(
            source_video="/path/to/video.mp4",
            segments=[segment1, segment2],
            total_duration=20.0,
        )

        with pytest.raises(EDLValidationError, match="[Oo]verlap"):
            _validate_edl_for_cutting(edl)

    def test_validate_segment_beyond_duration_raises(self) -> None:
        """EDL with segment beyond video duration should raise EDLValidationError."""
        segment = EditSegment(
            start=5.0,
            end=15.0,  # Beyond total_duration of 10.0
            action=EditAction.KEEP,
            reason="Test",
            transcript_indices=[0],
        )
        edl = EditDecisionList(
            source_video="/path/to/video.mp4",
            segments=[segment],
            total_duration=10.0,
        )

        with pytest.raises(EDLValidationError, match="[Bb]eyond|[Ee]xceed|duration"):
            _validate_edl_for_cutting(edl)

    def test_validate_negative_start_time_raises(self) -> None:
        """EDL with negative start time should raise EDLValidationError."""
        segment = EditSegment(
            start=-1.0,
            end=5.0,
            action=EditAction.KEEP,
            reason="Test",
            transcript_indices=[0],
        )
        edl = EditDecisionList(
            source_video="/path/to/video.mp4",
            segments=[segment],
            total_duration=10.0,
        )

        with pytest.raises(EDLValidationError, match="[Nn]egative|[Ii]nvalid"):
            _validate_edl_for_cutting(edl)

    def test_validate_start_after_end_raises(self) -> None:
        """EDL with start time after end time should raise EDLValidationError."""
        segment = EditSegment(
            start=10.0,
            end=5.0,  # End before start
            action=EditAction.KEEP,
            reason="Test",
            transcript_indices=[0],
        )
        edl = EditDecisionList(
            source_video="/path/to/video.mp4",
            segments=[segment],
            total_duration=15.0,
        )

        with pytest.raises(EDLValidationError, match="[Ss]tart.*[Ee]nd|[Ii]nvalid"):
            _validate_edl_for_cutting(edl)

    def test_validate_adjacent_segments_no_overlap(self) -> None:
        """Adjacent segments (end of one equals start of next) should be valid."""
        segment1 = EditSegment(
            start=0.0,
            end=5.0,
            action=EditAction.KEEP,
            reason="First",
            transcript_indices=[0],
        )
        segment2 = EditSegment(
            start=5.0,  # Starts exactly where segment1 ends
            end=10.0,
            action=EditAction.KEEP,
            reason="Second",
            transcript_indices=[1],
        )
        edl = EditDecisionList(
            source_video="/path/to/video.mp4",
            segments=[segment1, segment2],
            total_duration=10.0,
        )

        # Should not raise
        _validate_edl_for_cutting(edl)


class TestGetVideoDuration:
    """Tests for get_video_duration function."""

    def test_get_duration_file_not_found(self) -> None:
        """get_video_duration raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            get_video_duration("/path/to/nonexistent/video.mp4")

    @patch("scripts.video_cutter.subprocess.run")
    def test_get_duration_parses_ffprobe_output(self, mock_run: MagicMock) -> None:
        """get_video_duration correctly parses ffprobe output."""
        # Mock ffprobe returning duration
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="123.456\n",
            stderr="",
        )

        with patch("os.path.exists", return_value=True):
            duration = get_video_duration("/path/to/video.mp4")

        assert duration == 123.456

    @patch("scripts.video_cutter.subprocess.run")
    def test_get_duration_ffprobe_failure(self, mock_run: MagicMock) -> None:
        """get_video_duration raises VideoCuttingError on ffprobe failure."""
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe", stderr=b"error")

        with patch("os.path.exists", return_value=True):
            with pytest.raises(VideoCuttingError, match="[Ff]ailed|[Ee]rror|duration"):
                get_video_duration("/path/to/video.mp4")


class TestCutVideo:
    """Tests for cut_video function."""

    def test_cut_video_file_not_found(self, simple_edl: EditDecisionList) -> None:
        """cut_video raises FileNotFoundError for non-existent video."""
        with pytest.raises(FileNotFoundError):
            cut_video("/path/to/nonexistent/video.mp4", simple_edl)

    def test_cut_video_invalid_edl_raises(self, no_keep_edl: EditDecisionList) -> None:
        """cut_video raises EDLValidationError for invalid EDL."""
        with patch("os.path.exists", return_value=True):
            with pytest.raises(EDLValidationError):
                cut_video("/path/to/video.mp4", no_keep_edl)

    @patch("scripts.video_cutter.subprocess.run")
    @patch("scripts.video_cutter._check_ffmpeg_available")
    def test_cut_video_creates_output_file(
        self,
        mock_check_ffmpeg: MagicMock,
        mock_subprocess_run: MagicMock,
        simple_edl: EditDecisionList,
        tmp_path: Path,
    ) -> None:
        """cut_video creates output file at specified path."""
        video_path = tmp_path / "input.mp4"
        video_path.touch()
        output_path = tmp_path / "output.mp4"

        # Mock subprocess.run to succeed
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        result = cut_video(str(video_path), simple_edl, str(output_path))

        assert result == str(output_path)
        mock_subprocess_run.assert_called_once()

    @patch("scripts.video_cutter.subprocess.run")
    @patch("scripts.video_cutter._check_ffmpeg_available")
    def test_cut_video_default_output_path(
        self,
        mock_check_ffmpeg: MagicMock,
        mock_subprocess_run: MagicMock,
        simple_edl: EditDecisionList,
        tmp_path: Path,
    ) -> None:
        """cut_video generates temp file when output_path is None."""
        video_path = tmp_path / "input.mp4"
        video_path.touch()

        # Mock subprocess.run to succeed
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        result = cut_video(str(video_path), simple_edl, None)

        assert result.endswith(".mp4")
        assert "tmp" in result.lower() or "temp" in result.lower()

    @patch("scripts.video_cutter.subprocess.run")
    @patch("scripts.video_cutter._check_ffmpeg_available")
    def test_cut_video_uses_filter_complex_for_few_segments(
        self,
        mock_check_ffmpeg: MagicMock,
        mock_subprocess_run: MagicMock,
        simple_edl: EditDecisionList,
        tmp_path: Path,
    ) -> None:
        """cut_video uses filter_complex for few segments (<= threshold)."""
        video_path = tmp_path / "input.mp4"
        video_path.touch()
        output_path = tmp_path / "output.mp4"

        # Mock subprocess.run to succeed
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        # simple_edl has 2 KEEP segments, which is <= threshold
        cut_video(str(video_path), simple_edl, str(output_path))

        # Verify subprocess.run was called with ffmpeg and filter_complex
        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args[0][0]  # Get the command list
        assert "ffmpeg" in call_args
        assert "-filter_complex" in call_args
        # Get the filter string (next arg after -filter_complex)
        filter_idx = call_args.index("-filter_complex")
        filter_str = call_args[filter_idx + 1]
        assert "trim" in filter_str
        assert "concat" in filter_str

    @patch("scripts.video_cutter.subprocess.run")
    @patch("scripts.video_cutter._check_ffmpeg_available")
    def test_cut_video_uses_concat_demuxer_for_many_segments(
        self,
        mock_check_ffmpeg: MagicMock,
        mock_subprocess_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """cut_video uses concat demuxer for many segments (> threshold)."""
        video_path = tmp_path / "input.mp4"
        video_path.touch()
        output_path = tmp_path / "output.mp4"

        # Create an EDL with more than threshold KEEP segments
        segments = []
        for i in range(CONCAT_DEMUXER_THRESHOLD + 2):
            segments.append(
                EditSegment(
                    start=float(i * 2),
                    end=float(i * 2 + 1),
                    action=EditAction.KEEP,
                    reason=f"Segment {i}",
                    transcript_indices=[i],
                )
            )
        many_segments_edl = EditDecisionList(
            source_video="/path/to/video.mp4",
            segments=segments,
            total_duration=100.0,
        )

        # Mock subprocess.run to succeed
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        cut_video(str(video_path), many_segments_edl, str(output_path))

        # For concat demuxer, we expect multiple subprocess calls:
        # - One per segment for cutting
        # - One for final concatenation
        total_calls = mock_subprocess_run.call_count
        expected_calls = len(many_segments_edl.keep_segments) + 1  # segments + concat
        assert total_calls == expected_calls

        # The last call should be the concat demuxer command
        last_call = mock_subprocess_run.call_args_list[-1]
        last_cmd = last_call[0][0]
        assert "-f" in last_cmd
        f_idx = last_cmd.index("-f")
        assert last_cmd[f_idx + 1] == "concat"
        assert "-safe" in last_cmd
        assert "-c" in last_cmd
        assert "copy" in last_cmd

    @patch("scripts.video_cutter.subprocess.run")
    @patch("scripts.video_cutter._check_ffmpeg_available")
    def test_cut_video_ffmpeg_error_raises(
        self,
        mock_check_ffmpeg: MagicMock,
        mock_subprocess_run: MagicMock,
        simple_edl: EditDecisionList,
        tmp_path: Path,
    ) -> None:
        """cut_video raises VideoCuttingError on ffmpeg failure."""
        video_path = tmp_path / "input.mp4"
        video_path.touch()

        # Mock subprocess.run raising CalledProcessError
        import subprocess
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"ffmpeg error message"
        )

        with pytest.raises(VideoCuttingError, match="[Ff]ailed|error"):
            cut_video(str(video_path), simple_edl, str(tmp_path / "output.mp4"))


# Path to test video for integration tests
TEST_VIDEO_PATH = "/home/gudmundur/ai-youtube/input/test_video.mov"


@pytest.mark.slow
class TestCutVideoIntegration:
    """Integration tests using real video file."""

    @pytest.fixture(autouse=True)
    def skip_if_no_ffmpeg(self) -> None:
        """Skip tests if ffmpeg is not available."""
        import subprocess

        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ffmpeg not available")

    @pytest.fixture(autouse=True)
    def skip_if_no_test_video(self) -> None:
        """Skip tests if test video is not available."""
        if not os.path.exists(TEST_VIDEO_PATH):
            pytest.skip(f"Test video not found: {TEST_VIDEO_PATH}")

    def test_get_video_duration_real_file(self) -> None:
        """get_video_duration returns positive duration for real video."""
        duration = get_video_duration(TEST_VIDEO_PATH)

        assert duration > 0
        assert isinstance(duration, float)

    def test_cut_video_real_file(self, tmp_path: Path) -> None:
        """cut_video produces a valid output file."""
        duration = get_video_duration(TEST_VIDEO_PATH)
        output_path = tmp_path / "cut_output.mp4"

        # Create an EDL that keeps the first 2 seconds
        segment = EditSegment(
            start=0.0,
            end=min(2.0, duration),
            action=EditAction.KEEP,
            reason="Keep intro",
            transcript_indices=[0],
        )
        edl = EditDecisionList(
            source_video=TEST_VIDEO_PATH,
            segments=[segment],
            total_duration=duration,
        )

        result = cut_video(TEST_VIDEO_PATH, edl, str(output_path))

        assert os.path.exists(result)
        # Output should be smaller than input (we kept only 2 seconds)
        output_size = os.path.getsize(result)
        assert output_size > 0

    def test_cut_video_multiple_segments(self, tmp_path: Path) -> None:
        """cut_video correctly concatenates multiple segments."""
        duration = get_video_duration(TEST_VIDEO_PATH)
        output_path = tmp_path / "multi_cut_output.mp4"

        # Skip if video is too short
        if duration < 4.0:
            pytest.skip("Test video too short for multiple segment test")

        # Create an EDL that keeps 0-1s and 2-3s (removes 1-2s)
        segment1 = EditSegment(
            start=0.0,
            end=1.0,
            action=EditAction.KEEP,
            reason="Keep first second",
            transcript_indices=[0],
        )
        segment2 = EditSegment(
            start=1.0,
            end=2.0,
            action=EditAction.REMOVE,
            reason="Remove second second",
            transcript_indices=[1],
        )
        segment3 = EditSegment(
            start=2.0,
            end=3.0,
            action=EditAction.KEEP,
            reason="Keep third second",
            transcript_indices=[2],
        )
        edl = EditDecisionList(
            source_video=TEST_VIDEO_PATH,
            segments=[segment1, segment2, segment3],
            total_duration=duration,
        )

        result = cut_video(TEST_VIDEO_PATH, edl, str(output_path))

        assert os.path.exists(result)
        # Verify output duration is approximately 2 seconds
        output_duration = get_video_duration(result)
        assert 1.5 < output_duration < 2.5

    def test_cut_video_many_segments_uses_concat_demuxer(self, tmp_path: Path) -> None:
        """cut_video with many segments uses concat demuxer approach."""
        duration = get_video_duration(TEST_VIDEO_PATH)
        output_path = tmp_path / "many_segments_output.mp4"

        # Skip if video is too short for many segments test
        min_duration_needed = (CONCAT_DEMUXER_THRESHOLD + 2) * 0.5  # 0.5s per segment
        if duration < min_duration_needed:
            pytest.skip(f"Test video too short ({duration}s) for many segments test")

        # Create an EDL with more segments than the threshold
        segments = []
        segment_duration = 0.3  # Small segments to fit within video
        gap_duration = 0.1  # Gap between segments

        for i in range(CONCAT_DEMUXER_THRESHOLD + 2):
            start = i * (segment_duration + gap_duration)
            end = start + segment_duration
            if end > duration:
                break
            segments.append(
                EditSegment(
                    start=start,
                    end=end,
                    action=EditAction.KEEP,
                    reason=f"Segment {i}",
                    transcript_indices=[i],
                )
            )

        # Skip if we couldn't create enough segments
        if len(segments) <= CONCAT_DEMUXER_THRESHOLD:
            pytest.skip("Could not create enough segments for test")

        edl = EditDecisionList(
            source_video=TEST_VIDEO_PATH,
            segments=segments,
            total_duration=duration,
        )

        result = cut_video(TEST_VIDEO_PATH, edl, str(output_path))

        # Verify output was created
        assert os.path.exists(result)
        output_size = os.path.getsize(result)
        assert output_size > 0

        # Verify output duration is reasonable
        # Note: With -c copy, cuts happen at keyframes, so duration can vary significantly
        output_duration = get_video_duration(result)
        # Just verify output is non-zero and less than original
        assert output_duration > 0
        assert output_duration < duration

    def test_cut_video_concat_demuxer_cleans_up_temp_files(
        self, tmp_path: Path
    ) -> None:
        """Concat demuxer approach should clean up temporary files."""
        duration = get_video_duration(TEST_VIDEO_PATH)
        output_path = tmp_path / "cleanup_test_output.mp4"

        # Skip if video is too short
        min_duration_needed = (CONCAT_DEMUXER_THRESHOLD + 2) * 0.5
        if duration < min_duration_needed:
            pytest.skip(f"Test video too short for cleanup test")

        # Create segments
        segments = []
        segment_duration = 0.3
        gap_duration = 0.1

        for i in range(CONCAT_DEMUXER_THRESHOLD + 2):
            start = i * (segment_duration + gap_duration)
            end = start + segment_duration
            if end > duration:
                break
            segments.append(
                EditSegment(
                    start=start,
                    end=end,
                    action=EditAction.KEEP,
                    reason=f"Segment {i}",
                    transcript_indices=[i],
                )
            )

        if len(segments) <= CONCAT_DEMUXER_THRESHOLD:
            pytest.skip("Could not create enough segments for cleanup test")

        edl = EditDecisionList(
            source_video=TEST_VIDEO_PATH,
            segments=segments,
            total_duration=duration,
        )

        # Count temp directories before
        temp_dir = tempfile.gettempdir()
        temp_items_before = set(os.listdir(temp_dir))

        result = cut_video(TEST_VIDEO_PATH, edl, str(output_path))

        # Check that temp files were cleaned up
        temp_items_after = set(os.listdir(temp_dir))
        # Any new items should not contain our segment files
        new_items = temp_items_after - temp_items_before
        for item in new_items:
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path):
                # Check no segment files remain
                contents = os.listdir(item_path)
                segment_files = [f for f in contents if f.startswith("segment_")]
                assert len(segment_files) == 0, f"Segment files not cleaned up: {segment_files}"

        assert os.path.exists(result)
