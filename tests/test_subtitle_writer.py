"""Tests for the subtitle_writer module.

These tests follow TDD principles and are written BEFORE the implementation.
"""

import os
import tempfile

import pytest

from scripts.subtitle_writer import format_srt_timestamp, write_srt
from scripts.transcription import TranscriptSegment


class TestFormatSrtTimestamp:
    """Tests for the format_srt_timestamp function."""

    def test_format_srt_timestamp_zero(self) -> None:
        """Test that zero seconds formats correctly."""
        result = format_srt_timestamp(0.0)
        assert result == "00:00:00,000"

    def test_format_srt_timestamp_simple(self) -> None:
        """Test that 65.5 seconds formats to 00:01:05,500."""
        result = format_srt_timestamp(65.5)
        assert result == "00:01:05,500"

    def test_format_srt_timestamp_hours(self) -> None:
        """Test that timestamps with hours format correctly."""
        # 1 hour + 30 minutes + 45 seconds + 123 milliseconds
        seconds = 3600 + 1800 + 45 + 0.123
        result = format_srt_timestamp(seconds)
        assert result == "01:30:45,123"

    def test_format_srt_timestamp_rounds_milliseconds(self) -> None:
        """Test that milliseconds are properly rounded to 3 decimal places."""
        # 1.5555 seconds should round to 1.556 (556 milliseconds)
        result = format_srt_timestamp(1.5555)
        assert result == "00:00:01,556"

    def test_format_srt_timestamp_large_hours(self) -> None:
        """Test formatting with more than 10 hours."""
        # 12 hours + 34 minutes + 56 seconds + 789 milliseconds
        seconds = 12 * 3600 + 34 * 60 + 56 + 0.789
        result = format_srt_timestamp(seconds)
        assert result == "12:34:56,789"

    def test_format_srt_timestamp_exact_minute(self) -> None:
        """Test exact minute boundary."""
        result = format_srt_timestamp(60.0)
        assert result == "00:01:00,000"

    def test_format_srt_timestamp_exact_hour(self) -> None:
        """Test exact hour boundary."""
        result = format_srt_timestamp(3600.0)
        assert result == "01:00:00,000"


class TestWriteSrt:
    """Tests for the write_srt function."""

    def test_write_srt_creates_file(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that write_srt creates a file at the specified path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subtitles.srt")
            write_srt(sample_segments, output_path)
            assert os.path.exists(output_path)

    def test_write_srt_empty_segments_raises_error(self) -> None:
        """Test that write_srt raises ValueError for empty segments list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subtitles.srt")
            with pytest.raises(ValueError, match="empty"):
                write_srt([], output_path)

    def test_write_srt_correct_format(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that write_srt outputs correct SRT format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subtitles.srt")
            write_srt(sample_segments, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            expected = (
                "1\n"
                "00:00:00,000 --> 00:00:02,500\n"
                "Hello, world!\n"
                "\n"
                "2\n"
                "00:00:02,600 --> 00:00:05,100\n"
                "This is a test.\n"
                "\n"
                "3\n"
                "00:00:05,200 --> 00:00:08,000\n"
                "Goodbye!\n"
            )
            assert content == expected

    def test_write_srt_handles_special_characters(self) -> None:
        """Test that write_srt handles special characters in text."""
        segments = [
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="Hello <world> & \"friends\"!",
            ),
            TranscriptSegment(
                start=2.0,
                end=4.0,
                text="Line with\nnewline character",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subtitles.srt")
            write_srt(segments, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Special characters should be preserved as-is in SRT format
            assert 'Hello <world> & "friends"!' in content
            # Newlines within text should also be preserved
            assert "Line with\nnewline character" in content

    def test_write_srt_single_segment(
        self, single_segment: TranscriptSegment
    ) -> None:
        """Test write_srt with a single segment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subtitles.srt")
            write_srt([single_segment], output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            expected = (
                "1\n"
                "00:00:00,000 --> 00:00:02,500\n"
                "Hello, world!\n"
            )
            assert content == expected

    def test_write_srt_utf8_encoding(self) -> None:
        """Test that write_srt properly handles UTF-8 characters."""
        segments = [
            TranscriptSegment(start=0.0, end=2.0, text="Cafe is written as cafe"),
            TranscriptSegment(start=2.0, end=4.0, text="Chinese: \u4e2d\u6587"),
            TranscriptSegment(start=4.0, end=6.0, text="Emoji test: \u2764"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "subtitles.srt")
            write_srt(segments, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "Chinese: \u4e2d\u6587" in content
            assert "\u2764" in content
