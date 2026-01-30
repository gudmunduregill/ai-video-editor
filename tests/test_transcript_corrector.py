"""Tests for the transcript_corrector module.

These tests follow TDD principles and are written BEFORE the implementation.
"""

import pytest

from scripts.transcription import TranscriptSegment
from scripts.transcript_corrector import (
    correct_transcript,
    format_for_review,
    parse_corrected_transcript,
)


class TestFormatForReview:
    """Tests for the format_for_review function."""

    def test_format_includes_index_timestamps_and_text(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that formatted output includes segment index, timestamps, and text."""
        result = format_for_review(sample_segments)

        # Check that all segments are represented with their data
        assert "[0]" in result
        assert "[1]" in result
        assert "[2]" in result

        # Check timestamps are included
        assert "0.0" in result or "0.00" in result
        assert "2.5" in result or "2.50" in result

        # Check text content is included
        assert "Hello, world!" in result
        assert "This is a test." in result
        assert "Goodbye!" in result

    def test_format_includes_context_when_provided(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that context string is included when provided."""
        context = "This is a technical video about Python programming."
        result = format_for_review(sample_segments, context=context)

        assert context in result

    def test_format_without_context(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that formatting works without context parameter."""
        result = format_for_review(sample_segments)

        # Should still contain segment information
        assert "Hello, world!" in result
        # Should not contain any context marker or placeholder
        # (implementation detail: context should simply be omitted)

    def test_format_empty_segments_list(self) -> None:
        """Test that empty segments list returns appropriate output."""
        result = format_for_review([])

        # Should return empty or minimal output for empty input
        # The exact behavior depends on implementation, but it should not raise
        assert result is not None
        assert isinstance(result, str)

    def test_format_single_segment(self, single_segment: TranscriptSegment) -> None:
        """Test formatting with a single segment."""
        result = format_for_review([single_segment])

        assert "[0]" in result
        assert "Hello, world!" in result
        assert "0.0" in result or "0.00" in result
        assert "2.5" in result or "2.50" in result

    def test_format_preserves_segment_order(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that segments are formatted in order."""
        result = format_for_review(sample_segments)

        # Find positions of each segment's text
        pos_hello = result.find("Hello, world!")
        pos_test = result.find("This is a test.")
        pos_goodbye = result.find("Goodbye!")

        # They should appear in order
        assert pos_hello < pos_test < pos_goodbye


class TestParseCorrectedTranscript:
    """Tests for the parse_corrected_transcript function."""

    def test_parse_preserves_timestamps(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that parsing preserves original timestamps."""
        corrected_text = """[0] 0.0-2.5: Hello, world!
[1] 2.6-5.1: This is a test.
[2] 5.2-8.0: Goodbye!"""

        result = parse_corrected_transcript(corrected_text, sample_segments)

        assert len(result) == 3
        assert result[0].start == 0.0
        assert result[0].end == 2.5
        assert result[1].start == 2.6
        assert result[1].end == 5.1
        assert result[2].start == 5.2
        assert result[2].end == 8.0

    def test_parse_applies_text_corrections(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that corrected text is applied to segments."""
        corrected_text = """[0] 0.0-2.5: Hello, World!
[1] 2.6-5.1: This is a TEST.
[2] 5.2-8.0: Goodbye, everyone!"""

        result = parse_corrected_transcript(corrected_text, sample_segments)

        assert result[0].text == "Hello, World!"
        assert result[1].text == "This is a TEST."
        assert result[2].text == "Goodbye, everyone!"

    def test_parse_handles_different_line_formats(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test parsing handles various line format variations."""
        # Format with extra whitespace
        corrected_text = """[0]  0.0 - 2.5:  Hello, world!
[1]  2.6 - 5.1:  This is a test.
[2]  5.2 - 8.0:  Goodbye!"""

        result = parse_corrected_transcript(corrected_text, sample_segments)

        assert len(result) == 3
        assert result[0].text == "Hello, world!"

    def test_parse_returns_new_segment_objects(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that parse returns new TranscriptSegment objects."""
        corrected_text = """[0] 0.0-2.5: Hello, world!
[1] 2.6-5.1: This is a test.
[2] 5.2-8.0: Goodbye!"""

        result = parse_corrected_transcript(corrected_text, sample_segments)

        # Result should be a new list
        assert result is not sample_segments
        # Each segment should be a new object
        for i, segment in enumerate(result):
            assert segment is not sample_segments[i]

    def test_parse_handles_text_with_colons(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that text containing colons is parsed correctly."""
        corrected_text = """[0] 0.0-2.5: Hello: world!
[1] 2.6-5.1: Time is 12:30:00.
[2] 5.2-8.0: Goodbye!"""

        result = parse_corrected_transcript(corrected_text, sample_segments)

        assert result[0].text == "Hello: world!"
        assert result[1].text == "Time is 12:30:00."

    def test_parse_handles_empty_corrected_text(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test behavior with empty corrected text."""
        corrected_text = ""

        # Should return segments with original text or raise an appropriate error
        # Implementation can choose either approach
        with pytest.raises((ValueError, IndexError)):
            parse_corrected_transcript(corrected_text, sample_segments)


class TestCorrectTranscript:
    """Tests for the correct_transcript function."""

    def test_apply_single_correction(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test applying a single correction."""
        corrections = {0: "Hello, World!"}

        result = correct_transcript(sample_segments, corrections)

        assert result[0].text == "Hello, World!"
        # Original timestamps should be preserved
        assert result[0].start == 0.0
        assert result[0].end == 2.5

    def test_apply_multiple_corrections(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test applying multiple corrections at once."""
        corrections = {
            0: "Hello, World!",
            2: "Goodbye, everyone!",
        }

        result = correct_transcript(sample_segments, corrections)

        assert result[0].text == "Hello, World!"
        assert result[1].text == "This is a test."  # Unchanged
        assert result[2].text == "Goodbye, everyone!"

    def test_uncorrected_segments_remain_unchanged(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that segments not in corrections dict keep original text."""
        corrections = {1: "This is a modified test."}

        result = correct_transcript(sample_segments, corrections)

        assert result[0].text == "Hello, world!"  # Unchanged
        assert result[1].text == "This is a modified test."  # Changed
        assert result[2].text == "Goodbye!"  # Unchanged

    def test_empty_corrections_dict(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that empty corrections dict returns unchanged segments."""
        corrections: dict[int, str] = {}

        result = correct_transcript(sample_segments, corrections)

        # All segments should have original text
        assert result[0].text == "Hello, world!"
        assert result[1].text == "This is a test."
        assert result[2].text == "Goodbye!"

    def test_returns_new_list(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that correct_transcript returns a new list (immutable pattern)."""
        corrections = {0: "Modified"}

        result = correct_transcript(sample_segments, corrections)

        # Should return a new list
        assert result is not sample_segments
        # Original segments should be unchanged
        assert sample_segments[0].text == "Hello, world!"

    def test_returns_new_segment_objects(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that correct_transcript returns new TranscriptSegment objects."""
        corrections: dict[int, str] = {}

        result = correct_transcript(sample_segments, corrections)

        # Even with no corrections, segments should be new objects
        for i, segment in enumerate(result):
            assert segment is not sample_segments[i]

    def test_invalid_index_raises_key_error(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that invalid index in corrections raises KeyError."""
        corrections = {99: "Invalid index"}

        with pytest.raises(KeyError):
            correct_transcript(sample_segments, corrections)

    def test_negative_index_raises_key_error(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that negative index raises KeyError."""
        corrections = {-1: "Negative index"}

        with pytest.raises(KeyError):
            correct_transcript(sample_segments, corrections)

    def test_preserves_timestamps(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that all timestamps are preserved after corrections."""
        corrections = {0: "Modified", 1: "Also modified", 2: "All modified"}

        result = correct_transcript(sample_segments, corrections)

        for i, (original, corrected) in enumerate(zip(sample_segments, result)):
            assert corrected.start == original.start, f"Start mismatch at index {i}"
            assert corrected.end == original.end, f"End mismatch at index {i}"

    def test_empty_segments_with_empty_corrections(self) -> None:
        """Test that empty segments list with empty corrections returns empty list."""
        segments: list[TranscriptSegment] = []
        corrections: dict[int, str] = {}

        result = correct_transcript(segments, corrections)

        assert result == []

    def test_correction_with_empty_string(
        self, sample_segments: list[TranscriptSegment]
    ) -> None:
        """Test that corrections can set text to empty string."""
        corrections = {0: ""}

        result = correct_transcript(sample_segments, corrections)

        assert result[0].text == ""
        # Timestamps should still be preserved
        assert result[0].start == 0.0
        assert result[0].end == 2.5
