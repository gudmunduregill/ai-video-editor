"""Tests for the edit_analyzer module.

These tests follow TDD principles and are written BEFORE the implementation.
"""

import pytest

from scripts.edit_decision import EditAction, EditSegment
from scripts.edit_analyzer import (
    format_transcript_for_editing,
    merge_adjacent_segments,
    parse_edit_decisions,
)
from scripts.transcription import TranscriptSegment


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def editing_segments() -> list[TranscriptSegment]:
    """Return transcript segments for editing tests."""
    return [
        TranscriptSegment(start=0.0, end=2.0, text="Hello everyone"),
        TranscriptSegment(start=2.0, end=5.0, text="Let me try again"),
        TranscriptSegment(start=5.0, end=8.0, text="Welcome to the video"),
        TranscriptSegment(start=8.0, end=12.0, text="Today we will discuss"),
        TranscriptSegment(start=12.0, end=15.0, text="um, let me think"),
        TranscriptSegment(start=15.0, end=20.0, text="Python programming"),
    ]


@pytest.fixture
def simple_segments() -> list[TranscriptSegment]:
    """Return simple transcript segments for basic tests."""
    return [
        TranscriptSegment(start=0.0, end=3.0, text="First segment"),
        TranscriptSegment(start=3.0, end=6.0, text="Second segment"),
        TranscriptSegment(start=6.0, end=9.0, text="Third segment"),
    ]


# ============================================================================
# format_transcript_for_editing Tests
# ============================================================================


class TestFormatTranscriptForEditing:
    """Tests for the format_transcript_for_editing function."""

    def test_format_includes_index_timestamps_and_text(
        self, editing_segments: list[TranscriptSegment]
    ) -> None:
        """Test that formatted output includes segment index, timestamps, and text."""
        result = format_transcript_for_editing(editing_segments)

        # Check that all segments are represented with their data
        assert "[0]" in result
        assert "[1]" in result
        assert "[5]" in result

        # Check timestamps are included
        assert "0.0" in result
        assert "2.0" in result
        assert "5.0" in result

        # Check text content is included
        assert "Hello everyone" in result
        assert "Let me try again" in result
        assert "Welcome to the video" in result

    def test_format_matches_example_output(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test that output matches the expected format from requirements."""
        result = format_transcript_for_editing(simple_segments)

        # Should follow format: [0] 0.0-3.0: First segment
        assert "[0] 0.0-3.0: First segment" in result
        assert "[1] 3.0-6.0: Second segment" in result
        assert "[2] 6.0-9.0: Third segment" in result

    def test_format_includes_context_when_provided(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test that context string is included at the beginning."""
        context = "This video is about Python programming basics."
        result = format_transcript_for_editing(simple_segments, context=context)

        # Context should be at the beginning
        assert result.startswith(context)
        # Segments should still be included
        assert "[0]" in result
        assert "First segment" in result

    def test_format_context_followed_by_blank_line(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test that context is followed by a blank line before segments."""
        context = "Video context here."
        result = format_transcript_for_editing(simple_segments, context=context)

        # There should be a blank line between context and first segment
        lines = result.split("\n")
        assert lines[0] == context
        assert lines[1] == ""
        assert "[0]" in lines[2]

    def test_format_without_context(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test that formatting works without context parameter."""
        result = format_transcript_for_editing(simple_segments)

        # Should contain segment information without any context
        assert "First segment" in result
        # First line should be a segment line
        first_line = result.split("\n")[0]
        assert first_line.startswith("[0]")

    def test_format_empty_segments_list(self) -> None:
        """Test that empty segments list returns empty string."""
        result = format_transcript_for_editing([])

        assert result == ""

    def test_format_single_segment(self) -> None:
        """Test formatting with a single segment."""
        segment = TranscriptSegment(start=0.0, end=5.0, text="Only one segment")
        result = format_transcript_for_editing([segment])

        assert "[0]" in result
        assert "0.0-5.0" in result
        assert "Only one segment" in result

    def test_format_preserves_segment_order(
        self, editing_segments: list[TranscriptSegment]
    ) -> None:
        """Test that segments are formatted in order."""
        result = format_transcript_for_editing(editing_segments)

        # Find positions of each segment's text
        pos_hello = result.find("Hello everyone")
        pos_try = result.find("Let me try again")
        pos_welcome = result.find("Welcome to the video")

        # They should appear in order
        assert pos_hello < pos_try < pos_welcome

    def test_format_handles_text_with_special_characters(self) -> None:
        """Test formatting handles text with colons and other special characters."""
        segments = [
            TranscriptSegment(start=0.0, end=3.0, text="Time is 12:30:00"),
            TranscriptSegment(start=3.0, end=6.0, text="It's a test: really!"),
        ]
        result = format_transcript_for_editing(segments)

        assert "Time is 12:30:00" in result
        assert "It's a test: really!" in result


# ============================================================================
# parse_edit_decisions Tests
# ============================================================================


class TestParseEditDecisions:
    """Tests for the parse_edit_decisions function."""

    def test_parse_single_keep_range(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test parsing a single KEEP range."""
        ai_response = '[KEEP] 0-2: "First to third segments"'

        result = parse_edit_decisions(ai_response, simple_segments)

        assert len(result) == 1
        assert result[0].action == EditAction.KEEP
        assert result[0].transcript_indices == [0, 1, 2]
        assert result[0].start == 0.0
        assert result[0].end == 9.0

    def test_parse_single_remove(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test parsing a single REMOVE segment."""
        ai_response = "[REMOVE] 1: filler words"

        result = parse_edit_decisions(ai_response, simple_segments)

        assert len(result) == 1
        assert result[0].action == EditAction.REMOVE
        assert result[0].transcript_indices == [1]
        assert result[0].start == 3.0
        assert result[0].end == 6.0
        assert result[0].reason == "filler words"

    def test_parse_multiple_decisions(
        self, editing_segments: list[TranscriptSegment]
    ) -> None:
        """Test parsing multiple edit decisions."""
        ai_response = """[KEEP] 0: "Hello everyone"
[REMOVE] 1: false start
[KEEP] 2-3: "main content"
[REMOVE] 4: filler words
[KEEP] 5: "good ending"
"""

        result = parse_edit_decisions(ai_response, editing_segments)

        assert len(result) == 5

        # First KEEP
        assert result[0].action == EditAction.KEEP
        assert result[0].transcript_indices == [0]
        assert result[0].start == 0.0
        assert result[0].end == 2.0

        # First REMOVE
        assert result[1].action == EditAction.REMOVE
        assert result[1].transcript_indices == [1]
        assert result[1].reason == "false start"

        # KEEP range
        assert result[2].action == EditAction.KEEP
        assert result[2].transcript_indices == [2, 3]
        assert result[2].start == 5.0
        assert result[2].end == 12.0

    def test_parse_uses_original_timestamps(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test that parsed segments use timestamps from original segments."""
        ai_response = '[KEEP] 1: "middle segment"'

        result = parse_edit_decisions(ai_response, simple_segments)

        # Should use segment 1's timestamps (3.0-6.0)
        assert result[0].start == simple_segments[1].start
        assert result[0].end == simple_segments[1].end

    def test_parse_range_uses_first_and_last_timestamps(
        self, editing_segments: list[TranscriptSegment]
    ) -> None:
        """Test that ranges use start from first segment and end from last."""
        ai_response = '[KEEP] 2-4: "middle content"'

        result = parse_edit_decisions(ai_response, editing_segments)

        # Range 2-4 should span from segment 2's start to segment 4's end
        assert result[0].start == editing_segments[2].start  # 5.0
        assert result[0].end == editing_segments[4].end  # 15.0
        assert result[0].transcript_indices == [2, 3, 4]

    def test_parse_keep_with_quoted_text(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test parsing KEEP with quoted text description."""
        ai_response = '[KEEP] 0: "First segment"'

        result = parse_edit_decisions(ai_response, simple_segments)

        assert result[0].action == EditAction.KEEP
        # The quoted text is informational, reason might be None or contain the text
        assert result[0].transcript_indices == [0]

    def test_parse_remove_with_reason(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test parsing REMOVE with reason text."""
        ai_response = "[REMOVE] 1: contains um and uh filler words"

        result = parse_edit_decisions(ai_response, simple_segments)

        assert result[0].action == EditAction.REMOVE
        assert result[0].reason == "contains um and uh filler words"

    def test_parse_handles_extra_whitespace(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test parsing handles various whitespace variations."""
        ai_response = """  [KEEP]  0-1:  "some text"
[REMOVE]   2:   bad content
"""

        result = parse_edit_decisions(ai_response, simple_segments)

        assert len(result) == 2
        assert result[0].action == EditAction.KEEP
        assert result[1].action == EditAction.REMOVE

    def test_parse_ignores_non_matching_lines(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test that non-matching lines are ignored."""
        ai_response = """Here is my analysis:

[KEEP] 0: "good content"
Some explanation about the above...

[REMOVE] 1: filler
"""

        result = parse_edit_decisions(ai_response, simple_segments)

        assert len(result) == 2

    def test_parse_empty_response(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test parsing empty response returns empty list."""
        ai_response = ""

        result = parse_edit_decisions(ai_response, simple_segments)

        assert result == []

    def test_parse_response_with_no_valid_decisions(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test parsing response with no valid decision format."""
        ai_response = "This transcript looks good, no changes needed."

        result = parse_edit_decisions(ai_response, simple_segments)

        assert result == []

    def test_parse_invalid_index_raises_error(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test that invalid segment index raises IndexError."""
        ai_response = "[KEEP] 99: invalid"

        with pytest.raises(IndexError):
            parse_edit_decisions(ai_response, simple_segments)

    def test_parse_invalid_range_raises_error(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test that invalid range raises IndexError."""
        ai_response = "[KEEP] 0-99: invalid range"

        with pytest.raises(IndexError):
            parse_edit_decisions(ai_response, simple_segments)

    def test_parse_negative_index_raises_error(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test that negative index raises IndexError."""
        ai_response = "[REMOVE] -1: negative"

        with pytest.raises(IndexError):
            parse_edit_decisions(ai_response, simple_segments)

    def test_parse_returns_edit_segment_objects(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test that parse returns EditSegment objects."""
        ai_response = '[KEEP] 0: "test"'

        result = parse_edit_decisions(ai_response, simple_segments)

        assert isinstance(result[0], EditSegment)

    def test_parse_lowercase_action(
        self, simple_segments: list[TranscriptSegment]
    ) -> None:
        """Test parsing handles lowercase actions."""
        ai_response = """[keep] 0: "content"
[remove] 1: filler"""

        result = parse_edit_decisions(ai_response, simple_segments)

        assert len(result) == 2
        assert result[0].action == EditAction.KEEP
        assert result[1].action == EditAction.REMOVE


# ============================================================================
# merge_adjacent_segments Tests
# ============================================================================


class TestMergeAdjacentSegments:
    """Tests for the merge_adjacent_segments function."""

    def test_merge_adjacent_keeps(self) -> None:
        """Test merging adjacent KEEP segments."""
        segments = [
            EditSegment(
                start=0.0,
                end=3.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[0],
            ),
            EditSegment(
                start=3.0,
                end=6.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[1],
            ),
            EditSegment(
                start=6.0,
                end=9.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[2],
            ),
        ]

        result = merge_adjacent_segments(segments)

        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 9.0
        assert result[0].action == EditAction.KEEP
        assert result[0].transcript_indices == [0, 1, 2]

    def test_merge_adjacent_removes(self) -> None:
        """Test merging adjacent REMOVE segments."""
        segments = [
            EditSegment(
                start=0.0,
                end=2.0,
                action=EditAction.REMOVE,
                reason="filler",
                transcript_indices=[0],
            ),
            EditSegment(
                start=2.0,
                end=4.0,
                action=EditAction.REMOVE,
                reason="more filler",
                transcript_indices=[1],
            ),
        ]

        result = merge_adjacent_segments(segments)

        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 4.0
        assert result[0].action == EditAction.REMOVE
        assert result[0].transcript_indices == [0, 1]

    def test_no_merge_different_actions(self) -> None:
        """Test that segments with different actions are not merged."""
        segments = [
            EditSegment(
                start=0.0,
                end=3.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[0],
            ),
            EditSegment(
                start=3.0,
                end=6.0,
                action=EditAction.REMOVE,
                reason="filler",
                transcript_indices=[1],
            ),
            EditSegment(
                start=6.0,
                end=9.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[2],
            ),
        ]

        result = merge_adjacent_segments(segments)

        assert len(result) == 3
        assert result[0].action == EditAction.KEEP
        assert result[1].action == EditAction.REMOVE
        assert result[2].action == EditAction.KEEP

    def test_merge_multiple_groups(self) -> None:
        """Test merging with multiple groups of same action."""
        segments = [
            EditSegment(
                start=0.0,
                end=2.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[0],
            ),
            EditSegment(
                start=2.0,
                end=4.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[1],
            ),
            EditSegment(
                start=4.0,
                end=6.0,
                action=EditAction.REMOVE,
                reason="bad",
                transcript_indices=[2],
            ),
            EditSegment(
                start=6.0,
                end=8.0,
                action=EditAction.REMOVE,
                reason="also bad",
                transcript_indices=[3],
            ),
            EditSegment(
                start=8.0,
                end=10.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[4],
            ),
        ]

        result = merge_adjacent_segments(segments)

        assert len(result) == 3
        # First merged KEEP (0-4)
        assert result[0].start == 0.0
        assert result[0].end == 4.0
        assert result[0].action == EditAction.KEEP
        assert result[0].transcript_indices == [0, 1]
        # Merged REMOVE (4-8)
        assert result[1].start == 4.0
        assert result[1].end == 8.0
        assert result[1].action == EditAction.REMOVE
        assert result[1].transcript_indices == [2, 3]
        # Single KEEP (8-10)
        assert result[2].start == 8.0
        assert result[2].end == 10.0
        assert result[2].action == EditAction.KEEP
        assert result[2].transcript_indices == [4]

    def test_merge_empty_list(self) -> None:
        """Test merging empty list returns empty list."""
        result = merge_adjacent_segments([])

        assert result == []

    def test_merge_single_segment(self) -> None:
        """Test merging single segment returns same segment."""
        segment = EditSegment(
            start=0.0,
            end=5.0,
            action=EditAction.KEEP,
            reason=None,
            transcript_indices=[0],
        )

        result = merge_adjacent_segments([segment])

        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 5.0

    def test_merge_preserves_first_reason(self) -> None:
        """Test that merged segment keeps the first segment's reason."""
        segments = [
            EditSegment(
                start=0.0,
                end=2.0,
                action=EditAction.REMOVE,
                reason="first reason",
                transcript_indices=[0],
            ),
            EditSegment(
                start=2.0,
                end=4.0,
                action=EditAction.REMOVE,
                reason="second reason",
                transcript_indices=[1],
            ),
        ]

        result = merge_adjacent_segments(segments)

        assert result[0].reason == "first reason"

    def test_merge_returns_new_objects(self) -> None:
        """Test that merge returns new EditSegment objects."""
        segment = EditSegment(
            start=0.0,
            end=5.0,
            action=EditAction.KEEP,
            reason=None,
            transcript_indices=[0],
        )

        result = merge_adjacent_segments([segment])

        # Even single segment should return a new object
        assert result[0] is not segment

    def test_merge_does_not_modify_input(self) -> None:
        """Test that merge does not modify input list."""
        segments = [
            EditSegment(
                start=0.0,
                end=3.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[0],
            ),
            EditSegment(
                start=3.0,
                end=6.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[1],
            ),
        ]
        original_len = len(segments)
        original_indices = segments[0].transcript_indices.copy()

        merge_adjacent_segments(segments)

        assert len(segments) == original_len
        assert segments[0].transcript_indices == original_indices

    def test_merge_handles_non_adjacent_times(self) -> None:
        """Test merging segments that have same action but gaps in time."""
        # Even with gaps, if actions match they should merge
        # (gaps might indicate missing content in original)
        segments = [
            EditSegment(
                start=0.0,
                end=2.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[0],
            ),
            EditSegment(
                start=5.0,
                end=7.0,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[2],
            ),
        ]

        result = merge_adjacent_segments(segments)

        # Non-adjacent segments with same action should still merge
        # The merged segment spans from first start to last end
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 7.0
        assert result[0].transcript_indices == [0, 2]


# ============================================================================
# Integration Tests
# ============================================================================


class TestEditAnalyzerIntegration:
    """Integration tests for the edit analyzer workflow."""

    def test_full_workflow_format_parse_merge(
        self, editing_segments: list[TranscriptSegment]
    ) -> None:
        """Test the complete workflow from formatting to merging."""
        # Step 1: Format for AI
        formatted = format_transcript_for_editing(
            editing_segments,
            context="Review this video transcript for editing.",
        )

        assert "Review this video transcript" in formatted
        assert "[0]" in formatted
        assert "[5]" in formatted

        # Step 2: Simulate AI response
        ai_response = """[KEEP] 0: "Hello everyone" - good intro
[REMOVE] 1: false start, should be cut
[KEEP] 2-3: main content
[REMOVE] 4: filler words (um, let me think)
[KEEP] 5: good conclusion"""

        # Step 3: Parse AI response
        parsed = parse_edit_decisions(ai_response, editing_segments)

        assert len(parsed) == 5
        assert parsed[0].action == EditAction.KEEP
        assert parsed[1].action == EditAction.REMOVE

        # Step 4: Merge adjacent segments
        merged = merge_adjacent_segments(parsed)

        # After merging: KEEP(0), REMOVE(1), KEEP(2-3), REMOVE(4), KEEP(5)
        # No adjacent same-action segments to merge in this case
        assert len(merged) == 5

    def test_workflow_with_adjacent_keeps(
        self, editing_segments: list[TranscriptSegment]
    ) -> None:
        """Test workflow where multiple KEEPs can be merged."""
        ai_response = """[KEEP] 0: intro
[KEEP] 1: also keep
[KEEP] 2: and this
[REMOVE] 3: cut this
[KEEP] 4-5: ending"""

        parsed = parse_edit_decisions(ai_response, editing_segments)
        merged = merge_adjacent_segments(parsed)

        # Should merge 0,1,2 into one KEEP and 4,5 stays as one
        assert len(merged) == 3
        assert merged[0].action == EditAction.KEEP
        assert merged[0].transcript_indices == [0, 1, 2]
        assert merged[1].action == EditAction.REMOVE
        assert merged[2].action == EditAction.KEEP
