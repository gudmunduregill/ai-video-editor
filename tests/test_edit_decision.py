"""Tests for the edit_decision module.

These tests follow TDD principles and are written BEFORE the implementation.
"""

import json

import pytest

from scripts.edit_decision import (
    EditAction,
    EditDecisionList,
    EditSegment,
    apply_edl_corrections,
    edl_from_json,
    edl_to_json,
    format_edl_for_review,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_edit_segment() -> EditSegment:
    """Return a sample edit segment for testing."""
    return EditSegment(
        start=0.0,
        end=5.0,
        action=EditAction.KEEP,
        reason="Good content",
        transcript_indices=[0, 1],
    )


@pytest.fixture
def sample_remove_segment() -> EditSegment:
    """Return a sample remove edit segment for testing."""
    return EditSegment(
        start=5.0,
        end=10.0,
        action=EditAction.REMOVE,
        reason="Contains filler words",
        transcript_indices=[2],
    )


@pytest.fixture
def sample_edl(
    sample_edit_segment: EditSegment, sample_remove_segment: EditSegment
) -> EditDecisionList:
    """Return a sample edit decision list for testing."""
    return EditDecisionList(
        source_video="test_video.mp4",
        segments=[sample_edit_segment, sample_remove_segment],
        total_duration=10.0,
    )


@pytest.fixture
def multi_segment_edl() -> EditDecisionList:
    """Return an EDL with multiple segments for testing."""
    return EditDecisionList(
        source_video="video.mp4",
        segments=[
            EditSegment(
                start=0.0,
                end=5.0,
                action=EditAction.KEEP,
                reason="Intro",
                transcript_indices=[0],
            ),
            EditSegment(
                start=5.0,
                end=8.0,
                action=EditAction.REMOVE,
                reason="Coughing",
                transcript_indices=[1],
            ),
            EditSegment(
                start=8.0,
                end=15.0,
                action=EditAction.KEEP,
                reason="Main content",
                transcript_indices=[2, 3],
            ),
            EditSegment(
                start=15.0,
                end=20.0,
                action=EditAction.REMOVE,
                reason="Silence",
                transcript_indices=[4],
            ),
        ],
        total_duration=20.0,
    )


# ============================================================================
# EditAction Tests
# ============================================================================


class TestEditAction:
    """Tests for the EditAction enum."""

    def test_keep_action_value(self) -> None:
        """Test KEEP action has correct value."""
        assert EditAction.KEEP.value == "keep"

    def test_remove_action_value(self) -> None:
        """Test REMOVE action has correct value."""
        assert EditAction.REMOVE.value == "remove"

    def test_action_from_string(self) -> None:
        """Test creating action from string value."""
        assert EditAction("keep") == EditAction.KEEP
        assert EditAction("remove") == EditAction.REMOVE


# ============================================================================
# EditSegment Tests
# ============================================================================


class TestEditSegment:
    """Tests for the EditSegment dataclass."""

    def test_create_keep_segment(self) -> None:
        """Test creating a KEEP segment."""
        segment = EditSegment(
            start=0.0,
            end=5.0,
            action=EditAction.KEEP,
            reason="Good content",
            transcript_indices=[0, 1],
        )

        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.action == EditAction.KEEP
        assert segment.reason == "Good content"
        assert segment.transcript_indices == [0, 1]

    def test_create_remove_segment(self) -> None:
        """Test creating a REMOVE segment."""
        segment = EditSegment(
            start=5.0,
            end=10.0,
            action=EditAction.REMOVE,
            reason="Filler words",
            transcript_indices=[2],
        )

        assert segment.action == EditAction.REMOVE
        assert segment.reason == "Filler words"

    def test_segment_with_no_reason(self) -> None:
        """Test creating a segment with no reason (None)."""
        segment = EditSegment(
            start=0.0,
            end=5.0,
            action=EditAction.KEEP,
            reason=None,
            transcript_indices=[0],
        )

        assert segment.reason is None

    def test_segment_with_empty_transcript_indices(self) -> None:
        """Test creating a segment with empty transcript indices."""
        segment = EditSegment(
            start=0.0,
            end=5.0,
            action=EditAction.KEEP,
            reason="Silent section",
            transcript_indices=[],
        )

        assert segment.transcript_indices == []


# ============================================================================
# EditDecisionList Tests
# ============================================================================


class TestEditDecisionList:
    """Tests for the EditDecisionList dataclass."""

    def test_create_edl(self, sample_edl: EditDecisionList) -> None:
        """Test creating an EditDecisionList."""
        assert sample_edl.source_video == "test_video.mp4"
        assert len(sample_edl.segments) == 2
        assert sample_edl.total_duration == 10.0

    def test_keep_segments_property(self, multi_segment_edl: EditDecisionList) -> None:
        """Test keep_segments property returns only KEEP segments."""
        keep_segments = multi_segment_edl.keep_segments

        assert len(keep_segments) == 2
        for segment in keep_segments:
            assert segment.action == EditAction.KEEP

    def test_remove_segments_property(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test remove_segments property returns only REMOVE segments."""
        remove_segments = multi_segment_edl.remove_segments

        assert len(remove_segments) == 2
        for segment in remove_segments:
            assert segment.action == EditAction.REMOVE

    def test_kept_duration_property(self, multi_segment_edl: EditDecisionList) -> None:
        """Test kept_duration property calculates total kept time."""
        # Keep segments: 0-5 (5s) and 8-15 (7s) = 12s
        assert multi_segment_edl.kept_duration == 12.0

    def test_removed_duration_property(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test removed_duration property calculates total removed time."""
        # Remove segments: 5-8 (3s) and 15-20 (5s) = 8s
        assert multi_segment_edl.removed_duration == 8.0

    def test_empty_segments_list(self) -> None:
        """Test EDL with empty segments list."""
        edl = EditDecisionList(
            source_video="empty.mp4",
            segments=[],
            total_duration=0.0,
        )

        assert edl.keep_segments == []
        assert edl.remove_segments == []
        assert edl.kept_duration == 0.0
        assert edl.removed_duration == 0.0

    def test_all_keep_segments(self) -> None:
        """Test EDL with only KEEP segments."""
        edl = EditDecisionList(
            source_video="all_keep.mp4",
            segments=[
                EditSegment(
                    start=0.0,
                    end=10.0,
                    action=EditAction.KEEP,
                    reason=None,
                    transcript_indices=[0],
                ),
            ],
            total_duration=10.0,
        )

        assert len(edl.keep_segments) == 1
        assert len(edl.remove_segments) == 0
        assert edl.kept_duration == 10.0
        assert edl.removed_duration == 0.0

    def test_all_remove_segments(self) -> None:
        """Test EDL with only REMOVE segments."""
        edl = EditDecisionList(
            source_video="all_remove.mp4",
            segments=[
                EditSegment(
                    start=0.0,
                    end=10.0,
                    action=EditAction.REMOVE,
                    reason="All bad",
                    transcript_indices=[0],
                ),
            ],
            total_duration=10.0,
        )

        assert len(edl.keep_segments) == 0
        assert len(edl.remove_segments) == 1
        assert edl.kept_duration == 0.0
        assert edl.removed_duration == 10.0


# ============================================================================
# edl_to_json Tests
# ============================================================================


class TestEdlToJson:
    """Tests for the edl_to_json function."""

    def test_serialize_to_valid_json(self, sample_edl: EditDecisionList) -> None:
        """Test that output is valid JSON."""
        result = edl_to_json(sample_edl)

        # Should not raise
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_serialize_preserves_source_video(
        self, sample_edl: EditDecisionList
    ) -> None:
        """Test that source video path is preserved."""
        result = edl_to_json(sample_edl)
        parsed = json.loads(result)

        assert parsed["source_video"] == "test_video.mp4"

    def test_serialize_preserves_total_duration(
        self, sample_edl: EditDecisionList
    ) -> None:
        """Test that total duration is preserved."""
        result = edl_to_json(sample_edl)
        parsed = json.loads(result)

        assert parsed["total_duration"] == 10.0

    def test_serialize_preserves_segments(self, sample_edl: EditDecisionList) -> None:
        """Test that segments are preserved with all fields."""
        result = edl_to_json(sample_edl)
        parsed = json.loads(result)

        assert len(parsed["segments"]) == 2

        first_segment = parsed["segments"][0]
        assert first_segment["start"] == 0.0
        assert first_segment["end"] == 5.0
        assert first_segment["action"] == "keep"
        assert first_segment["reason"] == "Good content"
        assert first_segment["transcript_indices"] == [0, 1]

    def test_serialize_none_reason(self) -> None:
        """Test that None reason is serialized correctly."""
        edl = EditDecisionList(
            source_video="test.mp4",
            segments=[
                EditSegment(
                    start=0.0,
                    end=5.0,
                    action=EditAction.KEEP,
                    reason=None,
                    transcript_indices=[0],
                ),
            ],
            total_duration=5.0,
        )

        result = edl_to_json(edl)
        parsed = json.loads(result)

        assert parsed["segments"][0]["reason"] is None

    def test_serialize_empty_segments(self) -> None:
        """Test serialization of EDL with no segments."""
        edl = EditDecisionList(
            source_video="empty.mp4",
            segments=[],
            total_duration=0.0,
        )

        result = edl_to_json(edl)
        parsed = json.loads(result)

        assert parsed["segments"] == []


# ============================================================================
# edl_from_json Tests
# ============================================================================


class TestEdlFromJson:
    """Tests for the edl_from_json function."""

    def test_deserialize_basic_edl(self) -> None:
        """Test deserializing a basic EDL from JSON."""
        json_str = """{
            "source_video": "test.mp4",
            "total_duration": 10.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "action": "keep",
                    "reason": "Good",
                    "transcript_indices": [0]
                }
            ]
        }"""

        result = edl_from_json(json_str)

        assert isinstance(result, EditDecisionList)
        assert result.source_video == "test.mp4"
        assert result.total_duration == 10.0
        assert len(result.segments) == 1

    def test_deserialize_preserves_segment_data(self) -> None:
        """Test that all segment data is preserved."""
        json_str = """{
            "source_video": "test.mp4",
            "total_duration": 10.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "action": "keep",
                    "reason": "Good content",
                    "transcript_indices": [0, 1, 2]
                }
            ]
        }"""

        result = edl_from_json(json_str)
        segment = result.segments[0]

        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.action == EditAction.KEEP
        assert segment.reason == "Good content"
        assert segment.transcript_indices == [0, 1, 2]

    def test_deserialize_remove_action(self) -> None:
        """Test deserializing REMOVE action."""
        json_str = """{
            "source_video": "test.mp4",
            "total_duration": 5.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "action": "remove",
                    "reason": "Bad",
                    "transcript_indices": [0]
                }
            ]
        }"""

        result = edl_from_json(json_str)

        assert result.segments[0].action == EditAction.REMOVE

    def test_deserialize_none_reason(self) -> None:
        """Test deserializing segment with null reason."""
        json_str = """{
            "source_video": "test.mp4",
            "total_duration": 5.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "action": "keep",
                    "reason": null,
                    "transcript_indices": [0]
                }
            ]
        }"""

        result = edl_from_json(json_str)

        assert result.segments[0].reason is None

    def test_roundtrip_serialization(self, multi_segment_edl: EditDecisionList) -> None:
        """Test that serialize then deserialize returns equivalent EDL."""
        json_str = edl_to_json(multi_segment_edl)
        result = edl_from_json(json_str)

        assert result.source_video == multi_segment_edl.source_video
        assert result.total_duration == multi_segment_edl.total_duration
        assert len(result.segments) == len(multi_segment_edl.segments)

        for orig, restored in zip(multi_segment_edl.segments, result.segments):
            assert restored.start == orig.start
            assert restored.end == orig.end
            assert restored.action == orig.action
            assert restored.reason == orig.reason
            assert restored.transcript_indices == orig.transcript_indices

    def test_invalid_json_raises_error(self) -> None:
        """Test that invalid JSON raises an error."""
        invalid_json = "not valid json {"

        with pytest.raises(json.JSONDecodeError):
            edl_from_json(invalid_json)

    def test_missing_required_field_raises_error(self) -> None:
        """Test that missing required fields raise an error."""
        json_str = """{
            "source_video": "test.mp4"
        }"""

        with pytest.raises((KeyError, TypeError)):
            edl_from_json(json_str)


# ============================================================================
# format_edl_for_review Tests
# ============================================================================


class TestFormatEdlForReview:
    """Tests for the format_edl_for_review function."""

    def test_format_includes_segment_indices(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test that formatted output includes segment indices."""
        result = format_edl_for_review(multi_segment_edl)

        assert "[0]" in result
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_format_includes_timestamps(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test that formatted output includes timestamps."""
        result = format_edl_for_review(multi_segment_edl)

        assert "0.0" in result or "0.00" in result
        assert "5.0" in result or "5.00" in result
        assert "8.0" in result or "8.00" in result

    def test_format_includes_action(self, multi_segment_edl: EditDecisionList) -> None:
        """Test that formatted output includes action (KEEP/REMOVE)."""
        result = format_edl_for_review(multi_segment_edl)

        assert "KEEP" in result or "keep" in result
        assert "REMOVE" in result or "remove" in result

    def test_format_includes_reason(self, multi_segment_edl: EditDecisionList) -> None:
        """Test that formatted output includes reasons."""
        result = format_edl_for_review(multi_segment_edl)

        assert "Intro" in result
        assert "Coughing" in result
        assert "Main content" in result
        assert "Silence" in result

    def test_format_includes_source_video(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test that formatted output includes source video."""
        result = format_edl_for_review(multi_segment_edl)

        assert "video.mp4" in result

    def test_format_includes_duration_info(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test that formatted output includes duration information."""
        result = format_edl_for_review(multi_segment_edl)

        # Should include total, kept, and removed durations
        assert "20.0" in result or "20" in result  # Total duration
        assert "12.0" in result or "12" in result  # Kept duration
        assert "8.0" in result or "8" in result  # Removed duration

    def test_format_empty_edl(self) -> None:
        """Test formatting an empty EDL."""
        edl = EditDecisionList(
            source_video="empty.mp4",
            segments=[],
            total_duration=0.0,
        )

        result = format_edl_for_review(edl)

        assert result is not None
        assert isinstance(result, str)
        assert "empty.mp4" in result

    def test_format_handles_none_reason(self) -> None:
        """Test formatting segment with None reason."""
        edl = EditDecisionList(
            source_video="test.mp4",
            segments=[
                EditSegment(
                    start=0.0,
                    end=5.0,
                    action=EditAction.KEEP,
                    reason=None,
                    transcript_indices=[0],
                ),
            ],
            total_duration=5.0,
        )

        # Should not raise and should handle None gracefully
        result = format_edl_for_review(edl)

        assert result is not None
        assert "[0]" in result


# ============================================================================
# apply_edl_corrections Tests
# ============================================================================


class TestApplyEdlCorrections:
    """Tests for the apply_edl_corrections function."""

    def test_apply_single_correction(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test applying a single correction."""
        corrections = {0: EditAction.REMOVE}

        result = apply_edl_corrections(multi_segment_edl, corrections)

        assert result.segments[0].action == EditAction.REMOVE
        # Other segments unchanged
        assert result.segments[1].action == EditAction.REMOVE
        assert result.segments[2].action == EditAction.KEEP

    def test_apply_multiple_corrections(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test applying multiple corrections."""
        corrections = {
            0: EditAction.REMOVE,  # Was KEEP
            1: EditAction.KEEP,  # Was REMOVE
        }

        result = apply_edl_corrections(multi_segment_edl, corrections)

        assert result.segments[0].action == EditAction.REMOVE
        assert result.segments[1].action == EditAction.KEEP
        assert result.segments[2].action == EditAction.KEEP  # Unchanged
        assert result.segments[3].action == EditAction.REMOVE  # Unchanged

    def test_empty_corrections_returns_unchanged(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test that empty corrections dict returns unchanged EDL."""
        corrections: dict[int, EditAction] = {}

        result = apply_edl_corrections(multi_segment_edl, corrections)

        for orig, corrected in zip(multi_segment_edl.segments, result.segments):
            assert corrected.action == orig.action

    def test_returns_new_edl(self, multi_segment_edl: EditDecisionList) -> None:
        """Test that apply_edl_corrections returns a new EDL."""
        corrections = {0: EditAction.REMOVE}

        result = apply_edl_corrections(multi_segment_edl, corrections)

        assert result is not multi_segment_edl
        # Original should be unchanged
        assert multi_segment_edl.segments[0].action == EditAction.KEEP

    def test_returns_new_segment_objects(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test that corrections return new segment objects."""
        corrections: dict[int, EditAction] = {}

        result = apply_edl_corrections(multi_segment_edl, corrections)

        for orig, corrected in zip(multi_segment_edl.segments, result.segments):
            assert corrected is not orig

    def test_preserves_other_segment_fields(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test that other segment fields are preserved."""
        corrections = {0: EditAction.REMOVE}

        result = apply_edl_corrections(multi_segment_edl, corrections)

        orig = multi_segment_edl.segments[0]
        corrected = result.segments[0]

        assert corrected.start == orig.start
        assert corrected.end == orig.end
        assert corrected.reason == orig.reason
        assert corrected.transcript_indices == orig.transcript_indices

    def test_preserves_source_video_and_duration(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test that source video and total duration are preserved."""
        corrections = {0: EditAction.REMOVE}

        result = apply_edl_corrections(multi_segment_edl, corrections)

        assert result.source_video == multi_segment_edl.source_video
        assert result.total_duration == multi_segment_edl.total_duration

    def test_invalid_index_raises_key_error(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test that invalid index raises KeyError."""
        corrections = {99: EditAction.REMOVE}

        with pytest.raises(KeyError):
            apply_edl_corrections(multi_segment_edl, corrections)

    def test_negative_index_raises_key_error(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test that negative index raises KeyError."""
        corrections = {-1: EditAction.REMOVE}

        with pytest.raises(KeyError):
            apply_edl_corrections(multi_segment_edl, corrections)

    def test_empty_segments_with_empty_corrections(self) -> None:
        """Test empty EDL with empty corrections returns empty EDL."""
        edl = EditDecisionList(
            source_video="empty.mp4",
            segments=[],
            total_duration=0.0,
        )
        corrections: dict[int, EditAction] = {}

        result = apply_edl_corrections(edl, corrections)

        assert result.segments == []

    def test_updated_durations_after_corrections(
        self, multi_segment_edl: EditDecisionList
    ) -> None:
        """Test that duration properties update after corrections."""
        # Original: keep 0-5 (5s), 8-15 (7s) = 12s kept
        # Original: remove 5-8 (3s), 15-20 (5s) = 8s removed
        assert multi_segment_edl.kept_duration == 12.0
        assert multi_segment_edl.removed_duration == 8.0

        # Change first segment from KEEP to REMOVE
        corrections = {0: EditAction.REMOVE}

        result = apply_edl_corrections(multi_segment_edl, corrections)

        # New: keep 8-15 (7s) = 7s kept
        # New: remove 0-5 (5s), 5-8 (3s), 15-20 (5s) = 13s removed
        assert result.kept_duration == 7.0
        assert result.removed_duration == 13.0
