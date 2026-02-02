"""Edit decision module for video editing workflow.

This module provides data structures and functions for managing edit decisions
in a video editing workflow. An Edit Decision List (EDL) defines which segments
of a video should be kept or removed.
"""

import json
from dataclasses import dataclass
from enum import Enum


class EditAction(Enum):
    """Actions that can be applied to a video segment."""

    KEEP = "keep"
    REMOVE = "remove"


@dataclass
class EditSegment:
    """A single segment with an edit decision.

    Represents a portion of video that should either be kept or removed.
    """

    start: float  # Start time in seconds
    end: float  # End time in seconds
    action: EditAction  # KEEP or REMOVE
    reason: str | None  # Why this decision was made
    transcript_indices: list[int]  # Which transcript segments this covers


@dataclass
class EditDecisionList:
    """A complete edit decision list for a video.

    Contains all edit segments and metadata about the source video.
    """

    source_video: str  # Path or identifier for the source video
    segments: list[EditSegment]  # List of edit segments
    total_duration: float  # Total duration of the source video in seconds

    @property
    def keep_segments(self) -> list[EditSegment]:
        """Return only segments marked as KEEP."""
        return [s for s in self.segments if s.action == EditAction.KEEP]

    @property
    def remove_segments(self) -> list[EditSegment]:
        """Return only segments marked as REMOVE."""
        return [s for s in self.segments if s.action == EditAction.REMOVE]

    @property
    def kept_duration(self) -> float:
        """Calculate total duration of kept segments."""
        return sum(s.end - s.start for s in self.keep_segments)

    @property
    def removed_duration(self) -> float:
        """Calculate total duration of removed segments."""
        return sum(s.end - s.start for s in self.remove_segments)


def edl_to_json(edl: EditDecisionList) -> str:
    """Serialize an EditDecisionList to JSON string.

    Args:
        edl: The EditDecisionList to serialize

    Returns:
        JSON string representation of the EDL
    """
    data = {
        "source_video": edl.source_video,
        "total_duration": edl.total_duration,
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "action": segment.action.value,
                "reason": segment.reason,
                "transcript_indices": segment.transcript_indices,
            }
            for segment in edl.segments
        ],
    }
    return json.dumps(data, indent=2)


def edl_from_dict(data: dict) -> EditDecisionList:
    """Deserialize an EditDecisionList from a dictionary.

    This is the memory-efficient entry point when loading from a file,
    as it avoids creating an intermediate string representation.

    Args:
        data: Dictionary with EDL data

    Returns:
        EditDecisionList object

    Raises:
        KeyError: If required fields are missing
    """
    segments = [
        EditSegment(
            start=seg["start"],
            end=seg["end"],
            action=EditAction(seg["action"]),
            reason=seg["reason"],
            transcript_indices=seg["transcript_indices"],
        )
        for seg in data["segments"]
    ]

    return EditDecisionList(
        source_video=data["source_video"],
        segments=segments,
        total_duration=data["total_duration"],
    )


def edl_from_json(json_str: str) -> EditDecisionList:
    """Deserialize an EditDecisionList from JSON string.

    Note: For file loading, prefer using json.load(f) with edl_from_dict()
    to avoid loading the entire file content as a string first.

    Args:
        json_str: JSON string to deserialize

    Returns:
        EditDecisionList object

    Raises:
        json.JSONDecodeError: If JSON is invalid
        KeyError: If required fields are missing
    """
    data = json.loads(json_str)
    return edl_from_dict(data)


def format_edl_for_review(edl: EditDecisionList) -> str:
    """Format an EditDecisionList for human review.

    Produces a readable text format suitable for review by a human or AI agent.

    Args:
        edl: The EditDecisionList to format

    Returns:
        Human-readable string representation
    """
    lines = []

    # Header with source video info
    lines.append(f"Edit Decision List: {edl.source_video}")
    lines.append(f"Total Duration: {edl.total_duration}s")
    lines.append(f"Kept Duration: {edl.kept_duration}s")
    lines.append(f"Removed Duration: {edl.removed_duration}s")
    lines.append("")

    # Segments
    if not edl.segments:
        lines.append("No segments")
    else:
        lines.append("Segments:")
        for i, segment in enumerate(edl.segments):
            action_str = segment.action.name  # KEEP or REMOVE
            reason_str = segment.reason if segment.reason else "(no reason)"
            lines.append(
                f"[{i}] {segment.start}-{segment.end}: {action_str} - {reason_str}"
            )

    return "\n".join(lines)


def apply_edl_corrections(
    edl: EditDecisionList, corrections: dict[int, EditAction]
) -> EditDecisionList:
    """Apply corrections to edit decisions by segment index.

    Creates a new EditDecisionList with the specified action corrections applied.
    This follows an immutable pattern - the original EDL is not modified.

    Args:
        edl: Original EditDecisionList
        corrections: Dictionary mapping segment indices to new EditAction values

    Returns:
        New EditDecisionList with corrections applied

    Raises:
        KeyError: If any correction index is invalid (out of range or negative)
    """
    # Validate all indices first
    for index in corrections:
        if index < 0 or index >= len(edl.segments):
            raise KeyError(f"Invalid segment index: {index}")

    # Create new segments with corrections applied
    new_segments = []
    for i, segment in enumerate(edl.segments):
        if i in corrections:
            action = corrections[i]
        else:
            action = segment.action

        new_segments.append(
            EditSegment(
                start=segment.start,
                end=segment.end,
                action=action,
                reason=segment.reason,
                transcript_indices=list(segment.transcript_indices),
            )
        )

    return EditDecisionList(
        source_video=edl.source_video,
        segments=new_segments,
        total_duration=edl.total_duration,
    )
