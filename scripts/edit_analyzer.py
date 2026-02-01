"""Edit analyzer module for parsing AI agent edit decisions.

This module provides functions for formatting transcripts for AI review,
parsing AI edit decisions into EditSegment objects, and merging adjacent
segments to reduce FFmpeg complexity.
"""

import re

from scripts.edit_decision import EditAction, EditSegment
from scripts.transcription import TranscriptSegment


def format_transcript_for_editing(
    segments: list[TranscriptSegment], context: str | None = None
) -> str:
    """
    Format transcript segments for AI agent review.

    Produces output in the format:
        [0] 0.0-2.0: Hello everyone
        [1] 2.0-5.0: Let me try again

    Args:
        segments: List of TranscriptSegment objects to format
        context: Optional context string to include at the beginning

    Returns:
        Formatted string with segment indices, timestamps, and text
    """
    if not segments:
        return ""

    lines = []

    if context:
        lines.append(context)
        lines.append("")  # Blank line after context

    for i, segment in enumerate(segments):
        line = f"[{i}] {segment.start}-{segment.end}: {segment.text}"
        lines.append(line)

    return "\n".join(lines)


def parse_edit_decisions(
    ai_response: str, original_segments: list[TranscriptSegment]
) -> list[EditSegment]:
    """
    Parse AI agent output into EditSegment objects.

    Handles formats like:
        [KEEP] 0-5: "text description"
        [REMOVE] 6: reason for removal

    Args:
        ai_response: The AI agent's response text containing edit decisions
        original_segments: Original transcript segments to get timestamps from

    Returns:
        List of EditSegment objects with proper start/end times

    Raises:
        IndexError: If segment indices in the response are out of range
    """
    if not ai_response or not ai_response.strip():
        return []

    # Pattern to match: [KEEP] 0: reason  OR  [KEEP] 0-5: reason
    # Action can be uppercase or lowercase
    # Also matches negative numbers so we can raise proper errors
    pattern = r"\[([Kk][Ee][Ee][Pp]|[Rr][Ee][Mm][Oo][Vv][Ee])\]\s*(-?\d+)(?:-(-?\d+))?\s*:\s*(.+)"

    result = []
    lines = ai_response.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line)
        if match:
            action_str = match.group(1).upper()
            start_index = int(match.group(2))
            end_index_str = match.group(3)
            reason_text = match.group(4).strip()

            # Determine the range of indices
            if end_index_str:
                end_index = int(end_index_str)
            else:
                end_index = start_index

            # Validate indices
            if start_index < 0 or start_index >= len(original_segments):
                raise IndexError(f"Invalid start index: {start_index}")
            if end_index < 0 or end_index >= len(original_segments):
                raise IndexError(f"Invalid end index: {end_index}")

            # Get timestamps from original segments
            start_time = original_segments[start_index].start
            end_time = original_segments[end_index].end

            # Build list of transcript indices
            transcript_indices = list(range(start_index, end_index + 1))

            # Determine action
            action = EditAction.KEEP if action_str == "KEEP" else EditAction.REMOVE

            # For KEEP, quoted text is informational, not a reason
            # For REMOVE, the text is the reason
            if action == EditAction.KEEP:
                reason = None
            else:
                reason = reason_text

            result.append(
                EditSegment(
                    start=start_time,
                    end=end_time,
                    action=action,
                    reason=reason,
                    transcript_indices=transcript_indices,
                )
            )

    return result


def merge_adjacent_segments(segments: list[EditSegment]) -> list[EditSegment]:
    """
    Merge consecutive segments with the same action.

    This reduces FFmpeg complexity by combining multiple adjacent cuts
    into single operations.

    Args:
        segments: List of EditSegment objects to merge

    Returns:
        New list of EditSegment objects with adjacent same-action segments merged
    """
    if not segments:
        return []

    result = []
    current = EditSegment(
        start=segments[0].start,
        end=segments[0].end,
        action=segments[0].action,
        reason=segments[0].reason,
        transcript_indices=list(segments[0].transcript_indices),
    )

    for segment in segments[1:]:
        if segment.action == current.action:
            # Merge: extend end time and add indices
            current = EditSegment(
                start=current.start,
                end=segment.end,
                action=current.action,
                reason=current.reason,  # Keep first reason
                transcript_indices=current.transcript_indices + list(segment.transcript_indices),
            )
        else:
            # Different action: save current and start new
            result.append(current)
            current = EditSegment(
                start=segment.start,
                end=segment.end,
                action=segment.action,
                reason=segment.reason,
                transcript_indices=list(segment.transcript_indices),
            )

    # Don't forget the last segment
    result.append(current)

    return result
