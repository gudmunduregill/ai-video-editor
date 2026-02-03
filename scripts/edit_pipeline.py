"""Edit pipeline module for orchestrating the video editing workflow.

This module provides functions for the complete video editing pipeline:
1. Generating/loading transcripts
2. Formatting transcripts for AI review
3. Creating initial EDLs
4. Optionally using AI to analyze and suggest edits
5. Applying EDLs to cut videos
"""

import os
import re
import sys
from pathlib import Path
from typing import Generator, Optional

import json

from scripts.edit_decision import (
    EditAction,
    EditDecisionList,
    EditSegment,
    edl_from_dict,
    edl_to_json,
)
from scripts.llm_client import LLMClientError, analyze_transcript, load_agent_prompt
from scripts.pipeline import process_video
from scripts.transcription import TranscriptSegment
from scripts.video_cutter import adjust_srt_for_edl, cut_video, get_video_duration


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


def _parse_srt_timestamp(timestamp: str) -> float:
    """
    Parse an SRT timestamp string to seconds.

    Args:
        timestamp: Timestamp in format HH:MM:SS,mmm (e.g., "00:01:05,500")

    Returns:
        Time in seconds as a float

    Raises:
        ValueError: If timestamp format is invalid
    """
    # SRT format: HH:MM:SS,mmm
    pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    match = re.match(pattern, timestamp.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp format: {timestamp}")

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    milliseconds = int(match.group(4))

    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    return total_seconds


def _iter_srt_segments(transcript_path: str) -> "Generator[TranscriptSegment, None, None]":
    """
    Stream-parse an SRT file, yielding TranscriptSegment objects one at a time.

    This is memory-efficient as it reads the file line-by-line instead of
    loading the entire file into memory.

    Args:
        transcript_path: Path to the SRT file

    Yields:
        TranscriptSegment objects as they are parsed

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    # State for parsing SRT blocks
    # States: 'number', 'timestamp', 'text', 'blank'
    state = "number"
    current_start: float = 0.0
    current_end: float = 0.0
    text_lines: list[str] = []

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")

            if state == "number":
                # Expecting subtitle number (skip it)
                if line.strip().isdigit():
                    state = "timestamp"
                elif line.strip() == "":
                    # Extra blank line, stay in number state
                    pass
                # else: malformed, try next line

            elif state == "timestamp":
                # Expecting timestamp line
                timestamp_match = re.match(
                    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
                    line.strip(),
                )
                if timestamp_match:
                    current_start = _parse_srt_timestamp(timestamp_match.group(1))
                    current_end = _parse_srt_timestamp(timestamp_match.group(2))
                    text_lines = []
                    state = "text"
                else:
                    # Malformed, reset to looking for number
                    state = "number"

            elif state == "text":
                if line.strip() == "":
                    # Blank line signals end of this subtitle block
                    if text_lines:
                        yield TranscriptSegment(
                            start=current_start,
                            end=current_end,
                            text="\n".join(text_lines),
                        )
                    state = "number"
                else:
                    # Accumulate text lines
                    text_lines.append(line)

        # Handle last segment if file doesn't end with blank line
        if state == "text" and text_lines:
            yield TranscriptSegment(
                start=current_start,
                end=current_end,
                text="\n".join(text_lines),
            )


def _load_transcript(transcript_path: str) -> list[TranscriptSegment]:
    """
    Parse an SRT file into TranscriptSegment objects.

    Note: This returns a list because downstream callers (format_transcript_for_editing,
    _create_initial_edl) require indexed access to segments. The streaming parser
    _iter_srt_segments is used internally for memory efficiency during parsing.

    Args:
        transcript_path: Path to the SRT file

    Returns:
        List of TranscriptSegment objects

    Raises:
        FileNotFoundError: If the file does not exist
    """
    return list(_iter_srt_segments(transcript_path))


def _create_initial_edl(
    segments: list[TranscriptSegment],
    video_path: str,
    duration: float,
) -> EditDecisionList:
    """
    Create an initial EDL with all segments marked as KEEP.

    Args:
        segments: List of transcript segments
        video_path: Path to the source video
        duration: Total video duration in seconds

    Returns:
        EditDecisionList with all segments as KEEP
    """
    edit_segments: list[EditSegment] = []

    for i, segment in enumerate(segments):
        edit_segments.append(
            EditSegment(
                start=segment.start,
                end=segment.end,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=[i],
            )
        )

    return EditDecisionList(
        source_video=video_path,
        segments=edit_segments,
        total_duration=duration,
    )


def _parse_ai_response(
    response: str,
    segments: list[TranscriptSegment],
) -> list[EditSegment]:
    """
    Parse AI response into EditSegment objects.

    The AI response format looks like:
        [KEEP] 0-5: Main introduction
        [REMOVE] 6-7: "Let me try again, sorry" - Retake
        [KEEP] 8-15: Core content explaining the topic
        [REVIEW] 16-17: Borderline content

    Note: [REVIEW] segments are treated as [KEEP] since they need human decision.

    Args:
        response: The AI's response text containing edit decisions.
        segments: Original transcript segments to get timestamps from.

    Returns:
        List of EditSegment objects. Returns empty list if parsing fails completely.
    """
    if not response or not response.strip():
        return []

    # Pattern to match: [KEEP] 0: reason  OR  [KEEP] 0-5: reason
    # Also supports [REVIEW] which we treat as KEEP
    pattern = r"\[([Kk][Ee][Ee][Pp]|[Rr][Ee][Mm][Oo][Vv][Ee]|[Rr][Ee][Vv][Ii][Ee][Ww])\]\s*(-?\d+)(?:-(-?\d+))?\s*:\s*(.+)"

    result: list[EditSegment] = []
    lines = response.strip().split("\n")

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

            # Validate indices - skip invalid ones with warning
            if start_index < 0 or start_index >= len(segments):
                print(
                    f"Warning: Skipping invalid start index {start_index} in AI response",
                    file=sys.stderr,
                )
                continue
            if end_index < 0 or end_index >= len(segments):
                print(
                    f"Warning: Skipping invalid end index {end_index} in AI response",
                    file=sys.stderr,
                )
                continue

            # Get timestamps from original segments
            start_time = segments[start_index].start
            end_time = segments[end_index].end

            # Build list of transcript indices
            transcript_indices = list(range(start_index, end_index + 1))

            # Determine action - REVIEW is treated as KEEP
            if action_str == "REMOVE":
                action = EditAction.REMOVE
                reason = reason_text
            else:
                # KEEP or REVIEW
                action = EditAction.KEEP
                reason = None

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


def _analyze_with_ai(
    transcript: str,
    segments: list[TranscriptSegment],
    use_ai: bool,
) -> tuple[list[EditSegment], str | None]:
    """
    Analyze transcript with AI to suggest edits.

    When use_ai is True, calls the Claude API with the video-editor agent prompt
    to analyze the transcript and suggest KEEP/REMOVE decisions.

    When use_ai is False, returns an empty list (caller should use all-KEEP EDL).

    Args:
        transcript: The formatted transcript text.
        segments: Original transcript segments for timestamp lookup.
        use_ai: Whether to actually call the AI.

    Returns:
        Tuple of (edit_segments, raw_response):
        - edit_segments: List of EditSegment objects from AI analysis
        - raw_response: The raw AI response text, or None if use_ai=False

    Raises:
        LLMClientError: If the API call fails.
    """
    if not use_ai:
        return [], None

    # Load the video-editor agent prompt
    agent_prompt = load_agent_prompt("video-editor")

    # Call Claude API
    response = analyze_transcript(transcript, agent_prompt)

    # Parse the response
    edit_segments = _parse_ai_response(response, segments)

    return edit_segments, response


def _create_edl_from_ai_segments(
    ai_segments: list[EditSegment],
    all_segments: list[TranscriptSegment],
    video_path: str,
    duration: float,
) -> EditDecisionList:
    """
    Create an EDL from AI-analyzed segments, filling gaps with KEEP.

    If AI segments don't cover all transcript indices, the missing ones
    are filled in as KEEP segments.

    Args:
        ai_segments: EditSegments from AI analysis.
        all_segments: All original transcript segments.
        video_path: Path to the source video.
        duration: Total video duration in seconds.

    Returns:
        Complete EditDecisionList covering all transcript segments.
    """
    # Track which indices are covered by AI decisions
    covered_indices: set[int] = set()
    for seg in ai_segments:
        covered_indices.update(seg.transcript_indices)

    # Find gaps and create KEEP segments for them
    all_indices = set(range(len(all_segments)))
    missing_indices = sorted(all_indices - covered_indices)

    # Create KEEP segments for missing indices
    gap_segments: list[EditSegment] = []
    if missing_indices:
        # Group consecutive indices into segments
        start_idx = missing_indices[0]
        end_idx = missing_indices[0]

        for idx in missing_indices[1:]:
            if idx == end_idx + 1:
                # Consecutive, extend the range
                end_idx = idx
            else:
                # Gap - save current segment and start new one
                gap_segments.append(
                    EditSegment(
                        start=all_segments[start_idx].start,
                        end=all_segments[end_idx].end,
                        action=EditAction.KEEP,
                        reason=None,
                        transcript_indices=list(range(start_idx, end_idx + 1)),
                    )
                )
                start_idx = idx
                end_idx = idx

        # Add the last segment
        gap_segments.append(
            EditSegment(
                start=all_segments[start_idx].start,
                end=all_segments[end_idx].end,
                action=EditAction.KEEP,
                reason=None,
                transcript_indices=list(range(start_idx, end_idx + 1)),
            )
        )

    # Combine AI segments with gap segments and sort by start time
    all_edit_segments = ai_segments + gap_segments
    all_edit_segments.sort(key=lambda s: s.start)

    return EditDecisionList(
        source_video=video_path,
        segments=all_edit_segments,
        total_duration=duration,
    )


def edit_video(
    video_path: str,
    output_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
    edl_path: Optional[str] = None,
    auto_apply: bool = False,
    use_ai: bool = False,
) -> dict:
    """
    Orchestrate the video editing workflow.

    This function handles the full editing pipeline:
    1. Load or generate transcript (SRT file)
    2. Get video duration
    3. Format transcript for AI review
    4. Create EDL (all KEEP if use_ai=False, or AI-analyzed if use_ai=True)
    5. Save EDL to JSON
    6. Return information for review or apply cuts

    Args:
        video_path: Path to the input video file
        output_path: Optional path for edited video output (used if auto_apply=True)
        transcript_path: Optional path to existing SRT file. If None, generates one.
        edl_path: Optional path for EDL JSON file. If None, generates default path.
        auto_apply: If True, applies cuts immediately. If False (default), saves EDL
                   for human review.
        use_ai: If True, uses AI to analyze transcript and suggest edits.
                If False (default), creates all-KEEP EDL for manual review.

    Returns:
        Dictionary containing:
            - edl_path: Path to the saved EDL JSON file
            - transcript_for_review: Formatted transcript text for AI review
            - video_duration: Duration of the video in seconds
            - segment_count: Number of transcript segments
            - ai_used: Whether AI analysis was used

    Raises:
        FileNotFoundError: If video file does not exist
        LLMClientError: If use_ai=True and API call fails
    """
    # Validate video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Step 1: Load or generate transcript
    if transcript_path is None:
        # Generate transcript using the subtitle pipeline
        transcript_path = process_video(video_path)

    segments = _load_transcript(transcript_path)

    # Step 2: Get video duration
    duration = get_video_duration(video_path)

    # Step 3: Format transcript for AI review
    transcript_for_review = format_transcript_for_editing(segments)

    # Step 4: Create EDL - either AI-analyzed or all-KEEP
    if use_ai:
        ai_segments, raw_response = _analyze_with_ai(transcript_for_review, segments, use_ai=True)
        if ai_segments:
            edl = _create_edl_from_ai_segments(ai_segments, segments, video_path, duration)
            # Check if all segments are REMOVE
            if all(seg.action == EditAction.REMOVE for seg in edl.segments):
                print(
                    "Warning: AI suggested removing all segments. "
                    "Review carefully before applying.",
                    file=sys.stderr,
                )
        else:
            # Parse failure - show what we got and fall back to all-KEEP
            print(
                "Warning: Failed to parse AI response. Falling back to all-KEEP EDL.",
                file=sys.stderr,
            )
            if raw_response:
                # Show first 500 chars of response for debugging
                preview = raw_response[:500]
                if len(raw_response) > 500:
                    preview += "..."
                print(
                    f"AI response preview:\n{preview}",
                    file=sys.stderr,
                )
            edl = _create_initial_edl(segments, video_path, duration)
    else:
        edl = _create_initial_edl(segments, video_path, duration)

    # Step 5: Determine EDL path and save
    if edl_path is None:
        video_path_obj = Path(video_path)
        edl_path = str(video_path_obj.with_suffix(".edl.json"))

    with open(edl_path, "w", encoding="utf-8") as f:
        f.write(edl_to_json(edl))

    # Step 6: Return information for AI review (or apply cuts)
    result = {
        "edl_path": edl_path,
        "transcript_for_review": transcript_for_review,
        "video_duration": duration,
        "segment_count": len(segments),
        "video_path": video_path,
        "transcript_path": transcript_path,
        "ai_used": use_ai,
    }

    if auto_apply:
        # Apply cuts immediately
        edited_video_path = apply_edl_to_video(video_path, edl_path, output_path)
        result["edited_video_path"] = edited_video_path

    return result


def apply_edl_to_video(
    video_path: str,
    edl_path: str,
    output_path: Optional[str] = None,
    srt_path: Optional[str] = None,
) -> str | dict:
    """
    Apply a previously generated/reviewed EDL to a video.

    Args:
        video_path: Path to the input video file
        edl_path: Path to the EDL JSON file
        output_path: Optional path for output video. If None, generates temp file.
        srt_path: Optional path to input SRT file. If provided, generates adjusted SRT.

    Returns:
        If srt_path is None: Path to the edited video file (str) for backward compatibility
        If srt_path is provided: Dict with 'video_path' and 'srt_path' keys

    Raises:
        FileNotFoundError: If video, EDL, or SRT file does not exist
        json.JSONDecodeError: If EDL file is not valid JSON
    """
    # Validate files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(edl_path):
        raise FileNotFoundError(f"EDL file not found: {edl_path}")

    # Load EDL from JSON - use json.load directly on file handle for memory efficiency
    with open(edl_path, "r", encoding="utf-8") as f:
        edl_data = json.load(f)

    edl = edl_from_dict(edl_data)

    # Apply cuts using video_cutter
    edited_video_path = cut_video(video_path, edl, output_path)

    # If SRT file provided, adjust it for the cut video
    if srt_path is not None:
        # Generate output SRT path based on video output path
        video_output_path = Path(edited_video_path)
        srt_output_path = str(video_output_path.with_suffix(".srt"))

        adjusted_srt_path = adjust_srt_for_edl(srt_path, edl, srt_output_path)

        return {
            "video_path": edited_video_path,
            "srt_path": adjusted_srt_path,
        }

    # Return string for backward compatibility when no SRT provided
    return edited_video_path
