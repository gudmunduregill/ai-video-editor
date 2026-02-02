"""Edit pipeline module for orchestrating the video editing workflow.

This module provides functions for the complete video editing pipeline:
1. Generating/loading transcripts
2. Formatting transcripts for AI review
3. Creating initial EDLs
4. Applying EDLs to cut videos
"""

import os
import re
from pathlib import Path
from typing import Generator, Optional

from scripts.edit_analyzer import format_transcript_for_editing
import json

from scripts.edit_decision import (
    EditAction,
    EditDecisionList,
    EditSegment,
    edl_from_dict,
    edl_to_json,
)
from scripts.pipeline import process_video
from scripts.transcription import TranscriptSegment
from scripts.video_cutter import cut_video, get_video_duration


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


def edit_video(
    video_path: str,
    output_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
    edl_path: Optional[str] = None,
    auto_apply: bool = False,
) -> dict:
    """
    Orchestrate the video editing workflow.

    This function handles the full editing pipeline:
    1. Load or generate transcript (SRT file)
    2. Get video duration
    3. Format transcript for AI review
    4. Create initial EDL (all segments as KEEP)
    5. Save EDL to JSON
    6. Return information for AI review

    Args:
        video_path: Path to the input video file
        output_path: Optional path for edited video output (used if auto_apply=True)
        transcript_path: Optional path to existing SRT file. If None, generates one.
        edl_path: Optional path for EDL JSON file. If None, generates default path.
        auto_apply: If True, applies cuts immediately. If False (default), saves EDL
                   for human review.

    Returns:
        Dictionary containing:
            - edl_path: Path to the saved EDL JSON file
            - transcript_for_review: Formatted transcript text for AI review
            - video_duration: Duration of the video in seconds
            - segment_count: Number of transcript segments

    Raises:
        FileNotFoundError: If video file does not exist
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

    # Step 4: Create initial EDL (all KEEP)
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
) -> str:
    """
    Apply a previously generated/reviewed EDL to a video.

    Args:
        video_path: Path to the input video file
        edl_path: Path to the EDL JSON file
        output_path: Optional path for output video. If None, generates temp file.

    Returns:
        Path to the edited video file

    Raises:
        FileNotFoundError: If video or EDL file does not exist
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
    return cut_video(video_path, edl, output_path)
