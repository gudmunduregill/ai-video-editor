"""Video cutting module for cutting and concatenating video segments.

This module provides functions to cut videos based on an Edit Decision List (EDL).
It uses FFmpeg's filter_complex to trim and concatenate video segments.
"""

import os
import subprocess
import tempfile
from typing import Optional

import ffmpeg  # type: ignore[import-untyped]

from scripts.edit_decision import EditDecisionList, EditSegment
from scripts.exceptions import EDLValidationError, VideoCuttingError


def cut_video(
    video_path: str,
    edl: EditDecisionList,
    output_path: Optional[str] = None,
) -> str:
    """
    Cut a video based on an Edit Decision List.

    Takes a video file and an EDL, then produces a new video containing
    only the segments marked as KEEP, concatenated in order.

    Args:
        video_path: Path to the input video file
        edl: EditDecisionList defining which segments to keep
        output_path: Optional path for output video file.
                     If None, creates a temp file with .mp4 extension.

    Returns:
        Path to the edited video file

    Raises:
        FileNotFoundError: If video file doesn't exist
        EDLValidationError: If EDL is invalid for cutting
        VideoCuttingError: If FFmpeg fails to cut the video
    """
    # Validate input file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Validate EDL before cutting
    _validate_edl_for_cutting(edl)

    # Determine output path
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

    try:
        # Check if ffmpeg is available
        _check_ffmpeg_available()

        # Get only the KEEP segments
        keep_segments = edl.keep_segments

        # Build the filter_complex string
        filter_str = _build_ffmpeg_filter(keep_segments)

        # Build ffmpeg command directly (ffmpeg-python doesn't support multiple -map args)
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", video_path,
            "-filter_complex", filter_str,
            "-map", "[outv]",
            "-map", "[outa]",
            output_path,
        ]

        # Run the ffmpeg command
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8") if e.stderr else "Unknown error"
        raise VideoCuttingError(
            f"Failed to cut video {video_path}: {stderr}"
        ) from e
    except Exception as e:
        if isinstance(e, (EDLValidationError, VideoCuttingError, FileNotFoundError)):
            raise
        raise VideoCuttingError(
            f"Failed to cut video {video_path}: {str(e)}"
        ) from e

    return output_path


def _build_ffmpeg_filter(keep_segments: list[EditSegment]) -> str:
    """
    Build an FFmpeg filter_complex string for trimming and concatenating segments.

    Creates a filter that:
    1. Trims each segment from the input video/audio
    2. Resets timestamps with setpts/asetpts
    3. Concatenates all segments together

    Args:
        keep_segments: List of EditSegment objects to keep (must be non-empty)

    Returns:
        FFmpeg filter_complex string

    Raises:
        ValueError: If keep_segments is empty

    Example output for segments at [0-5s], [10-15s]:
        [0:v]trim=start=0:end=5,setpts=PTS-STARTPTS[v0];
        [0:a]atrim=start=0:end=5,asetpts=PTS-STARTPTS[a0];
        [0:v]trim=start=10:end=15,setpts=PTS-STARTPTS[v1];
        [0:a]atrim=start=10:end=15,asetpts=PTS-STARTPTS[a1];
        [v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]
    """
    if not keep_segments:
        raise ValueError("Must provide at least one segment to build filter")

    filter_parts = []

    # Build trim filters for each segment
    for i, segment in enumerate(keep_segments):
        # Video trim
        video_trim = (
            f"[0:v]trim=start={segment.start}:end={segment.end},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
        filter_parts.append(video_trim)

        # Audio trim
        audio_trim = (
            f"[0:a]atrim=start={segment.start}:end={segment.end},"
            f"asetpts=PTS-STARTPTS[a{i}]"
        )
        filter_parts.append(audio_trim)

    # Build concat filter
    # Input labels: [v0][a0][v1][a1]...
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(keep_segments)))
    concat_filter = f"{concat_inputs}concat=n={len(keep_segments)}:v=1:a=1[outv][outa]"
    filter_parts.append(concat_filter)

    return ";".join(filter_parts)


def _validate_edl_for_cutting(edl: EditDecisionList) -> None:
    """
    Validate an EDL before cutting.

    Checks:
    - At least one KEEP segment exists
    - No overlapping KEEP segments
    - All segments are within the video duration
    - Segment times are valid (start < end, non-negative)

    Args:
        edl: EditDecisionList to validate

    Raises:
        EDLValidationError: If validation fails
    """
    keep_segments = edl.keep_segments

    # Check for at least one KEEP segment
    if not keep_segments:
        raise EDLValidationError(
            "EDL must have at least one KEEP segment for cutting"
        )

    # Validate each segment
    for segment in keep_segments:
        # Check for negative times
        if segment.start < 0:
            raise EDLValidationError(
                f"Invalid segment: negative start time {segment.start}"
            )

        # Check start < end
        if segment.start >= segment.end:
            raise EDLValidationError(
                f"Invalid segment: start time {segment.start} must be before "
                f"end time {segment.end}"
            )

        # Check within duration
        if segment.end > edl.total_duration:
            raise EDLValidationError(
                f"Segment end time {segment.end} exceeds video duration "
                f"{edl.total_duration}"
            )

    # Check for overlapping KEEP segments
    # Sort by start time for overlap detection
    sorted_segments = sorted(keep_segments, key=lambda s: s.start)
    for i in range(len(sorted_segments) - 1):
        current = sorted_segments[i]
        next_seg = sorted_segments[i + 1]
        if current.end > next_seg.start:
            raise EDLValidationError(
                f"Overlapping KEEP segments detected: "
                f"[{current.start}-{current.end}] and [{next_seg.start}-{next_seg.end}]"
            )


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video file using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds as a float

    Raises:
        FileNotFoundError: If video file doesn't exist
        VideoCuttingError: If ffprobe fails to get duration
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration_str = result.stdout.strip()
        return float(duration_str)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if e.stderr else "Unknown error"
        raise VideoCuttingError(
            f"Failed to get duration for {video_path}: {stderr}"
        ) from e
    except ValueError as e:
        raise VideoCuttingError(
            f"Failed to parse duration for {video_path}: {str(e)}"
        ) from e


def _check_ffmpeg_available() -> None:
    """
    Check if ffmpeg binary is available in the system.

    Raises:
        VideoCuttingError: If ffmpeg is not found
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise VideoCuttingError(
            "ffmpeg is not installed or not found in PATH. "
            "Please install ffmpeg to cut videos."
        )
    except subprocess.CalledProcessError as e:
        raise VideoCuttingError(
            f"ffmpeg check failed: {e.stderr.decode('utf-8') if e.stderr else 'Unknown error'}"
        )
