"""Audio extraction module for extracting audio from video files."""

import os
import subprocess
import tempfile
from typing import Optional

import ffmpeg  # type: ignore[import-untyped]

from scripts.exceptions import AudioExtractionError


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio from a video file.

    Args:
        video_path: Path to the input video file
        output_path: Optional path for output audio file.
                     If None, creates a temp file with .wav extension.

    Returns:
        Path to the extracted audio file (WAV format, 16kHz mono for Whisper compatibility)

    Raises:
        FileNotFoundError: If video file doesn't exist
        AudioExtractionError: If ffmpeg fails to extract audio
    """
    # Validate input file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Determine output path
    if output_path is None:
        # Create a temporary file with .wav extension
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    try:
        # Check if ffmpeg is available
        _check_ffmpeg_available()

        # Build ffmpeg command using ffmpeg-python
        # Output: WAV format, 16kHz sample rate, mono channel
        stream = (
            ffmpeg
            .input(video_path)
            .output(
                output_path,
                acodec="pcm_s16le",  # PCM 16-bit little-endian (standard WAV)
                ar=16000,  # Sample rate: 16kHz (optimal for Whisper)
                ac=1,  # Audio channels: 1 (mono)
                format="wav",
            )
            .overwrite_output()  # Allow overwriting existing files
        )

        # Run the ffmpeg command
        stream.run(capture_stdout=True, capture_stderr=True)

    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8") if e.stderr else "Unknown error"
        raise AudioExtractionError(
            f"Failed to extract audio from {video_path}: {stderr}"
        ) from e
    except Exception as e:
        # Handle other unexpected errors
        if isinstance(e, AudioExtractionError):
            raise
        raise AudioExtractionError(
            f"Failed to extract audio from {video_path}: {str(e)}"
        ) from e

    return output_path


def _check_ffmpeg_available() -> None:
    """
    Check if ffmpeg binary is available in the system.

    Raises:
        AudioExtractionError: If ffmpeg is not found
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise AudioExtractionError(
            "ffmpeg is not installed or not found in PATH. "
            "Please install ffmpeg to extract audio from videos."
        )
    except subprocess.CalledProcessError as e:
        raise AudioExtractionError(
            f"ffmpeg check failed: {e.stderr.decode('utf-8') if e.stderr else 'Unknown error'}"
        )
