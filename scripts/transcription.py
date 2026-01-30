"""Transcription module for speech-to-text processing using faster-whisper."""

import os
from dataclasses import dataclass

from faster_whisper import WhisperModel  # type: ignore[import-untyped]

from scripts.exceptions import TranscriptionError


@dataclass
class TranscriptSegment:
    """A single segment of transcribed speech."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str  # Transcribed text


def transcribe(
    audio_path: str,
    model_size: str = "base",
    language: str | None = None,
) -> list[TranscriptSegment]:
    """
    Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large-v2")
        language: Optional language code (e.g., "en"). If None, auto-detect.

    Returns:
        List of TranscriptSegment objects with timestamps

    Raises:
        FileNotFoundError: If audio file doesn't exist
        TranscriptionError: If transcription fails
    """
    # Validate input file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        # Initialize the Whisper model
        # Use compute_type="int8" for CPU (no GPU on this system)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        # Transcribe the audio file
        # Returns an iterator of segments and transcription info
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
        )

        # Convert iterator to list of TranscriptSegment objects
        segments: list[TranscriptSegment] = []
        for segment in segments_iter:
            segments.append(
                TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                )
            )

        return segments

    except FileNotFoundError:
        # Re-raise FileNotFoundError as-is
        raise
    except Exception as e:
        # Wrap other errors in TranscriptionError
        raise TranscriptionError(f"Failed to transcribe {audio_path}: {str(e)}") from e
