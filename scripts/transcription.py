"""Transcription data structures for speech-to-text processing."""

from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    """A single segment of transcribed speech."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    text: str  # Transcribed text
