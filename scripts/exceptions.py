"""Custom exceptions for subtitle generation and audio processing."""


class SubtitleError(Exception):
    """Base exception for subtitle generation errors."""

    pass


class AudioExtractionError(SubtitleError):
    """Raised when audio extraction from video fails."""

    pass


class TranscriptionError(SubtitleError):
    """Raised when audio transcription fails."""

    pass


class TranscriptCorrectionError(SubtitleError):
    """Raised when transcript correction fails."""

    pass
