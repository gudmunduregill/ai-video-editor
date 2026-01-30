"""Shared pytest fixtures for the test suite."""

import pytest

from scripts.transcription import TranscriptSegment


@pytest.fixture
def sample_segments() -> list[TranscriptSegment]:
    """Return a list of sample transcript segments for testing."""
    return [
        TranscriptSegment(start=0.0, end=2.5, text="Hello, world!"),
        TranscriptSegment(start=2.6, end=5.1, text="This is a test."),
        TranscriptSegment(start=5.2, end=8.0, text="Goodbye!"),
    ]


@pytest.fixture
def single_segment() -> TranscriptSegment:
    """Return a single sample transcript segment."""
    return TranscriptSegment(start=0.0, end=2.5, text="Hello, world!")
