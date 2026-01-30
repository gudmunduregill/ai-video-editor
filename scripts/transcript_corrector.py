"""Module for correcting transcripts with LLM assistance."""

import re

from scripts.transcription import TranscriptSegment


def format_for_review(
    segments: list[TranscriptSegment], context: str | None = None
) -> str:
    """
    Format transcript segments for LLM review.

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


def parse_corrected_transcript(
    corrected_text: str, original_segments: list[TranscriptSegment]
) -> list[TranscriptSegment]:
    """
    Parse LLM-corrected text back into TranscriptSegment objects.

    Args:
        corrected_text: The corrected transcript text from LLM
        original_segments: Original segments to preserve timestamps from

    Returns:
        New list of TranscriptSegment objects with corrected text

    Raises:
        ValueError: If corrected_text is empty
        IndexError: If segment indices don't match
    """
    if not corrected_text or not corrected_text.strip():
        raise ValueError("Corrected text cannot be empty")

    # Pattern to match lines like: [0] 0.0-2.5: Hello, world!
    # or [0]  0.0 - 2.5:  Hello, world! (with extra whitespace)
    pattern = r"\[(\d+)\]\s*[\d.]+\s*-\s*[\d.]+\s*:\s*(.+)"

    result = []
    lines = corrected_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line)
        if match:
            index = int(match.group(1))
            text = match.group(2).strip()

            # Use original segment's timestamps
            original = original_segments[index]
            result.append(
                TranscriptSegment(
                    start=original.start,
                    end=original.end,
                    text=text,
                )
            )

    if not result:
        raise IndexError("No valid segments found in corrected text")

    return result


def correct_transcript(
    segments: list[TranscriptSegment], corrections: dict[int, str]
) -> list[TranscriptSegment]:
    """
    Apply specific corrections to segments by index.

    Args:
        segments: Original list of TranscriptSegment objects
        corrections: Dictionary mapping segment indices to corrected text

    Returns:
        New list of TranscriptSegment objects with corrections applied

    Raises:
        KeyError: If correction index is invalid (out of range or negative)
    """
    # Validate all indices first
    for index in corrections:
        if index < 0 or index >= len(segments):
            raise KeyError(f"Invalid segment index: {index}")

    # Create new segments with corrections applied
    result = []
    for i, segment in enumerate(segments):
        if i in corrections:
            text = corrections[i]
        else:
            text = segment.text

        result.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=text,
            )
        )

    return result
