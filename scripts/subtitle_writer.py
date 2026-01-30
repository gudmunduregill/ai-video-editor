"""Module for writing subtitle files in SRT format."""

from scripts.transcription import TranscriptSegment


def format_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds (e.g., 65.5)

    Returns:
        SRT-formatted timestamp (e.g., "00:01:05,500")
    """
    # Calculate hours, minutes, seconds, and milliseconds
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    # Extract milliseconds and round to 3 decimal places
    milliseconds = round((seconds - total_seconds) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_srt(segments: list[TranscriptSegment], output_path: str) -> None:
    """
    Write transcript segments to an SRT subtitle file.

    Args:
        segments: List of TranscriptSegment objects
        output_path: Path for the output .srt file

    Raises:
        ValueError: If segments list is empty
    """
    if not segments:
        raise ValueError("Cannot write SRT file with empty segments list")

    with open(output_path, "w", encoding="utf-8") as f:
        for index, segment in enumerate(segments, start=1):
            # Write subtitle number
            f.write(f"{index}\n")

            # Write timestamp line
            start_ts = format_srt_timestamp(segment.start)
            end_ts = format_srt_timestamp(segment.end)
            f.write(f"{start_ts} --> {end_ts}\n")

            # Write subtitle text
            f.write(f"{segment.text}\n")

            # Write blank line separator (except after last segment)
            if index < len(segments):
                f.write("\n")
