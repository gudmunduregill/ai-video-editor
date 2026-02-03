"""Pipeline module for orchestrating the video-to-subtitle workflow."""

import os
from pathlib import Path
from typing import Optional

from scripts.audio_extractor import extract_audio
from scripts.subtitle_writer import write_srt, write_vtt
from scripts.transcription import transcribe


SUPPORTED_SUBTITLE_FORMATS = ("srt", "vtt")


def derive_output_path(video_path: str, extension: str) -> str:
    """
    Derive output path from video path.

    If video is in a source/ directory, output goes to sibling output/ directory.
    Otherwise, output goes to same directory as video (backwards compatible).

    Args:
        video_path: Path to the input video file
        extension: File extension for output (e.g., ".srt", ".edl.json", "_edited.mp4")

    Returns:
        Path string for the output file
    """
    video_path_obj = Path(video_path)
    parent = video_path_obj.parent

    # Check if video is in a "source" directory
    if parent.name == "source":
        # Output goes to sibling "output" directory
        output_dir = parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        return str(output_dir / video_path_obj.stem) + extension
    else:
        # Same directory as video (backwards compatible)
        # Use stem + extension to handle complex suffixes like "_edited.mp4"
        return str(parent / video_path_obj.stem) + extension


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    model_size: str = "base",
    language: Optional[str] = None,
    subtitle_format: str = "srt",
) -> str:
    """
    Process a video file to generate subtitles.

    This function orchestrates the full video-to-subtitle workflow:
    1. Extracts audio from the video to a temporary WAV file
    2. Transcribes the audio using Whisper
    3. Writes the transcript to a subtitle file (SRT or VTT format)
    4. Cleans up the temporary audio file

    Args:
        video_path: Path to the input video file
        output_path: Optional path for output subtitle file.
                     If None, creates subtitle file with same name as video.
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large-v2")
                    Defaults to "base".
        language: Optional language code (e.g., "en"). If None, auto-detect.
        subtitle_format: Output subtitle format ("srt" or "vtt"). Defaults to "srt".

    Returns:
        Path to the generated subtitle file

    Raises:
        FileNotFoundError: If video file doesn't exist
        AudioExtractionError: If audio extraction fails
        TranscriptionError: If transcription fails
        ValueError: If transcription returns empty segments or invalid subtitle format
    """
    # Validate subtitle format
    if subtitle_format not in SUPPORTED_SUBTITLE_FORMATS:
        raise ValueError(
            f"Invalid subtitle format: '{subtitle_format}'. "
            f"Supported formats: {', '.join(SUPPORTED_SUBTITLE_FORMATS)}"
        )

    # Validate input file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Determine output subtitle path with appropriate extension
    if output_path is None:
        output_path = derive_output_path(video_path, f".{subtitle_format}")

    # Extract audio to temp file, then transcribe and write subtitle file
    temp_audio_path: Optional[str] = None
    try:
        # Step 1: Extract audio to temporary WAV file
        temp_audio_path = extract_audio(video_path)

        # Step 2: Transcribe the audio
        segments = transcribe(
            temp_audio_path,
            model_size=model_size,
            language=language,
        )

        # Step 3: Validate we have segments
        if not segments:
            raise ValueError(
                f"Transcription returned empty segments for {video_path}"
            )

        # Step 4: Write subtitle file in requested format
        if subtitle_format == "vtt":
            write_vtt(segments, output_path)
        else:
            write_srt(segments, output_path)

        return output_path

    finally:
        # Step 5: Clean up temporary audio file
        if temp_audio_path is not None and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
