"""Pipeline module for orchestrating the video-to-subtitle workflow."""

import os
from pathlib import Path
from typing import Optional

from scripts.audio_extractor import extract_audio
from scripts.subtitle_writer import write_srt
from scripts.transcription import transcribe


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    model_size: str = "base",
    language: Optional[str] = None,
) -> str:
    """
    Process a video file to generate subtitles.

    This function orchestrates the full video-to-subtitle workflow:
    1. Extracts audio from the video to a temporary WAV file
    2. Transcribes the audio using Whisper
    3. Writes the transcript to an SRT subtitle file
    4. Cleans up the temporary audio file

    Args:
        video_path: Path to the input video file
        output_path: Optional path for output SRT file.
                     If None, creates SRT file with same name as video.
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large-v2")
                    Defaults to "base".
        language: Optional language code (e.g., "en"). If None, auto-detect.

    Returns:
        Path to the generated SRT subtitle file

    Raises:
        FileNotFoundError: If video file doesn't exist
        AudioExtractionError: If audio extraction fails
        TranscriptionError: If transcription fails
        ValueError: If transcription returns empty segments
    """
    # Validate input file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Determine output SRT path
    if output_path is None:
        video_path_obj = Path(video_path)
        output_path = str(video_path_obj.with_suffix(".srt"))

    # Extract audio to temp file, then transcribe and write SRT
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

        # Step 4: Write SRT file
        write_srt(segments, output_path)

        return output_path

    finally:
        # Step 5: Clean up temporary audio file
        if temp_audio_path is not None and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
