"""Command-line interface for the video-to-subtitle pipeline."""

import argparse
import sys
from typing import List, Optional

from scripts.exceptions import AudioExtractionError, TranscriptionError
from scripts.pipeline import process_video


MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v2"]
FORMAT_CHOICES = ["srt", "vtt"]


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args: List of arguments to parse. If None, uses sys.argv[1:].

    Returns:
        Parsed arguments namespace with video, output, model, and language.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts",
        description="Generate subtitles from video files using Whisper.",
    )

    parser.add_argument(
        "video",
        help="Path to the input video file",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Path for the output SRT file (default: same name as video with .srt extension)",
        default=None,
    )

    parser.add_argument(
        "--model",
        "-m",
        help="Whisper model size (default: large-v2)",
        choices=MODEL_CHOICES,
        default="large-v2",
    )

    parser.add_argument(
        "--language",
        "-l",
        help="Language code for transcription (default: 'is' for Icelandic)",
        default="is",
    )

    parser.add_argument(
        "--format",
        "-f",
        help="Output subtitle format (default: srt)",
        choices=FORMAT_CHOICES,
        default="srt",
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: List of arguments to parse. If None, uses sys.argv[1:].

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    parsed_args = parse_args(args)

    print(f"Processing video: {parsed_args.video}")

    try:
        output_path = process_video(
            parsed_args.video,
            output_path=parsed_args.output,
            model_size=parsed_args.model,
            language=parsed_args.language,
            subtitle_format=parsed_args.format,
        )
        print(f"Subtitles saved to: {output_path}")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except AudioExtractionError as e:
        print(f"Error: Failed to extract audio from video: {e}", file=sys.stderr)
        return 1

    except TranscriptionError as e:
        print(f"Error: Failed to transcribe audio: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
