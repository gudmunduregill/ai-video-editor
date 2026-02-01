"""Command-line interface for the video-to-subtitle pipeline."""

import argparse
import sys
from typing import List, Optional

from scripts.edit_pipeline import apply_edl_to_video, edit_video
from scripts.exceptions import (
    AudioExtractionError,
    EDLValidationError,
    TranscriptionError,
    VideoCuttingError,
)
from scripts.pipeline import process_video


MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v2"]
FORMAT_CHOICES = ["srt", "vtt"]
SUBCOMMANDS = ["subtitle", "edit", "apply-edl"]


def _is_subcommand(arg: str) -> bool:
    """Check if an argument is a subcommand."""
    return arg in SUBCOMMANDS


def _preprocess_args(args: List[str]) -> List[str]:
    """
    Preprocess args for backward compatibility.

    If the first argument is not a subcommand, prepend 'subtitle'.
    """
    if not args:
        return args

    first_arg = args[0]

    # Check if first arg is a subcommand
    if _is_subcommand(first_arg):
        return args

    # Check if first arg is a help flag
    if first_arg in ["-h", "--help"]:
        return args

    # Otherwise, it's a video path - prepend 'subtitle' for backward compatibility
    return ["subtitle"] + args


def _create_subtitle_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create the subtitle subcommand parser."""
    subtitle_parser = subparsers.add_parser(
        "subtitle",
        help="Generate subtitles from video files using Whisper",
        description="Generate subtitles from video files using Whisper.",
    )

    subtitle_parser.add_argument(
        "video",
        help="Path to the input video file",
    )

    subtitle_parser.add_argument(
        "--output",
        "-o",
        help="Path for the output subtitle file (default: same name as video with .srt extension)",
        default=None,
    )

    subtitle_parser.add_argument(
        "--model",
        "-m",
        help="Whisper model size (default: large-v2)",
        choices=MODEL_CHOICES,
        default="large-v2",
    )

    subtitle_parser.add_argument(
        "--language",
        "-l",
        help="Language code for transcription (default: 'is' for Icelandic)",
        default="is",
    )

    subtitle_parser.add_argument(
        "--format",
        "-f",
        help="Output subtitle format (default: srt)",
        choices=FORMAT_CHOICES,
        default="srt",
    )


def _create_edit_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create the edit subcommand parser."""
    edit_parser = subparsers.add_parser(
        "edit",
        help="Generate EDL (Edit Decision List) for review",
        description="Generate an EDL (Edit Decision List) from video for review.",
    )

    edit_parser.add_argument(
        "video",
        help="Path to the input video file",
    )

    edit_parser.add_argument(
        "--output",
        "-o",
        help="Path for the output EDL file (default: video.edl.json)",
        default=None,
    )

    edit_parser.add_argument(
        "--transcript",
        "-t",
        help="Path to existing transcript (SRT file) to use",
        default=None,
    )

    edit_parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-apply cuts without review",
        default=False,
    )


def _create_apply_edl_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create the apply-edl subcommand parser."""
    apply_edl_parser = subparsers.add_parser(
        "apply-edl",
        help="Apply reviewed EDL to video",
        description="Apply a reviewed EDL (Edit Decision List) to cut the video.",
    )

    apply_edl_parser.add_argument(
        "video",
        help="Path to the input video file",
    )

    apply_edl_parser.add_argument(
        "edl",
        help="Path to the EDL JSON file",
    )

    apply_edl_parser.add_argument(
        "--output",
        "-o",
        help="Path for the output video file (default: video_edited.mp4)",
        default=None,
    )


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args: List of arguments to parse. If None, uses sys.argv[1:].

    Returns:
        Parsed arguments namespace with command-specific options.
    """
    if args is None:
        args = sys.argv[1:]

    # Preprocess args for backward compatibility
    args = _preprocess_args(args)

    parser = argparse.ArgumentParser(
        prog="python -m scripts",
        description="Video editing and subtitle generation CLI.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
    )

    _create_subtitle_parser(subparsers)
    _create_edit_parser(subparsers)
    _create_apply_edl_parser(subparsers)

    parsed = parser.parse_args(args)

    # If no command was specified (empty args), show help
    if parsed.command is None:
        parser.print_help()
        sys.exit(0)

    return parsed


def _run_subtitle(args: argparse.Namespace) -> int:
    """Run the subtitle subcommand."""
    print(f"Processing video: {args.video}")

    try:
        output_path = process_video(
            args.video,
            output_path=args.output,
            model_size=args.model,
            language=args.language,
            subtitle_format=args.format,
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


def _run_edit(args: argparse.Namespace) -> int:
    """Run the edit subcommand."""
    print(f"Generating EDL for video: {args.video}")

    try:
        result = edit_video(
            args.video,
            output_path=None,  # output_path is for auto-apply output video
            transcript_path=args.transcript,
            edl_path=args.output,
            auto_apply=args.auto,
        )
        print(f"EDL saved to: {result['edl_path']}")
        print(f"Segments: {result['segment_count']}")
        print(f"Duration: {result['video_duration']:.2f}s")

        if args.auto and "edited_video_path" in result:
            print(f"Edited video saved to: {result['edited_video_path']}")

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

    except VideoCuttingError as e:
        print(f"Error: Failed to cut video: {e}", file=sys.stderr)
        return 1

    except EDLValidationError as e:
        print(f"Error: Invalid EDL: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _run_apply_edl(args: argparse.Namespace) -> int:
    """Run the apply-edl subcommand."""
    print(f"Applying EDL to video: {args.video}")

    try:
        output_path = apply_edl_to_video(
            args.video,
            args.edl,
            args.output,
        )
        print(f"Edited video saved to: {output_path}")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except VideoCuttingError as e:
        print(f"Error: Failed to cut video: {e}", file=sys.stderr)
        return 1

    except EDLValidationError as e:
        print(f"Error: Invalid EDL: {e}", file=sys.stderr)
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: List of arguments to parse. If None, uses sys.argv[1:].

    Returns:
        Exit code: 0 for success, 1 for error.
    """
    parsed_args = parse_args(args)

    if parsed_args.command == "subtitle":
        return _run_subtitle(parsed_args)
    elif parsed_args.command == "edit":
        return _run_edit(parsed_args)
    elif parsed_args.command == "apply-edl":
        return _run_apply_edl(parsed_args)
    else:
        # Should not happen if argparse is configured correctly
        print(f"Error: Unknown command: {parsed_args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
