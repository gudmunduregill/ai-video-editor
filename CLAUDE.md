## Development Environment

**Package manager:** uv (not pip)
- Always use `uv run` to execute Python commands
- Example: `uv run pytest tests/` (NOT `python -m pytest`)
- Example: `uv run python scripts/pipeline.py` (NOT `python scripts/pipeline.py`)

**Language:** All video content is in **Icelandic** (language code: `is`)
- The CLI defaults to Icelandic (`-l is`)

---

## Project Status (Last Updated: 2026-02-01)

**Phase:** CLI complete, ready for additional features

**Completed:**
- ✅ Audio extraction (`scripts/audio_extractor.py`) - ffmpeg-based WAV extraction
- ✅ Transcription (`scripts/transcription.py`) - faster-whisper integration
- ✅ Subtitle writing (`scripts/subtitle_writer.py`) - SRT format output
- ✅ Transcript correction (`scripts/transcript_corrector.py`) - LLM-assisted review utilities
- ✅ Main pipeline (`scripts/pipeline.py`) - `process_video()` orchestrates full workflow
- ✅ CLI interface (`scripts/cli.py`) - `uv run python -m scripts video.mp4`
- ✅ Test suite - 113 tests passing

**Next up:**
- [ ] VTT subtitle format support
- [ ] Actual video editing features (beyond subtitles)

**Known issues:**
- Docs reference `src/` but code lives in `scripts/`

---

The objective is to create an application which edits videos for the user. This will adhere to modern software development standards
and includes multiple agents, each with a specific role and domain knowledge.

We are using test driven development.

## Agentic Workflow Rules

1. **Delegate code writing to subagents** - The main agent should not write code directly. Instead, use the Task tool to spawn subagents for all code writing tasks.

## Transcript Correction Workflow

### Workflow Steps

1. **Run transcription pipeline** - Extract audio from video, then run Whisper API to generate initial transcript
2. **Spawn reviewer subagent** - Pass transcript and video context to a reviewer agent for correction
3. **Reviewer returns corrections** - Subagent identifies errors and returns corrections
4. **Apply corrections** - Use correction functions to fix transcript and write final SRT

### Key Functions

- `format_for_review()` - Formats transcript segments for LLM review (numbered lines with timestamps)
- `parse_corrected_transcript()` - Parses corrected text back into segment objects
- `correct_transcript()` - Applies targeted corrections to specific segments by index

### Usage Example

```python
from src.transcription import transcribe_audio
from src.transcript_review import format_for_review, correct_transcript

# 1. Get initial transcript
segments = transcribe_audio("audio.wav")

# 2. Format for reviewer subagent
review_text = format_for_review(segments)

# 3. After reviewer returns corrections, apply them
corrections = {
    0: "Fixed text for first segment",
    5: "Fixed text for sixth segment"
}
corrected_segments = correct_transcript(segments, corrections)
```

In a Claude Code session, spawn a reviewer subagent with the formatted transcript and any relevant context (video topic, speaker names, technical terms). The reviewer analyzes and returns a corrections dict.
