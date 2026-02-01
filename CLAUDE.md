## Development Environment

**Package manager:** uv (not pip)
- Always use `uv run` to execute Python commands
- Example: `uv run pytest tests/` (NOT `python -m pytest`)
- Example: `uv run python scripts/pipeline.py` (NOT `python scripts/pipeline.py`)

**Language:** All video content is in **Icelandic** (language code: `is`)
- The CLI defaults to Icelandic (`-l is`)

**Model:** Always use `large-v2` for transcription (better accuracy for Icelandic)
- The CLI defaults to `large-v2`

---

## Project Status (Last Updated: 2026-02-01)

**Phase:** Subtitle pipeline complete and tested

**Completed:**
- ✅ Audio extraction (`scripts/audio_extractor.py`) - ffmpeg-based WAV extraction
- ✅ Transcription (`scripts/transcription.py`) - faster-whisper with large-v2 model
- ✅ Subtitle writing (`scripts/subtitle_writer.py`) - SRT and VTT format output
- ✅ Transcript correction (`scripts/transcript_corrector.py`) - LLM-assisted review utilities
- ✅ Main pipeline (`scripts/pipeline.py`) - `process_video()` orchestrates full workflow
- ✅ CLI interface (`scripts/cli.py`) - `uv run python -m scripts video.mp4`
- ✅ Browser-based user testing with video player
- ✅ Test suite - 143 tests passing

**Next up:**
- [ ] Actual video editing features (beyond subtitles)

---

The objective is to create an application which edits videos for the user. This will adhere to modern software development standards
and includes multiple agents, each with a specific role and domain knowledge.

We are using test driven development.

## Agentic Workflow Rules

1. **Delegate code writing to subagents** - The main agent should not write code directly. Instead, use the Task tool to spawn subagents for all code writing tasks.

2. **No browser testing** - To verify subtitle content, just read the SRT/VTT files directly. Don't use Playwright for verification.

3. **Browser is for the user** - If opening the browser to play video, just open and play it. Don't take screenshots or "verify" things - let the user watch.

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
from scripts.transcription import transcribe_audio
from scripts.transcript_corrector import format_for_review, correct_transcript

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
