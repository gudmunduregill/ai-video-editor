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

## Transcription Workflow

For any new video transcription, follow this complete workflow:

### Steps

1. **Transcribe** - Run CLI to generate SRT
   ```bash
   uv run python -m scripts video.mp4 -o output.srt
   ```

2. **Icelandic Review** - Spawn `icelandic-reviewer` agent (or `general-purpose` with reviewer prompt) to review the SRT file. The reviewer:
   - Reads entire transcript to understand context
   - Identifies Whisper errors (mishearings, gibberish, artifacts)
   - Returns corrections dict

3. **Apply Corrections** - Use `correct_transcript()` to apply the corrections

4. **Generate VTT** - Re-run CLI with `-f vtt` or use `write_vtt()` directly

5. **Open Browser** - Start HTTP server and open video player for user to watch (no screenshots, just play)

### Key Functions

- `correct_transcript(segments, corrections)` - Applies corrections by segment index

### Agents

- **icelandic-reviewer** - Reviews Icelandic subtitles, knows slang (e.g., "skvísa" = girl/chick), marks unclear text as [óljóst]
