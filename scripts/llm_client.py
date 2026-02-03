"""LLM client module for AI-powered video editing analysis.

This module provides functions for interacting with the Claude API
to analyze video transcripts and suggest edits.
"""

import os
import re
from pathlib import Path

import anthropic


class LLMClientError(Exception):
    """Raised when LLM client operations fail."""

    pass


def get_api_key() -> str:
    """
    Get the Anthropic API key from environment variables.

    Returns:
        The API key string.

    Raises:
        LLMClientError: If ANTHROPIC_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMClientError("ANTHROPIC_API_KEY environment variable not set")
    return api_key


def load_agent_prompt(agent_name: str) -> str:
    """
    Load an agent prompt from the .claude/agents directory.

    Loads the markdown file and strips YAML frontmatter (the part between
    --- lines at the start of the file).

    Args:
        agent_name: Name of the agent (e.g., "video-editor")

    Returns:
        The prompt content with YAML frontmatter removed.

    Raises:
        LLMClientError: If the agent file does not exist or cannot be read.
    """
    # Find the project root by looking for .claude directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    agent_path = project_root / ".claude" / "agents" / f"{agent_name}.md"

    if not agent_path.exists():
        raise LLMClientError(f"Agent prompt file not found: {agent_path}")

    try:
        content = agent_path.read_text(encoding="utf-8")
    except OSError as e:
        raise LLMClientError(f"Failed to read agent prompt file: {e}")

    # Strip YAML frontmatter (content between --- at the start)
    # Pattern matches: start of string, ---, any content, ---, newline
    frontmatter_pattern = r"^---\n.*?\n---\n"
    content = re.sub(frontmatter_pattern, "", content, count=1, flags=re.DOTALL)

    return content.strip()


def analyze_transcript(
    transcript: str,
    agent_prompt: str,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """
    Analyze a transcript using Claude API.

    Sends the transcript to Claude with the given agent prompt to get
    edit suggestions.

    Args:
        transcript: The formatted transcript text to analyze.
        agent_prompt: The system prompt for the agent.
        model: The Claude model to use. Defaults to claude-sonnet-4-20250514.

    Returns:
        The AI's response text containing edit decisions.

    Raises:
        LLMClientError: If the API call fails or returns an error.
    """
    try:
        api_key = get_api_key()
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system=agent_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Please analyze this transcript and provide edit decisions:\n\n{transcript}",
                }
            ],
        )

        # Extract text from the response
        if message.content and len(message.content) > 0:
            return message.content[0].text
        else:
            raise LLMClientError("Empty response from Claude API")

    except anthropic.AuthenticationError as e:
        raise LLMClientError(f"Authentication failed: {e}")
    except anthropic.APIError as e:
        raise LLMClientError(f"Claude API error: {e}")
