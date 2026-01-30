The objective is to create an application which edits videos for the user. This will adhere to modern software development standards
and includes multiple agents, each with a specific role and domain knowledge.

We are using test driven development.

## Agentic Workflow Rules

1. **Delegate code writing to subagents** - The main agent should not write code directly. Instead, use the Task tool to spawn subagents for all code writing tasks.
