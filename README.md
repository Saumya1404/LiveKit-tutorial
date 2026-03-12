# LiveKit Tutorial

This project is a LiveKit voice agent built with Python. It uses Sarvam for speech-to-text and text-to-speech, Gemini as the primary LLM, Groq as fallback LLMs, and a weather tool plus LiveKit docs MCP tools for documentation questions.

## Requirements

- Python 3.12
- `uv`
- LiveKit credentials
- API keys for Groq, Google, and Sarvam

## Setup

Create a `.env` file with:

```env
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
GROQ_API_KEY=
GOOGLE_API_KEY=
SARVAM_API_KEY=
```

Install dependencies:

```bash
uv sync
```

## Run

Run the agent in console mode:

```bash
uv run agent.py console
```

Other modes may also be available through the LiveKit CLI, but only `console` has been used here.

For example:

```bash
uv run agent.py dev
```

## Notes

- Do not commit `.env`.
- The agent can answer weather questions with a function tool.
- LiveKit documentation questions are handled through the LiveKit docs MCP server.
