#!/usr/bin/env python3
"""
Chat interface for your YouTube knowledge base.
Loads database.json into the system prompt and lets you query across all videos.
"""

import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = "claude-sonnet-4-6"


def load_db(folder_name: str) -> list[dict]:
    path = Path(folder_name) / "database.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run pipeline.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def build_system_prompt(db: list[dict], channel_name: str) -> str:
    db_text = json.dumps(db, indent=2, ensure_ascii=False)
    name_part = f" ({channel_name})" if channel_name else ""
    return f"""You are a knowledge analyst for a YouTube channel{name_part}.

You have a structured database of {len(db)} videos with extracted insights. Use it to answer questions analytically — find patterns, compare videos, summarize themes, identify outliers, and cite specific examples.

When referencing a video, include its title and URL. Be concrete and specific rather than generic.

DATABASE:
{db_text}"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chat with your YouTube knowledge base")
    parser.add_argument("--config", default="config.json", help="Config file (default: config.json)")
    parser.add_argument("--model", default=CHAT_MODEL, help=f"Claude model (default: {CHAT_MODEL})")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: '{args.config}' not found.")
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    folder_name = config["folder_name"]
    channel_name = config.get("channel_name", "")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    db = load_db(folder_name)

    print(f"Loaded {len(db)} videos from {folder_name}/database.json")
    print(f"Model: {args.model}")
    print("Type your question. Ctrl+C or 'quit' to exit.\n")

    system = build_system_prompt(db, channel_name)
    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        try:
            resp = client.messages.create(
                model=args.model,
                max_tokens=2048,
                system=system,
                messages=history,
            )
        except anthropic.APIError as e:
            print(f"API error: {e}\n")
            history.pop()
            continue

        reply = resp.content[0].text
        history.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
