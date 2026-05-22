#!/usr/bin/env python3
"""
CLI entry point for the YouTube KB pipeline.
All logic lives in pipeline_core.py — this file handles args, logging, and confirmation.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from pipeline_core import fetch_video_list, filter_videos, run_pipeline

load_dotenv()


def format_duration(secs) -> str:
    secs = int(secs or 0)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def print_video_table(videos: list[dict]):
    bar = "─" * 74
    print(f"\n{bar}")
    print(f"  {'#':<5}{'Title':<46}{'Date':<13}{'Dur'}")
    print(bar)
    for i, v in enumerate(videos, 1):
        title = (v.get("title") or "")[:45]
        date = v.get("upload_date", "")
        if len(date) == 8:
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        dur = format_duration(v.get("duration") or 0)
        print(f"  {i:<5}{title:<46}{date:<13}{dur}")
    print(bar)
    print(f"  Total: {len(videos)} videos\n")


def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="YouTube Channel Knowledge Base Pipeline")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--max", type=int, metavar="N")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--filter", dest="filter_kw", metavar="KEYWORD")
    parser.add_argument("--months", type=int)
    parser.add_argument("--cookies", metavar="FILE")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: '{args.config}' not found.")
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    channel_url: str = config["channel_url"]
    folder_name: str = config["folder_name"]
    extract_fields: dict = config["extract_fields"]
    filter_keywords: list[str] = config.get("filter_keywords", [])
    months: int = args.months if args.months is not None else config.get("months", 12)
    extract_model: str = config.get("extract_model", "claude-haiku-4-5-20251001")

    if args.filter_kw:
        filter_keywords = [args.filter_kw]

    out = Path(folder_name)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(str(out / "pipeline.log"))

    # Preview matched videos before processing
    all_videos = fetch_video_list(channel_url)
    matched = filter_videos(all_videos, filter_keywords, months)

    if not matched:
        print("No videos matched the filters.")
        sys.exit(0)

    print(f"\nMatched {len(matched)} videos:")
    print_video_table(matched)

    confirm = input("Process these videos? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    run_pipeline(
        channel_url=channel_url,
        folder_name=folder_name,
        extract_fields=extract_fields,
        filter_keywords=filter_keywords,
        months=months,
        max_videos=args.max,
        skip_existing=args.skip_existing,
        cookies_file=args.cookies,
        extract_model=extract_model,
    )


if __name__ == "__main__":
    main()
