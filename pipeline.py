#!/usr/bin/env python3
"""
YouTube Channel Knowledge Base Pipeline

Fetches videos from a channel, downloads transcripts, extracts structured
insights via Claude API, and saves results incrementally to database.json/csv.
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_TRANSCRIPT_CHARS = 12_000
MIN_DURATION_SECS = 600  # 10 minutes
RATE_LIMIT_SECS = 1.5

# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

logger = logging.getLogger(__name__)

# ─── Video fetching ───────────────────────────────────────────────────────────

def fetch_video_list(channel_url: str) -> list[dict]:
    """Fetch all videos from a channel using yt-dlp --flat-playlist."""
    logger.info(f"Fetching video list from: {channel_url}")
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--dump-json", channel_url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"yt-dlp error:\n{result.stderr}")
        sys.exit(1)

    videos = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            videos.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    logger.info(f"Found {len(videos)} total videos on channel")
    return videos


def filter_videos(
    videos: list[dict],
    filter_keywords: list[str],
    months: int,
) -> list[dict]:
    """Filter by duration (>10 min), date range, and title keywords."""
    cutoff = (datetime.now() - timedelta(days=months * 30)).strftime("%Y%m%d")

    matched = []
    for v in videos:
        # Skip shorts / trailers
        duration = v.get("duration") or 0
        if duration < MIN_DURATION_SECS:
            continue

        # Date filter
        upload_date = v.get("upload_date", "")
        if upload_date and upload_date < cutoff:
            continue

        # Keyword filter (any keyword must appear in title)
        if filter_keywords:
            title_lower = (v.get("title") or "").lower()
            if not any(kw.lower() in title_lower for kw in filter_keywords):
                continue

        matched.append(v)

    return matched


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

# ─── Transcript fetching ──────────────────────────────────────────────────────

def get_transcript(video_id: str, cookies_file: str | None = None) -> str | None:
    """Fetch and trim transcript. Falls back to yt-dlp on 429."""
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=["en"])
        text = " ".join(snippet.text for snippet in transcript).strip()
        return text[:MAX_TRANSCRIPT_CHARS]

    except Exception as e:
        err = str(e)
        if "429" in err or "Too Many Requests" in err:
            logger.warning(f"Rate-limited by YouTube for {video_id}. Trying yt-dlp fallback...")
            return _transcript_via_ytdlp(video_id, cookies_file)
        elif any(x in err for x in ("TranscriptsDisabled", "NoTranscriptFound", "no element found")):
            logger.warning(f"No transcript available: {video_id}")
        else:
            logger.warning(f"Transcript error for {video_id}: {e}")
        return None


def _transcript_via_ytdlp(video_id: str, cookies_file: str | None) -> str | None:
    """yt-dlp subtitle fallback (uses Chrome cookies if provided)."""
    import tempfile

    cmd = [
        "yt-dlp", "--skip-download",
        "--write-auto-sub", "--sub-lang", "en", "--sub-format", "json3",
        "-o", "%(id)s",
    ]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    else:
        # Try Chrome cookies automatically when no file specified
        cmd += ["--cookies-from-browser", "chrome"]
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
        sub_files = list(Path(tmpdir).glob(f"{video_id}*.json3"))
        if not sub_files:
            logger.warning(f"No subtitle file from yt-dlp for {video_id}")
            return None
        try:
            data = json.loads(sub_files[0].read_text())
            texts = [
                seg.get("utf8", "").replace("\n", " ").strip()
                for event in data.get("events", [])
                for seg in event.get("segs", [])
                if seg.get("utf8", "").strip()
            ]
            return " ".join(texts)[:MAX_TRANSCRIPT_CHARS]
        except Exception as e:
            logger.warning(f"Failed to parse yt-dlp subtitle for {video_id}: {e}")
            return None

# ─── Claude extraction ────────────────────────────────────────────────────────

def build_prompt(transcript: str, extract_fields: dict) -> str:
    fields_block = "\n".join(
        f'- "{name}": {desc}' for name, desc in extract_fields.items()
    )
    return f"""Analyze this YouTube video transcript and extract the following fields as a JSON object.

Fields:
{fields_block}

Rules:
- Return ONLY valid JSON — no markdown, no code fences, no explanation
- Use null for any field where information is absent
- Each value should be a string, number, or short list (not nested objects)

Transcript:
{transcript}"""


def extract_insights(
    client: anthropic.Anthropic,
    transcript: str,
    extract_fields: dict,
    title: str,
    model: str,
) -> dict | None:
    prompt = build_prompt(transcript, extract_fields)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for '{title}': {e}")
        return None
    except anthropic.APIError as e:
        logger.error(f"Claude API error for '{title}': {e}")
        return None

# ─── Database ─────────────────────────────────────────────────────────────────

def load_db(path: str) -> list[dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_db(db: list[dict], json_path: str, csv_path: str):
    with open(json_path, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    if not db:
        return

    all_keys: list[str] = []
    seen: set[str] = set()
    for entry in db:
        for k in entry:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for entry in db:
            row = {
                k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                for k, v in entry.items()
            }
            writer.writerow(row)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YouTube Channel Knowledge Base Pipeline")
    parser.add_argument("--config", default="config.json", help="Config file (default: config.json)")
    parser.add_argument("--max", type=int, metavar="N", help="Process at most N videos (test mode)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos already in database")
    parser.add_argument("--filter", dest="filter_kw", metavar="KEYWORD", help="Case-insensitive keyword filter on titles")
    parser.add_argument("--months", type=int, help="Only videos from last N months (overrides config)")
    parser.add_argument("--cookies", metavar="FILE", help="Cookies file path (for YouTube 429 fallback)")
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: '{args.config}' not found. Fill in config.json with your channel details.")
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

    # Output paths
    out = Path(folder_name)
    out.mkdir(parents=True, exist_ok=True)
    db_json = str(out / "database.json")
    db_csv = str(out / "database.csv")
    log_file = str(out / "pipeline.log")

    setup_logging(log_file)
    logger.info(f"Config: folder={folder_name} | months={months} | filter={filter_keywords or 'none'}")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Fetch and filter
    all_videos = fetch_video_list(channel_url)
    matched = filter_videos(all_videos, filter_keywords, months)

    if not matched:
        print("No videos matched the filters. Adjust --filter or --months.")
        sys.exit(0)

    print(f"\nMatched {len(matched)} videos:")
    print_video_table(matched)

    confirm = input("Process these videos? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    # Load existing DB and apply filters
    db = load_db(db_json)
    existing_ids = {e["video_id"] for e in db}

    to_process = matched
    if args.skip_existing:
        before = len(to_process)
        to_process = [v for v in to_process if v["id"] not in existing_ids]
        skipped = before - len(to_process)
        if skipped:
            logger.info(f"Skipping {skipped} already-processed videos")

    if args.max:
        to_process = to_process[: args.max]
        logger.info(f"--max {args.max}: capped to {len(to_process)} videos")

    if not to_process:
        print("Nothing new to process.")
        sys.exit(0)

    logger.info(f"Processing {len(to_process)} videos with model: {extract_model}")

    # Main loop
    success = 0
    for i, video in enumerate(to_process, 1):
        video_id = video["id"]
        title = video.get("title", video_id)
        upload_date = video.get("upload_date", "")
        duration = video.get("duration", 0)

        logger.info(f"[{i}/{len(to_process)}] {title}")

        transcript = get_transcript(video_id, args.cookies)
        if not transcript:
            logger.warning(f"  → Skipped (no transcript)")
            continue

        logger.info(f"  → Transcript: {len(transcript):,} chars")

        insights = extract_insights(client, transcript, extract_fields, title, extract_model)
        if insights is None:
            logger.warning(f"  → Skipped (extraction failed)")
            continue

        entry = {
            "video_id": video_id,
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "upload_date": upload_date,
            "duration_secs": int(duration or 0),
            **insights,
        }

        db.append(entry)
        save_db(db, db_json, db_csv)
        success += 1
        logger.info(f"  → Saved ({len(db)} total in database)")

        if i < len(to_process):
            time.sleep(RATE_LIMIT_SECS)

    logger.info(f"Pipeline complete. {success}/{len(to_process)} videos processed. {len(db)} total entries.")
    print(f"\nDone! {success} new entries → {db_json} and {db_csv}")


if __name__ == "__main__":
    main()
