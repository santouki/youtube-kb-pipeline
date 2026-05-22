"""
Core pipeline logic — importable by both pipeline.py (CLI) and api.py (microservice).
No argparse, no sys.exit, no interactive prompts. Raises exceptions on hard failures.
"""

import csv
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import anthropic
from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)

MAX_TRANSCRIPT_CHARS = 12_000
MIN_DURATION_SECS = 600       # 10 minutes
RATE_LIMIT_SECS = 1.5

# ─── Video fetching ───────────────────────────────────────────────────────────

def fetch_video_list(channel_url: str) -> list[dict]:
    """Fetch all videos from a channel using yt-dlp --flat-playlist."""
    logger.info(f"Fetching video list: {channel_url}")
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--dump-json", channel_url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

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
    """Filter by duration (>10 min), upload date, and title keywords."""
    cutoff = (datetime.now() - timedelta(days=months * 30)).strftime("%Y%m%d")

    matched = []
    for v in videos:
        if (v.get("duration") or 0) < MIN_DURATION_SECS:
            continue
        upload_date = v.get("upload_date", "")
        if upload_date and upload_date < cutoff:
            continue
        if filter_keywords:
            title_lower = (v.get("title") or "").lower()
            if not any(kw.lower() in title_lower for kw in filter_keywords):
                continue
        matched.append(v)

    return matched

# ─── Transcript ───────────────────────────────────────────────────────────────

def get_transcript(video_id: str, cookies_file: str | None = None) -> str | None:
    """Fetch transcript, trim to MAX_TRANSCRIPT_CHARS. Falls back to yt-dlp on 429."""
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=["en"])
        text = " ".join(snippet.text for snippet in transcript).strip()
        return text[:MAX_TRANSCRIPT_CHARS]

    except Exception as e:
        err = str(e)
        if "429" in err or "Too Many Requests" in err:
            logger.warning(f"Rate-limited for {video_id}. Trying yt-dlp fallback...")
            return _transcript_via_ytdlp(video_id, cookies_file)
        elif any(x in err for x in ("TranscriptsDisabled", "NoTranscriptFound", "no element found")):
            logger.warning(f"No transcript available: {video_id}")
        else:
            logger.warning(f"Transcript error for {video_id}: {e}")
        return None


def _transcript_via_ytdlp(video_id: str, cookies_file: str | None) -> str | None:
    import tempfile

    cmd = [
        "yt-dlp", "--skip-download",
        "--write-auto-sub", "--sub-lang", "en", "--sub-format", "json3",
        "-o", "%(id)s",
    ]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    else:
        cmd += ["--cookies-from-browser", "chrome"]
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
        sub_files = list(Path(tmpdir).glob(f"{video_id}*.json3"))
        if not sub_files:
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

def extract_insights(
    client: anthropic.Anthropic,
    transcript: str,
    extract_fields: dict,
    title: str,
    model: str,
) -> dict | None:
    fields_block = "\n".join(f'- "{k}": {v}' for k, v in extract_fields.items())
    prompt = (
        "Analyze this YouTube video transcript and extract the following fields as a JSON object.\n\n"
        f"Fields:\n{fields_block}\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON — no markdown, no code fences, no explanation\n"
        "- Use null for any field where information is absent\n"
        "- Each value should be a string, number, or short list (not nested objects)\n\n"
        f"Transcript:\n{transcript}"
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
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

# ─── Database I/O ─────────────────────────────────────────────────────────────

def load_db(json_path: str) -> list[dict]:
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return []


def save_db(db: list[dict], json_path: str, csv_path: str):
    with open(json_path, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    if not db:
        return

    # Preserve insertion-order column layout
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
            writer.writerow({
                k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                for k, v in entry.items()
            })

# ─── Pipeline runner ──────────────────────────────────────────────────────────

def run_pipeline(
    channel_url: str,
    folder_name: str,
    extract_fields: dict,
    filter_keywords: list[str] = [],
    months: int = 12,
    max_videos: int | None = None,
    skip_existing: bool = False,
    cookies_file: str | None = None,
    extract_model: str = "claude-haiku-4-5-20251001",
    api_key: str | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """
    Run the full pipeline. Returns a summary dict.

    on_progress is called with event dicts:
      {"type": "matched",   "count": N}
      {"type": "start",     "index": i, "total": N, "video_id": ..., "title": ...}
      {"type": "processed", "video_id": ..., "title": ...}
      {"type": "skipped",   "video_id": ..., "title": ..., "reason": ...}
      {"type": "failed",    "video_id": ..., "title": ..., "reason": ...}
      {"type": "done",      "processed": N, "skipped": N, "failed": N, "total_in_db": N}
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=key)

    out = Path(folder_name)
    out.mkdir(parents=True, exist_ok=True)
    db_json = str(out / "database.json")
    db_csv = str(out / "database.csv")

    def emit(event: dict):
        if on_progress:
            on_progress(event)

    # Fetch and filter
    all_videos = fetch_video_list(channel_url)
    matched = filter_videos(all_videos, filter_keywords, months)
    emit({"type": "matched", "count": len(matched)})
    logger.info(f"Matched {len(matched)} videos after filtering")

    # Load existing DB
    db = load_db(db_json)
    existing_ids = {e["video_id"] for e in db}

    to_process = matched
    if skip_existing:
        before = len(to_process)
        to_process = [v for v in to_process if v["id"] not in existing_ids]
        logger.info(f"Skipping {before - len(to_process)} already-processed videos")

    if max_videos is not None:
        to_process = to_process[:max_videos]

    logger.info(f"Processing {len(to_process)} videos")

    processed = skipped = failed = 0

    for i, video in enumerate(to_process, 1):
        video_id = video["id"]
        title = video.get("title", video_id)
        emit({"type": "start", "index": i, "total": len(to_process), "video_id": video_id, "title": title})
        logger.info(f"[{i}/{len(to_process)}] {title}")

        transcript = get_transcript(video_id, cookies_file)
        if not transcript:
            logger.warning(f"  → Skipped (no transcript)")
            emit({"type": "skipped", "video_id": video_id, "title": title, "reason": "no transcript"})
            skipped += 1
            continue

        insights = extract_insights(client, transcript, extract_fields, title, extract_model)
        if insights is None:
            logger.warning(f"  → Skipped (extraction failed)")
            emit({"type": "failed", "video_id": video_id, "title": title, "reason": "extraction failed"})
            failed += 1
            continue

        entry = {
            "video_id": video_id,
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "upload_date": video.get("upload_date", ""),
            "duration_secs": int(video.get("duration") or 0),
            **insights,
        }
        db.append(entry)
        save_db(db, db_json, db_csv)
        processed += 1
        emit({"type": "processed", "video_id": video_id, "title": title})
        logger.info(f"  → Saved ({len(db)} total in database)")

        if i < len(to_process):
            time.sleep(RATE_LIMIT_SECS)

    summary = {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_in_db": len(db),
    }
    emit({"type": "done", **summary})
    logger.info(f"Done. {processed} processed, {skipped} skipped, {failed} failed. {len(db)} total in DB.")
    return summary
