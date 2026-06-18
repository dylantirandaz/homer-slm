#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mvp.io import write_jsonl
from mvp.manifest import CORPUS
from mvp.text import strip_gutenberg_boilerplate


def candidate_urls(gutenberg_id: int) -> list[str]:
    gid = str(gutenberg_id)
    return [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-8.txt",
    ]


def fetch(url: str, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": "minimum-viable-philosophy/0.1"})
    with urlopen(request, timeout=timeout) as response:
        data = response.read()
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def download_text(gutenberg_id: int) -> tuple[str, str]:
    errors = []
    for url in candidate_urls(gutenberg_id):
        try:
            text = fetch(url)
        except (HTTPError, URLError, TimeoutError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        if "<html" in text[:500].lower():
            errors.append(f"{url}: html response")
            continue
        cleaned = strip_gutenberg_boilerplate(text)
        if len(cleaned.split()) < 500:
            errors.append(f"{url}: too short after cleanup")
            continue
        return cleaned, url
    raise RuntimeError("; ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public-domain philosophy texts.")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-success", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = CORPUS[: args.limit] if args.limit else CORPUS
    metadata = []
    failures = []

    for item in manifest:
        path = out_dir / f"{item['text_id']}.txt"
        if path.exists() and not args.force:
            text = path.read_text(encoding="utf-8")
            source_url = item.get("source_url") or candidate_urls(item["gutenberg_id"])[0]
            print(f"[exists] {item['text_id']} words={len(text.split())}")
        else:
            try:
                text, source_url = download_text(item["gutenberg_id"])
            except RuntimeError as exc:
                print(f"[fail] {item['text_id']}: {exc}")
                failures.append({"text_id": item["text_id"], "error": str(exc)})
                continue
            path.write_text(text + "\n", encoding="utf-8")
            print(f"[downloaded] {item['text_id']} words={len(text.split())}")
            time.sleep(args.sleep)

        metadata.append(
            {
                **item,
                "source": "Project Gutenberg",
                "source_url": source_url,
                "source_path": str(path.relative_to(ROOT)),
                "license": "Project Gutenberg; text cleaned for local research use",
                "language": "English",
                "translator": "unknown_or_project_gutenberg_metadata",
            }
        )

    write_jsonl(out_dir / "metadata.jsonl", metadata)
    (out_dir / "download_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"downloaded_or_existing={len(metadata)} failures={len(failures)}")
    if len(metadata) < args.min_success:
        print(f"Only {len(metadata)} texts available; required {args.min_success}.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

