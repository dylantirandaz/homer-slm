#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odyssey.text import strip_gutenberg_boilerplate, word_count


def candidate_urls(gutenberg_id: int) -> list[str]:
    gid = str(gutenberg_id)
    return [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ]


def fetch(url: str, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": "minimum-viable-odyssey/0.1"})
    with urlopen(request, timeout=timeout) as response:
        data = response.read()
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a public-domain Odyssey translation.")
    parser.add_argument("--gutenberg-id", type=int, default=1727, help="Default is Samuel Butler's prose Odyssey.")
    parser.add_argument("--out", default="data/raw/odyssey.txt")
    parser.add_argument("--metadata", default="data/raw/metadata.json")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = ROOT / args.out
    metadata_path = ROOT / args.metadata
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        print(f"[exists] {out.relative_to(ROOT)} words={word_count(out.read_text(encoding='utf-8'))}")
        return 0

    errors = []
    for url in candidate_urls(args.gutenberg_id):
        try:
            text = strip_gutenberg_boilerplate(fetch(url))
        except (HTTPError, URLError, TimeoutError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        if word_count(text) < 50_000:
            errors.append(f"{url}: too short after cleanup")
            continue
        out.write_text(text + "\n", encoding="utf-8")
        metadata = {
            "title": "The Odyssey",
            "author": "Homer",
            "translation": "Samuel Butler prose translation",
            "source": "Project Gutenberg",
            "gutenberg_id": args.gutenberg_id,
            "source_url": url,
            "path": str(out.relative_to(ROOT)),
            "word_count": word_count(text),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[downloaded] {out.relative_to(ROOT)} words={metadata['word_count']}")
        return 0

    print("Unable to download Odyssey:\n" + "\n".join(errors), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

