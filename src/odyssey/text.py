from __future__ import annotations

import re
from collections import Counter

ODYSSEY_TERMS = {
    "achaeans",
    "athene",
    "calypso",
    "circe",
    "cyclops",
    "eumaeus",
    "ithaca",
    "laertes",
    "menelaus",
    "mentor",
    "nausicaa",
    "odysseus",
    "penelope",
    "phaeacians",
    "polyphemus",
    "poseidon",
    "suitors",
    "telemachus",
    "troy",
    "ulysses",
    "zeus",
    "home",
    "guest",
    "ship",
    "sea",
    "cattle",
    "bow",
    "bed",
    "disguise",
    "return",
    "song",
    "omen",
    "feast",
}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_gutenberg_boilerplate(text: str) -> str:
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, re.I | re.S)
    if start:
        text = text[start.end() :]
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*", text, re.I | re.S)
    if end:
        text = text[: end.start()]
    return normalize_whitespace(text)


def word_count(text: str) -> int:
    return len(text.split())


def chunk_words(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step
    return chunks


def sentence_split(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", compact)
    return [piece.strip() for piece in pieces if len(piece.split()) >= 6]


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).strip()


def term_counts(text: str) -> Counter[str]:
    tokens = re.findall(r"[a-z][a-z'-]*", text.lower())
    return Counter(token for token in tokens if token in ODYSSEY_TERMS)


def score_sentence(sentence: str) -> float:
    lowered = sentence.lower()
    terms = sum(1 for term in ODYSSEY_TERMS if term in lowered)
    words = len(sentence.split())
    score = terms * 3.0
    if 12 <= words <= 45:
        score += 1.5
    elif words > 80:
        score -= 2.0
    if any(marker in lowered for marker in ("therefore", "but", "for", "then", "when", "after")):
        score += 0.75
    return score


def extractive_summary(text: str, target_words: int) -> str:
    sentences = sentence_split(text)
    if not sentences:
        return truncate_words(text, target_words)
    ranked = sorted(enumerate(sentences), key=lambda item: (score_sentence(item[1]), -item[0]), reverse=True)
    selected = []
    total = 0
    for index, sentence in ranked:
        words = sentence.split()
        if selected and total + len(words) > target_words:
            continue
        selected.append((index, sentence))
        total += len(words)
        if total >= target_words * 0.85:
            break
    if not selected:
        selected = ranked[:1]
    selected.sort(key=lambda item: item[0])
    return truncate_words(" ".join(sentence for _, sentence in selected), target_words)
