from __future__ import annotations

import re
from hashlib import sha1
from typing import Iterable

PHILOSOPHY_TERMS = {
    "appearance",
    "beauty",
    "being",
    "belief",
    "cause",
    "causation",
    "choice",
    "citizen",
    "city",
    "conscience",
    "death",
    "desire",
    "duty",
    "evil",
    "experience",
    "faith",
    "freedom",
    "god",
    "good",
    "happiness",
    "ignorance",
    "justice",
    "knowledge",
    "language",
    "law",
    "liberty",
    "life",
    "meaning",
    "mind",
    "nature",
    "necessity",
    "pleasure",
    "power",
    "reason",
    "reality",
    "right",
    "self",
    "soul",
    "state",
    "suffering",
    "truth",
    "virtue",
    "will",
    "wisdom",
}

ARGUMENT_MARKERS = {
    "because",
    "but",
    "therefore",
    "thus",
    "hence",
    "if",
    "then",
    "although",
    "however",
    "nevertheless",
    "for",
    "since",
    "unless",
    "must",
    "cannot",
    "contrary",
}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_gutenberg_boilerplate(text: str) -> str:
    start_patterns = [
        r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .*?\*\*\*",
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*",
    ]
    end_patterns = [
        r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .*?\*\*\*",
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*",
    ]

    stripped = text
    for pattern in start_patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE | re.DOTALL)
        if match:
            stripped = stripped[match.end() :]
            break

    for pattern in end_patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE | re.DOTALL)
        if match:
            stripped = stripped[: match.start()]
            break

    return normalize_whitespace(stripped)


def sentence_split(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", compact)
    return [piece.strip() for piece in pieces if len(piece.split()) >= 6]


def stable_hash(parts: Iterable[str], length: int = 12) -> str:
    digest = sha1()
    for part in parts:
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()[:length]


def chunk_words(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step
    return chunks


def truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).strip()


def score_sentence(sentence: str) -> float:
    words = [re.sub(r"[^a-z]", "", token.lower()) for token in sentence.split()]
    words = [word for word in words if word]
    if not words:
        return 0.0

    term_hits = sum(1 for word in words if word in PHILOSOPHY_TERMS)
    marker_hits = sum(1 for word in words if word in ARGUMENT_MARKERS)
    length = len(words)

    score = term_hits * 2.5 + marker_hits * 1.5
    if 16 <= length <= 42:
        score += 1.5
    elif length > 70:
        score -= 1.0
    if "?" in sentence:
        score += 1.0
    return score


def extractive_philosophy_digest(text: str, target_word_count: int) -> str:
    sentences = sentence_split(text)
    if not sentences:
        return truncate_words(text, target_word_count)

    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (score_sentence(item[1]), -item[0]),
        reverse=True,
    )

    selected: list[tuple[int, str]] = []
    selected_terms: set[str] = set()
    total_words = 0
    max_words = max(40, target_word_count)

    for index, sentence in ranked:
        words = sentence.split()
        if total_words + len(words) > max_words and selected:
            continue

        lowered = {
            re.sub(r"[^a-z]", "", token.lower())
            for token in words
            if re.sub(r"[^a-z]", "", token.lower()) in PHILOSOPHY_TERMS
        }
        novelty = len(lowered - selected_terms)
        if selected and novelty == 0 and score_sentence(sentence) < 3:
            continue

        selected.append((index, sentence))
        selected_terms.update(lowered)
        total_words += len(words)
        if total_words >= target_word_count * 0.85:
            break

    if not selected:
        selected = ranked[:3]

    selected.sort(key=lambda item: item[0])
    digest = " ".join(sentence for _, sentence in selected)
    return truncate_words(digest, target_word_count)

