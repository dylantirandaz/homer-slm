from __future__ import annotations

import re
from collections import Counter

from .io import word_count


PHILOSOPHY_AXES: dict[str, set[str]] = {
    "metaphysics": {
        "appearance",
        "being",
        "cause",
        "essence",
        "existence",
        "god",
        "nature",
        "necessity",
        "reality",
        "substance",
        "world",
    },
    "epistemology": {
        "belief",
        "certainty",
        "doubt",
        "experience",
        "idea",
        "judgment",
        "knowledge",
        "perception",
        "reason",
        "truth",
        "understanding",
    },
    "ethics": {
        "choice",
        "conscience",
        "desire",
        "duty",
        "evil",
        "good",
        "happiness",
        "justice",
        "pleasure",
        "right",
        "virtue",
    },
    "politics": {
        "authority",
        "citizen",
        "city",
        "freedom",
        "government",
        "law",
        "liberty",
        "power",
        "property",
        "rights",
        "state",
    },
    "self_and_mind": {
        "body",
        "consciousness",
        "death",
        "desire",
        "identity",
        "life",
        "mind",
        "passion",
        "self",
        "soul",
        "will",
    },
    "suffering_and_practice": {
        "action",
        "discipline",
        "earnestness",
        "liberation",
        "practice",
        "suffering",
        "tranquility",
        "wisdom",
    },
    "language_and_method": {
        "argument",
        "contradiction",
        "definition",
        "inference",
        "language",
        "meaning",
        "method",
        "proof",
        "question",
        "sign",
    },
}

ALL_TERMS = set().union(*PHILOSOPHY_AXES.values())


def tokenize_terms(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z_'-]*", text.lower())


def concept_counts(text: str) -> Counter[str]:
    return Counter(token for token in tokenize_terms(text) if token in ALL_TERMS)


def axis_counts(counts: Counter[str]) -> dict[str, int]:
    return {
        axis: sum(counts.get(term, 0) for term in terms)
        for axis, terms in PHILOSOPHY_AXES.items()
    }


def concept_coverage(source_text: str, output_text: str) -> dict:
    source_counts = concept_counts(source_text)
    output_counts = concept_counts(output_text)
    source_terms = set(source_counts)
    output_terms = set(output_counts)
    retained_terms = sorted(source_terms & output_terms)
    lost_terms = sorted(source_terms - output_terms)
    introduced_terms = sorted(output_terms - source_terms)
    retention_ratio = len(retained_terms) / len(source_terms) if source_terms else 1.0

    axes = {}
    for axis, terms in PHILOSOPHY_AXES.items():
        source_axis_terms = source_terms & terms
        output_axis_terms = output_terms & terms
        retained_axis_terms = source_axis_terms & output_axis_terms
        axes[axis] = {
            "source_terms": sorted(source_axis_terms),
            "retained_terms": sorted(retained_axis_terms),
            "lost_terms": sorted(source_axis_terms - output_axis_terms),
            "source_count": sum(source_counts[term] for term in source_axis_terms),
            "output_count": sum(output_counts[term] for term in output_axis_terms),
            "retention_ratio": (
                len(retained_axis_terms) / len(source_axis_terms)
                if source_axis_terms
                else 1.0
            ),
        }

    return {
        "source_word_count": word_count(source_text),
        "output_word_count": word_count(output_text),
        "source_concepts": sorted(source_terms),
        "output_concepts": sorted(output_terms),
        "retained_concepts": retained_terms,
        "lost_concepts": lost_terms,
        "introduced_concepts": introduced_terms,
        "retention_ratio": retention_ratio,
        "source_axis_counts": axis_counts(source_counts),
        "output_axis_counts": axis_counts(output_counts),
        "axes": axes,
    }


def understanding_prompt(output_text: str, coverage: dict) -> str:
    output_concepts = ", ".join(coverage["output_concepts"][:16]) or "none detected"
    return (
        "Read the compressed philosophical text below. Describe only what this "
        "compressed text appears to understand. Do not compare it to another text. "
        "Do not mention retention ratios. Do not write an introduction.\n\n"
        "Write exactly three short bullets:\n"
        "- Main claim:\n"
        "- Concepts it emphasizes:\n"
        "- What remains unclear:\n\n"
        f"Detected child concepts: {output_concepts}\n\n"
        "Compressed text:\n"
        f"{output_text}\n"
    )


def absorption_prompt(
    source_text: str,
    output_text: str,
    coverage: dict,
    max_source_words: int = 900,
) -> str:
    return understanding_prompt(output_text, coverage)
