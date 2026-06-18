from __future__ import annotations

from pathlib import Path

from .analysis import concept_coverage
from .compression import MlxGenerator
from .io import read_jsonl, utc_now, word_count
from .prompts import generation_essay_prompt
from .text import sentence_split, truncate_words


def generation_text(path: str | Path) -> str:
    records = list(read_jsonl(path))
    return "\n\n".join(record.get("text", "") for record in records if record.get("text")).strip()


def generation_number(path: str | Path) -> int:
    path = Path(path)
    name = path.parent.name
    if name.startswith("gen_"):
        return int(name.removeprefix("gen_"))
    records = list(read_jsonl(path))
    if records:
        return int(records[0].get("generation", 0))
    return 0


def generation_paths(generations_dir: str | Path) -> list[Path]:
    root = Path(generations_dir)
    return [
        gen_dir / "units.jsonl"
        for gen_dir in sorted(root.glob("gen_*"))
        if (gen_dir / "units.jsonl").exists()
    ]


def dedupe_repeated_paragraphs(text: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in text.strip().split("\n\n") if paragraph.strip()]
    kept = []
    seen = set()
    for paragraph in paragraphs:
        key = " ".join(paragraph.lower().split())
        if key in seen:
            continue
        seen.add(key)
        kept.append(paragraph)
    return "\n\n".join(kept).strip()


def essay_token_budget(target_word_count: int, max_tokens: int | None) -> int:
    if max_tokens:
        return max_tokens
    return max(420, int(target_word_count * 1.35))


def short_list(items: list[str], limit: int = 12) -> str:
    if not items:
        return "none detected"
    clipped = items[:limit]
    suffix = "" if len(items) <= limit else f", and {len(items) - limit} more"
    return ", ".join(clipped) + suffix


def top_axes(coverage: dict, limit: int = 3) -> list[str]:
    counts = coverage["output_axis_counts"]
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [axis for axis, count in ranked[:limit] if count > 0]


def evidence_sentences(text: str, coverage: dict, limit: int = 3) -> list[str]:
    output_terms = set(coverage["output_concepts"])
    sentences = sentence_split(text)
    scored = []
    for index, sentence in enumerate(sentences):
        terms = {term for term in output_terms if term in sentence.lower()}
        scored.append((len(terms), -index, sentence))
    scored.sort(reverse=True)
    return [sentence for score, _, sentence in scored[:limit] if score > 0]


def render_generation_essay(
    generation: int,
    current_text: str,
    coverage: dict,
    model_reading: str,
) -> str:
    axes = top_axes(coverage)
    axis_text = ", ".join(axis.replace("_", " ") for axis in axes) or "no tracked axis"
    evidence = evidence_sentences(current_text, coverage)
    evidence_block = "\n".join(f"- {sentence}" for sentence in evidence) or "- No high-signal sentence detected."
    questions = []
    for concept in coverage["lost_concepts"][:5]:
        questions.append(f"- Can the next generation recover or explain `{concept}` without reintroducing noise?")
    if not questions:
        questions.append("- Which surviving claim should be tested against its parent text next?")

    return (
        f"# Generation {generation} Essay\n\n"
        "## What This Generation Understands\n\n"
        f"This generation's surviving text is concentrated around {axis_text}. Its visible "
        f"concept vocabulary includes {short_list(coverage['output_concepts'])}. Read as a "
        "state report, it appears to understand only the ideas that remain available in its "
        "own compressed language, not the full parent corpus.\n\n"
        "## What It Preserves\n\n"
        f"Against its parent generation, it preserves {short_list(coverage['retained_concepts'])}. "
        f"The measured concept retention is {coverage['retention_ratio']:.2f}, so the essay should "
        "be read as a partial survival map rather than a verdict on the whole tradition.\n\n"
        "## What It Has Lost Or Flattened\n\n"
        f"The clearest losses or muted concepts are {short_list(coverage['lost_concepts'])}. "
        "These are the areas where the next run should inspect whether the compression removed "
        "a real argument, collapsed a distinction, or merely dropped vocabulary.\n\n"
        "## What It Overweights\n\n"
        f"New or newly dominant concepts include {short_list(coverage['introduced_concepts'])}. "
        "If these terms are not strongly supported by the parent text, they should be treated as "
        "possible drift rather than genuine philosophical survival.\n\n"
        "## Evidence From The Surviving Text\n\n"
        f"{evidence_block}\n\n"
        "## Model Reading\n\n"
        f"{model_reading.strip() or 'No model reading was generated.'}\n\n"
        "## Questions For The Next Generation\n\n"
        + "\n".join(questions)
    ).strip()


def write_generation_essay(
    parent_path: str | Path,
    generation_path: str | Path,
    essay_path: str | Path,
    model_name: str,
    compression_adapter_path: str | None,
    essay_adapter_path: str | None,
    generator: MlxGenerator | None = None,
    target_word_count: int = 750,
    max_input_words: int = 2400,
    max_tokens: int | None = None,
) -> dict:
    parent_text = generation_text(parent_path)
    current_text = generation_text(generation_path)
    generation = generation_number(generation_path)
    coverage = concept_coverage(parent_text, current_text)
    prompt = generation_essay_prompt(
        generation=generation,
        generation_text=truncate_words(current_text, max_input_words),
        target_word_count=target_word_count,
        retained_concepts=coverage["retained_concepts"],
        lost_concepts=coverage["lost_concepts"],
        introduced_concepts=coverage["introduced_concepts"],
        retention_ratio=coverage["retention_ratio"],
    )

    if generator is None:
        generator = MlxGenerator(
            model_name=model_name,
            adapter_path=essay_adapter_path,
            max_tokens=essay_token_budget(target_word_count, max_tokens),
            repetition_penalty=1.25,
            repetition_context_size=256,
        )

    model_reading = generator.generate(prompt, max_tokens=min(320, essay_token_budget(target_word_count, max_tokens)))
    model_reading = dedupe_repeated_paragraphs(model_reading)
    essay = render_generation_essay(
        generation=generation,
        current_text=current_text,
        coverage=coverage,
        model_reading=model_reading,
    )
    essay_path = Path(essay_path)
    essay_path.parent.mkdir(parents=True, exist_ok=True)
    essay_path.write_text(essay.strip() + "\n", encoding="utf-8")

    return {
        "generation": generation,
        "essay_path": str(essay_path),
        "generation_path": str(generation_path),
        "parent_path": str(parent_path),
        "model": model_name,
        "compression_adapter_path": compression_adapter_path,
        "essay_adapter_path": essay_adapter_path,
        "source_word_count": word_count(parent_text),
        "generation_word_count": word_count(current_text),
        "essay_word_count": word_count(essay),
        "retention_ratio": coverage["retention_ratio"],
        "retained_concepts": coverage["retained_concepts"],
        "lost_concepts": coverage["lost_concepts"],
        "introduced_concepts": coverage["introduced_concepts"],
        "timestamp": utc_now(),
    }


def write_essay_index(
    records: list[dict],
    index_path: str | Path,
    run_id: str,
    model_name: str,
    compression_adapter_path: str | None,
    essay_adapter_path: str | None,
) -> None:
    lines = [
        "# Run Essays",
        "",
        f"Run: `{run_id}`",
        f"Generated: {utc_now()}",
        f"Model: `{model_name}`",
        f"Compression adapter: `{compression_adapter_path or 'none'}`",
        f"Essay adapter: `{essay_adapter_path or 'none; base model'}`",
        "",
        "These essays are the primary narrative output of the run. They are not final answers; they are research logs showing what each compressed generation appears to understand and what has narrowed.",
        "",
        "| Generation | Essay | Retention | Generation Words | Essay Words |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    index_path = Path(index_path)
    for record in sorted(records, key=lambda item: item["generation"]):
        essay_path = Path(record["essay_path"])
        try:
            relative = essay_path.relative_to(index_path.parent)
        except ValueError:
            relative = essay_path
        lines.append(
            f"| {record['generation']} | [{essay_path.name}]({relative}) | "
            f"{record['retention_ratio']:.2f} | {record['generation_word_count']} | "
            f"{record['essay_word_count']} |"
        )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_generation_essays(
    initial_path: str | Path,
    generations_dir: str | Path,
    essays_dir: str | Path,
    index_path: str | Path,
    run_id: str,
    model_name: str,
    compression_adapter_path: str | None,
    essay_adapter_path: str | None,
    target_word_count: int = 750,
    max_input_words: int = 2400,
    max_tokens: int | None = None,
) -> list[dict]:
    paths = generation_paths(generations_dir)
    generator = MlxGenerator(
        model_name=model_name,
        adapter_path=essay_adapter_path,
        max_tokens=essay_token_budget(target_word_count, max_tokens),
        repetition_penalty=1.25,
        repetition_context_size=256,
    )
    records = []
    parent_path = Path(initial_path)
    essays_dir = Path(essays_dir)

    for path in paths:
        generation = generation_number(path)
        essay_path = essays_dir / f"gen_{generation:03d}.md"
        record = write_generation_essay(
            parent_path=parent_path,
            generation_path=path,
            essay_path=essay_path,
            model_name=model_name,
            compression_adapter_path=compression_adapter_path,
            essay_adapter_path=essay_adapter_path,
            generator=generator,
            target_word_count=target_word_count,
            max_input_words=max_input_words,
            max_tokens=max_tokens,
        )
        records.append(record)
        parent_path = path

    write_essay_index(
        records=records,
        index_path=index_path,
        run_id=run_id,
        model_name=model_name,
        compression_adapter_path=compression_adapter_path,
        essay_adapter_path=essay_adapter_path,
    )
    return records
