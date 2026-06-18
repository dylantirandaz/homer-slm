from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import append_jsonl, read_jsonl, utc_now, word_count
from .prompts import compression_prompt, final_sentence_prompt
from .text import truncate_words


def get_record_id(record: dict) -> str:
    return (
        record.get("unit_id")
        or record.get("chunk_id")
        or record.get("text_id")
        or record.get("id")
        or "unknown"
    )


def group_records(records: list[dict], batch_size: int) -> list[list[dict]]:
    return [records[index : index + batch_size] for index in range(0, len(records), batch_size)]


class MlxGenerator:
    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        max_tokens: int = 768,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.12,
        repetition_context_size: int = 96,
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.repetition_context_size = repetition_context_size
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def load(self) -> None:
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is not installed. Install requirements.txt inside the virtualenv first."
            ) from exc

        kwargs = {}
        if self.adapter_path:
            kwargs["adapter_path"] = self.adapter_path
        self.model, self.tokenizer = load(self.model_name, **kwargs)
        self._generate = generate
        self._make_sampler = make_sampler
        self._make_logits_processors = make_logits_processors

    def _format_prompt(self, prompt: str) -> str:
        assert self.tokenizer is not None
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return prompt

    def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()

        formatted = self._format_prompt(prompt)
        sampler = self._make_sampler(
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        logits_processors = self._make_logits_processors(
            repetition_penalty=self.repetition_penalty,
            repetition_context_size=self.repetition_context_size,
        )
        output = self._generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens or self.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False,
        )
        return output.strip()


def compress_generation(
    input_path: str | Path,
    output_path: str | Path,
    generation: int,
    target_word_count: int,
    batch_size: int,
    model_name: str,
    adapter_path: str | None,
    force: bool = False,
    max_tokens: int | None = None,
    max_input_words: int | None = None,
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    records = list(read_jsonl(input_path))
    groups = group_records(records, batch_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output_path.exists() and not force:
        for record in read_jsonl(output_path):
            completed.add(record.get("unit_id"))
    elif force and output_path.exists():
        output_path.unlink()

    generator = MlxGenerator(
        model_name=model_name,
        adapter_path=adapter_path,
        max_tokens=max_tokens or max(128, target_word_count * 3),
    )

    written = 0
    for index, group in enumerate(groups):
        unit_id = f"gen_{generation:03d}_unit_{index:04d}"
        if unit_id in completed:
            print(f"[gen {generation}] skip {unit_id}")
            continue

        parent_ids = [get_record_id(record) for record in group]
        input_text = "\n\n---\n\n".join(record["text"] for record in group if record.get("text"))
        if max_input_words:
            input_text = truncate_words(input_text, max_input_words)
        late_stage = target_word_count <= 80 or generation >= 4
        prompt = compression_prompt(input_text, target_word_count, late_stage=late_stage)

        print(
            f"[gen {generation}] compress {unit_id} "
            f"parents={len(parent_ids)} input_words={word_count(input_text)} "
            f"target_words={target_word_count}"
        )
        compressed = generator.generate(prompt, max_tokens=max_tokens or max(128, target_word_count * 3))
        output_words = word_count(compressed)

        record = {
            "generation": generation,
            "parent_generation": generation - 1,
            "unit_id": unit_id,
            "parent_ids": parent_ids,
            "compression_ratio": output_words / max(1, word_count(input_text)),
            "target_word_count": target_word_count,
            "method": "mlx_lora_summary",
            "model": model_name,
            "adapter_path": adapter_path,
            "prompt_version": "v1",
            "input_word_count": word_count(input_text),
            "output_word_count": output_words,
            "text": compressed,
            "timestamp": utc_now(),
        }
        append_jsonl(output_path, record)
        written += 1

    return {"input_records": len(records), "groups": len(groups), "written": written}


def write_final_outputs(
    final_generation_path: str | Path,
    model_name: str,
    adapter_path: str | None,
    paragraph_path: str | Path,
    sentence_path: str | Path,
) -> None:
    records = list(read_jsonl(final_generation_path))
    text = "\n\n".join(record["text"] for record in records)
    paragraph = text.strip()
    Path(paragraph_path).parent.mkdir(parents=True, exist_ok=True)
    Path(paragraph_path).write_text(paragraph + "\n", encoding="utf-8")

    generator = MlxGenerator(model_name=model_name, adapter_path=adapter_path, max_tokens=160)
    sentence = generator.generate(final_sentence_prompt(paragraph), max_tokens=160)
    Path(sentence_path).write_text(sentence.strip() + "\n", encoding="utf-8")
