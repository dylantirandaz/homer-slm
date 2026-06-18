#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"


SYSTEM_PROMPT = (
    "You are an Odyssey-specialized small language model. You were fine-tuned on "
    "Homer's Odyssey in English translation. Answer questions about the poem, its "
    "characters, plot, places, motifs, and scenes. Be direct and concrete. If a "
    "question asks for something outside the Odyssey, say that it is outside your "
    "Odyssey training focus. Use the provided Odyssey excerpts as grounding. If the "
    "excerpts do not support an answer, say what is unclear instead of inventing."
)


STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "answer",
    "briefly",
    "could",
    "does",
    "from",
    "have",
    "into",
    "odyssey",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z'-]*", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


class OdysseyRetriever:
    def __init__(self, path: Path, max_chunks: int = 3, max_words: int = 170):
        self.path = path
        self.max_chunks = max_chunks
        self.max_words = max_words
        self.records: list[dict[str, Any]] | None = None

    def load(self) -> None:
        if self.records is not None:
            return
        records = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        record["_tokens"] = tokenize(record.get("text", ""))
                        records.append(record)
        self.records = records

    def excerpts(self, query: str) -> list[str]:
        self.load()
        assert self.records is not None
        query_terms = tokenize(query)
        if not query_terms:
            return []
        ranked = []
        for record in self.records:
            overlap = len(query_terms & record.get("_tokens", set()))
            if overlap:
                ranked.append((overlap, record.get("chunk_index", 0), record))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        excerpts = []
        for _, _, record in ranked[: self.max_chunks]:
            words = record.get("text", "").split()
            text = " ".join(words[: self.max_words]).strip()
            excerpts.append(f"[{record.get('chunk_id')}] {text}")
        return excerpts


class OdysseyChatModel:
    def __init__(
        self,
        model_id: str,
        adapter_path: str | None,
        max_tokens: int,
        temperature: float,
        repetition_penalty: float,
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.retriever = OdysseyRetriever(CHUNKS_PATH)

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
        except ImportError as exc:
            raise RuntimeError("mlx-lm is not installed. Run `make setup` first.") from exc

        kwargs = {}
        if self.adapter_path:
            kwargs["adapter_path"] = self.adapter_path
        self.model, self.tokenizer = load(self.model_id, **kwargs)
        self._generate = generate
        self._make_sampler = make_sampler
        self._make_logits_processors = make_logits_processors

    def format_prompt(self, messages: list[dict[str, str]]) -> str:
        self.load()
        assert self.tokenizer is not None
        latest_question = messages[-1]["content"]
        excerpts = self.retriever.excerpts(latest_question)
        grounded_question = latest_question
        if excerpts:
            grounded_question = (
                "Relevant Odyssey excerpts:\n"
                + "\n\n".join(excerpts)
                + "\n\nQuestion:\n"
                + latest_question
                + "\n\nAnswer using the excerpts when they are relevant."
            )
        chat = [{"role": "system", "content": SYSTEM_PROMPT}]
        chat.extend(messages[-10:-1])
        chat.append({"role": "user", "content": grounded_question})
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        return "\n".join(f"{message['role']}: {message['content']}" for message in chat) + "\nassistant:"

    def reply(self, messages: list[dict[str, str]]) -> str:
        self.load()
        sampler = self._make_sampler(temp=self.temperature, top_p=0.0, top_k=0)
        logits_processors = self._make_logits_processors(
            repetition_penalty=self.repetition_penalty,
            repetition_context_size=160,
        )
        return self._generate(
            self.model,
            self.tokenizer,
            prompt=self.format_prompt(messages),
            max_tokens=self.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False,
        ).strip()


def clean_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        raise ValueError("messages must be a list")
    messages = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            messages.append({"role": role, "content": content[:4000]})
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("last message must be a user message")
    return messages


def make_handler(chat_model: OdysseyChatModel):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OdysseyChat/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_json(200, {"ok": True, "model_id": chat_model.model_id})
                return

            path = self.path.split("?", 1)[0]
            if path == "/":
                path = "/index.html"
            target = (WEB_ROOT / path.lstrip("/")).resolve()
            if not str(target).startswith(str(WEB_ROOT.resolve())) or not target.exists() or target.is_dir():
                self.send_error(404)
                return

            data = target.read_bytes()
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = clean_messages(payload.get("messages"))
                answer = chat_model.reply(messages)
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})
                return
            self.send_json(200, {"reply": answer})

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Odyssey SLM chat UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-id", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--adapter-path", default="outputs/adapters/odyssey-qwen25-0.5b")
    parser.add_argument("--max-tokens", type=int, default=420)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter_path = str(ROOT / args.adapter_path) if args.adapter_path else None
    chat_model = OdysseyChatModel(
        model_id=args.model_id,
        adapter_path=adapter_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(chat_model))
    print(f"Serving Odyssey SLM chat at http://{args.host}:{args.port}")
    print(f"Model: {args.model_id}")
    print(f"Adapter: {args.adapter_path or 'none'}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
