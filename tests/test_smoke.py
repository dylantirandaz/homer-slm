from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mvp.compression import group_records
from mvp.io import read_jsonl, word_count, write_jsonl
from mvp.text import chunk_words, extractive_philosophy_digest, truncate_words


class SmokeTests(unittest.TestCase):
    def test_jsonl_round_trip(self) -> None:
        records = [{"id": "a", "text": "knowledge and virtue"}, {"id": "b", "text": "truth"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            self.assertEqual(write_jsonl(path, records), 2)
            self.assertEqual(list(read_jsonl(path)), records)

    def test_text_helpers(self) -> None:
        chunks = chunk_words("one two three four five six", chunk_size=3, overlap=1)
        self.assertEqual(chunks, ["one two three", "three four five", "five six"])
        self.assertEqual(truncate_words("one two three", 2), "one two")
        self.assertEqual(word_count("one two three"), 3)

    def test_extractive_digest_returns_argument_text(self) -> None:
        text = (
            "Knowledge matters because it guides action. "
            "This short filler sentence is ordinary. "
            "Therefore virtue and reason must govern desire."
        )
        digest = extractive_philosophy_digest(text, target_word_count=16)
        self.assertIn("because", digest.lower())
        self.assertLessEqual(word_count(digest), 16)

    def test_group_records(self) -> None:
        groups = group_records([{"id": index} for index in range(5)], batch_size=2)
        self.assertEqual([[record["id"] for record in group] for group in groups], [[0, 1], [2, 3], [4]])


if __name__ == "__main__":
    unittest.main()

