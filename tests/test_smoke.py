from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mvp.analysis import concept_coverage
from mvp.compression import group_records
from mvp.essays import dedupe_repeated_paragraphs
from mvp.io import read_jsonl, word_count, write_jsonl
from mvp.prompts import generation_essay_prompt
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

    def test_concept_coverage_tracks_retention_and_loss(self) -> None:
        coverage = concept_coverage(
            "Knowledge, virtue, justice, and substance shape conduct.",
            "Knowledge and virtue remain.",
        )
        self.assertEqual(coverage["retained_concepts"], ["knowledge", "virtue"])
        self.assertIn("justice", coverage["lost_concepts"])
        self.assertIn("substance", coverage["lost_concepts"])
        self.assertAlmostEqual(coverage["retention_ratio"], 0.5)

    def test_generation_essay_prompt_requests_concise_model_reading(self) -> None:
        prompt = generation_essay_prompt(
            generation=2,
            generation_text="Knowledge remains, but justice is muted.",
            target_word_count=500,
            retained_concepts=["knowledge"],
            lost_concepts=["justice"],
            introduced_concepts=[],
            retention_ratio=0.5,
        )
        self.assertIn("one concise paragraph", prompt)
        self.assertIn("Do not discuss the experiment", prompt)
        self.assertIn("Knowledge remains", prompt)

    def test_dedupe_repeated_paragraphs(self) -> None:
        text = "A claim remains.\n\nA claim remains.\n\nA different question remains."
        self.assertEqual(
            dedupe_repeated_paragraphs(text),
            "A claim remains.\n\nA different question remains.",
        )


if __name__ == "__main__":
    unittest.main()
