from __future__ import annotations

SYSTEM_PROMPT = (
    "You summarize Homer's Odyssey. Preserve plot sequence, names, places, divine "
    "interventions, hospitality scenes, disguises, tests, speeches, and consequences. "
    "Do not modernize the story into generic advice."
)


def compression_prompt(text: str, target_words: int, generation: int) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Generation: {generation}\n"
        f"Target length: {target_words} words\n\n"
        "Summarize the following Odyssey material. Keep concrete narrative events and "
        "recurring motifs. Remove ornament, repetition, and commentary.\n\n"
        f"Text:\n{text}\n\n"
        "Summary:"
    )


def training_user_prompt(text: str, target_words: int) -> str:
    return (
        "Compress this Odyssey passage while preserving plot, characters, places, "
        "divine causality, speeches, and narrative consequences.\n\n"
        f"Target length: {target_words} words\n\n"
        f"Passage:\n{text}"
    )

