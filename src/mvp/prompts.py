from __future__ import annotations


TRAINING_SYSTEM_PROMPT = (
    "You compress philosophical writing. Preserve arguments, concepts, tensions, "
    "and questions. Remove ornamental phrasing, names unless essential, and generic advice."
)


def compression_prompt(input_text: str, target_word_count: int, late_stage: bool = False) -> str:
    if late_stage:
        rules = [
            "Remove historical references and author names unless they are philosophically necessary.",
            "Preserve the unresolved tensions, not just conclusions.",
            "Keep metaphysics, epistemology, ethics, selfhood, suffering, and action in play when present.",
            "Do not turn the output into self-help advice.",
            "Compress aggressively.",
        ]
    else:
        rules = [
            "Preserve philosophical substance over historical detail.",
            "Preserve arguments, not just conclusions.",
            "Preserve disagreement when disagreement is essential.",
            "Avoid direct quotation.",
            "Do not flatten the text into generic advice.",
            "Keep paradoxes and tensions alive.",
            "Compress aggressively.",
        ]

    rule_block = "\n".join(f"- {rule}" for rule in rules)
    return (
        "You are compressing philosophical knowledge for a civilization that may lose "
        "the original texts.\n\n"
        "Rules:\n"
        f"{rule_block}\n\n"
        f"Target length: {target_word_count} words\n\n"
        "Input:\n"
        f"{input_text}\n\n"
        "Output only the compressed philosophy:"
    )


def final_sentence_prompt(input_text: str) -> str:
    return (
        "Everything written by philosophers is gone except the text below.\n\n"
        "Compress it into one sentence that preserves the irreducible core of philosophy.\n"
        "The sentence should not be motivational.\n"
        "The sentence should not be religious unless the input demands it.\n"
        "The sentence should not be merely ethical.\n"
        "The sentence should preserve inquiry, truth, selfhood, suffering, reality, "
        "and action if possible.\n\n"
        "Input:\n"
        f"{input_text}\n\n"
        "Final sentence:"
    )

