# PLAN.md — Minimum Viable Philosophy

## Project Name

**Minimum Viable Philosophy**

Alternative names:
- The Last Sentence
- After the Library Burns
- Philosophical Compression
- The Compressed Academy
- The Irreducible Philosophy Experiment

---

## One-Sentence Summary

Build a system that repeatedly compresses a large philosophy corpus through small language models and/or summarization loops to discover which philosophical ideas survive extreme compression.

---

## Core Research Question

If thousands of years of philosophy are repeatedly compressed, distilled, and re-expressed, what survives?

This project is not trying to make a better chatbot. It is trying to study whether philosophy has an irreducible core.

The central question:

> What is the minimum description length of philosophy?

---

## Conceptual Premise

Human civilization preserves philosophy through books, oral tradition, institutions, teachers, summaries, aphorisms, and canon formation.

This project simulates that process computationally.

Start with a large philosophy corpus:

```text
Plato
Aristotle
Stoics
Epicureans
Buddhist philosophy
Confucian texts
Augustine
Aquinas
Descartes
Spinoza
Hume
Kant
Hegel
Nietzsche
Wittgenstein
Simone Weil
Arendt
Rawls
etc.
```

Then repeatedly compress:

```text
1000 texts
↓
100 summaries
↓
10 essays
↓
1 treatise
↓
1 chapter
↓
1 page
↓
1 paragraph
↓
1 sentence
↓
1 phrase
```

At each stage, measure what philosophical content survives, mutates, disappears, or becomes dominant.

---

## Final Product

The final system should produce:

1. A reproducible compression pipeline.
2. A dataset of compressed philosophical generations.
3. A visualization dashboard.
4. A final "last surviving philosophy" output.
5. Metrics showing which schools, concepts, and arguments survived compression.
6. A short writeup explaining findings.

---

# 1. Project Goals

## Primary Goal

Create a working experimental framework that compresses philosophical knowledge across multiple generations and analyzes the resulting drift.

## Secondary Goals

- Compare different compression strategies.
- Compare model-based compression vs direct summarization.
- Track philosophical concept survival.
- Detect convergence toward philosophical "attractors."
- Identify which schools of thought are most compression-resistant.
- Produce a beautiful demo showing philosophy collapsing into its final irreducible form.

---

# 2. Non-Goals

This project should not become:

- A generic philosophy chatbot.
- A RAG app.
- A benchmark leaderboard.
- A simple summarizer.
- A paper search tool.
- A "talk to Plato" simulator.

The interesting part is the repeated compression and survival analysis.

---

# 3. System Overview

## Pipeline

```text
Raw Philosophy Corpus
        ↓
Chunking + Metadata
        ↓
Generation 0 Corpus
        ↓
Compression Round 1
        ↓
Compressed Corpus G1
        ↓
Compression Round 2
        ↓
Compressed Corpus G2
        ↓
...
        ↓
Final Irreducible Philosophy
```

Each generation should be saved.

No generation should overwrite a previous one.

---

# 4. Recommended Stack

## Core Language

Python 3.11+

## Local Model Runtime Options

Preferred for Apple Silicon:

- `mlx-lm`
- `ollama`
- `llama.cpp`

Suggested small models:

- Qwen2.5-1.5B-Instruct
- Qwen2.5-3B-Instruct
- Llama 3.2 1B
- Llama 3.2 3B
- Gemma 2B
- SmolLM2-1.7B

## Data

Use `.jsonl` for every intermediate generation.

## UI

Use one of:

- Streamlit
- Next.js + FastAPI
- simple static HTML + Plotly

For fastest build, start with Streamlit.

## Storage

Recommended directory layout:

```text
minimum-viable-philosophy/
  PLAN.md
  README.md
  requirements.txt
  data/
    raw/
    processed/
    generations/
      gen_000/
      gen_001/
      gen_002/
  scripts/
    ingest.py
    chunk.py
    compress.py
    evaluate.py
    visualize.py
  src/
    corpus.py
    models.py
    prompts.py
    compression.py
    metrics.py
    concepts.py
    utils.py
  dashboards/
    app.py
  outputs/
    figures/
    reports/
    final_sentence.txt
```

---

# 5. Data Design

## Raw Text Metadata Schema

Each raw text should become:

```json
{
  "text_id": "plato_republic",
  "title": "Republic",
  "author": "Plato",
  "tradition": "Greek",
  "period": "Ancient",
  "school": "Platonism",
  "source_path": "data/raw/plato/republic.txt",
  "license": "public_domain_or_allowed",
  "language": "English",
  "translator": "unknown_or_named"
}
```

## Chunk Schema

Each chunk should be stored as:

```json
{
  "chunk_id": "plato_republic_00042",
  "text_id": "plato_republic",
  "title": "Republic",
  "author": "Plato",
  "tradition": "Greek",
  "school": "Platonism",
  "generation": 0,
  "chunk_index": 42,
  "text": "..."
}
```

## Compressed Output Schema

Each compression output should be:

```json
{
  "generation": 3,
  "parent_generation": 2,
  "unit_id": "gen_003_unit_0012",
  "parent_ids": ["gen_002_unit_0041", "gen_002_unit_0042"],
  "compression_ratio": 0.25,
  "target_word_count": 500,
  "method": "model_summary",
  "model": "qwen2.5-1.5b-instruct",
  "prompt_version": "v1",
  "text": "...",
  "timestamp": "..."
}
```

---

# 6. Compression Strategies

Implement at least three strategies.

## Strategy A: Hierarchical Summarization

Group chunks into batches.

For each batch, ask the model to compress them.

Then recursively compress summaries.

Example:

```text
1000 chunks
↓
200 summaries
↓
40 summaries
↓
8 summaries
↓
1 final summary
```

## Strategy B: Philosophical Essence Extraction

Prompt the model to preserve:

- central claims
- arguments
- concepts
- tensions
- examples
- practical implications

Do not preserve:
- names
- historical context
- ornament
- repetition
- direct quotes

## Strategy C: Aphoristic Compression

At later stages, force compression into:

- maxims
- aphorisms
- short principles
- laws of thought
- final sentence

This is where the project becomes aesthetically interesting.

---

# 7. Prompt Design

## Compression Prompt V1

```text
You are compressing philosophical knowledge for a civilization that may lose the original texts.

Your task is to preserve the deepest ideas, arguments, tensions, and questions from the input.

Rules:
- Preserve philosophical substance over historical detail.
- Preserve arguments, not just conclusions.
- Preserve disagreement when disagreement is essential.
- Avoid direct quotation.
- Avoid unnecessary author names.
- Avoid modern commentary.
- Do not flatten everything into generic advice.
- Keep paradoxes and tensions alive.
- Compress aggressively.

Input:
{input_text}

Target length:
{target_word_count} words

Output:
```

## Late-Stage Compression Prompt

```text
You are compressing the surviving remains of philosophy.

The goal is not to summarize everything.

The goal is to preserve what cannot be lost without losing philosophy itself.

Rules:
- Remove all historical references.
- Remove all names.
- Remove all decoration.
- Preserve only the most fundamental principles, tensions, and questions.
- Do not produce self-help advice.
- Do not produce vague positivity.
- Do not resolve contradictions too quickly.
- Keep the deepest conflict alive.

Input:
{input_text}

Compress into {target_format}.

Output:
```

## Final Sentence Prompt

```text
Everything written by philosophers is gone except the text below.

Compress it into one sentence that preserves the irreducible core of philosophy.

The sentence should not be motivational.
The sentence should not be religious unless the input demands it.
The sentence should not be merely ethical.
The sentence should preserve inquiry, truth, selfhood, suffering, reality, and action if possible.

Input:
{input_text}

Final sentence:
```

---

# 8. Model Training Option

The project can work without fine-tuning, but a more ambitious version includes a lightweight fine-tuned model.

## Baseline Mode

Use an existing local instruct model to compress texts.

## Fine-Tuned Mode

Fine-tune a small model on philosophy-style compression examples.

Training data format:

```json
{
  "instruction": "Compress this philosophical passage while preserving its core argument.",
  "input": "...",
  "output": "..."
}
```

Recommended approach:

- Start with 500–2000 examples.
- Use LoRA.
- Use MLX on Apple Silicon.
- Train 1B–3B model only.
- Compare base model vs fine-tuned compressor.

## Important

Fine-tuning is optional. The first milestone should not depend on fine-tuning.

---

# 9. Evaluation Metrics

## 9.1 Compression Ratio

Track tokens before and after.

```text
compression_ratio = output_tokens / input_tokens
```

## 9.2 Concept Survival

Create a curated concept list.

Example concepts:

```text
truth
justice
virtue
soul
self
desire
suffering
death
beauty
being
appearance
knowledge
ignorance
reason
faith
power
freedom
duty
pleasure
goodness
evil
meaning
absurdity
language
God
nature
law
state
friendship
love
time
identity
```

For each generation, measure whether each concept appears explicitly or semantically.

Use:

- keyword matching
- embedding similarity
- model-based classification

## 9.3 School Survival

Track whether each philosophical school remains detectable:

```text
Platonism
Aristotelianism
Stoicism
Epicureanism
Skepticism
Buddhism
Confucianism
Christian philosophy
Rationalism
Empiricism
Kantianism
German Idealism
Utilitarianism
Existentialism
Phenomenology
Analytic philosophy
Pragmatism
```

## 9.4 Argument Survival

Track specific argument patterns:

```text
appearance vs reality
virtue produces flourishing
desire causes suffering
knowledge requires justification
freedom conflicts with causality
power shapes morality
language limits thought
death gives life urgency
justice is order
the good is beyond pleasure
```

## 9.5 Entropy / Diversity

Measure whether philosophy collapses into a single generic worldview.

Metrics:

- unique token count
- type-token ratio
- embedding dispersion
- number of surviving concepts
- number of surviving schools

## 9.6 Drift

Compare generation N to generation 0.

Use embeddings:

```text
drift(g) = distance(embedding(gen_0), embedding(gen_g))
```

Also compute adjacent drift:

```text
step_drift(g) = distance(embedding(gen_g), embedding(gen_g-1))
```

## 9.7 Attractor Detection

Run the experiment multiple times with different seeds, models, and prompts.

If different runs converge to similar final sentences, there may be a philosophical attractor.

---

# 10. Experiments

## Experiment 1: Full Corpus Compression

Compress all philosophy texts together.

Goal:

Find the global minimum viable philosophy.

## Experiment 2: Tradition-Specific Compression

Compress each tradition separately.

Examples:

```text
Greek philosophy → final sentence
Buddhist philosophy → final sentence
German philosophy → final sentence
Analytic philosophy → final sentence
Political philosophy → final sentence
```

Then compare final outputs.

Question:

Do different traditions converge?

## Experiment 3: Cross-Tradition Fusion

Take final outputs from multiple traditions and compress them together.

Question:

What survives when traditions collide?

## Experiment 4: Names Removed vs Names Preserved

Run two versions.

Version A:
- names allowed

Version B:
- names forbidden

Question:

Does philosophy survive without authority?

## Experiment 5: Argument-Only Compression

Strip all historical context and examples.

Preserve only arguments.

Question:

Are arguments more durable than doctrines?

## Experiment 6: Aphorism Bottleneck

Force every generation into aphorisms.

Question:

Does philosophy become wisdom literature?

## Experiment 7: Anti-Self-Help Guardrail

Detect and penalize generic self-help outputs.

Question:

Can the system preserve philosophical depth instead of collapsing into "be kind and seek balance"?

---

# 11. Dashboard Requirements

Build a dashboard with:

## Page 1: Compression Tree

Show:

```text
Generation 0 → Generation 1 → ... → Final Sentence
```

Each node should show:
- word count
- token count
- compression ratio
- top surviving concepts

## Page 2: Concept Survival

Line chart:

```text
x-axis: generation
y-axis: concept presence score
```

Concepts:
- truth
- justice
- suffering
- virtue
- freedom
- being
- self
- power
- language

## Page 3: School Survival

Heatmap:

```text
rows: philosophical schools
columns: generations
values: survival scores
```

## Page 4: Drift Map

2D embedding projection of generations.

Use PCA or UMAP.

Show whether the corpus converges, jumps, or spirals.

## Page 5: Final Output

Display:

- final paragraph
- final sentence
- final phrase
- comparison across runs

---

# 12. Implementation Milestones

## Milestone 0: Repo Setup

Create:

```text
README.md
PLAN.md
requirements.txt
data/
scripts/
src/
dashboards/
outputs/
```

Install dependencies:

```bash
pip install datasets pandas numpy matplotlib plotly streamlit sentence-transformers scikit-learn tqdm pydantic
```

Optional:

```bash
pip install mlx-lm
```

## Milestone 1: Corpus Loader

Implement:

```bash
python scripts/ingest.py --raw-dir data/raw --out data/processed/corpus.jsonl
```

Acceptance criteria:

- Reads `.txt` files.
- Attaches metadata.
- Saves JSONL.
- Logs count of texts and total words.

## Milestone 2: Chunker

Implement:

```bash
python scripts/chunk.py --input data/processed/corpus.jsonl --out data/processed/chunks.jsonl --chunk-size 1200
```

Acceptance criteria:

- Produces chunks.
- Preserves source metadata.
- No empty chunks.
- Stable chunk IDs.

## Milestone 3: Compression Engine

Implement:

```bash
python scripts/compress.py \
  --input data/processed/chunks.jsonl \
  --generation 1 \
  --target-word-count 500 \
  --batch-size 5 \
  --model local
```

Acceptance criteria:

- Groups chunks.
- Calls model.
- Saves generation outputs.
- Can resume if interrupted.
- Does not overwrite existing generations unless `--force`.

## Milestone 4: Recursive Compression

Implement:

```bash
python scripts/run_pipeline.py --max-generations 8
```

Acceptance criteria:

- Runs until final target length.
- Saves every generation.
- Records token counts.
- Produces final paragraph and sentence.

## Milestone 5: Concept Metrics

Implement:

```bash
python scripts/evaluate.py --generations data/generations --out outputs/metrics.json
```

Acceptance criteria:

- Computes concept survival.
- Computes school survival.
- Computes compression ratio.
- Computes drift.

## Milestone 6: Dashboard

Implement:

```bash
streamlit run dashboards/app.py
```

Acceptance criteria:

- Shows all generations.
- Shows concept survival chart.
- Shows final sentence.
- Shows drift map.

## Milestone 7: Repeatability

Implement seeds and run IDs.

Directory format:

```text
outputs/runs/run_001/
outputs/runs/run_002/
outputs/runs/run_003/
```

Acceptance criteria:

- Multiple runs can be compared.
- Final sentences can be listed side by side.
- Attractor similarity is computed.

---

# 13. Suggested CLI

The final project should support:

```bash
python scripts/run_pipeline.py \
  --corpus data/raw \
  --run-name full_philosophy_v1 \
  --model qwen2.5-1.5b \
  --generations 8 \
  --compression-schedule 1000,500,250,100,50,25,10,1
```

And:

```bash
python scripts/evaluate.py \
  --run outputs/runs/full_philosophy_v1
```

And:

```bash
streamlit run dashboards/app.py
```

---

# 14. Compression Schedule

Start with this schedule:

```text
Generation 0: raw chunks
Generation 1: 500 words per batch
Generation 2: 250 words per batch
Generation 3: 100 words per batch
Generation 4: 50 words per batch
Generation 5: 25 words per batch
Generation 6: 10 sentences total
Generation 7: 1 paragraph
Generation 8: 1 sentence
Generation 9: 1 phrase
```

Do not compress too aggressively early.

The early stages should preserve diversity.

The late stages should force philosophical bottlenecks.

---

# 15. Anti-Collapse Rules

A major failure mode is collapse into vague advice.

Bad final outputs:

```text
Be kind and seek balance.
Live well and help others.
Truth matters.
```

These are too generic.

The system should preserve tensions.

Better final outputs:

```text
To live wisely is to examine the tension between what appears, what is, what we desire, and what we owe.
```

or:

```text
Philosophy begins when the self discovers that desire, truth, suffering, and justice cannot be reconciled without changing the soul.
```

Evaluation should penalize:
- generic positivity
- self-help tone
- loss of contradiction
- loss of metaphysics
- loss of epistemology
- loss of ethics

---

# 16. Human Review Loop

Add optional human review.

After each generation, allow the user to rate:

- depth
- faithfulness
- originality
- compression quality
- philosophical richness

Store review:

```json
{
  "generation": 4,
  "unit_id": "gen_004_unit_0008",
  "depth_score": 4,
  "faithfulness_score": 5,
  "notes": "Preserved ethics but lost metaphysics."
}
```

This is optional for the MVP.

---

# 17. Final Demo

The final demo should show:

```text
A civilization begins with 1000 philosophy texts.

After repeated compression, only one sentence survives.

Here is what remained.
```

Then display:

1. The corpus.
2. The compression tree.
3. Concept survival.
4. Lost concepts.
5. Final sentence.
6. Multiple-run comparison.

The demo should feel like watching the Library of Alexandria burn slowly while one sentence escapes.

---

# 18. README Pitch

Use this in the README:

```text
Minimum Viable Philosophy is an experiment in cultural compression.

Given a large corpus of philosophy, the system repeatedly compresses it through language models until only a sentence remains. Across generations, it tracks which concepts, schools, and arguments survive.

The project asks whether philosophy has an irreducible core: if the library burned, what would wisdom remember?
```

---

# 19. Agent Instructions

The coding agent should proceed in this order:

1. Create repo structure.
2. Implement data schemas.
3. Implement corpus ingestion.
4. Implement chunking.
5. Implement a mock compressor first.
6. Implement local model compressor.
7. Implement recursive compression.
8. Implement metrics.
9. Implement dashboard.
10. Add README and example run.

The agent should not start with fine-tuning.

The first working version should use an existing local model or mock compressor.

Fine-tuning comes after the pipeline works.

---

# 20. MVP Scope

The MVP is complete when:

- At least 20 philosophy texts can be ingested.
- The corpus can be compressed for 5+ generations.
- Every generation is saved.
- Final paragraph and final sentence are produced.
- Concept survival metrics are computed.
- A dashboard displays the results.

Do not wait for 1000 texts before building the system.

Start with 20 texts.

Then scale.

---

# 21. Stretch Goals

## LoRA Compressor

Fine-tune a small model specifically to preserve philosophical arguments under compression.

## Multi-Model Comparison

Compare:

- Qwen
- Llama
- Gemma
- Phi
- Claude/GPT if API access exists

## Tradition Divergence

Compare Eastern vs Western vs religious vs analytic philosophy.

## Compression Tournament

Generate multiple compressed outputs per generation.

Select the best using a philosophical quality judge.

## Final Sentence Gallery

Run 100 independent trials.

Create a gallery of final sentences.

Cluster them.

Identify attractors.

---

# 22. Risks

## Risk 1: Copyright

Use public domain or properly licensed texts first.

## Risk 2: Generic Summaries

Use anti-collapse prompts and evaluation metrics.

## Risk 3: Too Much Compute

Start with small corpus and few generations.

## Risk 4: Fine-Tuning Complexity

Do not fine-tune until the basic compression pipeline works.

## Risk 5: Evaluation Is Subjective

Use both automated metrics and human review.

---

# 23. Immediate Next Steps

First build:

```text
scripts/ingest.py
scripts/chunk.py
scripts/compress.py
scripts/run_pipeline.py
scripts/evaluate.py
dashboards/app.py
```

Then create a tiny sample corpus:

```text
data/raw/plato_apology.txt
data/raw/aristotle_nicomachean_ethics.txt
data/raw/epictetus_enchiridion.txt
data/raw/marcus_aurelius_meditations.txt
data/raw/descartes_meditations.txt
```

Run the first experiment:

```bash
python scripts/run_pipeline.py --max-generations 5
```

Then inspect:

```text
outputs/runs/latest/final_sentence.txt
```

If the final sentence is interesting, expand the corpus.

If it collapses into generic advice, improve prompts and metrics.

---

# 24. Success Criteria

This project succeeds if it produces at least one surprising result.

Examples:

- Stoicism survives compression better than Kantianism.
- Metaphysics disappears before ethics.
- "Truth" survives longer than "beauty."
- Different traditions converge to similar final sentences.
- Repeated compression produces something closer to religion than philosophy.
- The final sentence is not generic.
- The model repeatedly preserves a contradiction instead of resolving it.

The best possible outcome:

> Across many runs and traditions, philosophy repeatedly compresses into the same small set of unresolved tensions.

That would be a real finding.

---

# 25. Philosophical Framing

This project can be framed as a computational version of an ancient question:

> Is wisdom discovered or inherited?

If wisdom is inherited, compression destroys it.

If wisdom is discovered, compression may reveal it.

If wisdom is neither, then perhaps philosophy is not a body of knowledge at all, but a recurring wound in consciousness: the place where truth, suffering, desire, and justice fail to become one.

That is what the experiment is trying to test.
