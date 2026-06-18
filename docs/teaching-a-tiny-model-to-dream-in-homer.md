# Teaching a Tiny Model to Dream in Homer

I wanted to know what a language model learns if its entire world is one book.

Not a pretrained model. Not a tokenizer trained on the internet. Not a chat model with hidden background knowledge. Just a randomly initialized transformer, byte tokens, and the body text of Samuel Butler's English *Odyssey*.

The result was not a good language model. That is the wrong standard for this experiment. The interesting part was watching the model move from noise, to English-like rhythm, to Odyssey-flavored imitation, and then into overfitting.

## The Setup

The model was intentionally small:

| Setting | Value |
| --- | ---: |
| Parameters | 4,929,536 |
| Layers | 6 |
| Attention heads | 8 |
| Hidden size | 256 |
| Context | 256 bytes |
| Vocabulary | 256 raw byte values |
| Training text | 616,538 bytes |
| Training steps | 10,000 |

The important constraint: this model starts from random weights. It does not use Qwen, GPT, BPE tokenization, or any pretrained checkpoint. It sees byte sequences and learns to predict the next byte.

That makes the experiment closer to a toy microscope than a product demo. The question is not whether the model can answer questions about Homer. It cannot. The question is what statistical shape of the *Odyssey* appears in a tiny neural network trained from scratch.

## The First Surprise: More Training Was Not Better

The validation curve improved quickly, bottomed out around 4,000 steps, then got worse:

| Step | Validation loss |
| ---: | ---: |
| 1,000 | 1.683 |
| 2,000 | 1.358 |
| 3,000 | 1.265 |
| 4,000 | 1.236 |
| 5,000 | 1.261 |
| 6,000 | 1.330 |
| 7,000 | 1.411 |
| 8,000 | 1.553 |
| 9,000 | 1.736 |
| 10,000 | 1.894 |

The training loss kept falling all the way to the end. The final checkpoint reached a train loss of `0.458`, but by then the validation loss had degraded to `1.894`.

That is the central finding: after roughly 4k-5k steps, this model was no longer learning a more general Odyssey-like distribution. It was getting better at the training slice and worse at held-out text.

The most annoying part is also the most honest part: I did not save a 4,000-step checkpoint. I saved at 2,500, 5,000, 7,500, and 10,000. So the best observed checkpoint by validation is 4k, but the best saved checkpoint is probably 5k.

## What the Checkpoints Sounded Like

I used the same prompt each time:

```text
Tell me, O Muse,
```

And the same decoding settings:

```text
temperature=0.45
top_k=20
```

At 2.5k steps, the model has learned the local texture of the book: gods, suitors, ships, strangers, the sea. But grammar is still badly broken.

```text
Tell me, O Muse, sir, and I do not tell yet on who can shee the
first of the suitors and on rocks; the was as I am to be in the stranger
to asse a god diving to the sea...
```

At 5k steps, the model is still broken, but it has a better sense of scene structure. It has learned names, reported speech, and the recurring social machinery of the poem.

```text
Tell me, O Muse, the son of Atreus, and take the stranger
who was his perison are him with a pigs...

Then Ulysses answered, “King Alcinous...
```

At 7.5k and 10k, the model is not simply becoming better. Some passages are more confident, but they are also stranger and more overfit. The held-out loss agrees with that impression.

```text
Tell me, O Muse, and Telemachus wept for a fashion ship in
the far from lawful and in the house of king’...
```

## What It Learned

This model did not learn the *Odyssey* in the human sense. It did not learn that Odysseus wants home, that hospitality has moral weight, or that divine intervention structures the plot.

But it did learn statistical shadows of the poem:

- Names: `Ulysses`, `Telemachus`, `Eumaeus`, `Alcinous`, `Calypso`
- Settings: ships, sea, houses, shore, Troy, Ithaca
- Social scenes: strangers arriving, hosts answering, suitors speaking, people sitting down
- Repeated syntax: "Then X answered", "the son of", "the gods", "on board"
- Epic texture: speech, lineage, movement, return

The 5k sample had the strongest Odyssey-term density in my comparison: 12 Odyssey term hits in a 128-word sample. It also had fewer repeated bigrams than the 2.5k sample. That made it the most interesting saved checkpoint, even though the best validation score happened one thousand steps earlier.

## Was It Just Copying?

I checked generated 8-word sequences against the training text. For these samples, there were zero exact 8-gram overlaps with the source text.

That does not prove the model was not memorizing. A tiny byte model can memorize texture and fragments without reproducing long exact spans. But it does suggest the samples are not just copied chunks pasted out of the training file.

The better description is imitation under constraint. The model learned local rhythms and vocabulary, but not enough structure to sustain clean prose.

## The Actual Lesson

The best part of this experiment is that it makes overfitting visible.

At 2.5k steps, the model is undertrained.
At 5k steps, it is weird but lively.
At 10k steps, it has a much lower train loss and a much worse validation loss.

The final model is not the most interesting model. The middle checkpoint is.

That feels like a useful miniature of language modeling in general: the thing you optimize is not always the thing you want. A lower training loss can make the output less alive, less general, or simply more trapped inside the source.

## Reproducing It

Train the larger scratch model:

```bash
make train-scratch
```

Generate from the best saved checkpoint:

```bash
.venv/bin/python scripts/generate_scratch.py \
  --checkpoint outputs/scratch/odyssey-byte-gpt-10k \
  --weights 005000_weights.safetensors \
  --prompt "Tell me, O Muse," \
  --tokens 1000 \
  --temperature 0.45 \
  --top-k 20
```

Regenerate the blog artifacts:

```bash
make blog-artifacts
```

That writes:

```text
outputs/blog/loss_curve.svg
outputs/blog/scratch_checkpoint_report.md
outputs/blog/scratch_blog_metrics.json
outputs/blog/samples/
```

## Next Experiment

The obvious next run is not 20,000 steps. It is a better checkpoint schedule around the interesting region:

```text
save every 500 steps from 2,500 to 5,500
```

I would also try:

- a word-level or byte-pair tokenizer trained only on the *Odyssey*
- a slightly longer context window
- temperature sweeps for each checkpoint
- nearest-neighbor analysis over generated passages
- training on the *Iliad* plus the *Odyssey* to see whether the model learns a broader Homeric register

The core result still stands: a tiny model trained from scratch on one ancient epic does not understand Homer. But it does learn to dream in the statistical accent of Homer, and there is a narrow window before that dream turns into memorization.
