# Storyteller

A model fine-tuned to write fiction from a dog's first-person sensory
point of view — scent, sound, pheromone — instead of the
visually-dominant human frame general-purpose LLMs default to.

## Why fine-tune, instead of just prompting a frontier model?

Prompting a frontier model to "respond as a dumb, excitable dog" gets
you an *impression* of a dog. Fine-tuning shifts the actual next-token
distribution. A few axes the difference shows up on:

**Voice persistence.** A prompted frontier model is performing — every
other weight in its network is still pulling toward neutral,
well-edited prose. Voice drifts back to baseline as context grows or
the topic moves away from the persona instructions. With a LoRA
adapter trained on dog-POV fiction (Beautiful Joe, Black Beauty,
Call of the Wild), the cadence is baked in. The model isn't pretending
to be a dog; the next-token probabilities have actually shifted.

**Where the sensory detail comes from.** Asked to describe a smell
from a dog's perspective, a frontier model imagines what a dog might
smell — which means anthropomorphic projection, because its training
data is overwhelmingly human-authored prose about dogs. Storyteller's
training pairs are grounded in retrieved chunks from real canine
olfaction papers (EuropePMC) and a dog-behaviour Q&A dataset. The
training data itself contains canine-correct perceptual detail, so the
model learns the register rather than guessing it.

**The "dumb and confidently wrong" register, specifically.** A
prompted frontier model performs dumbness while its underlying weights
still understand cause and effect — the seams show as it knowingly
mis-describes a situation it clearly grasps. A small (8B 4-bit) model
fine-tuned into the persona genuinely loses some of that upstream
sophistication for this style. The wrongness reads as authentic
because in a real sense it is.

**Iterability.** Prompt engineering fails qualitatively ("this sounds
too sophisticated") and you patch with more prompt. Fine-tuning fails
*measurably* — persona-violation counts, dedup-collapse rates,
kept-fraction per stimulus angle. [docs/pilot-history.md](../pilot-history.md)
is the record of those iterations, each one a corpus delta with
numbers attached.

**Ops.** Prompted: you pay for a multi-thousand-token system prompt on
a frontier-sized model every call. Fine-tuned: an 8B 4-bit adapter
runs locally, no per-call persona tax.

## Base model

`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`, trained on Apple
silicon (M2 Max) via `mlx_lm.lora` — LoRA on the top 8 transformer
blocks, the standard configuration for stylistic adaptation.

## Training corpus

SFT pairs are synthesised by Claude from chunks retrieved across three
domain pillars, each chunk grounding the generated story in real
material so the model is learning a register, not hallucinating one.

### Style — public-domain animal-POV fiction (Project Gutenberg)

The voice the model is trying to inherit, from books written in (or
adjacent to) the first-person animal frame:

- *Beautiful Joe* — Marshall Saunders (canonical dog-POV sentimental novel)
- *A Dog's Tale* — Mark Twain (short, first-person dog narrator)
- *Bob, Son of Battle* — Alfred Ollivant (working sheepdog protagonist)
- *Black Beauty* — Anna Sewell (first-person animal autobiography)
- *The Call of the Wild* — Jack London (wild-survival register)

### Science — canine olfaction papers (EuropePMC)

Open-access papers about how dogs actually perceive the world — scent
thresholds, pheromone detection, auditory range. Grounds the sensory
detail so descriptions reflect real canine perception rather than
guesses about what dogs might smell.

### Behavior — dog Q&A dataset (HuggingFace `pawgaze/pawgaze`)

Real-world behaviour questions and answers — what dogs do, why, and in
what contexts. Anchors the cause-and-effect logic of stories to actual
canine behaviour patterns.

## How a training pair is built

For each stimulus (e.g. *the smell of bacon*), the pipeline retrieves
chunks from each pillar via angle-aware embedding queries, then asks
Claude to write a short story grounded in those chunks. The resulting
SFT pair feeds the LoRA fine-tune.

The angle design — one sensory + one positive-emotional + one
negative-emotional slice per stimulus — is what stops dedup from
collapsing variations into near-duplicates. See
[docs/pilot-history.md](../pilot-history.md) for the iteration-by-iteration
record of what changed and what it bought, including the v5 finding
that drove this rule.

## Pilot history

[docs/pilot-history.md](../pilot-history.md) tracks every training
run — corpus stats, kept fractions, persona-violation counts, and
findings.
