# Rosetta Bone

<p align="center">
  <img src="docs/images/rosetta-bone-hero.png" alt="Rosetta Bone — The LLM that thinks it's a Good Boy!" width="700">
</p>

An umbrella repo for niche dog-domain LLMs trained via the
[three-pillars-data-architecture] + [synthetic-data-sandwich] pattern.

The first sub-package is **storyteller** — a model fine-tuned to write
fiction from a dog's first-person sensory point of view (scent, sound,
pheromone), instead of the visually-dominant human frame that
general-purpose LLMs default to.

## Layout

- `src/rosetta_bone/storyteller/` — Dog-POV Storyteller v1
- `src/rosetta_bone/common/` — utilities shared across future sub-packages
- `config/` — TOML config + curated stimuli list
- `data/` — derived artifacts (gitignored)
- `docs/superpowers/specs/` — design specs

## Pipeline stages

The end-to-end build runs as six sequential CLI commands. Each writes
checkpoint files under `data/` and is idempotent — re-running a stage
picks up where it left off rather than refetching or regenerating.

### 1. `ingest` — fetch raw source material

Downloads the raw text for one of the three pillars and saves it to
disk untouched. Does not transform, chunk, or analyze anything.

| Pillar     | Source                                                                                              | Output                              |
|------------|-----------------------------------------------------------------------------------------------------|-------------------------------------|
| `style`    | Project Gutenberg — curated public-domain animal-POV fiction (Beautiful Joe, A Dog's Tale, etc.)    | `data/raw/style/{id}.txt`           |
| `science`  | EuropePMC — open-access papers matching `canine olfaction OR vomeronasal OR "dog scent" …`          | `data/raw/science/{pmcid}.pdf` (+ `.json` metadata sidecar) |
| `behavior` | Hugging Face — `pawgaze/pawgaze` visual-Q&A benchmark; extracts question + correct-answer narrative | `data/raw/behavior/pawgaze.jsonl`   |

HTTP responses are cached under `data/raw/_cache/` so re-runs skip
already-fetched URLs even when output files are deleted.

### 2. `chunk` — token-aware split into uniform records

Reads each pillar's raw directory and produces a single JSONL of
fixed-size, overlapping chunks. PDFs are text-extracted via
`pdfplumber`; the chunker (cl100k_base via `tiktoken`) splits on
paragraph then sentence boundaries, greedy-packs into ~600-token
chunks, and prepends an ~80-token tail of the previous chunk for
overlap. Chunk IDs are stable SHA-1-suffixed hashes — re-chunking the
same source produces identical IDs.

Output: `data/chunks/{pillar}.jsonl`, one line per chunk:
`{id, source, pillar, text, metadata}`.

### 3. `embed` — build per-pillar FAISS indexes

Encodes every chunk with the local `BAAI/bge-small-en-v1.5`
sentence-transformer (384-dim, L2-normalized) and stores one
`IndexFlatIP` per pillar so cosine-similarity retrieval is O(N) but
near-instant for tens of thousands of chunks.

Output: `data/embeddings/{pillar}.faiss` + `{pillar}.ids.json` (id
order needed to map FAISS row indices back to chunk IDs).

### 4. `sft generate` / `sft poll` / `sft merge` — synthetic SFT pairs (the load-bearing stage)

For each curated stimulus in `config/stimuli.yaml` (e.g., *"the mailman
arriving"*, *"a trip to the vet"*):

1. Retrieve the top-1 chunk from each pillar by cosine similarity to
   the stimulus text.
2. Inject all three chunks into a Claude Sonnet 4.6 prompt as
   **strict context** — `<science>…</science>`, `<style>…</style>`,
   `<behavior>…</behavior>` — with a non-negotiable instruction:
   *"Do NOT invent new science. Voice and sentence rhythm MUST echo
   `<style>`. Stimulus-to-reaction patterns MUST be plausible per
   `<behavior>`."* The persona + contract block is byte-stable across
   calls and cached server-side via Anthropic prompt caching.
3. Claude returns an `(instruction, story)` pair. The story is
   first-person dog-POV narration grounded in the retrieved chunks
   rather than in Claude's pretraining memory — this is *the*
   difference between a useful niche fine-tune and a smaller, slower
   copy of Claude.

`sft generate --count N` plans the (stimulus × variation) pairs and
submits them to Anthropic's **Message Batches API** (50 % discount,
async, no rate-limit gymnastics). A safety cap (default 1,000
requests per invocation) prevents runaway spend; raise with
`--max-requests 10000` for the full run.

`sft poll` checks batch status; downloaded results land in
`data/sft/batches/batch-NNNN.jsonl`. `sft merge` parses every batch
file, validates the JSON, dedupes by instruction hash, splits 90/10
into `data/sft/train.jsonl` + `data/sft/valid.jsonl` in mlx-lm chat
format, and logs token totals + estimated USD cost.

### 5. `train` — LoRA fine-tune on Apple Silicon

Shells out to `python -m mlx_lm.lora --train` against
`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`. The merged
`train.jsonl`/`valid.jsonl` from the previous stage is the training
set; LoRA adapter weights land under
`data/adapters/llama31-8b-storyteller-v1/`.

`--iters` controls training length (default 1,000 in
`config/default.toml`; a few hundred is enough to see meaningful
style transfer at 10 K pairs).

### 6. `generate` — inference

Loads the base model + LoRA adapter once (cached for repeated calls)
and renders a prompt like *"Write a diary entry from a dog's
first-person sensory point of view about: a trip to the vet."* Output
is sampled with creative-writing defaults (temp 0.85, top-p 0.95,
repetition penalty 1.05). Also exposed as a Python API:

```python
from rosetta_bone.storyteller import generate
text = generate("a trip to the vet", form="diary", max_tokens=600)
```

---

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). On macOS:

```sh
brew install python@3.12 uv
```

Then in the repo:

```sh
uv sync
cp .env.example .env && $EDITOR .env   # add ANTHROPIC_API_KEY (and HF_TOKEN — see below)

uv run rosetta-storyteller ingest --pillar style --limit 3
uv run rosetta-storyteller ingest --pillar science --limit 5
uv run rosetta-storyteller ingest --pillar behavior --limit 50
uv run rosetta-storyteller chunk --all
uv run rosetta-storyteller embed

uv run rosetta-storyteller sft generate --count 10 --phase pilot
uv run rosetta-storyteller sft poll       # repeat until "All batches downloaded."
uv run rosetta-storyteller sft merge

uv run rosetta-storyteller train --iters 200
uv run rosetta-storyteller generate "a trip to the vet"
```

`uv run` runs the command inside the project's venv. Alternatively
`source .venv/bin/activate` once per shell session and drop the `uv run`
prefix.

### Recommended: set `HF_TOKEN`

Without an HF token, downloads of the embedding model (~130 MB) and the
Llama-3.1-8B base model (~4.5 GB) hit anonymous rate limits and emit
this warning on every run:

> Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

The warning is emitted by `huggingface_hub` via raw `print()` and can't
be filtered through Python's logging/warnings system — the fix is to
authenticate. Create a free Read-scope token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
and add it to `.env`:

```
HF_TOKEN=hf_...
```

No code changes are needed — `huggingface_hub` picks it up
automatically.

See [docs/superpowers/specs/](docs/superpowers/specs/) for the v1 design and
[docs/superpowers/plans/](docs/superpowers/plans/) for the implementation plan.

## Verbose / debugging output

By default the CLI is quiet — only Rosetta Bone's own structured events
plus genuine warnings are printed. Chatty third-party loggers (httpx,
huggingface_hub, sentence_transformers, transformers, urllib3,
datasets) are suppressed at INFO level.

To see everything (HTTP requests, download progress, library warnings)
pass `-v` / `--verbose` before the subcommand:

```sh
uv run rosetta-storyteller -v ingest --pillar science --limit 5
```

## Iterating: pilot → full

The 1000-request cap is the safety net. Recommended workflow:

1. **Pilot:** `uv run rosetta-storyteller sft generate --count 500 --phase pilot`
2. Inspect `data/sft/train.jsonl` by hand. Confirm sensory grounding,
   look for canned phrases, check `cache_read_input_tokens > 0` in
   `data/sft/manifest.jsonl` (if it's `0`, prompt caching is broken
   and you're not getting the 50% batch + cache discount).
3. Iterate `config/stimuli.yaml` and the persona text.
4. **Full:** `uv run rosetta-storyteller sft generate --count 10000 --phase full --max-requests 10000`

Cost estimate: pilot ≈ $3-5, full ≈ $20-60 (Sonnet 4.6 batch pricing).

## Tests

```sh
# Unit tests (fast — 60 tests, ~8s, no network)
uv run pytest tests/unit -q

# Integration smoke test (slow, costs ~$0.10, downloads model weights)
ANTHROPIC_API_KEY=... uv run pytest tests/integration -m slow -v
```

[three-pillars-data-architecture]: https://github.com/agileedge/llm-wiki
[synthetic-data-sandwich]: https://github.com/agileedge/llm-wiki
