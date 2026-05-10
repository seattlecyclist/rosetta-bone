# Dog-POV Storyteller v1 — Design

**Status:** approved
**Date:** 2026-05-10
**Sub-package:** `rosetta_bone.storyteller`

## Context

Build a Python application that fine-tunes a small open-source LLM into a "Dog-POV Storyteller" — a model that writes fiction from a dog's first-person sensory perspective (scent, sound, pheromone) instead of the visually-dominant human frame that all general-purpose models default to.

The product spec is fully described in three wiki concept files (separate repo at `~/devel/agileedge/llm-wiki/`):

- `wiki/concepts/dog-pov-storyteller.md` — the idea
- `wiki/concepts/three-pillars-data-architecture.md` — the data design pattern (science + style + behavior pillars)
- `wiki/concepts/synthetic-data-sandwich.md` — the training pipeline (scrape → frontier API generates SFT pairs strictly grounded in retrieved chunks → LoRA fine-tune small open model)

The wiki is unambiguous about the **load-bearing detail**: pillar chunks must be injected as **strict context** in the SFT-generation prompt with explicit "do not invent — base sensory details strictly on the provided text" instructions. Without that, the frontier model writes from its own pretraining memory and the corpus contributes nothing — the project becomes wasted API spend producing a smaller copy of the data generator.

## Decisions

| Axis | Choice |
|---|---|
| Scope | Full end-to-end pipeline (ingest → SFT-gen → LoRA → infer) |
| Persona for v1 | Lighthearted pampered house pet (Beautiful Joe / A Dog's Tale style, suburban stimuli) |
| Hardware target | Apple Silicon via mlx-lm (M-series, 32GB+) |
| Frontier API | Anthropic Claude Sonnet 4.6, Message Batches API, prompt caching on system block |
| Data acquisition | App auto-downloads (Project Gutenberg, EuropePMC, Hugging Face datasets) |
| Base model | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| Eval | Held-out test split, perplexity only (no LLM-judge / human UI / classifier in v1) |
| Dataset cadence | Pilot 500 pairs (~$3–5) → inspect + iterate prompt → scale to ~10K |
| API request cap | Hard cap of 1000 requests per `sft generate` invocation by default; explicit override required to scale |
| Project structure | `rosetta_bone` umbrella package; this v1 lives at `src/rosetta_bone/storyteller/`; shared utils at `src/rosetta_bone/common/` for future siblings |
| Inference UX | CLI (`rosetta-storyteller generate "..."`) + Python API. No web UI, no notebook. |
| Tooling | uv + pyproject.toml, Python 3.12+ |

## Approach (TL;DR)

1. **Pipeline-as-stages, file-as-checkpoint.** Four CLI subcommands (ingest, sft, train, generate); each reads/writes well-defined JSONL/manifest files in `data/`. Crash-resume = "look at what's on disk."
2. **The strict-context contract is enforced in one file** (`sft/prompt_builder.py`). All Anthropic calls go through it. A snapshot test guards against silent regression of the contract language.
3. **Stimulus-list × per-pillar retrieval** drives generation. ~100 curated dog-life stimuli in `config/stimuli.yaml`; for each, FAISS embedding similarity (BAAI/bge-small-en-v1.5, local) selects top-1 chunk per pillar.
4. **Anthropic Message Batches** for fan-out (~10K calls), with prompt caching on the persona+contract system block; per-stimulus mini-batches let the chunks block also benefit from cache hits across variations.
5. **mlx-lm LoRA** invoked via subprocess wrapper (cleaner than depending on internal APIs). Test perplexity is the entire eval surface for v1.

## Directory layout

```
rosetta_bone/
├── pyproject.toml
├── README.md
├── .env.example                       # ANTHROPIC_API_KEY=
├── config/
│   ├── default.toml                   # paths, model IDs, hyperparameters
│   └── stimuli.yaml                   # curated ~100 dog-life prompts
├── data/                              # gitignored; all derived artifacts
│   ├── raw/{science,style,behavior}/
│   ├── chunks/{science,style,behavior}.jsonl
│   ├── embeddings/{pillar}.faiss
│   ├── sft/
│   │   ├── batches/batch-NNNN.jsonl
│   │   ├── manifest.jsonl             # one line per submitted batch
│   │   ├── train.jsonl                # merged + deduped (mlx-lm chat format)
│   │   └── valid.jsonl
│   └── adapters/llama31-8b-storyteller-v1/
├── src/rosetta_bone/
│   ├── common/                        # shared across future sub-packages
│   │   ├── http.py                    # httpx + retry + on-disk cache
│   │   ├── chunking.py                # token-aware splitter (~80 lines, tiktoken)
│   │   ├── jsonl.py
│   │   ├── config.py                  # tomllib loader → dataclass
│   │   └── logging.py                 # structlog
│   └── storyteller/
│       ├── __init__.py                # public API: generate(stimulus, **kw)
│       ├── cli.py                     # Typer app, entry point
│       ├── ingest/
│       │   ├── science.py             # EuropePMC fetcher + pdfplumber
│       │   ├── style.py               # Project Gutenberg fetcher
│       │   ├── behavior.py            # HF datasets loader (pawgaze)
│       │   └── pipeline.py
│       ├── retrieval/
│       │   ├── embed.py               # sentence-transformers wrapper
│       │   ├── index.py               # FAISS build + query
│       │   └── select.py              # stimulus → {Pillar: Chunk}
│       ├── sft/
│       │   ├── prompt_builder.py      # ★ THE strict-context contract
│       │   ├── persona.py             # lighthearted-pampered-pet spec text
│       │   ├── stimuli.py             # load + expand stimuli.yaml
│       │   ├── generate.py            # build batch requests, submit, poll, save
│       │   ├── merge.py               # batches/*.jsonl → train/valid + dedup
│       │   └── cost.py                # token + dollar accounting
│       ├── train/
│       │   ├── lora.py                # mlx_lm.lora subprocess wrapper
│       │   └── eval.py                # perplexity on valid.jsonl
│       └── infer/
│           ├── model.py               # lazy load base + adapter, cached
│           └── generate.py
└── tests/
    ├── unit/
    │   ├── test_chunking.py
    │   ├── test_prompt_builder.py     # asserts strict-context language present
    │   ├── test_select.py
    │   └── test_merge.py
    ├── integration/
    │   └── test_e2e_tiny.py           # @pytest.mark.slow — Llama-3.2-3B, 5 pairs
    └── conftest.py
```

## Stage 1 — Ingestion

Three pillar fetchers under `src/rosetta_bone/storyteller/ingest/`, each writing to `data/raw/{pillar}/` and skipping work when output already exists. Uniform chunk format `{id, source, pillar, text, metadata}` written to `data/chunks/{pillar}.jsonl`.

- **science.py** — EuropePMC REST search (`canine olfaction OR vomeronasal OR "dog scent" AND OPEN_ACCESS:Y`), download PDFs, extract with `pdfplumber`.
- **style.py** — hardcoded list of Gutenberg book IDs biased to gentle/sentimental tone (Beautiful Joe #440, A Dog's Tale #1059, Bob Son of Battle #3007). Strip standard `*** START OF` / `*** END OF` headers/footers.
- **behavior.py** — `datasets.load_dataset("pawgaze/pawgaze")` to HF cache, serialize relevant text columns.

Chunker: roll our own in `common/chunking.py` using `tiktoken` — ~600 tokens/chunk, 80 overlap, paragraph-then-sentence boundaries. Avoids langchain/llama-index dependencies for ~80 lines of code.

## Stage 2 — SFT-pair generation (load-bearing)

**Stimulus list.** `config/stimuli.yaml` holds ~100 entries; each entry can request N variations (diary entry / vignette / short story; varied lengths). Provides instruction diversity without re-curating stimuli.

**Retrieval.** `sentence-transformers` with `BAAI/bge-small-en-v1.5` (384-dim, fast on Apple Silicon). One `IndexFlatIP` per pillar built at ingest time. `select.py` returns top-1 chunk per pillar — more chunks dilute grounding and inflate token cost. Sub-threshold matches (cosine < 0.25) log warnings indicating corpus gaps.

**Prompt builder** (`sft/prompt_builder.py`) — the only module allowed to call `anthropic`. The system block (cached) contains:

```
<persona>…lighthearted pampered house pet voice spec…</persona>
<contract>
Write one (instruction, story) pair for fine-tuning…
Ground strictly in source material below. Do NOT invent new science.
Sensory mechanisms (VOCs, scent plumes, vomeronasal cues, frequency-shifted
hearing) MUST be drawn ONLY from <science>. Voice MUST echo <style>.
Stimulus-to-reaction patterns MUST be plausible per <behavior>.
Return JSON only: {"instruction": "...", "story": "..."}.
</contract>
<science>{sci_chunk.text}</science>
<style>{sty_chunk.text}</style>
<behavior>{beh_chunk.text}</behavior>
```

User block: `Stimulus: "{stimulus}". Variation: {variation_spec}.`

Persona + contract are byte-identical across all calls → cache them. Per-stimulus mini-batches let the chunks block also stay cached across the N variations of one stimulus.

**Output format** (mlx-lm chat format):

```json
{"messages": [{"role": "user", "content": "<instruction>"},
              {"role": "assistant", "content": "<story>"}]}
```

**Concurrency.** Anthropic **Message Batches API** — submit up to 10K requests/batch, ~24h SLA, 50% discount, no rate-limit gymnastics. `custom_id = stimulus_slug + variation_idx` for reconciliation. Workflow:

1. **plan + submit** — enumerate (stimulus × variation), retrieve chunks, build messages, submit batch, append to `data/sft/manifest.jsonl` *before* network call.
2. **poll** — separate subcommand; reads manifest, asks Anthropic for status, downloads completed results to `data/sft/batches/batch-NNNN.jsonl`, writes token+cost stats to manifest.
3. **merge** — pure-function over `batches/*.jsonl`; validates JSON in each assistant reply (re-queue malformed via a follow-up batch), dedupes by SHA-1 of instruction text, splits 90/10 (seeded) into `train.jsonl` + `valid.jsonl`.

Two-phase support = `--count 500 --phase pilot` vs `--count 10000 --phase full`; phase tag in manifest lets pilot batches be excluded from final training set when prompt was iterated.

**API request cap (safety net).** `sft generate` enforces a hard cap on total requests submitted per invocation. Default `max_requests_per_run = 1000` (configurable in `config/default.toml`, overridable via `--max-requests N`). If `--count` would exceed the cap the CLI refuses to submit and prints the override needed. Exists to prevent runaway spend from a stimuli.yaml expansion bug or a typo on `--count`. The full 10K production run requires an explicit `--max-requests 10000`.

**Throughput throttle.** For any synchronous fallback (retry of malformed JSON via the regular Messages API, not Batches), an in-process token-bucket rate limiter caps requests-per-minute at a configurable value (default `requests_per_minute = 50`, comfortably under Anthropic tier-1 limits). Batch submissions don't need throttling — Anthropic queues them server-side — but the cap is enforced symmetrically so the budget is consistent across paths.

**Cost telemetry** (`cost.py`) reads `usage.{input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens}` from each result; price-table keyed by model. Per-batch totals on poll completion; per-run totals on merge.

## Stage 3 — LoRA fine-tune

Model: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` via `huggingface_hub.snapshot_download`. Pin `mlx-lm` to a known-good revision matching the chosen weights revision (mlx-lm tokenizer-config support occasionally lags new Llama variants).

`train/lora.py` shells out to `mlx_lm.lora` (cleaner than importing churning internal APIs). Defaults: rank 8, alpha 16, target `q_proj`/`v_proj`, batch size 4, lr 1e-5, 1000 iters. Adapter saved to `data/adapters/llama31-8b-storyteller-v1/adapters.safetensors`.

Eval: `mlx_lm.lora --test --data data/sft` → parse stdout perplexity → write `eval.json`. That is the entire eval surface for v1.

## Stage 4 — Inference

Public Python API:

```python
from rosetta_bone.storyteller import generate
text = generate("a trip to the vet", max_tokens=600, temperature=0.85, top_p=0.95)
```

`infer/model.py` lazily loads via `mlx_lm.load(model, adapter_path=...)` and caches a module-level singleton. CLI: `rosetta-storyteller generate "..." [--temperature ...] [--max-tokens ...]`. Defaults tuned for creative writing (temp 0.85, top-p 0.95, repetition penalty 1.05).

## Configuration & secrets

`config/default.toml` is the single source of truth (paths, model IDs, generation params, training hyperparameters). Loaded via stdlib `tomllib` into a frozen dataclass. API key strictly from `ANTHROPIC_API_KEY`; `python-dotenv` loads `.env` if present. No CLI flag for the key.

## Resumability

Every stage is idempotent at the file level:

- Ingest fetchers skip if raw file exists.
- Chunk IDs are stable hashes of source-id + offset.
- SFT generation logs every batch in `manifest.jsonl` *before* submission; crash mid-poll just leaves the batch on Anthropic's side, recoverable next `poll`. Already-downloaded result files never re-fetched.
- Merge is pure over `batches/*.jsonl`; safe to re-run.
- Train uses mlx-lm checkpoints; resume by pointing at last checkpoint.

A single `manifest.jsonl` per stage's data dir is the canonical "what's done" log. No external state store.

## Critical files

- `src/rosetta_bone/storyteller/sft/prompt_builder.py` — the strict-context contract; the load-bearing module
- `src/rosetta_bone/storyteller/sft/generate.py` — Anthropic batch submission, polling, telemetry
- `src/rosetta_bone/storyteller/retrieval/select.py` — stimulus → per-pillar chunk selection
- `src/rosetta_bone/storyteller/train/lora.py` — mlx-lm LoRA wrapper
- `src/rosetta_bone/storyteller/cli.py` — Typer entry point
- `config/stimuli.yaml` — corpus's leverage point; iterate between phases

## Dependencies

Runtime: `anthropic>=0.40`, `httpx>=0.27`, `pdfplumber>=0.11`, `datasets>=2.20`, `huggingface_hub>=0.24`, `sentence-transformers>=3.0`, `faiss-cpu>=1.8`, `tiktoken>=0.7`, `mlx>=0.18`, `mlx-lm>=0.20`, `typer>=0.12`, `rich>=13`, `structlog>=24`, `pydantic>=2.7`, `python-dotenv>=1.0`, `pyyaml>=6`.

Dev: `pytest>=8`, `pytest-asyncio`, `ruff>=0.6`, `mypy>=1.11`.

Entry: `[project.scripts] rosetta-storyteller = "rosetta_bone.storyteller.cli:app"`.

## Testing strategy

- **Unit (always):** chunker boundaries; `prompt_builder` snapshot test asserting strict-context language and pillar tags are present (this is the alarm bell for contract regression); `select_chunks` returns one chunk per pillar; `merge` dedupes correctly and rejects malformed JSON.
- **Integration (`@pytest.mark.slow`):** ingest 3 Gutenberg books + 5 EuropePMC papers, generate 5 SFT pairs against live API (gated on `ANTHROPIC_API_KEY`), train 50 iters of LoRA on Llama-3.2-3B-4bit (smaller than v1 default for speed), one inference call. Target: <10 min, <$0.10.

## Verification (end-to-end smoke test)

After implementation, the user runs:

```sh
uv sync
cp .env.example .env && $EDITOR .env       # add ANTHROPIC_API_KEY

# 1. Ingest (small)
rosetta-storyteller ingest --pillar style --limit 3
rosetta-storyteller ingest --pillar science --limit 5
rosetta-storyteller ingest --pillar behavior --limit 50
rosetta-storyteller chunk --all
rosetta-storyteller embed --all

# 2. Pilot SFT generation (~$0.50)
rosetta-storyteller sft generate --count 10 --phase pilot
rosetta-storyteller sft poll                # waits / re-runs until done
rosetta-storyteller sft merge

# Inspect data/sft/train.jsonl by hand. Confirm sensory grounding.

# 3. Tiny train (~5 min on M-series)
rosetta-storyteller train --iters 200

# 4. Generate
rosetta-storyteller generate "a trip to the vet"
```

**Pass criteria:** the generated text contains scent/sound vocabulary, stays in first-person dog POV for the full passage, and references mechanisms/imagery traceable to the retrieved chunks for at least one of the pilot stimuli.

## Risks

1. **Strict-context contract silently degrades.** Most likely failure mode: a developer "improves" the prompt and weakens the grounding language, or a pillar chunk is empty and the model falls back to pretraining. Mitigations: snapshot test; runtime assertion that each pillar block has ≥100 chars; the `merge` step computes a "% of stories containing a 5-gram from the science chunk" stat and warns if <30%.
2. **Anthropic batch behavior.** Eventually-consistent with up-to-24h SLA. Malformed JSON in `assistant` content is a real failure mode → validator + retry path from day one. Verify `cache_read_input_tokens > 0` in the first batch's telemetry; if zero, the system-block ordering is wrong and we're paying for nothing.
3. **mlx-lm + Llama-3.1-8B-Instruct compatibility.** Pin `mlx-lm==0.20.x` with the exact `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` revision. Integration test catches breakage on upgrade.
4. **Stimulus list curation is the corpus's leverage point.** Bland stimuli yield a bland model regardless of pillar quality. Plan to iterate `stimuli.yaml` between phase-1 (500) and phase-2 (10K).
5. **Pillar coverage gaps.** If science chunks don't cover a stimulus ("dog dreams"), low-similarity retrieval produces awkward grafts under strict context. Sub-threshold warning surfaces this; fix is to drop the stimulus or expand the science fetch query.
6. **Cap is a guardrail, not a budget.** The 1000-request cap prevents accidental fan-out, but it does not enforce a dollar limit. Cost is reported per batch by `cost.py`; an explicit `--max-cost-usd` flag could be added in v1.1 if the cap-by-count proves too coarse.
