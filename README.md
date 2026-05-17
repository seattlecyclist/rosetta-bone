# Rosetta Bone

<p align="center">
  <img src="docs/images/rosetta-bone-hero.png" alt="Rosetta Bone — The LLM that thinks it's a Good Boy!" width="700">
</p>

Home for a pack of niche dog-domain LLMs that are dumb, excitable,
and confidently wrong about cause-and-effect — exactly what every dog
is.

The first sub-package is **storyteller** — a model fine-tuned to write
fiction from a dog's first-person sensory point of view (scent, sound,
pheromone), instead of the visually-dominant human frame that
general-purpose LLMs default to. See
[docs/models/storyteller.md](docs/models/storyteller.md) for what it
was trained on and how.

## Why fine-tune instead of just prompting a frontier model?

Prompting a frontier model to "be a dumb, excitable dog" gets you an
impression of a dog — a sophisticated model performing dumbness while
its weights still understand cause and effect. Voice drifts back to
neutral as context grows, sensory detail collapses into
anthropomorphic projection, and you patch failures qualitatively with
more prompt.

Fine-tuning a small (8B 4-bit) model on dog-POV fiction
(Beautiful Joe, Black Beauty, Call of the Wild), canine olfaction
papers, and a real dog-behaviour Q&A dataset shifts the actual
next-token distribution. The cadence is baked in. The sensory detail
is grounded. The confidently-wrong register reads as authentic because
the model genuinely loses some of the upstream logical sophistication
for this style. And failures show up as numbers — persona-violation
counts, kept-fraction per stimulus — so each iteration is a measurable
corpus delta rather than another prompt tweak.

And it runs locally. There is no frontier-model inference bill — no
per-token cost, no multi-thousand-token persona prompt re-billed on
every call. A one-time training spend (a few dollars of Claude API
time to synthesise the SFT corpus, plus a few hours of local GPU
time on an M2 Max) buys an adapter you can run forever for free.

See [docs/models/storyteller.md](docs/models/storyteller.md#why-fine-tune-instead-of-just-prompting-a-frontier-model)
for the longer version.

## Sample outputs

*The three samples below were generated from the v11 adapter
(`20260516T195645Z`, current `latest`, trained 2026-05-16). v11
rewrote the persona to braid smell and sound — the previous
"scent-first" framing was replaced with a three-rule decision tree
(sound leads at distance, smell leads at proximity, both fire at
once for embodied events). The mailman sample below illustrates
the rule directly: the dog hears the footsteps before smelling the
mailman, where v9/v10 adapters opened with smell. See
[docs/pilot-history.md](docs/pilot-history.md) and
[docs/comedic-modes.md](docs/comedic-modes.md) for the full
iteration history, and the "Trained adapters" section below for
the version table.*

### Stimulus: `the smell of bacon`

```text
Dear Diary.

I am lying on the floor. I am doing a lot of lying on the floor. This
is my job. I have a very important job.

Then.

A smell.

NOT A POND SMELL. Not the dead leaves in the park. Not the rabbit that
was here Friday. A different smell altogether.

BACON.

The smell comes under the door. Up through the gap. It is a warm smell.
It is a hot smell. It is almost alive. It is coming in fast now.

I know this smell. I have always known this smell. It means someone is
close. It means someone has bacon and is willing to share. probably
not. But a dog can hope.

I am on it. My nose is going very fast. It sniffed the door gap.
Sniffed the floor. Sniffed the air under the door. And then I sniffed
toward the window. Both windows. Then the back door.

The back door.

I can smell it from here. Through the door. A cold draft comes in and
pushes the hot against me a little. This is great actually. More
smell. This is a very good day so far.

BACON.

BACON BACON BACON.

My tail is going. I am doing the low one first. Then the high one. I
do not know which one to do first but that is how it goes.

I walked to the kitchen. I sniffed the floor near the stove. He did
it. HE KEPT IT BEHIND THE STOVE. On purpose. To torture a dog.

I sat. I sat very nicely. I sat so nice that if he was not busy he
would have given me one already. He open the pan and took out a strip
and put it right on the counter.

I sat more nice.

Then he gave me the evidence. The evidence was a very small
grayish-brown strip. I do not know how to eat it yet but I am thinking
about it a lot.

It was in my mouth. It was going very fast. I do not know how to make
it stop.

It was gone.

I still have feelings about it. Very strong feelings. Not good. Not
bad. Complicated.

I am lying on the floor again now.

The couch is safe. I have always been the dog who guards this couch.
No one has ever come in and taken it.

But I fear always that someone will take the smell away.

The pool table has a smell. A ghost smell. A laptop has two smells.
The screen smell and the bottom smell. The toilet has many smells. The
good ones, the bad ones, I will not say which is which.

But there is only one bacon smell.

I think about it a lot.
```

### Stimulus: `the mailman has arrived`

```text
Dear Diary.

It starts before the door even opens. I hear it. Far away. Footsteps
on the path. Not my walks. Not a neighbor. The footsteps are on the
wrong side of the door.

Then the smell hits me. Rubber. And outside. And something else.
Something I cannot name but it is WRONG. My nose goes fast. Sniff
sniff sniff sniff. I pull the leash tight but it is not the time for
a walk.

Then the sound comes again. Closer now. Thump thump thump. I stand
very still. My whole body is a statue. One wrong move.

The door opens.

I BARK.

BARK BARK BARK BARK.

I have told them many times. The mailman should not come here. This
was not a designated walk. This was an emergency. I tilt my head
left. I tilt it right. I move one inch toward the door and away from
the couch. One inch. That is all the range I allowed.

The mailman stands there. So calm. His hands are full and he smells
like many things. Many wrong things. Sticking out from under his bag.
Fat little packets. I read about these. The paper kind. I am not
afraid of the paper but I am afraid of what it means.

He says some words. He says my name and also a bad word.

I bark again.

He says it louder this time.

BARK BARK BARK BARK BARK.

He goes away. He always goes away. I have always been the dog who
wins this fight. The mailman knows this. He is very smart for not
running.

I wait one more second. Two more. I did not get enough justice.

The door closes. The smell goes away all at once. One second. Very
tidy.

I will remember today.

I will also be at the door tomorrow.
```

### Stimulus: `a trip to the vet`

```text
The car stops.

I know this place. It smells wrong. Too many other dogs. Old grease.
Fear from the straggle of cats in the waiting room. Not here. Not a
playroom. Here is about pins in my back.

The man lifts me off the back seat. He smells like dinner and soap
and him and him alone. I press my nose into his chest. He says words
I know. GOOD BOY. GOOD BOY. He goes away.

The table. Cold. Bright. The lady in the blue shirt touches my ear.
I am still smelling him but that's fading now. The table is what they
put me onto. It is a hard flat thing. My legs just sit there now,
won't work.

The ear thing touches my side.

Oh.

OH.

It is very bright. The table is everywhere, it is what I am made of.
I close my eyes. The smell comes back. Him. He is here.

I hear his voice again from far away. I am getting off the table. My
legs know what to do. I am allowed back down.

The man picks me up again and puts me back in the car. I turn three
times on the seat. I am already smelling dinner from the glove box.
That is a good smell. That is the best smell.

I put my head back down on his chest.

He says GOOD BOY again.

He goes around the corner.

I have made it. I am home.
```

## Layout

- `src/rosetta_bone/storyteller/` — Dog-POV Storyteller v1
- `src/rosetta_bone/common/` — utilities shared across future sub-packages
- `config/` — TOML config + curated stimuli list
- `data/` — derived artifacts (gitignored)
- `docs/superpowers/specs/` — design specs

## Architecture: end-to-end flow

![Six-stage end-to-end pipeline (ingest → chunk → embed → SFT generate → train → generate)](docs/images/architecture-flow.png)

The pipeline runs as six sequential CLI commands. Stages 1-3 build
static, idempotent corpus artifacts that don't depend on a frontier
model or stimuli. Stage 4 is where everything fuses — the curated
stimuli, the three pillar chunks (selected via FAISS retrieval), and
the persona+contract are all assembled into Anthropic prompts, and the
resulting `(instruction, story)` pairs become the training data.
Stages 5-6 are strictly downstream — they only see those pairs, never
the persona, never the pillar chunks.

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: ingest --pillar {style|science|behavior}                  │
│                                                                     │
│  Project Gutenberg          EuropePMC              pawgaze/pawgaze  │
│  (curated public-domain     (open-access papers   (visual-Q&A       │
│   animal-POV fiction)        on canine olfaction)  benchmark on HF) │
│         │                          │                     │         │
│         ▼                          ▼                     ▼         │
│  data/raw/style/            data/raw/science/      data/raw/        │
│   {id}.txt                   {pmcid}.pdf            behavior/       │
│                              {pmcid}.json           pawgaze.jsonl   │
│                                                                     │
│  Idempotent: existing files are skipped. HTTP cache under           │
│  data/raw/_cache/ avoids refetching even if outputs are deleted.    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│  STAGE 2: chunk --all                                               │
│                                                                     │
│  Reads each pillar's raw files (PDFs via pdfplumber, behavior       │
│  JSONL row-by-row), splits to ~600-token chunks with 80-token       │
│  overlap on paragraph→sentence boundaries (tiktoken cl100k_base).   │
│  Chunk IDs are stable hashes — re-chunking the same source gives    │
│  identical IDs.                                                     │
│                                                                     │
│         │                                                           │
│         ▼                                                           │
│  data/chunks/{style,science,behavior}.jsonl                         │
│    {id, source, pillar, text, metadata}                             │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│  STAGE 3: embed                                                     │
│                                                                     │
│  Encodes every chunk with sentence-transformers                     │
│  BAAI/bge-small-en-v1.5 (384-dim, L2-normalized). Builds one        │
│  FAISS IndexFlatIP per pillar so cosine-similarity retrieval        │
│  is fast (inner product on unit vectors == cosine).                 │
│                                                                     │
│         │                                                           │
│         ▼                                                           │
│  data/embeddings/{style,science,behavior}.faiss + .ids.json         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│  STAGE 4: sft generate / poll / merge  (the load-bearing stage)     │
│                                                                     │
│  ┌─ config/stimuli.yaml ──────────────────────────────────────┐     │
│  │ - prompt: "the mailman arriving"  variations: 8  form: ... │     │
│  │ - prompt: "a trip to the vet"     variations: 8  form: ... │     │
│  │ - ...                                                      │     │
│  └────────────────────────┬───────────────────────────────────┘     │
│                           │  expand to (stimulus, variation,        │
│                           ▼  form) triples                          │
│                                                                     │
│  ─── For each UNIQUE stimulus (per-stimulus retrieval cache) ───    │
│                                                                     │
│                ┌──────────────────────────────┐                     │
│                │  Embedder.embed(             │                     │
│                │     "the mailman arriving")  │ ◀── same BAAI/bge   │
│                │                              │     model used in   │
│                │  → 384-dim unit vector       │     Stage 3         │
│                └──────────────┬───────────────┘                     │
│                               │                                     │
│         ┌─────────────────────┼─────────────────────┐               │
│         ▼                     ▼                     ▼               │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐       │
│  │science.faiss │      │ style.faiss  │      │behavior.faiss│       │
│  │ IndexFlatIP  │      │ IndexFlatIP  │      │ IndexFlatIP  │       │
│  │              │      │              │      │              │       │
│  │.query(qvec,  │      │.query(qvec,  │      │.query(qvec,  │       │
│  │  top_k=1)    │      │  top_k=1)    │      │  top_k=1)    │       │
│  │  → cos sim   │      │  → cos sim   │      │  → cos sim   │       │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘       │
│         │ chunk_id            │ chunk_id            │ chunk_id      │
│         ▼                     ▼                     ▼               │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐       │
│  │ id → Chunk   │      │ id → Chunk   │      │ id → Chunk   │       │
│  │   map        │      │   map        │      │   map        │       │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘       │
│         │                     │                     │               │
│         ▼                     ▼                     ▼               │
│  e.g. the              e.g. the                e.g. the             │
│  vomeronasal           mailman scene           pawgaze row about    │
│  passage from a        from Beautiful Joe      a dog rushing the    │
│  PMC paper             (style chunk)           door at a visitor    │
│  (science chunk)                               (behavior chunk)     │
│                                                                     │
│         │                     │                     │               │
│         └─────────────────────┼─────────────────────┘               │
│                               ▼                                     │
│                                                                     │
│  ─── Build ONE Claude request per (stimulus, variation, form) ───   │
│                                                                     │
│  prompt_builder.py:                                                 │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │ System block  (cacheable; identical persona+contract     │      │
│   │ across all requests; chunks identical for variations of  │      │
│   │ the same stimulus)                                       │      │
│   │                                                          │      │
│   │   <persona>dumb/funny dog spec</persona>                 │      │
│   │   <contract>"Do NOT invent — base sensory details        │      │
│   │       strictly on the provided text..."</contract>       │      │
│   │   <science> {retrieved science chunk} </science>         │      │
│   │   <style>   {retrieved style chunk}   </style>           │      │
│   │   <behavior>{retrieved behavior chunk}</behavior>        │      │
│   └──────────────────────────────────────────────────────────┘      │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │ User block                                               │      │
│   │   "Stimulus: 'the mailman arriving'.                     │      │
│   │    Form: diary.  Variation: 0."                          │      │
│   └──────────────────────────────────────────────────────────┘      │
│                               │                                     │
│                               ▼                                     │
│  Anthropic Message Batches API (claude-sonnet-4-6)                  │
│   • 50 % batch discount                                             │
│   • Cached system prefix → 90 % discount on persona + contract      │
│   • Within a batch, cached chunks block reused across all N         │
│     variations of the same stimulus                                 │
│   • Returns: {"instruction": "...", "story": "..."} per request     │
│                                                                     │
│                               │                                     │
│                               ▼                                     │
│  data/sft/batches/{batch_id}.jsonl   (raw API results)              │
│  data/sft/manifest.jsonl             (status + token + cost log)    │
│                                                                     │
│  merge: parse, validate JSON, dedup by instruction SHA-1,           │
│         90/10 split into mlx-lm chat format                         │
│                               │                                     │
│                               ▼                                     │
│  data/sft/train.jsonl   +   data/sft/valid.jsonl                    │
│   {messages: [{role: user, content: instruction},                   │
│               {role: assistant, content: story}]}                   │
│                                                                     │
│                               │                                     │
│                               ▼                                     │
│  ── sft stats  (pre-training inspection — run BEFORE train) ───     │
│                                                                     │
│  Joins raw batch results (custom_id → stimulus + angle) with the    │
│  merged train+valid (survivorship after dedup) to surface:          │
│   • overall dedup rate                                              │
│   • per-stimulus and per-angle pair counts + kept fractions         │
│     (angles producing low kept% are candidates to redesign)         │
│   • story token length distribution (p10/p50/p90/max)               │
│   • persona-violation flags (substring scan for "olfactory plume",  │
│     "I contemplated", etc.)                                         │
│  Writes data/sft/stats-<sha>.json next to the corpus.               │
│                                                                     │
│  ⚠ Persona, contract, and pillar chunks exist ONLY in this          │
│    stage's prompts. The trained model never sees them again.        │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│  STAGE 5: train --iters N                                           │
│                                                                     │
│  Shells out to `python -m mlx_lm.lora --train ...`                  │
│                                                                     │
│  Base: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit                │
│  LoRA on top 8 transformer blocks; rank 8, alpha 16, AdamW.         │
│  --grad-checkpoint + --max-seq-length 1024 to fit 32 GB.            │
│                                                                     │
│  Each iter samples a batch from train.jsonl, runs the chat-         │
│  formatted prompt through base+LoRA, computes loss against the      │
│  assistant turn, updates the (small) LoRA weights. Periodic eval    │
│  on valid.jsonl.                                                    │
│                                                                     │
│  mlx-lm sees only the literal (user → assistant) message pairs.     │
│  No persona, no contract, no pillar chunks at this stage.           │
│                                                                     │
│         │                                                           │
│         ▼                                                           │
│  data/adapters/llama31-8b-storyteller-v1/{ISO-timestamp}/           │
│    adapters.safetensors   ◀── LoRA weight delta                     │
│    metadata.json          ◀── base_model, iters, batch_size,        │
│                               data hashes, duration, mlx-lm version │
│  data/adapters/llama31-8b-storyteller-v1/latest → {timestamp}/      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│  STAGE 6: generate "<stimulus>" [--form ...]                        │
│                                                                     │
│  Resolves `latest` symlink → most recent adapter.                   │
│  Loads base model + LoRA into MLX (cached after first call in this  │
│  process).                                                          │
│                                                                     │
│  Prompt template:                                                   │
│    "Write a {form} entry from a dog's first-person sensory point    │
│     of view about the following stimulus: {stimulus}."              │
│                                                                     │
│  Streams tokens through mlx_lm.generate with the configured         │
│  creative-writing sampler (temp=0.85, top-p=0.95) and a repetition  │
│  penalty (1.05) via logits_processors.                              │
│                                                                     │
│  Persona, contract, pillar chunks: all absent. The trained LoRA     │
│  has imprinted those patterns into its weights — what comes out     │
│  is the model's learned approximation of the register Claude        │
│  produced during Stage 4.                                           │
│                                                                     │
│         │                                                           │
│         ▼                                                           │
│  Dog-POV story text                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Reading this diagram

- **Stages 1-3 build static data** that doesn't depend on Anthropic,
  stimuli, or training. Cheap and idempotent to rebuild.
- **Stage 4 is where everything fuses** — stimuli meet pillars (via
  FAISS) meet persona+contract (via `prompt_builder`) meet Claude
  (via Batches). It's the only stage that touches all three pillars,
  a frontier model, and the persona spec simultaneously.
- **Stages 5-6 are downstream of Stage 4** and never see the upstream
  context — they only see the `(instruction, story)` pairs.

### Iteration leverage points

| Want to change                         | Edit                  | Re-run from |
|----------------------------------------|-----------------------|-------------|
| The narrator's voice / register        | `persona.py`          | Stage 4     |
| The kinds of scenes the model handles  | `config/stimuli.yaml` | Stage 4     |
| Grounding diversity / corpus depth     | Add sources (Stage 1) | Stage 2     |
| Training duration / batch / LR / rank  | `default.toml`        | Stage 5     |
| Sampling at inference                  | `default.toml`        | Stage 6     |

---

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

`sft stats` is the pre-training inspector. Run it **between `merge`
and `train`** to catch a bad pilot before spending GPU time. It joins
raw batch results (which carry `custom_id → stimulus + angle`
attribution) with the merged train+valid (which carries dedup
survivorship), then prints:

- Overall counts: raw, errored, invalid-JSON, generated-valid, kept,
  persona-violation totals.
- Per-stimulus pair counts + kept fractions — exposes which stimuli
  hit dedup hardest.
- Per-(stimulus, angle) breakdown — angles producing low kept% are
  candidates to redesign or drop in `config/stimuli.yaml`.
- Story token length distribution (p10/p50/p90/max).
- Persona-violation flags (substring scan for `"olfactory plume"`,
  `"I contemplated"`, etc. — markers the persona explicitly forbids).

A JSON copy is written to `data/sft/stats-<sha>.json` next to the
corpus for archival/comparison across pilots.

### 5. `train` — LoRA fine-tune on Apple Silicon

Shells out to `python -m mlx_lm.lora --train` against
`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`. The merged
`train.jsonl`/`valid.jsonl` from the previous stage is the training
set; LoRA adapter weights land under
`data/adapters/llama31-8b-storyteller-v1/`.

`--iters` controls training length (default 1,000 in
`config/default.toml`; a few hundred is enough to see meaningful
style transfer at 10 K pairs).

Every run tees mlx-lm's stdout to `<adapter_dir>/train.log` and
auto-prints a parsed summary (train + validation loss series,
throughput, peak memory, overfit verdict) at the end. To re-inspect
a past run: `uv run rosetta-storyteller train-inspect [--adapter X]`.
See [docs/runbook.md](docs/runbook.md) for the report format and
verdict heuristics.

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
uv run rosetta-storyteller sft poll --wait     # blocks until "All batches downloaded."
uv run rosetta-storyteller sft merge
uv run rosetta-storyteller sft stats           # inspect BEFORE training

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
2. `uv run rosetta-storyteller sft poll --wait` — blocks until downloaded.
3. `uv run rosetta-storyteller sft merge`
4. **`uv run rosetta-storyteller sft stats`** — read this output carefully.
   Things to look for:
   - **Dedup rate.** Kept fraction below ~60% means too many variations
     are collapsing — review the per-angle table and redesign weak
     angles in `stimuli.yaml` before the full run.
   - **Per-stimulus balance.** Stimuli with very low kept counts may
     need additional angles or different `embed_queries`.
   - **Persona violations.** Any non-zero count means the persona
     is leaking ("olfactory plume", "I contemplated", etc.). Tighten
     `persona.py` before training.
   - **Cache health.** Check `cache_read_input_tokens > 0` in
     `data/sft/manifest.jsonl` — if it's `0`, prompt caching is broken
     and you're paying 2× what you should be.
   - **Eyeball a few stories** with
     `head -3 data/sft/train.jsonl | jq -r '.messages[1].content'`.
5. Iterate `config/stimuli.yaml` and the persona text if any of the
   above looks off. Re-run from step 1.
6. **Full:** `uv run rosetta-storyteller sft generate --count 10000 --phase full --max-requests 10000` →
   `sft poll --wait` → `sft merge` → `sft stats` → `train`.

Cost estimate: pilot ≈ $3-5, full ≈ $20-60 (Sonnet 4.6 batch pricing).

## Pilot history

Each pilot is logged as a self-contained snapshot — what changed
(schema / code / config delta + commit SHAs), the resulting `sft
stats` numbers, findings and lessons learned, and a pointer to the
raw `data/sft/stats-<sha>.json` artifact. The goal is that every
future pilot can be compared like-for-like against the prior one.

See [docs/pilot-history.md](docs/pilot-history.md) for the full log —
including the v5 angle-aware retrieval change that took
kept-after-dedup from ~55 % to ~75 % and absolute kept-pair count
from 57 to 269 with zero persona violations.

## Trained adapters

Every `train` run writes a versioned adapter directory under
`data/adapters/<adapter-name>/<timestamp>/`, with a `metadata.json`
sidecar capturing the hyperparameters, training sha, and (for runs
from 2026-05-12 onward) the corpus-token and tokens-seen counters.
A `latest` symlink in the same directory points at the most recent
run.

There are two product lines, routed at runtime via the `--config`
flag:

- **Adult** (`config/default.toml`) — the original Marley-ish
  register, hosted at `data/adapters/llama31-8b-storyteller-v1/`.
- **Kids** (`config/default-kids.toml`, ages 4-8) — warm, gentle,
  ~500-word lexicon, hosted at `data/adapters/llama31-8b-
  storyteller-kids-v1/`. See the [kids-v1 pilot
  entry](docs/pilot-history.md) for the rationale and audit.

### Adult adapters

Listed in chronological order; the **bold** row is `latest` for
the adult product.

| Adapter timestamp     | Pilot label  | Iters  | Train pairs | Description                                                                                                                            |
| --------------------- | ------------ | ------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `20260511T184645Z`    | bootstrap    | ~300   | unknown     | Pre-versioning shake-out run, before the metadata sidecar landed. No reliable stats. Kept for reproducibility.                         |
| `20260511T194239Z`    | bootstrap    | ~400   | unknown     | Second pre-versioning shake-out. Same caveat as above.                                                                                  |
| `20260511T221504Z`    | **v1 pilot** | 500    | 52          | First metadata-tracked run. Pre-angle-retrieval corpus (~55 % kept fraction). Mostly stylistic noise; baseline for measuring later gains. |
| `20260512T034042Z`    | small-corpus | 500    | 36          | Tight-loop iteration on a stripped corpus — used while debugging the SFT pipeline. Not a published pilot.                              |
| `20260512T042405Z`    | "funny baseline" | 2000 | 35       | Surprise comedic hit on a tiny corpus — deep memorization on 35 pairs produced the funniest mailman story we'd seen. Became the eval-set comedic touchstone for later runs. |
| `20260512T173159Z`    | v6 adapter   | 1000   | 249         | First train on the full 50-stimulus angle-redesigned corpus. Style transferred but the comic voice flattened — surfaced the "more pairs ≠ more humor" lesson.            |
| `20260512T210203Z`    | **v7 pilot** | 1000   | 249         | Style pillar swap (Call of the Wild in, Wind in the Willows out) + per-pilot token telemetry. Kept fraction held at 77 %; humor still flat at 1000 iters / 16 epochs.    |
| `20260513T020823Z`    | v8 pilot     | 2000   | 261         | Comic-pointed angle rewrites + 2000-iter deep-memorization regime. Kept fraction 77 % → 81 %; humor measurably back. One-mode-per-story tendency.                       |
| `20260513T191317Z`    | v9 pilot     | 2000   | 276         | Comedic-mode-tagged angles (delusion / coward / absurd / rationalizer / dissociator). Kept fraction 81 % → 85 %. Single stories now stack 2-3 modes.                    |
| `20260515T180408Z`    | v10 pilot    | 2000   | 309         | Canine-hearing science papers added to the science pillar (50/50 olfaction/audition split) + 5 auditory stimuli (storm, vacuum, footsteps, doorbell, fireworks). Modality-tagged science chunks let stimuli with `modality: hearing` route to hearing chunks at retrieval time. Smell-overweight stories fixed; auditory imagery is load-bearing on the new stimuli but persona still defaulted to scent-first on prompts without an explicit auditory stimulus.                                 |
| **`20260516T195645Z`**| **v11 pilot** (latest) | 2000 | 314 | Persona rewrite — replaced the single "Perceptual frame is scent-first" sentence with parity rules (sound leads at distance, smell leads at proximity, both at once for embodied events) plus an explicit "How a real dog hears" section (high/low frequency facts, onomatopoeia conventions, ear-swivel body-direction, sounds-as-learned-meanings). Same 55 stimuli + corpus as v10; the delta is system-prompt only. Sound now leads the mailman, footsteps, doorbell, and owner-returning stories — the persona-level distance rule is visible end-to-end. Currently `latest`. |

### Kids adapters

| Adapter timestamp     | Pilot label          | Iters | Train pairs | Description                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------- | -------------------- | ----- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`20260517T184857Z`**| **kids-v1** (latest) | 200   | 111         | First kids-product adapter. New persona (`persona_kids.py`) replaces dread/grievance/enemy register with curious / playful / loving / silly / sleepy stances + warm-resolution story shape, same sense-priority + "How a real dog hears" rules from v11. 20 stimuli / 53 angles / 123 kept pairs. Shared adult pillars in v1. Audit clean: zero persona violations across all stories. 200 iters (not 2000) because the smaller corpus bottoms its validation curve at iter 200 — see kids-v1 entry in pilot-history.md for the full overfit trajectory. |

### Loading a specific adapter

```sh
# Default — uses the adult 'latest' symlink:
uv run rosetta-storyteller generate "the mailman arriving"

# Kids product — route via --config:
uv run rosetta-storyteller generate "the new puppy" \
    --form diary \
    --config config/default-kids.toml

# Pin to a specific adapter (timestamp or full path):
uv run rosetta-storyteller generate "the mailman arriving" \
    --adapter 20260512T042405Z

# Inspect a past training log (runs after the tee-to-file change
# in commit ac19e59 only — earlier runs have no train.log):
uv run rosetta-storyteller train-inspect --adapter 20260513T020823Z
```

See [docs/runbook.md](docs/runbook.md) for the `train-inspect`
report format and verdict heuristics.

## Tests

```sh
# Unit tests (fast — 60 tests, ~8s, no network)
uv run pytest tests/unit -q

# Integration smoke test (slow, costs ~$0.10, downloads model weights)
ANTHROPIC_API_KEY=... uv run pytest tests/integration -m slow -v
```

## License

Source code in this repository is released under the [Apache License 2.0](LICENSE).

Trained adapters and model artifacts published from this project (e.g.
on Hugging Face Hub) are covered by the license stated at the
publication site, not by the repo license.

[three-pillars-data-architecture]: https://github.com/agileedge/llm-wiki
[synthetic-data-sandwich]: https://github.com/agileedge/llm-wiki
