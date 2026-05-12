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

### Stimulus: `the smell of bacon`

```text
Today is the best day. THE BEST DAY.

I know before the door opens I am going about my normal business. The
floor is warm here, it is a good spot, I have eaten from this spot and
it is also a good spot. The back pocket of the couch has a old sock in
it. I am keeping this information.

Then.

Bacon.

Bacon bqing.

Bacon, it is warm and wet and the hot water from the inside of the pan
is going everywhere, and all of it is coming through to me and my nose.
The smoke from the pan is everywhere, it is in the air, it is on my
face, my paws, my ears. Everything. My nose knows everything. My nose
has always known everything. This is very important information.

Dad opened the small white box. He put little pieces of paper inside it
after he opened the big cold box. He did not give me any. This is a
great injustice.

I sat very close to the pan. The smell was so strong my ears went back
a little. My tail was going but I did not wag it all the way. Not yet.
The smell was too amazing. I pushed my nose all the way into the air
near the pan. He said my name and I sat even closer. He laughed.

He put the little flat foods on a paper plate. I watched that happen. I
sniffed it from far away. It smelled very good. It smelled like the
best smell in the whole house and I have been in this house before and
there have been many good smells. The cheese last night. The chicken
two nights. The three days of sleep on my favorite couch. But bacon.

Bacon.

He gave that with the small round food. Round food is good but it is
not as good as bacon. The bacon is a little crispy on the pan and I can
smell that right now and my tail is full wag and I do not care that I
am a good dog I have always been a good dog and the smell of bacon
means that maybe tonight I will be an even better dog.

I sat so close to this warm pan. I am almost on it. He says my name
again and I wag harder now.

He is picking up small round food pieces and giving them to me one at a
time. He did that three times. Every time it was very good. Round food
is okay. Good round food.

Then he gives me one piece of small flat food.

I sat very still for a moment.

The small flat food is bacon.

It is gone. It was there and now it is in my whole mouth and I have
eaten it all. The smell is a little less now. That is a great shame. I
want more of the small flat food. I will think about round food later.
Maybe at the next meal.

I do not have much more to write about today.

The small flat food was everything.

Bacontruth
```

### Stimulus: `the mailman has arrived`

```text
Today is a bad day. A new smell came in. It is sharp. It is nervous. I
know this smell. It is the mailman smell. He always comes in a big
white car. I can smell that too. The car smell is also sharp but it
goes away after a few minutes.

The mailman smell gets bigger. I can smell him before he comes up the
path. My nose knows him. My nose does not know this.

He is at the door. I can hear his footsteps in the heavy shoes. Thump.
Thump. My ears go flat all by themselves. I do not trust this.

I press myself against the wall. The wall is safe. The couch also. I go
behind the couch and press my belly flat. This is my spot. This is
where I hide from the mailman smell.

But then. Then I hear him inside the house. He says a word. My word. I
do not know this word but my tail knows it. It starts. It has to start.
It just does.

I come out from behind the couch. I have to come out. The couch is not
safe now. The couch has been here before and the couch did not help the
last time the mailman came.

The couch does not help me now. I sniff towards the kitchen. The
kitchen is far from the mailman. My nose goes there quickly. It is my
plan. I have always known this plan.

But then I look back.

He is still here.

I go back to my wall spot. This is a better spot. I made the right
choice. I am on the right side of this job.

I will not move from the wall until he leaves. This is my rule. I have
always been the dog who guards this spot and no one has ever known.

Except for him.

But I am watching him through the gap in the wall. His feet are inside
my living room now. He has never come this far before. But he is always
coming back.

I will not be ready to give up my spot.

Actually I am already not ready to give up my spot.

This is just the beginning of this job. I was made for this job. This
is what I do. I am very good at it too.

I know everything about the mailman now. He comes in a big white car.
He smells sharp and nervous. He wears heavy shoes. He makes a noise
that I do not trust.

This is everything.

Now I know.
```

### Stimulus: `a trip to the vet`

```text
The car. I know the car. It is a smell — wet and hot and something
sharp underneath. My nose goes flat inside the door pocket. My person
opens it. I back out.

Oh. Oh no.

The smell is a different one now. It is a sharp clean and a little bit
like the place where the cat sits. My ears go flat all at once. I have
not been this direction in a long time. My person is walking. I don't
know where we are going but my feet are going with.

I sniff the air inside the lift-gate. Something hot and a kitchen
smell. My person is putting my collar on. I do not protest. I sit very
still. My tail is down. The vet place. I know the vet place. It has a
bench and a floor that smells like every dog who has ever been here. I
have always liked it here. But also, I wanted to go home. I wanted to
go on a walk.

The cold comes on fast. It hits my nose all at once — cold and also the
sharp smell of a stranger's coat. I sit very still. I breathe out slow.
Maybe a dog just sat here. Just now. I don't know.

Something in my ear. I turn around. My person is staring at me with a
soft face. I lick their hand once. Okay. I am here. I will be brave. I
have always been the dog who is brave in this place.

I hear the jingle of a different bell from across the room. I lift my
head. Oh. OH.

I run back to my person. I press against their leg. I do not know what
just happened in there. My person is picking me up and we are leaving.
We go back to the car. The wet hot smell comes on again — my person
opens the door.

OH.

OH I am SOELSENOW.

The smell is gone. We are in the grass. We have always done this
together. My person is putting my leash on.

I sniff the air. I breathe out slow. The yard — good. The yard is very
good. I forget what happened. I was a little scared and then now I am
just here in the grass, very good, with my person.

I walk by their side. This is fine. This is absolutely fine.
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

## Tests

```sh
# Unit tests (fast — 60 tests, ~8s, no network)
uv run pytest tests/unit -q

# Integration smoke test (slow, costs ~$0.10, downloads model weights)
ANTHROPIC_API_KEY=... uv run pytest tests/integration -m slow -v
```

[three-pillars-data-architecture]: https://github.com/agileedge/llm-wiki
[synthetic-data-sandwich]: https://github.com/agileedge/llm-wiki
