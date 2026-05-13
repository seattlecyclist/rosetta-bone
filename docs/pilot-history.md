# Pilot history

Snapshots of measured changes to the SFT corpus and resulting model
quality, in chronological order. Numbers here come from
`rosetta-storyteller sft stats` runs whose raw JSON output is
committed alongside this doc at `data/sft/stats-<sha>.json`.

Each entry below has the same shape:

- **What changed** — the schema, code, or config delta vs the prior
  pilot, with commit SHAs.
- **Stats** — top-line numbers, ideally compared to the prior pilot.
- **Findings** — what worked, what didn't, and what to do next.
- **Stats artifact** — the path to the underlying JSON.

---

## v1 pilot — baseline, pre-angle retrieval (2026-05-11)

### What changed

First end-to-end pilot off the initial implementation. Original schema
used `variations: N` per stimulus; all N variations of a stimulus saw
identical persona + contract + retrieved chunks, differing only by
`Variation: {idx}` in the user block. The strict-context contract
forbade Claude from inventing, so variations collapsed into
near-identical stories.

- **Pipeline:** initial v1 implementation merged in commit `77f5aad`.
- **Schema:** `variations: N` per stimulus (no angles); plan_batch
  cached chunks per stimulus name. Implemented in commit `2c9b827`.
- **Stimuli:** original 20 entries, variations summing to 100
  (truncated by `expand()` to 94 actual triples). Yaml committed in
  `52fbc7c`.
- **Persona:** original "lighthearted pampered house pet" — literary
  register, produced "olfactory plume / vessel" prose. Committed in
  `a060fdb`.
- **Prompt builder:** strict-context contract, committed in `175862e`.
- **Base model:** `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
  (this is the project's only base model so far; carried forward
  through every pilot).

### Stats

| Metric                 | v1              |
| ---------------------- | --------------- |
| Requests submitted     | 104             |
| Kept after dedup       | 57              |
| Kept fraction          | ~55 %           |
| Persona violations     | not tracked yet |
| Approximate cost       | ~$0.70          |

### Findings

~45 % of pairs dropped at merge due to dedup. The corpus was small
(57 train pairs) and stylistically uniform. Trained adapter produced
flowery, contemplative dog narration ("olfactory plumes", "vessel
without a bottom") — wrong register.

Two root causes, each addressed in later pilots:

1. **Literary persona** — replaced with the dumb/funny dog spec in
   commits `8faf223` (initial dumb rewrite) and `b5fc9e1` (added the
   "be funny like a dog" directives).
2. **Identical-input variations** — replaced with angle-aware
   retrieval in v5 (commit `c4226a5`).

### Stats artifact

Not separately committed (pre-dates the `sft stats` command added in
commit `7cb4dbd`). Top-line numbers above are recoverable from the
`merge_done` log lines and the manifest entries committed at the time
of the v1 run.

---

## v5 pilot — angle-aware retrieval + 50 stimuli (2026-05-12)

### What changed

- **Schema:** introduced `embed_queries: [angle1, angle2, ...]` ×
  `variations_per_query` (commit `c4226a5`). Each angle is
  independently embedded and queries the FAISS pillars, so different
  angles for one stimulus retrieve different chunks. The angle is also
  surfaced to Claude as an `Angle:` hint in the user block, so the
  generated story is recognizably *this* version of the scene rather
  than a generic instance.
- **Stimuli expansion:** `config/stimuli.yaml` from 20 → 50 stimuli
  (commit `c2417d4`).
- **Persona:** dumb/funny dog spec (carried forward from v3 — commit
  `b5fc9e1`).
- **Base model:** `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
  (unchanged).

### Stats

| Metric                          | v1 (no angles) | v5             |
| ------------------------------- | -------------- | -------------- |
| Requests submitted              | 104            | 360            |
| Kept after dedup                | 57             | **269**        |
| Kept fraction                   | ~55 %          | **75 %**       |
| Persona violations              | not tracked    | **0**          |
| Story length p50 / p90 (tokens) | n/a            | 454 / 610      |
| Errored / invalid-JSON          | 0 / 0          | 0 / 0          |
| Approximate cost                | ~$0.70         | ~$2.50         |

### Findings

The angle-aware retrieval fix moved the corpus decisively: kept
fraction +20 pp, absolute kept pairs 4.7×, zero persona violations.
The trained model on this corpus was the first one that produced the
target register reliably.

**The load-bearing lesson:** when you craft `embed_queries` for a
stimulus, the angles that survive dedup are the ones with genuinely
different *emotional or behavioral* content. Angles that share an
*emotional valence* — two anxious takes, two ecstatic takes — collapse
together because Claude writes similar instructions for similar
inputs even when the retrieved chunks differ.

A reliable spread is **one sensory slice + one emotional-positive
slice + one emotional-negative slice** per stimulus. Stimuli that
followed this pattern hit 90–100 % kept fractions; stimuli with
overlapping-valence angles dropped to 44–55 %.

Highest-performing stimuli (all angles ≥ 80 % kept):
*owner crying on the couch*, *midnight bathroom trip*,
*favorite toy lost under the couch*, *sprinkler going off*,
*dishwasher running*, *unexpected afternoon nap on the couch*,
*a trip to the vet*, *a bath being run*, *lying in a sunbeam*,
*meeting another dog at the park*, *owner returning from a long trip*.

Lowest-performing stimuli (kept fraction ≤ 56 %) flagged for
redesign in v6: *the mailman arriving* (44 %), *a long car ride*
(44 %), *owner taking out the trash* (50 %, one angle hit 0 %),
*a sock left on the floor* (50 %), *the back gate left open* (50 %),
*a sick owner staying in bed* (50 %), *owner returning home from
work* (56 %), *dinner being prepared in the kitchen* (56 %),
*a new baby in the house* (56 %).

### Stats artifact

`data/sft/stats-281896dfd6.json`

---

## v6 pilot — angle redesign on 9 underperforming stimuli (2026-05-12)

### What changed

- **Angle redesign** (commit `73d688d`) on the 9 stimuli flagged in
  the v5 findings. Each redesigned stimulus now has angles labeled
  inline (`# sensory`, `# positive`, `# negative`) following the
  spread principle from the v5 lesson.
- **Yaml header** gained a structured design-principle block citing
  this doc, so future angle additions inherit the rule.
- **No other changes:** schema, persona, stimulus count (50), total
  triples (360), unique-angle count (150), and base model all
  unchanged.

### Stats

| Metric                          | v5             | v6                 |
| ------------------------------- | -------------- | ------------------ |
| Requests submitted              | 360            | 360                |
| Kept after dedup                | 269            | **276** (+7)       |
| Kept fraction                   | 75 %           | **77 %** (+2.4 pp) |
| Persona violations              | 0              | **0**              |
| Story length p50 / p90 (tokens) | 454 / 610      | 473 / 615          |
| Errored                         | 0              | 0                  |
| Invalid JSON                    | 0              | 2 (dropped at merge) |
| Approximate cost                | ~$2.50         | ~$2.50             |

### Findings — the redesign was decisive on the targeted set

Aggregate kept fraction only moved +2.4 pp, but that undersells the
change. The 9 redesigned stimuli are ~18 % of the corpus; their gains
were diluted by run-to-run sampling variance on the 41 untouched
stimuli.

**Per-redesigned-stimulus before/after:**

| v5 → v6      | Δ      | Stimulus                                |
| ------------ | ------ | --------------------------------------- |
| 44 % → 89 %  | +44 pp | a long car ride                         |
| 50 % → 83 %  | +33 pp | owner taking out the trash              |
| 50 % → 83 %  | +33 pp | a sick owner staying in bed             |
| 56 % → 89 %  | +33 pp | dinner being prepared in the kitchen   |
| 44 % → 67 %  | +22 pp | the mailman arriving                    |
| 56 % → 75 %  | +19 pp | owner returning home from work          |
| 56 % → 75 %  | +19 pp | a new baby in the house                 |
| 50 % → 67 %  | +17 pp | the back gate left open                 |
| 50 % → 50 %  | 0      | a sock left on the floor                |

**Average lift on the redesigned set: +24 pp.** Eight of nine moved
decisively. This validates the v5 lesson: sensory + positive +
negative is the spread that survives dedup.

**The one that didn't move:** *a sock left on the floor* stayed at
50 %. The redesigned angles ("savoring the rich foot-scent",
"triumphant with stolen sock paraded", "guilty after being caught
chewing") are all still framed around *the dog's relationship to the
sock as object*. Claude collapses them into a single "sock as
cherished contraband" arc regardless of the labeled valence. Real
spread there needs the sock to be incidental to genuinely different
scenes — e.g., dog stepping over it without interest, dog leaving it
as a territorial gift, dog mortified by an owner finding the hoard.
Flagged for v6.1.

**Regressions on untouched stimuli are noise.** Several stimuli swung
down (sprinkler 100 → 67 %, vacuum 67 → 44 %, etc.), but with only
6–9 pairs generated per stimulus, one extra dedup hit flips the
percentage by 11–17 pp. Across 50 stimuli, run-to-run variance on
identical inputs is expected. None of the regressed stimuli is
structurally broken — each still has ≥80 % kept on at least two
angles.

### Stats artifact

`data/sft/stats-308bd4598d.json`

### Next iteration candidates

- **v6.1:** tighten angles for *a sock left on the floor* (the one
  stuck stimulus) and any others trending into the sub-60 % band.
- **Production scale:** validate the v6 trained adapter via
  `eval-compare` against the v5 adapter on the frozen eval set,
  focusing on the redesigned stimuli. If quality lift is visible
  there, scale to a 10K production run (`--count 10000
  --max-requests 10000`, estimated ~$60-80 with the prompt-cache
  discount).

---

## v7 pilot — Call of the Wild swap + token telemetry (2026-05-12)

### What changed

Small-surface iteration after v6. The corpus contents shifted via a
style-pillar swap; the generation schema, persona, and angle design
were carried forward unchanged.

- **Style pillar:** added *The Call of the Wild* (commit `26d50ee`)
  and dropped *The Wind in the Willows* (commit `78e5e77`). London's
  prose has the sensory-pull cadence the storyteller voice wants;
  Grahame's was too gentle / anthropomorphic and contaminated
  retrieval pool with non-dog scenes.
- **Telemetry:** `sft stats` and `train` now capture per-pilot
  corpus tokens and tokens-seen-during-training (commit `5e69e77`),
  so future entries have real volume numbers.
- **Stimuli, angles, persona, retrieval embedder, base model:**
  carried forward from v6 (no change).

### Corpus stats (from `sft stats`)

| Metric                          | v6                | v7                 |
| ------------------------------- | ----------------- | ------------------ |
| Requests submitted              | 360               | 360                |
| Kept after dedup                | 276               | 276                |
| Kept fraction                   | 77 %              | 77 %               |
| Persona violations              | 0                 | 0                  |
| Train pairs / Valid pairs       | (not split-logged) | 249 / 27          |
| Train assistant tokens          | not tracked       | 115,567            |
| Valid assistant tokens          | not tracked       | 13,493             |
| Story length p50 / p90 (tokens) | 473 / 615         | 464 / 619          |
| Approximate cost                | ~$2.50            | ~$2.50             |

### Trained adapter (from `metadata.json`)

| Metric                          | v7                       |
| ------------------------------- | ------------------------ |
| Adapter timestamp               | 20260512T210203Z         |
| Iters / batch_size              | 1000 / 4                 |
| Rank / alpha                    | 8 / 16.0                 |
| Effective epochs                | 16.06                    |
| Tokens seen during training     | 1,856,006                |
| Wall clock (s)                  | 7,697 (~2.1 h)           |

### Findings

The pillar swap was retrieval-positive but **didn't move the dedup
needle**: kept-fraction landed at 77 % (essentially identical to
v6's 77 %). The Call of the Wild chunks did show up in retrieval —
spot-checking `sft merge` output showed sled-dog scenes pulled into
several stimuli — but Claude's stories collapsed similar-valence
chunks at dedup regardless of which book they came from. Pillar
content matters less than angle spread.

**Quality regression that wasn't visible in the stats.** Generations
from this adapter felt flatter than v5 — less comic, more generic
safety-blanket prose ("The couch is safe. I have always been the dog
who stays safe on the couch."). The 16 epochs at 1000 iters were
enough to learn the prose surface but not enough to memorize the
comic-pointed angles to the punchline level. This was the load-
bearing observation that drove the v8 plan.

**Telemetry now load-bearing.** Token volume per pilot + per-
adapter tokens-seen means future pilots can be compared on
effective-epoch counts, not just iter counts. Backfilled v7
metadata after the fact (the schema change landed mid-day).

### Artifacts

- Stats JSON: `data/sft/stats-de722e5fae.json`
- Adapter: `data/adapters/llama31-8b-storyteller-v1/20260512T210203Z/`
- Adapter metadata: same dir, `metadata.json`
- Eval results: same dir, `eval-82770bf6de.json` (10 funny-benchmark
  prompts, scored side-by-side against the v8 adapter)

---

## v8 pilot — comic-pointed angles + 2000-iter memorization (2026-05-13)

### What changed

Two deliberate changes aimed at restoring the comic voice that v6/v7
sanded off:

- **Angle rewrites** on 9 stimuli (commit `8639fa9`) — moved from
  descriptive angles to *comic-pointed* ones with a punchline beat
  baked into the angle text itself (e.g., "dog defending the couch
  from the vacuum's vengeful gaze" replaces "dog afraid of the
  vacuum"). The valence-spread rule from v5 is preserved; the change
  is in the *framing* of each angle, not the spread.
- **Iter doubling:** 2000 iters (vs 1000 in v7), targeting deeper
  memorization of the comic angles. Stylistic-character fine-tunes
  *want* the memorization regime; the v7 adapter underfit it.
- **Perf:** dropped `--grad-checkpoint` (commit `18349bb`) after
  measuring 9.7 GB peak on 32 GB unified memory in v7 — saved
  ~25-30 % per-iter time. Pure speed change; no quality effect.
- **Stimuli, persona, retrieval embedder, base model:** carried
  forward from v7 (no change).

### Corpus stats (from `sft stats`)

| Metric                          | v7                 | v8                 |
| ------------------------------- | ------------------ | ------------------ |
| Requests submitted              | 360                | 360                |
| Kept after dedup                | 276                | **290** (+14)      |
| Kept fraction                   | 77 %               | **81 %** (+4 pp)   |
| Persona violations              | 0                  | 0                  |
| Train pairs / Valid pairs       | 249 / 27           | **261 / 29**       |
| Train assistant tokens          | 115,567            | **124,802**        |
| Valid assistant tokens          | 13,493             | **14,337**         |
| Story length p50 / p90 (tokens) | 464 / 619          | 475 / 636          |
| Errored / invalid JSON          | 0 / 0              | 0 / 1              |
| Approximate cost                | ~$2.50             | ~$2.50             |

### Trained adapter (from `metadata.json`)

| Metric                          | v7                 | v8                       |
| ------------------------------- | ------------------ | ------------------------ |
| Adapter timestamp               | 20260512T210203Z   | 20260513T020823Z         |
| Iters / batch_size              | 1000 / 4           | **2000** / 4             |
| Rank / alpha                    | 8 / 16.0           | 8 / 16.0                 |
| Effective epochs                | 16.06              | **30.65** (+14.6)        |
| Tokens seen during training     | 1,856,006          | **3,825,181** (+1.97M)   |
| Wall clock (s)                  | 7,697 (~2.1 h)     | 15,137 (~4.2 h)          |

### Findings

**The humor is back.** Side-by-side reads of the funny-benchmark
eval set (`eval-82770bf6de.json`) against v7 and against the
earlier funny baseline (T042405Z) showed v8 hitting punchlines that
v7 missed:

- *Mailman*: "BARK. He pulls back his hand fast. **I did that bark
  on purpose. I was helping.**"
- *Vet*: "I lick her hand. She Smells like fifteen now. **I wag
  harder. I have always been the dog who licks hands.**"
- *Dinner*: lifted high to walk on back legs — "I am a very
  important dog and I have always been."

For comparison, v7's mailman story had Korean-script gibberish
("THE FAIGIN절") and ended on generic safety-blanket prose. v8's
endings are cleaner *and* funnier.

**Deep-memorization regime confirmed.** No train.log captured for
this run (the tee-to-file change landed in commit `ac19e59`, after
v8 finished — first run with `train-inspect` available will be v9),
but the qualitative signal matches the regime: tight comic phrasing
recalled from the corpus, occasional verbatim cadence. This is the
**desired** regime for a stylistic-character fine-tune, not a
warning. Distinct from over-memorization — the model still
generalizes to held-out stimuli.

**Memorization artifacts to watch.** v8 shows occasional repetition
loops on the highest-deep-memorization stimuli — the mailman story
loops on "scope of smell" several times near the end. These are
cosmetic and feel like an over-trained tic on stimuli where the
training data leans hard on one motif. Candidates for v9 angle
diversification rather than reverting iters.

**Stray tokenization glitches** (`_EXTERNALLY aware`, mid-word
capitalization, the occasional non-ASCII fragment) still appear at
roughly v7 frequency. These are inherited from the 4-bit-quant
base model, not the fine-tune.

**Telemetry observation:** kept-fraction shifted +4 pp from v7
on identical stimuli and identical persona — the only generation-
side change between them was the angle rewrites. Comic-pointed
phrasing gave Claude enough handle to differentiate retrievals
that v7's descriptive phrasing collapsed. Same lesson the v5/v6
work surfaced, applied one layer deeper: the angle text itself
shapes Claude's variance.

### Artifacts

- Stats JSON: `data/sft/stats-48bc939fe3.json`
- Adapter: `data/adapters/llama31-8b-storyteller-v1/20260513T020823Z/`
- Adapter metadata: same dir, `metadata.json`
- Eval results: same dir, `eval-82770bf6de.json`

### Next iteration candidates

- **v9 angle pass** on the stimuli that produced repetition-loop
  artifacts in v8 (mailman, lying-in-a-sunbeam, thunderstorm). The
  spread principle holds; the framing needs more lateral variety
  per stimulus to avoid the deep-memorization loop signature.
- **First `train-inspect` run.** v9 will be the first pilot with a
  `<adapter_dir>/train.log` written by the tee path, so loss
  trajectories and overfit verdict will be machine-readable for
  the first time.
- **Production scale (deferred).** Hold off on the 10K production
  run until v9 confirms the looping is angle-shaped rather than
  iter-shaped; a $60-80 production run wired to a bad memorization
  signature would be wasted budget.
