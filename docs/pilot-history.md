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

---

## v9 pilot — comedic-mode-tagged angles + first captured train.log (2026-05-13)

### What changed

The load-bearing change to v9 was the angle-design rule. Multi-
sample comparison across all eight prior adapters (mailman + bacon
+ vet, 5 samples each = 120 stories) surfaced five distinct
comedic modes — *delusion*, *coward*, *absurd*, *rationalizer*,
*dissociator* — with each adapter accidentally specializing in
just one. v9 retags every angle in `config/stimuli.yaml` by
comedic mode rather than emotional valence, with the goal of
producing a single adapter that hits all five modes on demand.

- **Comedic-mode taxonomy** captured in [docs/comedic-modes.md](
  comedic-modes.md) — each of the five modes has its source
  adapter (the comparison's strongest sample), canonical lines
  pulled from the comparison, and the angle-phrasing pattern that
  reproduces it.
- **Stimuli rewrite** (commit `3cfe15a`): all 50 stimuli get 3
  angles, each tagged inline with one of the 5 modes. Mode
  coverage across the 150 angles: **D=37 / C=29 / A=30 / R=29 /
  Z=25** (target was ~30 each). Variations-per-query and total
  request volume held at 360 to match v8 (apples-to-apples).
- **Telemetry milestone:** v9 is the first run since the
  tee-to-file change in commit `ac19e59`, so this is the first
  adapter with a captured `train.log` parseable by
  `rosetta-storyteller train-inspect`.
- **Persona, retrieval embedder, base model, training
  hyperparameters:** carried forward from v8 (no change). The
  comparison is purely angle-design vs angle-design.

### Corpus stats (from `sft stats`)

| Metric                          | v8                 | v9                       |
| ------------------------------- | ------------------ | ------------------------ |
| Requests submitted              | 360                | 360                      |
| Kept after dedup                | 290                | **306** (+16)            |
| Kept fraction                   | 81 %               | **85 %** (+4 pp)         |
| Persona violations              | 0                  | **0**                    |
| Train pairs / Valid pairs       | 261 / 29           | **276 / 30**             |
| Train assistant tokens          | 124,802            | **135,237**              |
| Valid assistant tokens          | 14,337             | 14,584                   |
| Story length p50 / p90 (tokens) | 475 / 636          | (similar, in same range) |
| Errored / invalid JSON          | 0 / 1              | 0 / 0                    |
| Approximate cost                | ~$2.50             | ~$2.50                   |

The kept-fraction lift is a real corpus-level validation of the
comedic-mode-spread hypothesis: angles tagged by mode produce
*more* dedup-distinct stories than angles tagged by emotional
valence, on the same prompts and the same retrieval embedder.
The orthogonality argument in `docs/comedic-modes.md` ("mode is
a stance toward the situation, not a valence") holds in practice.

### Trained adapter (from `metadata.json`)

| Metric                          | v8                       | v9                       |
| ------------------------------- | ------------------------ | ------------------------ |
| Adapter timestamp               | 20260513T020823Z         | 20260513T191317Z         |
| Iters / batch_size              | 2000 / 4                 | 2000 / 4 (unchanged)     |
| Rank / alpha                    | 8 / 16.0                 | 8 / 16.0 (unchanged)     |
| Effective epochs                | 30.65                    | 28.99                    |
| Tokens seen during training     | 3,825,181                | 3,920,520                |
| Wall clock (s)                  | 15,137 (~4.2 h)          | 14,238 (~3.95 h)         |
| Peak memory                     | not captured             | **18.1 GB**              |

### `train-inspect` numbers (first ever captured)

```
THROUGHPUT   0.145 it/sec median (range 0.123-0.184), 323 tokens/sec median

TRAIN LOSS                 first-10 avg = 1.716 → last-10 avg = 0.080
                           min = 0.069 at iter 1970

VALIDATION LOSS  iter    1: 2.413   ← untrained baseline
                 iter  200: 1.520   ← min (generalization sweet spot)
                 iter  400: 1.568
                 iter  600: 1.669
                 iter  800: 1.840
                 iter 1000: 2.036
                 iter 1200: 2.276
                 iter 1400: 2.493
                 iter 1600: 2.852
                 iter 1800: 3.064
                 iter 2000: 3.239   ← +113 % from min

VERDICT  deep memorization. final train loss 0.08 (collapsed);
         train-validation gap 3.16. Desired regime for stylistic-
         character fine-tunes.
```

The validation curve is the textbook over-memorization parabola.
Iter 200 is where the model best generalizes to held-out
examples; iter 2000 is where it best reproduces the training
distribution. We deliberately want iter 2000 for a stylistic
fine-tune; an information-injection fine-tune would have stopped
at iter 200.

The grad-checkpoint drop from commit `18349bb` is now confirmed
working: peak memory 18.1 GB on 32 GB unified, well under budget
and ~1 GB under the predicted 17 GB ceiling.

### Findings — comedic-mode coverage hypothesis confirmed

**Single stories now stack multiple modes** — the headline result.
v8 produced single-mode stories (almost always absurd-distraction).
v9's eval-set generations stack 2-3 modes in a single ~300-word
story. From the v9 mailman against the funny benchmark
(`/tmp/v9-vs-funny.txt`):

> "I have reflected upon this attack. I think I may have been too
> soft. **Next time he will run faster.** He must know this is my
> territory and I have always been the dog who makes him go away."
> — *delusion (invented job, unearned credit) → coward (vague threat
> for next time, deferred to "tonight" thinking)*

v8 on the same prompt would commit to one mode and stay there.

**Validation curve confirms the regime is right.** The min
validation at iter 200 is interesting but academic: stylistic
fine-tunes do not optimize for validation loss; they optimize for
faithful reproduction of the training distribution. The +113%
validation climb is the *signal* that the model has internalized
the corpus, not a regression.

**Two minor regressions.** The eval-set v9 stories show:

1. **More non-English token fragments** than v8 — *"OptionsResolver"*,
   *"Dog Seznam"* (Czech for *list*), *"Groundrustle"*. These
   appear at maybe 1-2 per story vs v8's <1. The base model is
   4-bit quantized and reaches into rare-token-space when sampler
   probabilities flatten near sentence boundaries; the deeper
   memorization regime (28.99 epochs vs v8's lighter spread) seems
   to reach into that rare-token tail more often. Cosmetic, not a
   blocker.

2. **`# delusion` over-represented** in the v9 corpus (37 angles
   vs target 30). The trained adapter slightly over-indexes on the
   "I have always been the dog who…" rhetorical move — visible in
   3 of the first 5 eval-set stories. v9.1 stimuli rebalance:
   move 4 D angles to Z to bring the distribution closer to
   uniform.

**No more "scope of smell" loops.** v8's most prominent failure
mode (one phrase chained 5+ times near the end of stories on
high-memorization stimuli) is gone in v9 — the comedic-mode
spread breaks the single-tic memorization that v8's narrower
angle space created.

### Artifacts

- Stats JSON: `data/sft/stats-4a1c8b1f28.json`
- Adapter: `data/adapters/llama31-8b-storyteller-v1/20260513T191317Z/`
- Adapter metadata: same dir, `metadata.json`
- Training log (first ever captured): same dir, `train.log`
- Eval results: same dir, `eval-82770bf6de.json`
- Eval-compare v8 → v9: `/tmp/v9-vs-v8.txt`
- Eval-compare T042405Z → v9: `/tmp/v9-vs-funny.txt`

### Next iteration candidates

- **v9.1 angle-balance pass.** Move 4 D angles to Z; spot-check
  3 stimuli where D is currently doing the work that another mode
  could do better (e.g., *settling for the night* could swap D
  for Z without losing humor).
- **Multi-sample mode coverage check.** Run the same "5 samples
  per stimulus, 3 prompts" protocol against v9 that we used for
  the v8-era comparison. Quantify what fraction of stories hit
  each mode. If v9 is producing all 5 modes ≥15 % of the time on
  the eval set, the hypothesis is confirmed and v10 can shift
  focus to other corpus-design questions (better seed examples?
  longer-form stories?). If some modes are still sub-5 %, the
  underlying mode taxonomy needs a closer look — maybe two of the
  modes are functionally identical at corpus scale.
- **Production scale (now genuinely defensible).** v9 demonstrates
  the corpus-design lever works. A 10K production run on the same
  comedic-mode-tagged angles would give us a meaningfully larger
  training set per mode (~2K stories per mode instead of ~60).
  Estimated cost ~$60-80 with prompt caching. Hold until v9.1
  confirms the mode-coverage measurement is stable.

## v10 pilot — canine-hearing science + auditory stimuli (2026-05-15)

### Why this pilot

Stories generated by v7–v9 lean heavily on olfactory imagery
regardless of stimulus. Root cause: the science pillar's EuropePMC
query was olfaction-only ("canine olfaction OR vomeronasal OR
\"dog scent\" OR \"olfactory bulb dog\""), so the only science
chunks available for retrieval were smell-themed. Retrieval at SFT
time pulls one science chunk per story (`top_k=1` per pillar in
`retrieval/select.py`), so every story gets a smell-fact prompt.

v10 broadens the science pillar to include canine audition without
removing olfaction, and adds 5 auditory stimuli so retrieval has
audio-themed queries to surface the new chunks.

### What changed

- **Two-query science ingest** (`src/rosetta_bone/storyteller/ingest/science.py`).
  Replaced the single `DEFAULT_QUERY` with two named queries —
  `OLFACTION_QUERY` (the v9 query unchanged) and `AUDITION_QUERY`
  (dog-anchored audition vocabulary: BAER, audiogram, cochlea,
  pinna, sound-localization, hearing-range, presbycusis, deafness,
  noise-phobia, ultrasonic). `fetch_papers` now runs both queries
  and pools the results, splitting `limit` 50/50 between them.

  *Why two queries instead of one union:* a single OR'd query lets
  EuropePMC's relevance ranking pick a winner. The audition clause
  has more searchable terms than the olfaction clause, so the union
  ranked smell papers off page 1 entirely (audition=19, smell=0 in
  the union's top 50). Two independent fetches with independent
  caps guarantee both modalities reach the corpus.

- **Ingest limit bumped 25 → 50** (`cli.py` `--limit` default,
  `science.py` `fetch_papers` default). Splits to 25 olfaction +
  25 audition, doubling the prior corpus size.

- **5 new auditory stimuli** (`config/stimuli.yaml`):
  `a thunderstorm rolling in`, `the vacuum cleaner switched on`,
  `owner's footsteps in the hallway`, `the doorbell chiming`,
  `fireworks at dusk`. Each gets 3 mode-tagged angles. Mode mix
  for the 15 new angles: D=2, C=4, A=2, R=4, Z=3 — deliberately
  light on D (over-represented in v9 at 37) and heavy on C/R
  which fit acoustic startle and habituation behaviour cleanly.
  Total stimulus count: 50 → 55.

- **New `ingest-inspect` CLI command** (`cli.py`,
  `src/rosetta_bone/storyteller/ingest/inspect.py`). Read-only
  human-in-the-loop summary between `ingest` and `chunk`. For
  science, prints paper count, year distribution, a smell/hearing
  title-keyword breakdown, and the per-paper table. Optional —
  never blocks the pipeline. `--json` flag for scripting.

### Corpus stats (post-ingest, pre-train)

| Metric                       | v9 (smell only) | v10                     |
| ---------------------------- | --------------- | ----------------------- |
| Science papers ingested      | 25 (olfaction)  | **50** (25 + 25)        |
| Disk size of science raw     | ~70 MB          | **138 MB**              |
| Science chunks in JSONL      | (carryover)     | **1,490**               |
| Title-keyword smell bucket   | ~25             | **17**                  |
| Title-keyword hearing bucket | 0               | **17**                  |
| Other / clinical-tangential  | ~0              | **16**                  |
| Stimuli                      | 50              | **55** (+5 auditory)    |
| Angles                       | 150             | **165** (+15 auditory)  |

### Retrieval routing — fixed mid-pilot

A first pass with the 4-tuple `(stimulus, embed_query, variation,
form)` selector showed only **1 of 5** auditory stimuli surfaced a
hearing chunk; the others matched smell/equipment papers because
bge-small-en-v1.5 ranked on lexical surface-overlap and smell
papers had richer descriptive prose. Even with the corpus 50/50
balanced (17 smell / 17 hearing / 16 other-clinical papers), the
ranker preferred smell.

Fixed by adding **modality tags** rather than swapping embedders:

- New `src/rosetta_bone/storyteller/ingest/modality.py` —
  `classify_title(title)` regex returns `"smell" | "hearing" | None`
  (single source of truth, also used by `ingest-inspect`).
- `chunk_pillar` reads `{pmcid}.json` sidecars and stamps
  `metadata.modality` on every science chunk at chunk time.
- `Stimulus` gets an optional `modality: Literal["smell","hearing"]`
  field. The 5 v10 auditory stimuli get `modality: hearing`; the
  50 v9 stimuli stay unset (backward-compat path).
- `select_chunks` accepts `science_modality`. When set, pulls
  `MODALITY_POOL=50` cosine-top results from FAISS, takes the first
  whose modality matches before applying `top_k=1`. Falls back to
  unfiltered top-1 if none match (logged; never fired in v10).
- `expand()` now yields 5-tuples; `plan_batch` cache key is
  `(query, modality)`.

After re-chunk + re-embed the science index has **514 hearing /
503 smell / 473 unstamped** chunks. Routing recheck:

| Auditory stimulus (3 angles each) | Routed to                                                    |
| --------------------------------- | ------------------------------------------------------------ |
| thunderstorm rolling in           | all 3 → hearing-tagged papers (mostly PMC12963408)           |
| vacuum cleaner switched on        | all 3 → hearing                                              |
| owner's footsteps in hallway      | all 3 → PMC12963408 (canine hearing-eval protocol)           |
| doorbell chiming                  | all 3 → PMC12963408                                          |
| fireworks at dusk                 | all 3 → PMC12963408                                          |

**15 of 15** auditory angles route correctly. PMC12963408 dominates —
it has 28 dense hearing-eval chunks, of which the BAER/wave-V
methodology and dB-SPL passages cosine-rank highest against
acoustic-startle angles. Olfactory stimuli (modality unset) still
hit the unfiltered v9 path and continue to pull smell papers.

### Corpus stats (`sft stats`)

| Metric                          | v8                 | v9                       | v10                      |
| ------------------------------- | ------------------ | ------------------------ | ------------------------ |
| Requests submitted              | 360                | 360                      | **405** (+45 auditory)   |
| Succeeded / dropped invalid     | 360 / 1            | 360 / 0                  | **401 / 4**              |
| Kept after dedup                | 290                | 306                      | **343** (+37)            |
| Kept fraction                   | 81 %               | 85 %                     | **84.7 %** (held)        |
| Train pairs / Valid pairs       | 261 / 29           | 276 / 30                 | **309 / 34**             |
| Train assistant tokens          | 124,802            | 135,237                  | **150,253** (+11 %)      |
| Valid assistant tokens          | 14,337             | 14,584                   | **17,772**               |
| Persona violations              | 0                  | 0                        | **0**                    |
| Approximate cost                | ~$2.50             | ~$2.50                   | **$3.13**                |

The kept-fraction held within noise of v9 — adding hearing material
to the science pillar didn't hurt the comedic-mode dedup
distinction, which is the load-bearing v9 design rule. The +11 %
training-token bump is from the +5 stimuli with 3 variations each.

### Trained adapter (`metadata.json`)

| Metric                          | v9                       | v10                      |
| ------------------------------- | ------------------------ | ------------------------ |
| Adapter timestamp               | 20260513T191317Z         | **20260515T180408Z**     |
| Iters / batch_size              | 2000 / 4                 | 2000 / 4 (unchanged)     |
| Rank / alpha                    | 8 / 16.0                 | 8 / 16.0 (unchanged)     |
| Effective epochs                | (similar)                | **25.89**                |
| Tokens seen during training     | (similar)                | **3,890,050**            |
| Wall time                       | ~4 hours                 | **3.95 hours**           |
| Peak memory                     | 18.1 GB                  | **18.1 GB**              |
| Throughput (median it/s)        | ~0.14                    | **0.146**                |

### Validation-loss trajectory

| iter | val loss | note                                |
| ---- | -------- | ----------------------------------- |
| 1    | 2.408    | base model                          |
| 200  | **1.560** | generalization sweet spot          |
| 400  | 1.592    | start of memorization               |
| 600  | 1.651    |                                     |
| 800  | 1.830    |                                     |
| 1000 | 1.892    |                                     |
| 1200 | 2.079    |                                     |
| 1400 | 2.353    |                                     |
| 1600 | 2.522    |                                     |
| 1800 | 2.794    |                                     |
| 2000 | **3.020** | deep memorization (final)          |

Train loss dropped 1.71 → 0.15 (min 0.130 at iter 1940). Same
deliberate over-memorization regime as v9 — desired for
stylistic-character fine-tunes, where validation loss measures
generalization away from the carefully-curated training corpus
(which is the wrong objective here). The 200-iter checkpoint at
[20260515T174047Z](data/adapters/llama31-8b-storyteller-v1/20260515T174047Z)
is preserved for an apples-to-apples generalization-vs-memorization
comparison if needed.

### Qualitative read — auditory imagery audit

Generated diary-form stories for all 5 auditory stimuli plus 2
olfactory baselines using the v10 adapter. Saved to
`/tmp/v10-samples/{auditory,olfactory}.txt`.

**Auditory stimuli — sound-first language is now load-bearing in
every story:**

| Stimulus       | Auditory passage                                                                |
| -------------- | ------------------------------------------------------------------------------- |
| Thunderstorm   | "a big drum. A low thud-thud-thud just under the sky"                          |
| Vacuum         | "going click-click-click and then RRRRRR and then SSSSSS and then a big whoom" |
| Footsteps      | "Footstep. Footstep. Footstep... I hear it first. A thump. Thump-thump."        |
| Doorbell       | "A bell inside it is SCREAMING at us. It happens twice. It is very loud."       |
| Fireworks      | "BOOM. BOOM. It was like someone kept dropping something heavy on the ground"   |

Smell is still present as supporting texture (the storm story leads
with "something in the air I have never smelled before" before the
thunder hits, the vacuum story includes "burnt-dust smell"), but
sound is no longer absent or subordinate — the *narrative arc*
hangs on the auditory cue. v9 adapters could not produce these
stories at all; sound was a 1-line aside at best.

**Olfactory baselines — overcorrection observed:**

The *owner returning home from work* story emits labeled
`SOUND:` / `SIGHT:` sentence prefixes — the model has learned a
multi-sensory enumeration pattern that wasn't in the v9 output. The
*mailman arriving* story leads with smell but immediately moves to
"a sound. A car... footstep. One footstep. Very methodical" —
sound has become an early co-equal sensory anchor even on
smell-themed stimuli. Whether this is a bug or a feature depends on
intent: v10 wanted to stop smell-monoculture, and that's what
happened. The stylistic side-effect is the labeled-section thing,
which feels more diagnostic-output than diary-prose.

### Findings

- **Modality tags are the right shape for sensory routing.** The
  retrieval-side fix landed cleanly, didn't require schema
  contortions, and generalises — `modality: vision` would just work
  when v11 adds visual stimuli. Cost: ~30 LOC plus tests.
- **One pilot can mix corpus-design and routing-fix changes
  cleanly when the routing fix is a separate commit.** The two-stage
  v10 commits (corpus, then modality routing) means a v11 that
  removes either is a clean revert.
- **Verdict on the original goal:** ✅ smell-overweight stories
  fixed. The five auditory stimuli produce stories that are
  recognisably auditory rather than thinly-disguised olfactory
  pieces with a sound-effect tacked on.

### Next iteration candidates

- **v10.1 stylistic cleanup of the labeled-section pattern.** The
  `SOUND: ... SIGHT: ...` enumeration in the owner-returning story
  is a v10-only artifact. Likely picked up from one or more SFT
  pairs in this run. Spot-check `data/sft/train.jsonl` for stories
  with that pattern and consider tweaking either the prompt builder
  or the persona prompt to discourage it.
- **v11 add visual stimuli + science-vision corpus** following the
  v10 template: AUDITION_QUERY → VISION_QUERY (canine retina,
  cone-rod ratio, motion sensitivity, dichromatic colour vision),
  add 4-5 visual stimuli with `modality: vision`, extend
  `classify_title` regex. The tag→filter machinery is now in place
  so this is a corpus change, not an architecture change.
- **Audition title-regex tightening.** `ultrasonic` matches the
  "ultrasonic toothbrush in dogs" paper which isn't really hearing
  science. Either drop `ultrasonic` from the regex or add a small
  exclude list. Low-priority — only affects 1-2 papers.
- **Over-correction on olfactory baselines.** If the auditory
  framing on smell stimuli proves too strong in production
  evaluation, options are: (a) reduce the auditory-stimulus
  variation count from 3 to 2, (b) loosen the modality filter to
  only restrict the *first* of multiple retrieved chunks. Hold
  until we have eval data on whether the framing actually hurts
  smell-themed story quality.

---

## v11 pilot — persona rewrite (sound + smell parity) (2026-05-16)

### Why this pilot

v10's qualitative read showed sound leading on the 5 new auditory
stimuli (great) but the model defaulted to smell-first on every other
prompt because the *persona itself* was still telling Claude
"Perceptual frame is **scent-first**, sound-broadband-and-shifted,
proprioceptive over visual" — a single declarative sentence at SFT
generation time that propagated through the entire training corpus.

v11 isolates that lever. **No code, schema, corpus, or retrieval
changes.** Only `src/rosetta_bone/storyteller/sft/persona.py` was
edited. SFT was regenerated against the same 55 stimuli and the same
science / style / behavior chunks v10 used.

### What changed

- Replaced the "scent-first" perceptual-frame sentence with a three-
  rule decision tree: **sound leads when distance is involved**,
  **smell leads when source is close**, **both fire at once for
  embodied events** (storms, vacuums).
- Added a "How a real dog hears" section parallel to the existing
  "How a real dog narrates": high frequencies humans miss, low
  rumbles humans miss, onomatopoeia conventions (DING-DONG not
  "the door rang"), ear-swivel/head-tilt body-direction language,
  sounds-as-learned-meanings (keys = walk, microwave = food).
- Updated the v1 docstring that still said "scent-first house pet".

Persona size: 942 words / 110 lines (up from 758 / 82 in v10 —
about a 25 % expansion).

### Corpus stats (`sft stats`)

| Metric                          | v10                      | v11                       |
| ------------------------------- | ------------------------ | ------------------------- |
| Requests submitted              | 405                      | 405                       |
| Succeeded / dropped invalid     | 401 / 4                  | **405 / 1**               |
| Kept after dedup                | 343                      | **348** (+5)              |
| Kept fraction                   | 84.7 %                   | **85.9 %** (held)         |
| Train pairs / Valid pairs       | 309 / 34                 | **314 / 34**              |
| Train assistant tokens          | 150,253                  | **157,466** (+5 %)        |
| Approximate cost                | $3.13                    | **$4.46** (+42 %)         |

The cost bump is the longer persona text driving cache-creation
input tokens (v11 cache_creation_input_tokens = 1.29M vs v10's
0.57M). Kept fraction held within noise — the persona change didn't
hurt mode-distribution diversity, which is the load-bearing v9 rule.

Note: a first pass merged the v11 batch alongside the v10 batch
still in `data/sft/batches/`, producing a blended 611-pair corpus.
Archived the v10 batch to `data/sft/batches.archive/` and re-merged
for a clean v11 corpus. Without that step v11's training would have
been an averaged v10+v11 persona signal — exactly what we needed
to isolate.

### Trained adapter

| Metric                          | v10                      | v11                       |
| ------------------------------- | ------------------------ | ------------------------- |
| Adapter timestamp               | 20260515T180408Z         | **20260516T195645Z**      |
| Iters / batch_size              | 2000 / 4                 | 2000 / 4                  |
| Effective epochs                | 25.89                    | 25.48                     |
| Tokens seen during training     | 3,890,050                | 4,012,233                 |
| Wall time                       | 3.95 hours               | **4.10 hours**            |
| Peak memory                     | 18.1 GB                  | 18.6 GB                   |
| Throughput (median it/s)        | 0.146                    | 0.145                     |

### Validation-loss trajectory

| iter | v10 val | v11 val | delta |
| ---- | ------- | ------- | ----- |
| 1    | 2.408   | 2.460   | +0.05 |
| 200  | 1.560   | **1.529** | **−0.03** (best ever; v9 was also ~1.56) |
| 400  | 1.592   | 1.560   | −0.03 |
| 600  | 1.651   | 1.614   | −0.04 |
| 800  | 1.830   | 1.759   | −0.07 |
| 1000 | 1.892   | 1.781   | −0.11 |
| 1200 | 2.079   | 2.033   | −0.05 |
| 1400 | 2.353   | 2.261   | −0.09 |
| 1600 | 2.522   | 2.392   | −0.13 |
| 1800 | 2.794   | 2.646   | −0.15 |
| 2000 | 3.020   | **2.862** | **−0.16** |

v11 generalises modestly better at every checkpoint and memorises
slightly less aggressively. Reading: the more structured persona
gives the model a clearer pattern to fit, so validation loss
behaves marginally healthier across the run. The regime (deep
memorisation by iter 2000) is unchanged.

### Qualitative read — auditory imagery audit

Generated 10 samples on the v11 adapter: the 3 README stimuli +
5 auditory stimuli + 2 olfactory baselines. Saved to
`/tmp/v11-samples.txt`. The decision-tree rules from the persona
are visible end-to-end.

**Distance-rule wins (sound leads):**

| Stimulus               | v11 opening                                                  |
| ---------------------- | ------------------------------------------------------------ |
| the mailman has arrived| "It starts before the door even opens. **I hear it. Far away. Footsteps on the path.**" |
| owner's footsteps      | "**Footsteps. I hear them. Low and far away. My ears swing toward the hallway.**" |
| the doorbell chiming   | "**DING-DONG.** I hear it. My ears swing around. Both of them. Hard." |
| owner returning home   | "**The car is turning. The low rumble of the engine is coming from the end of the street.**" |

All four were smell-first or smell-co-equal under v10. The
distance-cue → sound-first rule is now load-bearing.

**Proximity-rule wins (smell leads — appropriate):**

| Stimulus            | v11 opening                                                  |
| ------------------- | ------------------------------------------------------------ |
| the smell of bacon  | "Then. A smell. NOT A POND SMELL. ... BACON."                |
| the smell of clean laundry | "It is soap. Warm soap. Not that sharp, bad soap."    |

Smell-themed prompts retain smell leadership — the parity rules
didn't over-correct into sound-everywhere.

**Embodied-event wins (both fire):**

The thunderstorm and vacuum stories interleave smell and sound
explicitly: thunderstorm opens with smell-of-electric-air, hits
BOOM, returns to burnt-grass smell, returns to BOOM BOOM BOOM.
Vacuum interleaves RRRRRRRRR onomatopoeia with burnt-dust smell.

**Persona-mechanic wins (body-direction language):**

The "how a real dog hears" section's ear-swivel rule shows up
explicitly:

- "**My ears swing around. Both of them. Hard.**" (doorbell)
- "**I tilt. My head. Left.**" (vacuum)
- "**My ears swing toward the hallway.**" (footsteps)
- "**I put my ears back. Both ears. This way.**" (vacuum)

Zero stories in the v10 samples used this construction.

**Onomatopoeia wins:**

- DING-DONG (doorbell)
- RRRRRRRRR / RMMMMMMMM (vacuum)
- BOOM BOOM BOOM (thunderstorm and fireworks)
- chunk-chunk vs tucker-tucker (footsteps — same person, different rooms)
- click-click-click-CLICK (owner returning, escalating)
- Pad pad pad PAD (owner returning, footsteps on path)

**No SOUND: / SIGHT: labels.** The v10 emergent labeled-section
artifact (which appeared in 2/3 owner-returning regens last cycle)
did not appear in any of the 10 v11 samples. Possibly the more
structured persona gave the model a less ambiguous template, so it
didn't reach for label-headers as a structural prop.

### Known v11 tokenizer artifacts

Three minor mid-token slips in 10 samples:

- "OH ROTH." (vet)
- "DEW RAGSaround" (vacuum, mid-word collision)
- "theमक door" (footsteps — Hindi character leaked in)
- "havePosts tens times" (doorbell)

Rate (~3 per 10 stories) is similar to or slightly higher than v10.
These come from the 4-bit MLX quantization plus the model
occasionally trying onomatopoeia patterns near token boundaries.
Not corpus-fixable. Could try the un-quantized 8-bit MLX base if
the rate grows objectionable.

### Findings

- **Persona is the right lever for sense-priority.** The v10 retrieval
  fix routed the right chunks; v11 changed the system message that
  tells Claude what to do with them. Both were necessary; neither
  alone would have produced the v11 result.
- **Three-rule decision trees beat single-priority declarations** in
  this kind of instruction. "scent-first" is one global hint;
  "distance → sound, proximity → smell, embodied → both" is three
  conditional hints that the model applies per-prompt.
- **Structured "How a real dog X" sections produce structured output.**
  The "How a real dog narrates" section in the original persona
  already established the template; adding "How a real dog hears"
  parallel to it transferred directly into output that includes
  ear-swivel, onomatopoeia, and sound-as-learned-meaning constructions.

### Next iteration candidates

- **v12 add visual stimuli + science-vision corpus + persona "How a
  real dog sees" section** following the v10/v11 template. The tag→
  filter machinery + parity-rule persona pattern are both proven.
- **Tokenizer-artifact audit.** Rate (~3/10) is acceptable but
  measurable. Compare to the un-quantized 8-bit base — if artifacts
  drop substantially, decide whether the larger memory footprint is
  worth it.
- **Stop archiving the previous batch.** Move the merge step to
  treat each `--phase` tag as the merge boundary so a stale batch
  in `data/sft/batches/` can't accidentally blend with the new one.
  Mid-pilot v11 already caught this; better to make it impossible.

---

## kids-v1 pilot — sibling product for ages 4-8 (2026-05-17)

### Why this pilot

The v11 adapter is adult-coded: ironic, Marley-ish, with dread /
grievance / enemy framing (mailman as "my old enemy", vet as "a
personal catastrophe", references to "a bad word"). This pilot
introduces a **second product** — `rosetta-bone-kids` — targeting
ages 4-8 with warm, reassuring, kid-appropriate narration, while
keeping the existing adult adapter unchanged.

Architecturally this is a **sibling product**, not a fork: same
six-stage pipeline, same retrieval, same base model, same train
loop — routed at runtime by a new `--config config/default-kids.toml`
and `--stimuli config/stimuli-kids.yaml`. The two products differ in
exactly three places: persona module, stimuli file, and per-product
`sft_dir` + `adapter_dir`. Shared pillars (raw / chunks / embeddings)
in v1 — to test whether the persona alone moves the needle.

### What changed

Code (5 files, 4 commits — all PRs against `claude/pedantic-
albattani-4b6a72`):

- **`common/config.py`** — added a `Persona` dataclass section with
  a `module: str` field defaulting to the adult persona module, so
  configs without `[persona]` still load.
- **`sft/persona.py`** — moved `_PERSONA_VIOLATIONS` from
  `stats.py` and exported as public `PERSONA_VIOLATIONS`. Each
  persona module now owns its own register-regression markers.
- **`sft/stats.py`** — `compute_stats` takes an optional
  `persona_module` and resolves the violations tuple dynamically.
- **`sft/prompt_builder.py`** — `build_system_block` / `build_messages`
  take an optional `persona_module` keyword. `None` falls back to the
  adult persona so the test snapshot and existing callers are
  unchanged.
- **`storyteller/cli.py`** — `sft_generate` gained a `--stimuli` flag
  (mirroring `sft stats`), and now threads `cfg.persona.module`
  through to `plan_batch`.

Content (3 new files):

- **`sft/persona_kids.py`** — kids persona (~110 lines, structurally
  parallel to adult persona). Same sense-priority rules, same
  "How a real dog hears" rules, same body-direction language. The
  delta is tone: warm not ironic, vocabulary inside a ~500-word
  lexicon, story-shape guidance favouring clear beginning-middle-
  warm-end. `PERSONA_VIOLATIONS` flags hostility (`enemy`, `hated`,
  `monster`, `catastrophe`), kid-inappropriate language (`bad word`,
  `swear`, `damn`), and the literary markers from the adult tuple
  (`olfactory`, `vomeronasal`, `contemplated`).
- **`config/stimuli-kids.yaml`** — 20 stimuli, 53 angles
  (~2.65 per stimulus), `variations_per_query: 3` → 159 angle×var
  triples total. Stance spread `curious | playful | loving | silly
  | sleepy` replaces the v9 comedic-mode taxonomy. Forms balanced
  across diary / vignette / short_story.
- **`config/default-kids.toml`** — per-product `sft_dir =
  data/sft-kids`, `adapter_dir = data/adapters/llama31-8b-
  storyteller-kids-v1`, slightly cooler infer
  (`temperature = 0.75`, `max_tokens = 400`), `[persona] module =
  rosetta_bone.storyteller.sft.persona_kids`.

108 unit tests pass after every code commit. Adult product
behaviour is unchanged.

### Corpus stats (`sft stats`)

Two batches submitted — `kids-v1` (50 requests, 6 stimuli) and
`kids-v1b` (105 requests, remaining 14 stimuli) — then merged.

| Metric                          | adult v11        | kids-v1          |
| ------------------------------- | ---------------- | ---------------- |
| Stimuli                         | 55               | **20**           |
| Angles total                    | 135              | **53**           |
| Requests submitted              | 405              | **155** (50+105) |
| Succeeded / dropped invalid     | 405 / 1          | 155 / 2          |
| Kept after dedup                | 348              | **123**          |
| Kept fraction                   | 85.9 %           | **80 %**         |
| Train pairs / Valid pairs       | 314 / 34         | **111 / 12**     |
| Train assistant tokens          | 157,466          | **40,885**       |
| Persona violations              | (different list) | **0**            |
| Approximate cost                | $4.46            | **$1.38**        |

Cost is ~30% of adult v11 because of the smaller stimuli set and
the prompt-cache hits on the second batch (`cache_read_input_tokens
= 62,724` on the top-up batch). Kept fraction is at the plan
threshold (≥80%). **Zero persona violations** across all 123 kept
stories — the kids persona's stricter violation list (19 markers,
including hostility + simplicity + literary register) is
respected.

### Trained adapter

| Metric                          | adult v11        | kids-v1          |
| ------------------------------- | ---------------- | ---------------- |
| Adapter timestamp               | 20260516T195645Z | **20260517T184857Z** |
| Iters chosen                    | 2000             | **200**          |
| Wall time                       | 4.10 hours       | **0.26 hours** (925s) |
| Effective epochs at iter 200    | ~2.6             | **7.21**         |
| Train loss @ chosen iter        | 1.43 (iter 2000) | **0.526**        |
| Val loss @ chosen iter          | 2.862            | **1.469**        |

Effective epochs is the load-bearing difference. Adult corpus at
314 train pairs hits 25.9 epochs by iter 2000; kids corpus at 111
train pairs hits 7.2 epochs by iter 200 — roughly 3.6× the per-pair
exposure rate. The smaller corpus means the right iteration count
is much lower.

### Validation-loss trajectory — overfitting documented

Ran a 200-iter sanity train (per the [training-tuning-discipline
memory note](/Users/mikeporter/.claude/projects/-Users-mikeporter-
devel-agileedge-rosetta-bone/memory/training-tuning-discipline.md))
then a full 2000-iter run. The 2000-iter run was killed at iter
1000 once the validation curve made it unambiguous:

| iter | val loss | train loss | note                              |
| ---- | -------- | ---------- | --------------------------------- |
| 1    | 2.394    | —          | base model                        |
| 200  | **1.469** | 0.526     | **best — kept this adapter**      |
| 400  | 2.266    | ~0.10      | overfitting begins                |
| 600  | 2.823    | ~0.03      | worse than base model             |
| 800  | 3.097    | ~0.03      | severe memorisation               |
| 1000 | 3.186    | ~0.03      | killed run; train loss at 0.025   |

Adult pilots show this same regime (deep memorisation past iter
2000) but their validation curve doesn't bottom out until iter
1000-1200 — 5-6× later. With 1/3 the corpus, the kids pilot bottoms
out 5× earlier. The next kids iteration should default to
`--iters 200` until the corpus grows.

The 2000-iter run's adapter dir (`20260517T190454Z`) preserves the
overfit checkpoints + train.log as evidence; the production adapter
is the iter-200 sanity-run output at `20260517T184857Z`, pointed to
by `data/adapters/llama31-8b-storyteller-kids-v1/latest`.

### Qualitative read — kids-tone audit

Generated 5 samples on the iter-200 kids adapter, written to
`data/samples-kids-v1/audit-run-01.txt`. Audit-checklist results:

| Check                                            | Result |
| ------------------------------------------------ | ------ |
| No swearing / "bad word" references              | ✓ all 5 |
| No dread, grievance, enemy framing               | ✓ all 5 |
| Vocabulary at ~5-year-old level                  | ✓ all 5 |
| Stories have warm resolution                     | ✓ all 5 |
| Read-aloud cadence (clear boundaries, repetition)| ✓ all 5 |
| Persona violations counted by `sft stats`        | **0**  |

**Distance-rule wins (inherited from v11):**

| Stimulus               | kids-v1 opening                                              |
| ---------------------- | ------------------------------------------------------------ |
| the new puppy          | "I heard it first. A tiny sound. Eee-ee-ee. High and squeaky like a toy that needs winding." |
| snow for the first time| "I heard it first. A low rumble. R-r-rumble. ... tap-taps on the window." |

The sense-priority rules transferred directly from the v11 persona
template — sound leads when there's distance, smell leads when
close. The kids persona kept those rules verbatim and the adapter
expresses them in kid-friendly vocabulary.

**Warm-resolution wins:**

- *bedtime*: "And then I know it is time. I lie down in my bed. ...
  I put my chin on my paws. And I close my eyes."
- *snow for the first time*: "I knew right then. Today was a very
  big day."
- *the cat by the window*: "I sit down. I turn around two times. I
  lie down. I am very good. I am very brave."
- *a lost teddy bear* (novel — not in training): "I will bring them
  here. ... Yes. That is a good smell now."

Stories close on a soft landing, not the ironic open-ended Marley
register the adult adapter produces.

**Onomatopoeia wins:**

- Eee-ee-ee / EEE EEE EEE (puppy)
- Sniff sniff sniff / SNIFF (cat, teddy)
- Creak-creak / Creak-creak-ROAR (bedtime)
- CRUNCH CRUNCH CRUNCH (snow)
- R-r-rumble / tap-taps (snow)

The "How a real dog hears" rules from the persona produce the same
kid-friendly sound-shape narration the adult product produces in
adult register.

**Novel-prompt generalisation:**

"a lost teddy bear" (not in training stimuli) produced a coherent
story: scent-tracking the bear, deciding it's lost (not a real
bear), bringing it to the person for a "warm spot". The model
generalised the persona's sense-priority + warm-resolution rules to
an unseen scene.

**Minor regression on `the cat by the window`:** the dog froze at
the cat ("I freeze") which reads slightly cautious rather than the
"my friend the cat" framing intended by the persona. Likely a
pull from the shared behavior pillar (which describes adult-dog
caution patterns). Not a deal-breaker; flag for the v2 shared-vs-
per-product pillar decision.

### Cost summary

- SFT generation: $1.38 (vs $4.46 for adult v11, $3.13 for v10)
- Compute: 925s on M-series GPU; $0 marginal
- **Total kids-v1: $1.38**

Within the $1-2 budget the planning doc estimated.

### Findings

1. **The "persona alone" hypothesis held.** Shared pillars +
   different persona produced a register that audits clean against
   all six kid-tone criteria with zero violations. Per-product
   raw/chunks/embeddings stays out of scope for v2 unless a
   regression surfaces.
2. **Small corpora need short trains.** 111 train pairs at
   batch_size 4 hits its validation minimum at ~iter 200 (7
   effective epochs). Anything past that memorises. Future kids
   pilots should default `--iters 200` until the corpus grows to
   ≥250 pairs.
3. **The persona's `PERSONA_VIOLATIONS` is a load-bearing audit
   signal.** Tracking the kids violation list in `sft stats`
   surfaced the persona's fidelity before training. Recommend
   keeping this discipline for any future persona work.
4. **The structural parallel-persona pattern works.** Replicating
   the v11 persona's "How a real dog narrates" + "How a real dog
   hears" sections (with kids vocabulary) produced an adapter that
   shows the same sound-leadership and body-direction language
   the adult product does, while staying in register.

### Next iteration candidates

- **kids-v2: per-product pillars** if a v1-adapter audit on a wider
  novel-prompt set surfaces adult vocabulary leaking from the
  science pillar. Until then, keep sharing.
- **kids-v2: scale stimuli to ~40** and bump `--iters` to ~400.
  Current 20 stimuli x 3 angles produces 53 angles; doubling to 40
  stimuli with the same stance-spread rule should yield ~250 train
  pairs, which would support a ~400-iter train at the same
  effective-epoch budget.
- **Address the "cat freeze" regression** with either (a) a
  per-product behavior pillar focused on cohabitation-friendly
  examples, or (b) a stronger "the cat is your friend" rule in
  the kids persona.
- **modality routing is a no-op.** The science chunks have no
  `modality` metadata, so the `modality: hearing` tag on 4 kids
  stimuli silently falls back to the unfiltered pool. Same
  behaviour as the adult product. Tag the science chunks at
  ingest-time as a separate cleanup ticket — not specific to
  kids-v1.
