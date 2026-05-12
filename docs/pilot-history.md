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

## v1 pilot — baseline (pre-angle retrieval)

### What changed

The original schema: `variations: N` per stimulus. All N variations
of a stimulus saw identical persona + contract + retrieved chunks,
differing only by `Variation: {idx}` in the user block. The
strict-context contract forbade Claude from inventing, so variations
collapsed into near-identical stories.

- 20 stimuli, variations summing to 100 (clamped to 94 by the
  pipeline at the time)
- Persona: original "lighthearted pampered house pet" (literary register)

### Stats

| Metric                 | v1     |
| ---------------------- | ------ |
| Requests submitted     | 104    |
| Kept after dedup       | 57     |
| Kept fraction          | ~55 %  |
| Persona violations     | not tracked yet |
| Approximate cost       | ~$0.70 |

### Findings

~45 % of pairs dropped at merge due to dedup. The corpus was
small (57 train pairs) and stylistically uniform. Trained adapter
produced flowery, contemplative dog narration ("olfactory plumes",
"vessel without a bottom") — wrong register. Driver: the literary
persona text. Replaced in v3 (commit `8faf223` and `b5fc9e1`).

### Stats artifact

Not separately committed (pre-dates `sft stats` command). Recoverable
from `data/sft/manifest.jsonl` history if needed.

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
