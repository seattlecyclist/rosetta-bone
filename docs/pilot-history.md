# Pilot history

Snapshots of measured changes to the SFT corpus and resulting model
quality, in chronological order. Numbers here are from
`rosetta-storyteller sft stats` runs and are committed alongside
their underlying stats JSON in `data/sft/stats-<sha>.json`.

---

## v5 pilot — angle-aware retrieval + 50 stimuli (2026-05-12)

**Change:** introduced `embed_queries: [angle1, angle2, ...]` ×
`variations_per_query` schema (commit `c4226a5`) and expanded
`config/stimuli.yaml` from 20 → 50 stimuli (commit `c2417d4`).
Each angle independently queries the FAISS pillars AND is surfaced
to Claude as an `Angle:` hint in the user block, so different
angles for one stimulus produce different stories.

**Stats file:** `data/sft/stats-281896dfd6.json`

| Metric                          | v1 pilot (no angles) | v5 pilot                  |
| ------------------------------- | -------------------- | ------------------------- |
| Requests submitted              | 104                  | 360                       |
| Kept after dedup                | 57                   | **269**                   |
| Kept fraction                   | ~55 %                | **75 %**                  |
| Persona violations              | not tracked          | **0** across 269 stories  |
| Story length p50 / p90 (tokens) | n/a                  | 454 / 610                 |
| Errored / invalid-JSON          | 0 / 0                | 0 / 0                     |
| Approximate cost                | ~$0.70               | ~$2.50                    |

### The load-bearing lesson

When you craft `embed_queries` for a stimulus, the angles that
survive dedup are the ones with genuinely different emotional or
behavioral content. Angles that share an *emotional valence* —
two anxious takes, two ecstatic takes — collapse together at
dedup, because Claude writes similar instructions for similar
inputs even when the retrieved chunks differ.

A reliable spread is **one sensory slice + one emotional-positive
slice + one emotional-negative slice** per stimulus. Stimuli with
that pattern hit 90–100 % kept fractions in the v5 pilot; stimuli
with overlapping-valence angles dropped to 44–55 %.

Highest-performing stimuli (all angles ≥ 80 % kept):
*owner crying on the couch*, *midnight bathroom trip*,
*favorite toy lost under the couch*, *sprinkler going off*,
*dishwasher running*, *unexpected afternoon nap on the couch*,
*a trip to the vet*, *a bath being run*, *lying in a sunbeam*,
*meeting another dog at the park*, *owner returning from a long trip*.

Lowest-performing stimuli (kept fraction ≤ 56 %) — candidates to
redesign angles for the next pilot:
*the mailman arriving* (44 % — all three angles confrontational),
*a long car ride* (44 % — two anxious takes),
*a sock left on the floor* (50 %),
*the back gate left open* (50 %),
*a sick owner staying in bed* (50 %),
*owner taking out the trash* (50 % — one angle hit 0 %),
*owner returning home from work* (56 %),
*dinner being prepared in the kitchen* (56 %),
*a new baby in the house* (56 %).

### Action

Run `rosetta-storyteller sft stats` after every `sft merge` and
review per-angle kept fractions before training. It's the cheapest
place to catch retrieval-collapse before spending GPU time.
