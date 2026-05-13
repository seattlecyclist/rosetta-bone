# Comedic modes

Distilled from the full multi-sample mailman + bacon + vet comparison
across all 8 storyteller-v1 adapters (2026-05-13). Each adapter
specialized in a different comedic register; the funniest individual
stories in the comparison were each examples of *one* of the modes
below, never a blend.

This file is the angle-design rule for the corpus from v9 onward.
Every embed_query in [config/stimuli.yaml](../config/stimuli.yaml)
gets tagged with exactly one of these modes, and each stimulus is
designed to span three distinct modes so the trained adapter can
produce all of them on demand rather than collapsing to one.

---

## The five modes

### 1. Delusions of grandeur

The dog has invented a job, an authority, or a special status, and
performs it gravely. The humor is in the gap between the dog's
self-image and reality.

**Source adapter (strongest in comparison):** `20260512T042405Z`
("funny baseline", small-corpus 2000-iter run).

**Canonical lines:**

- "The couch is safe. **I have always been the dog who keeps the
  couch safe.** No squirrels have ever gotten on the couch. That is
  because of me."
- "I am the dog who guards the food. **The diary is closed.**"

**Angle phrasing pattern:**
*"dog [appointing itself / officially in charge of / acting as
the household's / handling on behalf of] [some unearned authority]"*

---

### 2. Cowardly deflation

The dog builds dread or commits to bravery, then bails — usually
with a face-saving rationalization. The humor is in the abrupt
retreat and the dignified spin on it.

**Source adapter:** v7 pilot (`20260512T210203Z`).

**Canonical lines:**

- "I go to the couch very fast. I circle it three times. I lie down
  facing the wall. **The couch was my hero today.**"
- "I sit down very fast. Then. The pincushion. I bite my chew toy.
  **I have to. It is survival.**"

**Angle phrasing pattern:**
*"dog [committing to / charging at / very brave about] [the
threat] and then [immediately retreating / pretending it didn't /
strategically losing / claiming a higher-priority concern]"*

---

### 3. Absurd logic & distraction

The dog reasons via free association, gets hyper-fixated on
irrelevant details, or makes wildly out-of-frame deductions
delivered as if obvious. The humor is in the off-axis seriousness.

**Source adapter:** v8 LATEST (`20260513T020823Z`).

**Canonical lines:**

- "Paper and a little bit of ham. **HAM. I know that ham smell.
  That is the most important part right now.**"
- "I look back at the couch. **The couch will still be here when I
  get back. This is a fact I have always known.** I do not need to
  check it right now."
- "Tricolor flag means three colors side by side. **Tricolor flag
  means danger.**"

**Angle phrasing pattern:**
*"dog [hyper-focused on / fact-checking / forensically tracking]
[some specific overweighted detail] [and forgetting / and pivoting
away from / instead of] [the obvious main thing]"*

---

### 4. Bizarre rationalizer

The dog states a mistaken premise as fact, often with a casual
self-deprecating correction or aside. The humor is in the
confidence of the wrong model of the world plus the willingness to
volunteer embarrassing context.

**Source adapter:** v6-era (`20260512T173159Z`).

**Canonical lines:**

- "**At first I thought the neighbor had a new dog and was grilling
  a whole sock. I know. I know.** I once ate part of one. It was
  not good."
- "He puts on gloves. **This is confusing. He put on the gloves and
  he is still suspicious.**"

**Angle phrasing pattern:**
*"dog [initially convinced / at first thought / mistaking X for Y]
[some plausible but wrong premise] [and behaving accordingly]"*

---

### 5. Coping through dissociation

The dog refuses to engage with the situation — pivots to something
else, pointedly ignores, or chooses inner peace as defiance. The
humor is in the dignified refusal.

**Source adapter:** v7 pilot (`20260512T210203Z`).

**Canonical lines:**

- "I have survived this. I will think about something else. **I
  will think about bacon. BAKON.**"
- "I do not befriend these dogs any better than I befriend this
  place. **I do not befriend this place.** This is my worst place."

**Angle phrasing pattern:**
*"dog [pretending not to / refusing to engage with / tactfully
ignoring / choosing inner peace over] [the situation], [and just
doing something else instead]"*

---

## Why this works

The dedup pass in `sft merge` collapses generations that share
emotional valence and behavioral shape, even when the retrieved
chunks differ. That's the v5 lesson: angles need to differ in
*content* to survive dedup.

Comedic modes are a stronger content discriminant than emotional
valence because they encode the dog's **stance toward the
situation**, not just whether the situation is good or bad.

- Two anxious angles ("dog afraid of mailman", "dog fearful of
  mailman") collapse.
- Two negative-valence angles ("dog afraid", "dog suspicious")
  also tend to collapse — both produce defensive prose.
- Two distinct comedic modes on the same stimulus ("dog charging
  the mailman and then hiding" vs "dog deciding the mailman
  situation is too much and thinking about lunch instead")
  produce structurally different stories: one has an action arc,
  one has a dissociation pivot. Neither is reducible to the other.

The orthogonality is what makes a comedic-mode-spread corpus
dedup-resilient *and* mode-balanced.

## The corpus design rule

For each of the 50 stimuli in `config/stimuli.yaml`:

1. Pick **three** of the five comedic modes that fit the situation
   naturally. Not every mode works for every stimulus — *lying in
   a sunbeam* has no real "cowardly deflation" angle.
2. Write one angle per chosen mode, in the angle's phrasing pattern
   above.
3. Tag the angle inline with a `# delusion`, `# coward`, `# absurd`,
   `# rationalizer`, or `# dissociator` comment so the design intent
   is visible in the file.
4. Aim for roughly even mode coverage across the 50 stimuli (target
   ~30 angles per mode out of 150 total). Over-indexing on one mode
   in the corpus produces an adapter that does that mode well and
   nothing else — exactly the v8 trap (deadpan-distraction-rich,
   delusion-poor).

## When to update this doc

- New comedic mode surfaces in an adapter that none of the existing
  five capture cleanly.
- An existing mode turns out to be a special case of another after
  more data (e.g., if "rationalizer" collapses into "absurd logic"
  at corpus scale).
- The phrasing patterns themselves get sharpened by a successful
  pilot — append the better phrasing to the relevant mode section,
  don't replace.
