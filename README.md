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

## Quickstart

```sh
uv sync
cp .env.example .env && $EDITOR .env   # add ANTHROPIC_API_KEY

rosetta-storyteller ingest --pillar style --limit 3
rosetta-storyteller ingest --pillar science --limit 5
rosetta-storyteller ingest --pillar behavior --limit 50
rosetta-storyteller chunk --all
rosetta-storyteller embed

rosetta-storyteller sft generate --count 10 --phase pilot
rosetta-storyteller sft poll
rosetta-storyteller sft merge

rosetta-storyteller train --iters 200
rosetta-storyteller generate "a trip to the vet"
```

See [docs/superpowers/specs/](docs/superpowers/specs/) for the v1 design and
[docs/superpowers/plans/](docs/superpowers/plans/) for the implementation plan.

## Iterating: pilot → full

The 1000-request cap is the safety net. Recommended workflow:

1. **Pilot:** `rosetta-storyteller sft generate --count 500 --phase pilot`
2. Inspect `data/sft/train.jsonl` by hand. Confirm sensory grounding,
   look for canned phrases, check `cache_read_input_tokens > 0` in the
   manifest entry (if not, prompt caching is broken).
3. Iterate `config/stimuli.yaml` and the persona text.
4. **Full:** `rosetta-storyteller sft generate --count 10000 --phase full --max-requests 10000`

Cost estimate: pilot ≈ $3-5, full ≈ $20-60 (Sonnet 4.6 batch pricing).

## Tests

```sh
# Unit tests (fast)
uv run pytest tests/unit -v

# Integration smoke test (slow, costs ~$0.10, downloads model weights)
ANTHROPIC_API_KEY=... uv run pytest tests/integration -m slow -v
```

[three-pillars-data-architecture]: https://github.com/agileedge/llm-wiki
[synthetic-data-sandwich]: https://github.com/agileedge/llm-wiki
