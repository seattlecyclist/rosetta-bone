# Rosetta Bone

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
rosetta-storyteller embed --all

rosetta-storyteller sft generate --count 10 --phase pilot
rosetta-storyteller sft poll
rosetta-storyteller sft merge

rosetta-storyteller train --iters 200
rosetta-storyteller generate "a trip to the vet"
```

See [docs/superpowers/specs/](docs/superpowers/specs/) for the v1 design.

[three-pillars-data-architecture]: https://github.com/agileedge/llm-wiki
[synthetic-data-sandwich]: https://github.com/agileedge/llm-wiki
