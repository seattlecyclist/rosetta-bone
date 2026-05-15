# RunPod Training Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move LoRA fine-tuning of the storyteller adapter from a local M2 Max (`mlx_lm.lora`) to an on-demand RunPod GPU pod (PyTorch + HuggingFace PEFT), driven by a new `train --remote` CLI flag. Datasets and adapters round-trip through an S3-compatible bucket (Cloudflare R2). The trained adapter is converted back to mlx-lm format and dropped into `data/adapters/.../{timestamp}/` so the existing inference path (`rosetta-storyteller infer`) is unchanged.

**Non-goals:** Moving inference off-device. Multi-node training. Hyperparameter sweep orchestration (single-pod, single-job is the unit of work; sweeps are a follow-up that loops over this primitive).

**Why now:** Wall-clock — an M2 Max run takes a few hours; a single 4090 finishes the same LoRA in roughly 10–15 minutes for ~$0.10. The existing README cost story ("a few dollars … a few hours of local GPU time") gets strictly better.

**Architecture:** Same pipeline-as-stages discipline as the rest of the repo. The remote training path is four idempotent stages:

```
sft/* (local) ──► [stage R1] upload data        ──► R2:datasets/<hash>/
                  [stage R2] launch pod         ──► RunPod pod id
                  [stage R3] pod runs train.py  ──► R2:adapters/<key>/peft/
                  [stage R4] fetch + convert    ──► data/adapters/.../<ts>/  (mlx format)
```

Each stage is resumable from on-disk / on-bucket state. Re-running with the same inputs is a no-op (content-addressed adapter key).

**The framework gap:** mlx-lm is Apple-silicon-only. RunPod is NVIDIA CUDA. This is not a lift-and-shift — it's a small framework port. The training arithmetic (LoRA rank 8, top-8 layers, lr 1e-5, seq 1024) is preserved; the trainer changes from `mlx_lm.lora` to `transformers.Trainer` + `peft.LoraConfig`. Adapter weights then need a one-shot conversion back to mlx-lm's key naming/shape convention before they can be loaded for local inference.

**Tech stack additions:** `boto3` (R2 client), `runpod` (pod lifecycle SDK). Inside the RunPod image: `torch`, `transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `safetensors`. No new local CUDA deps — all GPU-touching code lives inside the container.

---

## File Structure

### New source files (`src/rosetta_bone/storyteller/train/remote/`)

| File | Responsibility |
|---|---|
| `__init__.py` | Re-export `remote_train()` entry point used by the CLI |
| `config.py` | `RemoteConfig` dataclass (R2 + RunPod settings); reads `[train.remote]` in `default.toml` + env vars |
| `keys.py` | Content-address helper: `adapter_key(train_sha, valid_sha, base_model, hyperparams) -> str` |
| `storage.py` | Thin boto3 wrapper: `upload_dir(local, prefix)`, `download_dir(prefix, local)`, `exists(prefix)`, `list_prefix(prefix)` — all against an S3-compatible endpoint |
| `runpod_client.py` | `launch_pod(image, env, command, gpu_type) -> PodHandle`; `wait_for_completion(handle, timeout) -> ExitStatus`; `fetch_logs(handle) -> str`; `terminate(handle)` — wraps the `runpod` SDK |
| `orchestrator.py` | `remote_train(cfg, train_path, valid_path, adapter_dir, hyperparams)` — sequences R1→R4, handles resume |
| `convert.py` | `peft_to_mlx(peft_dir, mlx_dir, base_model)` — rename/reshape PEFT LoRA tensors to mlx-lm's `adapters.safetensors` schema; write `adapter_config.json` |

### New files inside the RunPod container (`docker/runpod-trainer/`)

| File | Responsibility |
|---|---|
| `Dockerfile` | Based on `runpod/pytorch:2.4.0-py3.12-cuda12.4.1-devel-ubuntu22.04`; layers in `transformers`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `safetensors`, `boto3` |
| `train.py` | Entry point inside the pod: `boto3` download dataset → `transformers.Trainer` + `peft.LoraConfig` → `boto3` upload adapter; reads job spec from env vars (no CLI args, no secrets in argv) |
| `README.md` | Image build/push recipe + the one-liner that runs locally with `docker run --gpus all` for a parity smoke test |

### Modified files

| File | Change |
|---|---|
| `src/rosetta_bone/storyteller/cli.py` | `train_cmd` grows a `--remote/--local` flag (default `--local`). When `--remote`, dispatches to `remote_train()`; the metadata-writing and `latest` symlink code is factored into a helper used by both paths. |
| `src/rosetta_bone/common/config.py` | Add `RemoteTrain` dataclass + `Train.remote: RemoteTrain \| None` field |
| `config/default.toml` | New `[train.remote]` section with `image`, `gpu_type`, `bucket`, `endpoint_url` (no secrets) |
| `.env.example` | Add `RUNPOD_API_KEY`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT_URL`, `R2_BUCKET` |
| `pyproject.toml` | Add `boto3>=1.34`, `runpod>=1.7` to the `[project]` deps. Keep `mlx`/`mlx-lm` — local path still works. |
| `README.md` | New "Training remotely on RunPod" subsection alongside the existing local training story |
| `docs/runbook.md` | Operator notes: how to rotate R2 keys, kill a stuck pod, force-resume from a partial run |

### New tests

| File | Coverage |
|---|---|
| `tests/unit/test_remote_keys.py` | `adapter_key()` is stable across hyperparam dict ordering; changes when any input changes |
| `tests/unit/test_remote_storage.py` | boto3 calls go to the configured endpoint; mocked with `moto` |
| `tests/unit/test_remote_runpod_client.py` | Pod lifecycle calls hit the right SDK methods; `runpod` mocked |
| `tests/unit/test_remote_orchestrator.py` | R1→R4 sequencing; existing-adapter short-circuit; cleanup on failure; all I/O mocked |
| `tests/unit/test_convert_peft_to_mlx.py` | Golden-file test: a tiny synthetic PEFT adapter converts to the expected mlx-lm `adapters.safetensors` keys + shapes |
| `tests/integration/test_e2e_remote_tiny.py` | `@pytest.mark.slow + @pytest.mark.runpod`. End-to-end against real RunPod + R2 with a 50-iter run on the 3B model. Skipped by default; runs only when both secrets are present. |

---

## Phase 0 — Decisions to lock before code

- [ ] **GPU class.** Recommend RTX 4090 (24 GB, Community Cloud, ~$0.34/hr). LoRA on Llama-3.1-8B in bf16 needs ~18 GB; 4090 fits with headroom. A40 (48 GB, ~$0.45/hr) is the fallback if we hit OOM on longer sequences.
- [ ] **Precision parity decision.** Local training quantizes the base model to 4-bit (mlx-community `…-4bit`). On CUDA, train the LoRA against the same unquantized Llama-3.1-8B in bf16, then load the merged adapter against the 4-bit base for inference. Acceptable because LoRA only sees frozen base weights as a feature extractor; the small dtype gap on the base is the same trick mlx-lm's QLoRA path uses.
- [ ] **Adapter conversion strategy.** Two options: (a) write our own PEFT-→mlx-lm renamer in `convert.py`, or (b) train on the mlx-community 4-bit model via `mlx`-on-CUDA (does not exist) / `bitsandbytes` 4-bit + a different converter. Pick (a) — it's a ~50-line rename + reshape, fully testable, no surprise dependencies. Bake a snapshot test against a 2-layer toy model.
- [ ] **Image hosting.** Push to `ghcr.io/seattlecyclist/rosetta-bone-trainer:vN`. Use a CalVer tag (`2026.05.15`) so the trainer version is grep-able in adapter metadata.
- [ ] **Cost / safety rails.** Hard ceiling on pod lifetime (default 30 min) enforced via the RunPod API's `terminationGracePeriodSeconds` + an in-pod watchdog that self-terminates after `MAX_TRAIN_SECONDS`. Cost cap printed before launch (`pod_seconds * gpu_$/sec`).

---

## Phase 1 — Storage layer (R2)

- [ ] **R1.1** Add `boto3` to `pyproject.toml`; `uv sync`.
- [ ] **R1.2** Add `[train.remote]` section to `config/default.toml`. Fields: `image`, `gpu_type`, `bucket`, `endpoint_url`, `pod_timeout_seconds`. Secrets stay in env.
- [ ] **R1.3** Add R2 env vars to `.env.example` with comments pointing at the Cloudflare R2 dashboard URL pattern.
- [ ] **R1.4** Implement `storage.py`. The S3-compatibility surface is small: `head_object`, `put_object`, `get_object`, `list_objects_v2`. Use the `endpoint_url` kwarg on `boto3.client("s3", ...)`.
- [ ] **R1.5** Unit test `storage.py` with `moto` (mocks S3). Assert: uploading a directory results in one object per file with the prefix preserved; `exists(prefix)` returns true iff at least one object matches; download is the inverse of upload byte-for-byte.
- [ ] **R1.6** Manual smoke test against real R2: `python -c "from rosetta_bone.storyteller.train.remote import storage; storage.upload_dir('data/sft', 'smoke/')"`. Verify in dashboard; delete.

## Phase 2 — Content-addressed keys

- [ ] **R2.1** Implement `keys.adapter_key()`. Inputs: `train_sha1`, `valid_sha1`, base model id, hyperparams dict. Output: short hex prefix of `sha256(canonical_json)`. Canonical JSON = sorted keys, no whitespace.
- [ ] **R2.2** Unit test: same inputs → same key; reordering the hyperparams dict → same key; changing learning rate → different key.
- [ ] **R2.3** Wire `adapter_key()` into the orchestrator's resume check: if `adapters/<key>/peft/adapter_model.safetensors` already exists in R2, skip straight to R4 (fetch + convert).

## Phase 3 — RunPod container image

- [ ] **R3.1** Write `docker/runpod-trainer/Dockerfile`. Base `runpod/pytorch:2.4.0-py3.12-cuda12.4.1-devel-ubuntu22.04`. Pip install pinned versions of transformers, peft, trl, accelerate, bitsandbytes, safetensors, boto3. Copy `train.py` to `/app/train.py`. `ENTRYPOINT ["python", "/app/train.py"]`.
- [ ] **R3.2** Write `train.py`. Reads from env: `R2_*`, `DATASET_PREFIX`, `ADAPTER_PREFIX`, `BASE_MODEL`, `HYPERPARAMS_JSON`. Downloads dataset prefix → `/workspace/data`. Builds `LoraConfig(r=rank, lora_alpha=alpha, target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")`. Loads tokenizer + model in bf16. Trains with `transformers.Trainer` mimicking mlx-lm's CLI semantics: same `--num-layers 8` (freeze all but top 8), `--max-seq-length 1024`, same lr, same batch size, same iter count. Writes `adapter_model.safetensors` + `adapter_config.json` to `/workspace/out` + tees stdout/stderr to `/workspace/out/train.log`. Uploads the directory to `adapters/<key>/peft/`. Self-terminates on success or failure (does NOT keep the pod alive on error — we have logs and the orchestrator can re-launch).
- [ ] **R3.3** Local image smoke test: `docker build -t rosetta-bone-trainer:dev .`; `docker run --rm rosetta-bone-trainer:dev --help` succeeds. (Skip the `--gpus all` actual training smoke unless a CUDA machine is available; that's covered by the e2e test.)
- [ ] **R3.4** Push the first version: `docker push ghcr.io/seattlecyclist/rosetta-bone-trainer:2026.05.15`. Tag is also baked into the image as `IMAGE_TAG` env var so metadata.json on the produced adapter records exactly which trainer built it.

## Phase 4 — RunPod client

- [ ] **R4.1** Implement `runpod_client.py`. `launch_pod()` returns a `PodHandle` (id + start time). `wait_for_completion()` polls `runpod.get_pod(id)` every 10 s until status is `EXITED` or until `pod_timeout_seconds`. On timeout, calls `terminate()` and raises. Log retrieval uses the SDK's logs endpoint.
- [ ] **R4.2** Unit test against the `runpod` SDK mocked at the module level. Cover: success path, timeout-then-terminate, non-zero exit, transient API error retried with backoff.
- [ ] **R4.3** Manual smoke: launch a 1-minute hello-world container (override the entrypoint with `sh -c 'echo hi; sleep 60'`), confirm logs come back, confirm pod is gone afterward, confirm cost on the RunPod dashboard is ~$0.01.

## Phase 5 — Adapter conversion (PEFT → mlx-lm)

- [ ] **R5.1** Document the key/shape mapping in `convert.py` as a module docstring. PEFT names like `base_model.model.model.layers.{N}.self_attn.{q,v}_proj.lora_{A,B}.weight` map to mlx-lm `model.layers.{N}.self_attn.{q,v}_proj.lora_{a,b}`. Shapes already match for rank-only LoRA; verify by loading both and printing keys.
- [ ] **R5.2** Implement `peft_to_mlx()`. Reads PEFT's `adapter_model.safetensors` + `adapter_config.json`. Writes mlx-lm's `adapters.safetensors` + `adapter_config.json` in the schema mlx-lm expects (look at the actual contents of a known-good local adapter under `data/adapters/llama31-8b-storyteller-v1/latest/` to lock the format).
- [ ] **R5.3** Build the golden-file test fixture: train a 5-iter LoRA locally on a 2-layer dummy model with both mlx-lm and PEFT against synthetic data. Snapshot the mlx adapter; the test asserts conversion of the PEFT adapter produces an equivalent state-dict (same keys, shape-matched).
- [ ] **R5.4** Wire conversion into the orchestrator after R4 download.
- [ ] **R5.5** Parity verification: load the converted adapter via the existing `infer` path; generate one completion for a fixed prompt; sanity-check that output is fluent (not gibberish). Numerical equivalence to a local mlx-trained adapter is NOT expected (different optimizer numerics on different hardware) — fluency + reasonable loss are the bar.

## Phase 6 — Orchestrator + CLI integration

- [ ] **R6.1** Implement `orchestrator.remote_train()`. Sequencing:
  1. Compute `adapter_key`. If `adapters/<key>/peft/` already in R2 → skip to step 5.
  2. `storage.upload_dir(sft_dir, f"datasets/{dataset_sha}/")` if not already there.
  3. `runpod_client.launch_pod(image, env=...)` with env carrying `DATASET_PREFIX`, `ADAPTER_PREFIX`, `BASE_MODEL`, `HYPERPARAMS_JSON`, R2 creds.
  4. `wait_for_completion()`. On non-zero exit, fetch logs, write to `versioned_dir/train.log`, raise.
  5. `storage.download_dir(f"adapters/{key}/peft/", tmp)`; `convert.peft_to_mlx(tmp, versioned_dir, base_model)`.
  6. Write `metadata.json` (same shape as the local path, with extra `remote: {pod_id, gpu_type, image_tag, key}` block).
- [ ] **R6.2** Refactor `cli.py:train_cmd` so the metadata + symlink code is shared between local and remote paths. Add `--remote/--local` flag (default `--local`). When `--remote`, call `remote_train()` instead of `lora.train()`.
- [ ] **R6.3** Cost preview: before launching the pod, print the dataset hash, adapter key, expected GPU-minutes, and dollar cap. Require `--yes` to skip the prompt in CI / automation.
- [ ] **R6.4** Wire log retrieval: at the end of a remote run, the pod's `train.log` is downloaded to `versioned_dir/train.log` so `train-inspect` works unchanged.

## Phase 7 — Tests + docs

- [ ] **R7.1** All unit tests above pass under `pytest tests/unit -q`.
- [ ] **R7.2** Local path regression: existing `tests/integration/test_e2e_tiny.py` still passes on an M2 Max (`--local` default).
- [ ] **R7.3** New integration test `tests/integration/test_e2e_remote_tiny.py`: 50 iters on the 3B base model, real RunPod + R2. Marked `@pytest.mark.runpod`. Skips unless both `RUNPOD_API_KEY` and `R2_*` are present.
- [ ] **R7.4** README "Training remotely" subsection. Show the one-liner: `rosetta-storyteller train --remote --iters 1000`. Note the cost estimate and the unchanged inference command.
- [ ] **R7.5** Update `docs/runbook.md` with: "I cancelled mid-run, now what?" (pods self-terminate; orchestrator is safe to re-launch — re-running with the same data is content-addressed and free if it completed, or a fresh run if it didn't).

## Phase 8 — Cutover

- [ ] **R8.1** Train v2 of the adapter remotely. Compare validation loss curves to the latest local-trained v1 (via `train-inspect`). Expect comparable shape; absolute loss may differ slightly due to dtype.
- [ ] **R8.2** Spot-check inference quality on the standard stimuli set against both adapters; flag any obvious regressions.
- [ ] **R8.3** Commit + push. PR title: "Add RunPod remote training path". Land behind `--remote` flag — local stays the default until v2 is in production for at least one week.

---

## Risks & open questions

- **Adapter format drift.** mlx-lm has rev'd its adapter schema before (the comment in `train/lora.py` flags this for the trainer; the loader has the same risk). Our converter must target the **current** mlx-lm schema; if mlx-lm bumps, the converter needs a matching bump. Mitigation: snapshot test on every CI run pins us to a known shape.
- **Tokenizer parity.** The `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` repo and the upstream `meta-llama/Meta-Llama-3.1-8B-Instruct` use the same tokenizer, but we should confirm by hash. If they diverge, SFT pairs trained against the wrong tokenizer will load fine and degrade subtly.
- **HF auth on the pod.** Loading Llama-3.1 needs `HF_TOKEN`. Pass it via RunPod pod env, not baked in the image.
- **R2 vs S3 pricing.** R2 has zero egress fees, which matters because we're pulling adapters from the pod (US) → bucket (anywhere) → laptop (anywhere). Egress on real S3 would be ~$0.09/GB; an 80 MB adapter is negligible, but if we ever upload the base model it adds up.
- **What if the local mlx path drifts?** Two trainers, two skews. Mitigation: the local path is the source of truth for hyperparameter semantics, and the integration test re-runs both on the tiny 3B model — divergence in loss shape fails CI.
- **Reproducibility across GPU types.** A run on a 4090 and the same run on an A40 won't be bitwise identical (different kernels). The orchestrator's content-addressed key intentionally **excludes** `gpu_type` — different GPUs producing functionally equivalent adapters get the same key and dedupe correctly. If we later want strict per-GPU reproducibility, add `gpu_type` to the key inputs.

---

## Out of scope (deliberate)

- Hyperparameter sweeps. The single-job primitive lands first; a sweep is "loop over `remote_train()` with different hyperparams" and adds nothing structurally new.
- Moving inference to a hosted endpoint. README's "runs locally, no inference bill" story is load-bearing; remote inference is a separate decision with separate trade-offs.
- Multi-GPU / multi-node training. LoRA-8 on Llama-3.1-8B fits on a single 4090; distributed training is unnecessary complexity for this workload.
- A general-purpose "job runner" abstraction. We have one job (LoRA fine-tune) and one backend (RunPod). YAGNI until we have a second of either.
