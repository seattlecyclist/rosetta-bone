# RunPod LoRA-trainer image

Container that runs one LoRA fine-tune on RunPod and uploads the
adapter to R2. Invoked by `rosetta-storyteller train --remote`.

## Build & push

```sh
# Tag in CalVer; bump on every dep change.
TAG=2026.05.15

docker build \
    --build-arg IMAGE_TAG=$TAG \
    -t ghcr.io/seattlecyclist/rosetta-bone-trainer:$TAG \
    docker/runpod-trainer

docker push ghcr.io/seattlecyclist/rosetta-bone-trainer:$TAG
```

After pushing, update `train.remote.image` in `config/default.toml`
to point at the new tag.

## Local CUDA smoke test (optional)

If you have a CUDA box handy, you can exercise the image end-to-end
against R2 before paying for a RunPod pod:

```sh
docker run --rm --gpus all \
    -e R2_ENDPOINT_URL=https://<acct>.r2.cloudflarestorage.com \
    -e R2_ACCESS_KEY_ID=... \
    -e R2_SECRET_ACCESS_KEY=... \
    -e R2_BUCKET=rosetta-bone \
    -e BASE_MODEL=meta-llama/Meta-Llama-3.2-3B-Instruct \
    -e HF_TOKEN=... \
    -e DATASET_PREFIX=datasets/<sha> \
    -e ADAPTER_PREFIX=adapters/test/peft \
    -e HYPERPARAMS_JSON='{"rank":8,"alpha":16.0,"iters":50,"batch_size":2,"learning_rate":1e-5}' \
    -e MAX_TRAIN_SECONDS=600 \
    ghcr.io/seattlecyclist/rosetta-bone-trainer:dev
```

## Environment contract

The container reads its entire job spec from env vars; nothing is
passed as argv (keeps secrets out of the process table).

| Var | Purpose |
|---|---|
| `R2_ENDPOINT_URL` | S3-compatible endpoint, e.g. `https://<acct>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | R2 credentials |
| `R2_BUCKET` | Bucket name |
| `DATASET_PREFIX` | Where to pull `train.jsonl` + `valid.jsonl` from |
| `ADAPTER_PREFIX` | Where to push the PEFT adapter (`adapter_model.safetensors`, etc.) |
| `BASE_MODEL` | HF repo id (gated; `HF_TOKEN` must have access) |
| `HF_TOKEN` | HuggingFace read token |
| `HYPERPARAMS_JSON` | JSON: `{rank, alpha, iters, batch_size, learning_rate, target_modules, num_layers, max_seq_length}` |
| `MAX_TRAIN_SECONDS` | Watchdog ceiling (default 1800) |
| `IMAGE_TAG` | Baked at build time; recorded in `trainer_info.json` |
