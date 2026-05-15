"""In-pod entry point: download dataset, train LoRA, upload adapter.

Reads its entire spec from environment variables (no argv, no
secrets-in-process-table). Tees all stdout/stderr to
/workspace/out/train.log so the orchestrator can pull the log back
and feed it to `train-inspect` for parity with the local path.

Environment contract:
    R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
    DATASET_PREFIX        e.g. "datasets/<sha>"
    ADAPTER_PREFIX        e.g. "adapters/<key>/peft"
    BASE_MODEL            HF repo id, e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"
    HF_TOKEN              HuggingFace read token (Llama is gated)
    HYPERPARAMS_JSON      JSON: {rank, alpha, iters, batch_size,
                                 learning_rate, target_modules,
                                 num_layers, max_seq_length}
    MAX_TRAIN_SECONDS     watchdog ceiling; pod self-terminates on overshoot
    IMAGE_TAG             baked at build time; written into metadata
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

DATA_DIR = Path("/workspace/data")
OUT_DIR = Path("/workspace/out")


def _env(key: str, *, required: bool = True, default: str | None = None) -> str:
    v = os.environ.get(key, default)
    if required and not v:
        raise SystemExit(f"missing required env var: {key}")
    return v or ""


def _start_watchdog(max_seconds: int) -> None:
    """Background thread that kills the process after max_seconds.

    Defense against a wedged training loop — independent of RunPod's
    own pod-lifetime ceiling, so we get a clean exit + uploadable log
    instead of an opaque pod-terminated state.
    """
    def _kill() -> None:
        time.sleep(max_seconds)
        sys.stderr.write(f"\n[watchdog] {max_seconds}s elapsed, exiting\n")
        sys.stderr.flush()
        # _exit so we bypass atexit handlers that might hang on the
        # CUDA driver shutdown. Logs are already flushed to disk.
        os._exit(124)

    t = threading.Thread(target=_kill, daemon=True)
    t.start()


def _s3_client():  # type: ignore[no-untyped-def]
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=_env("R2_ENDPOINT_URL"),
        aws_access_key_id=_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


def _download_prefix(prefix: str, dest: Path) -> int:
    client = _s3_client()
    bucket = _env("R2_BUCKET")
    prefix = prefix.rstrip("/") + "/"
    dest.mkdir(parents=True, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            relpath = obj["Key"][len(prefix):]
            if not relpath:
                continue
            target = dest / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, obj["Key"], str(target))
            count += 1
    return count


def _upload_dir(local: Path, prefix: str) -> int:
    client = _s3_client()
    bucket = _env("R2_BUCKET")
    prefix = prefix.rstrip("/")
    count = 0
    for path in sorted(local.rglob("*")):
        if not path.is_file():
            continue
        key = f"{prefix}/{path.relative_to(local).as_posix()}"
        client.upload_file(str(path), bucket, key)
        count += 1
    return count


def _setup_logging() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "train.log"
    # Tee: write to original stdout AND to file. Implemented as a
    # write-multiplexer rather than `tee` subprocess because we want
    # transformers' tqdm-progress lines captured too.
    class _Tee:
        def __init__(self, *streams):  # type: ignore[no-untyped-def]
            self._streams = streams
        def write(self, s):  # type: ignore[no-untyped-def]
            for st in self._streams:
                st.write(s)
                st.flush()
        def flush(self):  # type: ignore[no-untyped-def]
            for st in self._streams:
                st.flush()

    log_file = log_path.open("w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)


def _train(hyperparams: dict, base_model: str) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    rank = int(hyperparams["rank"])
    alpha = float(hyperparams["alpha"])
    iters = int(hyperparams["iters"])
    batch_size = int(hyperparams["batch_size"])
    learning_rate = float(hyperparams["learning_rate"])
    target_modules = list(hyperparams.get("target_modules", ["q_proj", "v_proj"]))
    num_layers = int(hyperparams.get("num_layers", 8))
    max_seq_length = int(hyperparams.get("max_seq_length", 1024))

    print(f"[train] loading tokenizer + model: {base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=_env("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=_env("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Match mlx-lm's --num-layers N semantics: LoRA only on the top N
    # transformer blocks. PEFT calls this `layers_to_transform`.
    total_layers = model.config.num_hidden_layers
    layers_to_transform = list(range(total_layers - num_layers, total_layers))
    print(
        f"[train] LoRA rank={rank} alpha={alpha} targets={target_modules} "
        f"layers={layers_to_transform[0]}..{layers_to_transform[-1]} "
        f"({len(layers_to_transform)}/{total_layers})",
        flush=True,
    )

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def _format(row):  # type: ignore[no-untyped-def]
        text = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False,
        )
        enc = tokenizer(
            text, truncation=True, max_length=max_seq_length, padding=False,
        )
        return enc

    train_ds = load_dataset(
        "json", data_files=str(DATA_DIR / "train.jsonl"), split="train",
    ).map(_format, remove_columns=["messages"])
    valid_ds = load_dataset(
        "json", data_files=str(DATA_DIR / "valid.jsonl"), split="train",
    ).map(_format, remove_columns=["messages"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        max_steps=iters,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(10, iters // 10),
        save_strategy="no",
        report_to=[],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    print(f"[train] saved adapter to {OUT_DIR}", flush=True)


def main() -> int:
    _setup_logging()
    max_seconds = int(os.environ.get("MAX_TRAIN_SECONDS", "1800"))
    _start_watchdog(max_seconds)

    base_model = _env("BASE_MODEL")
    dataset_prefix = _env("DATASET_PREFIX")
    adapter_prefix = _env("ADAPTER_PREFIX")
    hyperparams = json.loads(_env("HYPERPARAMS_JSON"))
    image_tag = os.environ.get("IMAGE_TAG", "unknown")

    print(f"[setup] image_tag={image_tag}", flush=True)
    print(f"[setup] downloading dataset from {dataset_prefix}", flush=True)
    n = _download_prefix(dataset_prefix, DATA_DIR)
    print(f"[setup] downloaded {n} files", flush=True)

    started = time.monotonic()
    _train(hyperparams, base_model)
    duration = time.monotonic() - started

    (OUT_DIR / "trainer_info.json").write_text(json.dumps({
        "image_tag": image_tag,
        "base_model": base_model,
        "hyperparams": hyperparams,
        "duration_seconds": round(duration, 2),
    }, indent=2))

    print(f"[upload] pushing adapter to {adapter_prefix}", flush=True)
    n = _upload_dir(OUT_DIR, adapter_prefix)
    print(f"[upload] uploaded {n} files in {duration:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
