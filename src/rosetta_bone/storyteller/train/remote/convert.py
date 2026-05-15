"""Convert a PEFT LoRA adapter to mlx-lm's adapter format.

PEFT (what runs on RunPod) writes:
    base_model.model.model.layers.{N}.self_attn.{q,v}_proj.lora_A.weight   shape (r, D_in)
    base_model.model.model.layers.{N}.self_attn.{q,v}_proj.lora_B.weight   shape (D_out, r)
    adapter_config.json with {"r": ..., "lora_alpha": ..., "target_modules": ...}

mlx-lm (what runs on the laptop) wants:
    model.layers.{N}.self_attn.{q,v}_proj.lora_a   shape (D_in, r)
    model.layers.{N}.self_attn.{q,v}_proj.lora_b   shape (r, D_out)
    adapter_config.json with mlx-lm's schema (num_layers, lora_parameters, ...)

The PEFT alpha and rank map onto mlx-lm's `scale` via scale = alpha / r,
which preserves the magnitude of the LoRA delta exactly. We DON'T copy
PEFT's `alpha` into mlx's config because mlx's loader doesn't understand
that key — it only consumes `scale`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

# Match either single or double "model." prefix. HuggingFace Llama
# nests the transformer under model.model (outer wrapper + inner
# config), but bare-bones test fixtures often only have one level.
# Either way, mlx-lm wants exactly one "model." in the output key.
_PEFT_KEY = re.compile(
    r"^base_model\.model\.(?:model\.)?"
    r"layers\.(?P<layer>\d+)\.(?P<rest>.+)\.lora_(?P<ab>[AB])\.weight$"
)


@dataclass(frozen=True)
class ConvertResult:
    """Returned by `peft_to_mlx()` so the caller can sanity-check the result."""

    n_tensors: int
    n_layers: int
    rank: int
    scale: float


def peft_to_mlx(peft_dir: Path, mlx_dir: Path) -> ConvertResult:
    """Convert PEFT adapter in `peft_dir` to mlx-lm format in `mlx_dir`.

    Writes:
        mlx_dir/adapters.safetensors
        mlx_dir/adapter_config.json
    """
    peft_config = json.loads((peft_dir / "adapter_config.json").read_text())
    rank = int(peft_config["r"])
    alpha = float(peft_config["lora_alpha"])
    target_modules = sorted(peft_config["target_modules"])  # sort for stability
    scale = alpha / rank

    src = peft_dir / "adapter_model.safetensors"
    mlx_dir.mkdir(parents=True, exist_ok=True)
    new_tensors: dict = {}
    layer_indices: set[int] = set()

    with safe_open(src, framework="pt") as f:
        for peft_key in f.keys():  # noqa: SIM118
            m = _PEFT_KEY.match(peft_key)
            if m is None:
                # Bias terms, embedding LoRAs, modules_to_save, etc.
                # mlx-lm only consumes lora_a/lora_b on the configured
                # target modules; surface unknowns so we don't silently
                # drop something important on a future PEFT upgrade.
                raise ValueError(
                    f"unrecognised PEFT key (no q/v_proj LoRA shape): {peft_key}",
                )
            layer = int(m["layer"])
            rest = m["rest"]  # e.g. "self_attn.q_proj"
            ab = m["ab"].lower()  # "a" or "b"
            mlx_key = f"model.layers.{layer}.{rest}.lora_{ab}"
            # PEFT A is (r, D_in); mlx a is (D_in, r). Transpose handles both.
            new_tensors[mlx_key] = f.get_tensor(peft_key).t().contiguous()
            layer_indices.add(layer)

    save_file(new_tensors, str(mlx_dir / "adapters.safetensors"))

    keys = [f"self_attn.{m}" for m in target_modules]
    # mlx-lm's --num-layers N means "top N transformer blocks." We don't
    # know the total block count from the PEFT adapter alone, so we
    # record the count of layers we have — accurate under the convention
    # "PEFT trained on layers [total-N, total)", which is what our
    # trainer does.
    num_layers = len(layer_indices)

    mlx_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": rank,
            "dropout": float(peft_config.get("lora_dropout", 0.0)),
            "scale": scale,
            "keys": keys,
        },
    }
    (mlx_dir / "adapter_config.json").write_text(json.dumps(mlx_config, indent=2))

    return ConvertResult(
        n_tensors=len(new_tensors),
        n_layers=num_layers,
        rank=rank,
        scale=scale,
    )
