"""Tests for the PEFT -> mlx-lm adapter converter.

We build a real PEFT adapter via `get_peft_model()` on a tiny dummy
model so we exercise the actual on-disk schema PEFT produces, not a
schema we hand-rolled and would need to keep in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from rosetta_bone.storyteller.train.remote.convert import peft_to_mlx


class _MiniLlama(nn.Module):
    """Two-layer dummy whose module tree mirrors HF Llama's relevant shape."""

    def __init__(self, dim: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.model = nn.ModuleDict({
            "layers": nn.ModuleList([
                nn.ModuleDict({
                    "self_attn": nn.ModuleDict({
                        "q_proj": nn.Linear(dim, dim, bias=False),
                        "v_proj": nn.Linear(dim, dim, bias=False),
                    }),
                })
                for _ in range(n_layers)
            ]),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make_peft_adapter(tmp_path: Path, *, r: int = 4, alpha: int = 8) -> Path:
    base = _MiniLlama(dim=8, n_layers=2)
    cfg = LoraConfig(
        r=r, lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"], bias="none",
    )
    m = get_peft_model(base, cfg)
    # Deterministic non-zero weights so the conversion is observable.
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.copy_(torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape) * 0.01)
    out = tmp_path / "peft"
    m.save_pretrained(out)
    return out


def test_converts_keys_to_mlx_lm_convention(tmp_path):
    peft_dir = _make_peft_adapter(tmp_path)
    mlx_dir = tmp_path / "mlx"

    result = peft_to_mlx(peft_dir, mlx_dir)

    sd = load_file(str(mlx_dir / "adapters.safetensors"))
    expected_keys = {
        "model.layers.0.self_attn.q_proj.lora_a",
        "model.layers.0.self_attn.q_proj.lora_b",
        "model.layers.0.self_attn.v_proj.lora_a",
        "model.layers.0.self_attn.v_proj.lora_b",
        "model.layers.1.self_attn.q_proj.lora_a",
        "model.layers.1.self_attn.q_proj.lora_b",
        "model.layers.1.self_attn.v_proj.lora_a",
        "model.layers.1.self_attn.v_proj.lora_b",
    }
    assert set(sd.keys()) == expected_keys
    assert result.n_tensors == 8
    assert result.n_layers == 2
    assert result.rank == 4


def test_transposes_shapes_correctly(tmp_path):
    peft_dir = _make_peft_adapter(tmp_path)
    mlx_dir = tmp_path / "mlx"

    peft_to_mlx(peft_dir, mlx_dir)

    sd = load_file(str(mlx_dir / "adapters.safetensors"))
    # PEFT A: (r, D_in)=(4,8) -> mlx a: (D_in, r)=(8,4)
    assert tuple(sd["model.layers.0.self_attn.q_proj.lora_a"].shape) == (8, 4)
    # PEFT B: (D_out, r)=(8,4) -> mlx b: (r, D_out)=(4,8)
    assert tuple(sd["model.layers.0.self_attn.q_proj.lora_b"].shape) == (4, 8)


def test_delta_is_preserved_through_conversion(tmp_path):
    """The whole point: PEFT delta == mlx delta for the same input."""
    peft_dir = _make_peft_adapter(tmp_path, r=4, alpha=8)
    mlx_dir = tmp_path / "mlx"
    peft_to_mlx(peft_dir, mlx_dir)

    # PEFT side: load the raw tensors and reconstruct the delta the
    # way PEFT's LoraLayer applies it (alpha/r * x @ A.T @ B.T).
    from safetensors.torch import load_file as _load
    peft_sd = _load(str(peft_dir / "adapter_model.safetensors"))
    a_peft = peft_sd["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    b_peft = peft_sd["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    peft_delta = (8 / 4) * (x @ a_peft.t() @ b_peft.t())

    # mlx side: scale * x @ a @ b (per LoRALinear.__call__ in mlx-lm).
    mlx_sd = load_file(str(mlx_dir / "adapters.safetensors"))
    a_mlx = mlx_sd["model.layers.0.self_attn.q_proj.lora_a"]
    b_mlx = mlx_sd["model.layers.0.self_attn.q_proj.lora_b"]
    cfg = json.loads((mlx_dir / "adapter_config.json").read_text())
    scale = cfg["lora_parameters"]["scale"]
    mlx_delta = scale * (x @ a_mlx @ b_mlx)

    assert torch.allclose(peft_delta, mlx_delta, atol=1e-6)


def test_adapter_config_schema_matches_mlx_lm(tmp_path):
    peft_dir = _make_peft_adapter(tmp_path, r=8, alpha=16)
    mlx_dir = tmp_path / "mlx"
    peft_to_mlx(peft_dir, mlx_dir)

    cfg = json.loads((mlx_dir / "adapter_config.json").read_text())
    assert cfg["fine_tune_type"] == "lora"
    assert cfg["num_layers"] == 2
    assert cfg["lora_parameters"]["rank"] == 8
    assert cfg["lora_parameters"]["scale"] == pytest.approx(2.0)  # alpha/r
    assert cfg["lora_parameters"]["dropout"] == 0.0
    assert sorted(cfg["lora_parameters"]["keys"]) == [
        "self_attn.q_proj", "self_attn.v_proj",
    ]


def test_raises_on_unrecognised_peft_key(tmp_path):
    peft_dir = _make_peft_adapter(tmp_path)
    # Repack the safetensors with one extra junk key.
    from safetensors.torch import load_file as _load
    from safetensors.torch import save_file as _save
    sd = _load(str(peft_dir / "adapter_model.safetensors"))
    sd["unexpected.key.weight"] = torch.zeros(2, 2)
    _save(sd, str(peft_dir / "adapter_model.safetensors"))

    with pytest.raises(ValueError, match="unrecognised PEFT key"):
        peft_to_mlx(peft_dir, tmp_path / "mlx")
