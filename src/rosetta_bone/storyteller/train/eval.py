"""Perplexity eval: run `mlx_lm.lora --test`, parse stdout."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_PPL_RE = re.compile(r"Test ppl\s+([0-9.]+)")


def parse_perplexity(stdout: str) -> float | None:
    m = _PPL_RE.search(stdout)
    return float(m.group(1)) if m else None


def evaluate(
    *,
    base_model: str,
    data_dir: Path,
    adapter_dir: Path,
    out_path: Path,
) -> float | None:
    argv = [
        sys.executable, "-m", "mlx_lm.lora",
        "--test",
        "--model", base_model,
        "--adapter-path", str(adapter_dir),
        "--data", str(data_dir),
    ]
    res = subprocess.run(argv, check=False, capture_output=True, text=True)
    ppl = parse_perplexity(res.stdout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"perplexity": ppl, "stdout_tail": res.stdout[-2000:]}))
    return ppl
