"""Parse and analyze an mlx-lm training log.

Used by `rosetta-storyteller train-inspect` and by `train_cmd`'s
completion summary. Reads a captured mlx-lm stdout log (one log per
adapter, written to `<adapter_dir>/train.log`) and extracts:

- Train loss series (every reported iter)
- Validation loss series (every eval iter)
- Iteration throughput, token throughput, peak memory
- Total trained-token counter (mlx-lm's own value, includes user+assistant)
- A heuristic overfit verdict

mlx-lm's stdout uses the abbreviation "Val loss" in its lines; this
analyzer translates to "validation loss" in everything it renders.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# Two line shapes from mlx_lm.lora --train. The regexes are anchored
# loosely so we don't break if mlx-lm tacks more fields on later.
_TRAIN_LINE_RE = re.compile(
    r"Iter (?P<iter>\d+): Train loss (?P<loss>[\d.]+),"
    r".*?It/sec (?P<ips>[\d.]+),"
    r".*?Tokens/sec (?P<tps>[\d.]+),"
    r".*?Trained Tokens (?P<tokens>\d+),"
    r".*?Peak mem (?P<mem>[\d.]+) GB"
)

# mlx-lm calls it "Val loss"; we translate everywhere we render.
_VAL_LINE_RE = re.compile(
    r"Iter (?P<iter>\d+): Val loss (?P<loss>[\d.]+),"
    r"\s*Val took (?P<took>[\d.]+)s"
)


@dataclass
class TrainReport:
    log_path: Path
    train_loss_series: list[tuple[int, float]] = field(default_factory=list)
    validation_loss_series: list[tuple[int, float]] = field(default_factory=list)
    iter_per_sec_series: list[float] = field(default_factory=list)
    tokens_per_sec_series: list[float] = field(default_factory=list)
    peak_memory_gb: float = 0.0
    trained_tokens: int = 0  # mlx-lm's counter — user+assistant+special
    final_iter: int = 0

    @property
    def n_train_reports(self) -> int:
        return len(self.train_loss_series)

    @property
    def n_validation_reports(self) -> int:
        return len(self.validation_loss_series)


def parse_log(log_path: Path) -> TrainReport:
    """Read an mlx-lm training log and return a structured TrainReport.

    Robust to extra lines (banner, validation-progress tqdm bars,
    checkpoint-save lines). Only the two regex shapes are extracted.
    """
    rep = TrainReport(log_path=log_path)
    if not log_path.exists():
        return rep
    for line in log_path.read_text(errors="replace").splitlines():
        m = _TRAIN_LINE_RE.search(line)
        if m:
            iter_n = int(m["iter"])
            rep.train_loss_series.append((iter_n, float(m["loss"])))
            rep.iter_per_sec_series.append(float(m["ips"]))
            rep.tokens_per_sec_series.append(float(m["tps"]))
            rep.trained_tokens = max(rep.trained_tokens, int(m["tokens"]))
            rep.peak_memory_gb = max(rep.peak_memory_gb, float(m["mem"]))
            rep.final_iter = max(rep.final_iter, iter_n)
            continue
        m = _VAL_LINE_RE.search(line)
        if m:
            iter_n = int(m["iter"])
            rep.validation_loss_series.append((iter_n, float(m["loss"])))
    return rep


def overfit_verdict(rep: TrainReport) -> str:
    """One-line heuristic diagnosis of the train/validation gap."""
    if not rep.train_loss_series or not rep.validation_loss_series:
        return "insufficient data — need both train + validation measurements to judge fit"

    last_train = rep.train_loss_series[-1][1]
    last_validation = rep.validation_loss_series[-1][1]
    min_validation = min(v for _, v in rep.validation_loss_series)
    gap = last_validation - last_train

    if last_train < 0.3 and last_validation > min_validation * 1.2:
        return (
            f"deep memorization. final train loss {last_train:.2f} (collapsed); "
            f"validation loss climbed from min {min_validation:.2f} to "
            f"{last_validation:.2f} (+{(last_validation/min_validation - 1)*100:.0f}%). "
            f"train-validation gap {gap:.2f}. Desired regime for stylistic-"
            "character fine-tunes; not the regime for new-knowledge fine-tunes."
        )
    if last_train > 0.7:
        return (
            f"underfit. final train loss {last_train:.2f} suggests the model "
            "has not fully learned the corpus pattern; consider more iters or "
            "a higher learning rate."
        )
    if last_validation <= min_validation * 1.05:
        return (
            f"healthy fit. validation loss held near its minimum "
            f"({min_validation:.2f}); train-validation gap {gap:.2f}."
        )
    return (
        f"mild overfit. final train {last_train:.2f}, final validation "
        f"{last_validation:.2f}, gap {gap:.2f}."
    )


def format_report(rep: TrainReport) -> str:
    """Render a TrainReport as a terminal-friendly multi-line string."""
    lines: list[str] = []

    lines.append(f"TRAINING RUN  {rep.log_path}")
    lines.append(f"  final iter:           {rep.final_iter}")
    lines.append(
        f"  trained tokens:       {rep.trained_tokens:,}  "
        "(mlx-lm counter, all token types)"
    )
    lines.append(f"  peak memory:          {rep.peak_memory_gb:.1f} GB")
    lines.append("")

    if rep.iter_per_sec_series:
        ips_med = statistics.median(rep.iter_per_sec_series)
        ips_lo = min(rep.iter_per_sec_series)
        ips_hi = max(rep.iter_per_sec_series)
        tps_med = statistics.median(rep.tokens_per_sec_series)
        lines.append("THROUGHPUT")
        lines.append(
            f"  iterations per second:  {ips_med:.3f} median   "
            f"range {ips_lo:.3f}-{ips_hi:.3f}"
        )
        lines.append(f"  tokens per second:      {tps_med:.0f} median")
        lines.append("")

    if rep.train_loss_series:
        first_n = [v for _, v in rep.train_loss_series[:10]]
        last_n = [v for _, v in rep.train_loss_series[-10:]]
        min_iter, min_loss = min(rep.train_loss_series, key=lambda x: x[1])
        lines.append("TRAIN LOSS  (reported every 10 iters)")
        lines.append(f"  first 10 reports avg:  {statistics.mean(first_n):.3f}")
        lines.append(f"  last  10 reports avg:  {statistics.mean(last_n):.3f}")
        lines.append(f"  min observed:          {min_loss:.3f}  at iter {min_iter}")
        lines.append("")

    if rep.validation_loss_series:
        lines.append("VALIDATION LOSS  (reported every 200 iters)")
        for iter_n, loss in rep.validation_loss_series:
            lines.append(f"  iter {iter_n:>5d}:  {loss:.3f}")
        lines.append("")

    lines.append(f"VERDICT  {overfit_verdict(rep)}")
    return "\n".join(lines)
