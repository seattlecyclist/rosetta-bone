# Runbook

Operational tasks for diagnosing and inspecting training runs.
Each entry: when to use it, the command, and what to look for in
the output.

---

## `train-inspect` — read a training log

**When to use:** after any `train` run, to check whether the run
converged, overfit, or underfit — without re-reading the raw
mlx-lm log by hand (or burning AI tokens parsing it every time).

Every `train` invocation tees mlx-lm's stdout to
`<adapter_dir>/train.log`. `train-inspect` parses that log and
renders a fixed report.

```sh
# Latest adapter (resolved through data/adapters/<run>/latest):
uv run rosetta-storyteller train-inspect

# Specific adapter by timestamp:
uv run rosetta-storyteller train-inspect --adapter 20260513T010203Z

# Arbitrary log file (e.g., from a workflow script that wrote
# its log somewhere other than the adapter dir):
uv run rosetta-storyteller train-inspect --log-file /tmp/v8.log
```

### What the report contains

```
TRAINING RUN  <log path>
  final iter:           <int>
  trained tokens:       <int>  (mlx-lm counter, all token types)
  peak memory:          <float> GB

THROUGHPUT
  iterations per second:  <median>  range <min>-<max>
  tokens per second:      <median>

TRAIN LOSS  (reported every 10 iters)
  first 10 reports avg:  <float>
  last  10 reports avg:  <float>
  min observed:          <float>  at iter <int>

VALIDATION LOSS  (reported every 200 iters)
  iter     1:  <float>
  iter   200:  <float>
  ...

VERDICT  <one-line heuristic>
```

Train and validation loss are summarized differently on purpose:
mlx-lm reports train loss every 10 iters but validation only every
200 iters, so first/last-10 averaging is meaningful for train but
not for validation. The validation series is shown in full.

The report uses **"validation loss"** throughout. mlx-lm itself
abbreviates to "Val loss" in its stdout; the analyzer translates.

### Verdict heuristics

| Condition                                                | Verdict                |
| -------------------------------------------------------- | ---------------------- |
| `last_train < 0.3` AND `last_val > min_val * 1.2`        | **deep memorization**  |
| `last_train > 0.7`                                       | **underfit**           |
| `last_val <= min_val * 1.05`                             | **healthy fit**        |
| anything else                                            | **mild overfit**       |

"Deep memorization" is the **desired** regime for stylistic-
character fine-tunes like the dog storyteller — we want the model
to over-memorize the prose pattern, not generalize. It would be
the **wrong** regime for a fact-injection fine-tune.

The same report is auto-printed at the end of every successful
`train` run, so usually you don't need to invoke `train-inspect`
manually — only when you want to revisit an old run.
