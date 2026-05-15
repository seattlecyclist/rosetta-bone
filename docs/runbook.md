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

## Remote training (RunPod)

### One-time setup

1. **Container image.** Build and push the trainer image:

   ```sh
   TAG=$(date -u +%Y.%m.%d)
   docker build --build-arg IMAGE_TAG=$TAG \
       -t ghcr.io/seattlecyclist/rosetta-bone-trainer:$TAG \
       docker/runpod-trainer
   docker push ghcr.io/seattlecyclist/rosetta-bone-trainer:$TAG
   ```

   Update `train.remote.image` in `config/default.toml` to the new tag.

2. **R2 bucket.** Create a bucket (one is enough — `datasets/` and
   `adapters/` are sibling prefixes). Generate an API token with
   *Object Read & Write* scope. Put the keys in `.env`.

3. **RunPod API key.** [RunPod settings →
   API Keys](https://www.runpod.io/console/user/settings). Put in
   `.env` as `RUNPOD_API_KEY`.

4. **HF token.** Llama-3.1-8B is gated; the pod needs `HF_TOKEN`
   to download it. Already in `.env.example`.

### Operator answers

**I cancelled mid-run, now what?** The pod self-terminates on the
watchdog (default 29 min). Storage is content-addressed by
`(train_sha + valid_sha + base_model + hyperparams)`, so re-running
`train --remote` with the same inputs is safe: if the previous run
finished, you get a free download; if it didn't, you get a fresh pod.

**I want to force a re-train of an existing key.** Either change a
hyperparameter (any change re-keys), or delete the prefix manually:
`aws s3 rm --recursive s3://rosetta-bone/adapters/<key>/ --endpoint-url=...`.

**A pod is stuck and burning money.** `runpod.terminate_pod(pod_id)`
in a quick Python REPL, or use the RunPod console. The pod id is
in `metadata.json` of the in-flight versioned adapter dir (it's
written only on success, so a stuck run won't have one yet —
check the RunPod dashboard).

**The conversion step failed but training succeeded.** The PEFT
adapter is safe in R2 under `adapters/<key>/peft/`. Re-running the
same `train --remote` invocation short-circuits past training and
retries only the convert step.

**Rotate R2 keys.** Generate new tokens in the Cloudflare dashboard,
update `.env`, revoke the old token. No code change needed; the
client reads from env on every invocation.

**Bumping the trainer image.** Bump the CalVer tag, build + push,
update `train.remote.image`, and add a line to `docs/pilot-history.md`
noting which image produced which adapter version. The image tag is
also baked into `trainer_info.json` inside the produced adapter.
