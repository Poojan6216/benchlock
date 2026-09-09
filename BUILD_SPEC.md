# BUILD SPEC — Benchlock
### Your eval score dropped. Benchlock tells you whether your system regressed or the judge changed underneath you — with a guarantee that survives looking every day.

---

## HOW TO USE THIS FILE (read first, agent)

You are building this project end to end. Work through the phases in order. Do not skip ahead, do not ask which phase to start with, do not stop to ask for approval between tasks.

**Your working loop:**

1. Read the phase you're on. Read every task in it.
2. Implement each task in order.
3. After each task, run its **Verify** step. If it fails, fix it before moving on.
4. Tick the checkbox in the task list and append one line to the **Progress Log** at the bottom of this file.
5. When a phase's **Phase Gate** passes, move to the next phase.
6. When all phases are done, write the **Final Report** section and stop.

**Rules for the whole build:**

- Do not ask for permission to continue. Keep going until every phase is complete.
- If a decision isn't specified here, pick the simplest option that satisfies the Hard Rules, and log the decision in the Progress Log. Don't stall on it.
- If you get genuinely blocked (an API doesn't exist, a dependency is broken, a dataset won't download), write the blocker in the Progress Log, implement the closest working alternative, and continue. Don't halt the whole build over one task.
- Commit after each completed task. Conventional Commits.
- Every phase must leave the tool in a working, installable state. Never end a phase with a broken build.
- **Numbers in this file that come from published research are marked `[cited]`. Never invent a number. Any figure that appears in the README or `RESULTS.md` must come from a run you actually executed, and the command that produced it must be in the repo.** This project's entire value is that its numbers are real. One fabricated number destroys all of it.
- **Statistical claims are held to the same standard as benchmark numbers.** If you claim a test is anytime-valid, there must be a simulation in the repo that demonstrates the false-alarm rate is bounded. Do not assert a guarantee you have not empirically checked.

**Do not build:** a SaaS control plane, a hosted dashboard, user accounts, telemetry, a "Pro" tier, a Kubernetes operator, an LLM that decides verdicts, a new eval framework, a new judge model, or a general-purpose observability platform. If you find yourself writing a login page, a billing table, or a prompt that asks a model "did the system regress?", stop — you've misread the spec.

---

## 1. What this is

Teams run LLM-as-judge evaluations in CI. A model or prompt changes, the suite runs, a number comes out. The number moves. And then three questions arrive that nobody can currently answer:

1. **Is the move real, or is it noise?** You have 200 samples and a nondeterministic judge behind an HTTP API.
2. **If it's real, what moved — your system, or the judge?** The judge is itself a hosted model. Providers rotate snapshots, change safety post-processing, and adjust defaults. Your rubric prompt gets edited by a teammate. Any of these moves the score with zero change to the thing you are trying to measure.
3. **You have looked at this dashboard every day for forty days. What is your actual false-alarm rate?**

Current practice answers none of them. Teams eyeball deltas, set arbitrary thresholds ("alert if it drops more than five points"), or — the sophisticated-looking option — run a fixed-α t-test at every CI run. That last one is worse than the eyeball, because it looks principled and is not: a test with α = 0.05 re-run at every accumulating observation has no type-I error control at all. Under optional stopping the false-alarm probability climbs toward 1. Peeking is not a misuse of the tool here; peeking is the entire workflow. CI runs on every commit.

The consequence is expensive in both directions. A judge that silently drifts produces a phantom regression, and the team rolls back a healthy release, or spends a week bisecting a change that was never there. A real regression arrives during a period of judge noise and gets waved off as "the judge being weird again," and ships.

**The gap this project fills.** The problem has been named — one 2026 preprint formalises it (§3) — and the statistical machinery to solve it has existed in the sequential-analysis literature for years and is pip-installable today. But no tool combines them. Every LLM drift tool on PyPI *detects* movement; none *attributes* it. Every eval framework compares numbers; none is valid under repeated looking. The primitive that a working AI engineer needs — "your score moved; here is which of the two things moved, and here is why you can believe that after two hundred peeks" — does not exist as an installable thing.

Benchlock is that thing. It sits beside your existing eval pipeline, ingests the score stream you already produce, and returns a verdict from a pure function with no model in it.

**The mechanism, in three lines.** Hold a frozen anchor set that the system under test never touches, and have the judge re-score it on every run.

- The anchor score moved ⇒ **only the judge can have moved it.**
- The system score moved and the anchor score did not ⇒ **the system moved.**
- Both moved ⇒ **confounded**, and Benchlock says so rather than picking.

That is a difference-in-differences design where the anchor set is the control group and the judge is the shared time effect. It is not clever. It is the obvious thing, and it works precisely because it is deterministic and requires no model to adjudicate it. Wrapping it in anytime-valid sequential tests is what makes the verdict survive daily peeking.

**Before writing any README copy, read §3.** A preprint got here first and deserves prominent credit. Our contribution is narrower and more specific than "we solved judge drift," and §3 states exactly how narrow.

---

## 2. The promise: the dashboard says roll back, and it is wrong

Three demos. All three must work end to end, all three go in the README, and all three are generated by `uv run python bench/demo.py` with every line captured from a real run.

### Demo 1 — the phantom regression

A team's eval pass rate falls from 0.83 to 0.71 over two weeks. Their dashboard flags a regression. The on-call engineer starts bisecting.

```text
benchlock verdict

verdict=judge          confidence: anytime-valid at alpha=0.05
  E_anchor  = 41.7   (threshold 20.0, crossed at run 118)
  E_system  =  8.3   (threshold 20.0, not crossed)

- the anchor set moved -0.118  [CS: -0.149, -0.087]
  the system under test does not touch the anchor set; only the judge can move it
- your system moved -0.121 raw; anchor-corrected -0.003  [CS: -0.041, +0.035]
- anchor provisioning: ADEQUATE
  minimum detectable judge shift at this anchor size = 0.021 < observed 0.118
- baseline judge: claude-sonnet-4-5-20250929
  current judge:  claude-sonnet-4-5-20260114   (declared config changed at run 116)

DO NOT roll back. Re-baseline against the new judge:
  benchlock rebaseline --reason judge-version-change
```

Ground truth: the provider rotated the judge snapshot. The system under test was byte-identical throughout. The rollback would have been pure waste.

### Demo 2 — the real regression, correctly named

Same pipeline. This time the system's retrieval top-k was cut from 8 to 4 in a "cleanup" PR.

```text
verdict=system         confidence: anytime-valid at alpha=0.05
  E_anchor  =  1.2   (threshold 20.0, not crossed)
  E_system  = 63.9   (threshold 20.0, crossed at run 47)

- the anchor set is stable: -0.004  [CS: -0.021, +0.013]
- your system moved -0.089  [CS: -0.117, -0.061]
- first run outside the baseline confidence sequence: run 44
- anchor provisioning: ADEQUATE

CI gate: FAIL (exit 1)
```

### Demo 3 — the honest one, which is the point of the project

The same real regression, but the anchor set was configured at 40 items instead of the 260 that `benchlock plan` asked for.

```text
verdict=indeterminate  confidence: anytime-valid at alpha=0.05
  E_anchor  =  0.9   (threshold 20.0, not crossed)
  E_system  = 63.9   (threshold 20.0, crossed at run 47)

- your system score moved, and the anchor set did not.
- BUT the anchor set is under-provisioned: at n=40 the minimum judge shift this
  anchor process could have detected by run 47 is 0.094. The observed system
  move is 0.089. A judge shift of that size would be INVISIBLE here.
- Benchlock will not attribute this to your system. It cannot rule out the judge.

Fix the provisioning, then this becomes decidable:
  benchlock plan --target-shift 0.05    ->  anchor n>=260, cadence every run
```

Most tools would say "regression" and be right by luck. Benchlock refuses, tells you why, and tells you exactly what to change. **A tool that is confidently wrong 10% of the time is worse than useless in a CI gate, because people learn to ignore it.** Demo 3 is the one to lead the write-up with.

---

## 3. Prior art — who is already here, and exactly where they stop

Read this before designing anything. The eval-tooling space is crowded and the sequential-testing literature is deep and old. Building "another drift detector" is worthless. Knowing precisely where each neighbour stops is the reason this project is interesting.

**The direct predecessor.** *"Who Drifted: the System or the Judge? Anytime-Valid Attribution in LLM Evaluation Pipelines"* (arXiv 2606.15474, June 2026) states the problem, proposes the anchor-set + e-process construction, and reports a three-way verdict. It is the intellectual source for this project and must be credited in the README's first prior-art paragraph, by name, in the first sentence. `[cited]`

*Where it stops:* it is a solo-authored, unaffiliated preprint with no public code, no released implementation, and no third-party reproduction. Its anchor sets are **human-labeled**, which is a multi-week adoption cost for any real team. It returns three verdicts and has no state for "I cannot tell." It gives no method for sizing the anchor set, which is the difference between the guarantee being real and being decorative.

**Treat reproduction as part of the contribution, not as a given.** If its numbers do not reproduce, that is a publishable finding and it goes in `RESULTS.md` plainly. Do not tune your implementation until it agrees with an unverified preprint.

**Anytime-valid inference / e-values.** This is the load-bearing mathematics and it is not ours. Ville's inequality (1939) is the whole guarantee. Howard, Ramdas, McAuliffe & Sekhon gave time-uniform Chernoff bounds and confidence sequences. Waudby-Smith & Ramdas, *Estimating means of bounded random variables by betting*, gives the betting construction we use. Grünwald, de Heide & Koolen's safe testing frames e-values as the right object. Shin, Ramdas & Rinaldo's **e-detectors** are the correct primitive for sequential *change* detection built from e-processes — that is what the monitoring layer is. The `confseq` package (Howard/Waudby-Smith/Ramdas) is on PyPI and is our test oracle.
*Where it stops:* none of it knows what an LLM judge is. The contribution is not the mathematics; it is the identification argument that makes the mathematics answer a question AI engineers actually have, plus the provisioning rule that makes it deployable.

**Prediction-powered inference** (Angelopoulos et al., *Science* 2023) is the closest statistical cousin: valid inference from many cheap model labels plus a few gold ones. Our `human` anchor mode is essentially PPI applied to judge scores. Credit it; consider using it for the corrected effect-size estimate in that mode.

**LLM drift tools on PyPI.** `llm-drift`, `driftmonitor-ai`, `dedrift`.
*Where they stop:* embedding and output diffing, and classical fixed-sample tests. `dedrift` explicitly paywalls its e-process layer — its own repo says anytime-valid sequential inference is in a separate commercial tier and not in the open-source package `[cited]`. **None of them attribute.** They tell you a number moved. That is the easy half.

**Eval and observability platforms.** promptfoo (acquired by OpenAI, March 2026 `[cited]`), DeepEval, Braintrust, LangSmith, Arize Phoenix, Langfuse, W&B Weave, Evidently, RAGAS, Inspect AI.
*Where they stop:* they compute, store and compare scores, and several have "regression" alerts driven by fixed thresholds or fixed-α tests. **None is valid under repeated looking, and none has a control group.** They are the substrate Benchlock plugs into, not competitors — build adapters for them (Phase 8) and say so.

**Classical change detection.** Page's CUSUM, ADWIN (Bifet & Gavaldà), DDM/EDDM, available in `river`.
*Where they stop:* excellent single-stream change detectors with no notion of *which of two causes* produced the change, and mostly without time-uniform error control in the form we need. They are baselines in Phase 6, not prior art we are extending.

**LLM-as-judge validity literature.** Zheng et al. (MT-Bench / Chatbot Arena) on position and verbosity bias; Guerdan et al. (NeurIPS 2025) on rating indeterminacy; the broad self-inconsistency literature.
*Where they stop:* they establish that judges are biased and unstable — which is the premise of this project — and then propose better judges. Benchlock takes judge instability as a given and makes it *attributable* instead of trying to remove it.

**Benchlock's actual contribution, in three sentences.** It is the first installable implementation of judge-versus-system attribution for LLM evaluation. It removes the human-labeling requirement by observing that attribution needs judge *stability*, not judge *validity*, and therefore that a frozen snapshot of the judge's own scores is a sufficient control. And it adds the provisioning rule and the `indeterminate` verdict, which together are the difference between a guarantee and a decoration — a tool that says `system` when its anchor set was too small to rule out the judge is guessing with a straight face.

**Naming.** A *bench mark* is a surveyor's fixed reference, historically a mark cut into stone from which all other measurements are taken. Benchlock cuts one into your eval pipeline and locks it, so that when a number moves you can tell whether the wall shifted or the ruler did. Use this in the README; it is one sentence and it explains the entire product.

---

## 4. Hard rules — never violate these

These are not preferences. Breaking any one makes the tool dishonest, which for a measurement tool is the same as making it useless.

1. **No model in the decision path.** The judge produces numbers. `decide()` consumes numbers and returns a verdict using arithmetic. A verdict must never be produced, adjusted, explained-away or softened by asking an LLM. There is no prompt anywhere in `src/benchlock/stats/` or `src/benchlock/attribute/`.
2. **Refuse to attribute rather than guess.** `INDETERMINATE` is a first-class verdict and must fire whenever the anchor process lacked the power to have detected a judge shift of the observed magnitude. Never emit `SYSTEM` to look decisive. This rule will cost you verdicts in the benchmark; that cost is the honest number and it gets published.
3. **Anytime-validity is non-negotiable.** Every test that gates a verdict must be valid under optional stopping. No fixed-α test, no bootstrap CI recomputed each run, no "we corrected with Bonferroni over the runs so far" may ever produce a verdict. Fixed-sample tests appear in this repo **only** as measured baselines in Phase 6.
4. **Conservative truncation.** The e-detector retains a bounded set of change-point hypotheses. Pruning it may only *delay* detection; it may never *create* an alarm that a full-memory detector would not have raised. There is a property test for this and it must never be skipped.
5. **Never report a number you didn't measure.** Every figure in `README.md` and `RESULTS.md` traces to a committed command and a committed JSON file. `RESULTS.md` is generated. It is never hand-edited.
6. **Never claim causality beyond the identification argument.** Benchlock separates judge from system *given a fixed eval suite*. If the eval inputs are sampled from shifting production traffic, a third cause exists that Benchlock does not model, and the tool must detect that configuration and warn loudly. Do not let the README imply otherwise.
7. **Determinism and replay.** Same ledger + same config = same verdicts, forever. No clock, no randomness, no network, no I/O inside `decide()`. `benchlock replay` re-derives every historical verdict and a mismatch is a build failure.
8. **Never mutate a baseline silently.** Anchor sets, judge configuration and rubric text are hashed and pinned. A change to any of them invalidates the comparison. Rebaselining is an explicit command, logged, versioned, and visible in every subsequent verdict. If a pinned hash changes without a rebaseline, that is an error, not a warning.
9. **Never store raw eval content.** The ledger holds scores, hashes, shapes and metadata. Eval suites routinely contain customer data. There must be a test that seeds recognisable secrets into an eval run and asserts none appear in the ledger.
10. **Fail loud on assumption violation.** Unbounded or undeclared score ranges, a changed anchor hash, a judge config that moved without being declared, fewer observations than the configured minimum — all raise, none silently continue. A statistical tool that quietly runs outside its assumptions is worse than no tool.
11. **No telemetry, no hosted components, no accounts, no network calls we own.** API calls go to the user's chosen judge provider and nowhere else.

---

## 5. Locked technical decisions

Don't re-litigate these.

| Decision | Choice |
|---|---|
| Language | Python 3.12+, `from __future__ import annotations`, full type hints |
| Typing | mypy `--strict` on `src/`, no `Any` in public signatures |
| Packaging | `uv`, `pyproject.toml`, hatchling backend |
| Numerics | `numpy` + `scipy`. **No torch, no GPU, anywhere, ever** |
| Stats core | Implemented in-house in `stats/`. Cross-validated against `confseq` as a test oracle |
| Test oracle | `confseq` (Howard/Waudby-Smith/Ramdas), dev dependency only |
| Baselines | `river` for ADWIN/DDM, `scipy.stats` for t-test/CUSUM. Dev dependency only |
| Config | YAML validated by Pydantic v2. Schema versioned |
| Ledger | Append-only JSONL, hash-chained (each record carries SHA-256 of the previous) |
| Lint/format | `ruff` (lint + format), line length 100 |
| Tests | `pytest`, `pytest-asyncio`, `hypothesis` for the stats and attribution engines |
| Judge providers | Provider-agnostic adapter. Anthropic first, OpenAI second. Adapter interface is 4 methods |
| Plots | `matplotlib` only, no seaborn, no plotly. Committed as PNG + the script that made them |
| License | Apache-2.0 |
| Package name | `benchlock`. CLI: `benchlock`. Fall back to `benchlock-eval` only if squatted before publish |
| Config location | `./benchlock.yaml`, then `$XDG_CONFIG_HOME/benchlock/config.yaml` |
| Ledger location | `./.benchlock/ledger.jsonl` |

**Why we implement the e-processes rather than only calling `confseq`:** we need specific composite hypotheses (a baseline mean known only up to a confidence sequence), a change-point-robust detector layer, and a bounded-memory pruning rule with a conservativeness guarantee. None of that is a `confseq` call. But `confseq`'s Hedged and empirical-Bernstein confidence sequences are a correct, peer-reviewed implementation of the underlying primitive, so every construction we write is checked against it in tests. Writing the core and validating against the reference is the right split; writing the core and validating against nothing is not.

**Why no GPU and no fine-tuning:** the decision path is arithmetic on scalars. The only model calls in the whole project are Phase 7's judge calls to a hosted API, and those produce data, not decisions.

**Why a fixed suite is required, not optional:** see Hard Rule 6. If the inputs move, score movement has a third cause and the identification argument collapses. The tool must check for this and refuse to run in attribution mode against a stream whose item-set hash changes between runs.

---

## 6. Project structure

Create exactly this. Don't reorganise.

```
benchlock/
├── pyproject.toml
├── README.md
├── LICENSE                          # Apache-2.0
├── CHANGELOG.md
├── BUILD_SPEC.md                    # this file
├── RESULTS.md                       # generated by Phases 6/7/8, never hand-written
├── benchlock.yaml.example
├── src/benchlock/
│   ├── __init__.py
│   ├── cli.py                       # init plan baseline observe verdict gate replay rebaseline report
│   ├── config.py                    # Pydantic v2 models, schema version
│   ├── model/
│   │   ├── streams.py               # Observation, RunRecord, StreamId
│   │   ├── verdict.py               # Verdict, Attribution, Evidence
│   │   └── pins.py                  # AnchorPin, JudgePin — hashing and comparison
│   ├── stats/
│   │   ├── betting.py               # wealth process, betting strategies
│   │   ├── eprocess.py              # EProcess: composite-null bounded-mean test
│   │   ├── edetector.py             # change detection, bounded memory, conservative pruning
│   │   ├── confseq.py               # our confidence sequence (checked against `confseq`)
│   │   └── power.py                 # minimum detectable effect, provisioning calculator
│   ├── attribute/
│   │   ├── engine.py                # decide() — PURE. the heart of the project.
│   │   ├── lattice.py               # the 5 verdicts and their preconditions
│   │   └── race.py                  # the anchor race condition / provisioning check
│   ├── anchor/
│   │   ├── modes.py                 # frozen-self | human | replicate
│   │   ├── select.py                # stratified anchor selection from a suite
│   │   ├── coverage.py              # anchor coverage measurement + warnings
│   │   └── noisefloor.py            # K-replicate within-judge variance estimation
│   ├── ledger/
│   │   ├── log.py                   # append-only, hash-chained JSONL
│   │   └── replay.py                # re-derive every historical verdict
│   ├── judge/
│   │   ├── base.py                  # provider adapter interface (4 methods)
│   │   ├── anthropic.py
│   │   ├── openai.py
│   │   └── cache.py                 # local cache + cache-busting nonce (see 7.9)
│   ├── adapters/                    # ingest from existing eval frameworks
│   │   ├── jsonl.py                 # the universal one. scores in, that's it.
│   │   ├── promptfoo.py
│   │   ├── inspect_ai.py
│   │   └── deepeval.py
│   └── report/
│       ├── human.py                 # the verdict block in §2
│       └── markdown.py              # PR comment / RESULTS.md fragments
├── bench/
│   ├── demo.py                      # generates docs/demo.md — the §2 three demos
│   ├── sim/                         # Tier 1: simulation study
│   │   ├── generate.py              # ground-truth stream generator
│   │   ├── baselines.py             # B0..B5
│   │   └── run_sim.py
│   ├── real/                        # Tier 2: real judges
│   │   ├── build_pool.py            # score pool construction
│   │   ├── compose.py               # stream construction from the pool
│   │   └── run_real.py
│   ├── adversarial/                 # Tier 3: break your own attributor
│   │   └── strategies/              # one file per strategy, see Phase 7
│   ├── results/                     # committed JSON. the source of every number
│   └── plots/                       # committed PNG + the script that made each
├── docs/
│   ├── demo.md                      # captured transcript, generated
│   ├── identification.md            # the identification argument, written out properly
│   ├── statistical-guarantees.md    # what is guaranteed, what is not, and the proofs we rely on
│   ├── provisioning.md              # how to size your anchor set
│   ├── threat-model.md              # what breaks this
│   └── writeup.md
├── tests/
│   ├── fixtures/
│   │   ├── streams/                 # golden streams with known ground truth
│   │   └── secrets/seeded.json      # Hard Rule 9
│   └── ...
└── .github/workflows/ci.yml
```

---

## 7. Core contracts

Define these before writing implementations so nothing drifts.

```python
# src/benchlock/model/streams.py


class StreamKind(StrEnum):
    SYSTEM = "system"  # scores of the system under test
    ANCHOR = "anchor"  # scores of the frozen anchor set. the system never touches these.


@dataclass(frozen=True, slots=True)
class Observation:
    """One judged item. Scores are normalised to [0, 1] at ingest; the original
    scale and bounds are recorded so the normalisation is reversible and auditable."""

    item_id: str  # stable id of the eval item
    score: float  # in [0, 1]
    raw_score: float
    scale: tuple[float, float]


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One CI run. This is the unit of the stream."""

    run_id: str  # ULID
    run_index: int  # monotonic, 0-based
    kind: StreamKind
    observations: tuple[Observation, ...]
    suite_hash: str  # SHA-256 of the sorted item_ids. Hard Rule 6 checks this.
    judge_pin: JudgePin
    anchor_pin: AnchorPin | None  # None for SYSTEM streams
```

```python
# src/benchlock/model/pins.py


@dataclass(frozen=True, slots=True)
class JudgePin:
    """Everything about the judge that, if changed, invalidates comparison.
    Hard Rule 8: a change here without an explicit rebaseline is an ERROR."""

    provider: str
    model: str  # the exact dated snapshot where the provider offers one
    rubric_hash: str  # SHA-256 of the full rubric/system prompt
    params_hash: str  # temperature, top_p, max_tokens, response format, seed
    scale: tuple[float, float]

    def differs_from(self, other: JudgePin) -> tuple[str, ...]:
        ...
        # returns the names of the fields that changed, for the verdict message


@dataclass(frozen=True, slots=True)
class AnchorPin:
    mode: AnchorMode  # FROZEN_SELF | HUMAN | REPLICATE
    item_set_hash: str  # SHA-256 of sorted anchor item_ids
    baseline_scores_hash: str
    n: int
    noise_floor: NoiseFloor  # from K replicates at baseline
```

```python
# src/benchlock/model/verdict.py


class Verdict(StrEnum):
    STABLE = "stable"  # neither process has crossed
    JUDGE = "judge"  # the judge moved
    SYSTEM = "system"  # the system moved, and the judge demonstrably did not
    BOTH = "both"  # both moved; anchor correction does not explain the system move
    INDETERMINATE = "indeterminate"  # cannot rule out the judge. Hard Rule 2.


@dataclass(frozen=True, slots=True)
class Evidence:
    e_system: float  # e-detector value for the system stream
    e_anchor: float  # e-detector value for the anchor stream
    threshold: float  # 1/alpha
    system_shift: Interval  # confidence sequence, anytime-valid
    anchor_shift: Interval
    corrected_shift: Interval  # system shift with the anchor shift removed
    crossed_at: dict[StreamKind, int | None]
    min_detectable_judge_shift: float  # what the anchor process could have caught by now
    provisioning: Provisioning  # ADEQUATE | UNDER_PROVISIONED
    judge_pin_delta: tuple[str, ...]  # which judge fields changed, if declared


@dataclass(frozen=True, slots=True)
class Attribution:
    verdict: Verdict
    rule_id: str  # which lattice rule fired. never empty.
    reasons: tuple[str, ...]  # human-readable, shown in the verdict block
    evidence: Evidence
    alpha: float
```

```python
# src/benchlock/attribute/engine.py


def decide(
    system: Sequence[RunRecord],
    anchor: Sequence[RunRecord],
    config: AttributionConfig,
) -> Attribution:
    """Pure function. No I/O, no clock, no randomness, no network, no LLM.

    Hard Rule 7: same (system, anchor, config) MUST yield the same Attribution,
    forever. This is what makes `benchlock replay` possible.

    Hard Rule 2: if the anchor process could not have detected a judge shift
    of the magnitude observed in the system stream, the verdict is
    INDETERMINATE. Never SYSTEM.
    """
```

### The verdict lattice

Evaluated in order, first match wins, every branch names its `rule_id`. `E_x` is the e-detector value for stream `x`; the threshold is `1/alpha`.

| order | rule_id | condition | verdict |
|---|---|---|---|
| 1 | `pin_violation` | `judge_pin` or `anchor_pin` changed without a logged rebaseline | **raise**, not a verdict (Hard Rule 8) |
| 2 | `suite_drift` | `suite_hash` differs between runs in the system stream | **raise** (Hard Rule 6) |
| 3 | `insufficient_data` | fewer runs than `config.min_runs`, or fewer observations than `config.min_obs` | `STABLE` with a `low_power` reason |
| 4 | `both_crossed` | `E_anchor ≥ thr` and `E_system ≥ thr` and the anchor-corrected system interval excludes 0 | `BOTH` |
| 5 | `judge_only` | `E_anchor ≥ thr` (whether or not `E_system` crossed, provided the corrected interval contains 0) | `JUDGE` |
| 6 | `race_failed` | `E_system ≥ thr`, `E_anchor < thr`, and `min_detectable_judge_shift ≥ \|system_shift\|` | `INDETERMINATE` |
| 7 | `system_only` | `E_system ≥ thr`, `E_anchor < thr`, and `min_detectable_judge_shift < \|system_shift\|` | `SYSTEM` |
| 8 | `no_crossing` | neither crossed | `STABLE` |

Rule 6 before rule 7 is the entire ethical content of the tool. Do not reorder them.

### The anchor race — state this as a design law

> **Design law.** The anchor process must be provisioned to reach its threshold no later than the system process would, for any judge shift large enough to move the system stream detectably.

If it isn't, a judge shift moves the system stream before the anchor stream can prove it, and a naive implementation returns `SYSTEM` — a confident, wrong rollback recommendation. Rule 6 is the runtime enforcement of this law; `benchlock plan` is the design-time enforcement.

```python
# src/benchlock/stats/power.py


def min_anchor_size(
    target_shift: float,  # smallest judge shift we must be able to attribute
    noise_floor: NoiseFloor,  # measured, not assumed
    alpha: float,
    horizon: int,  # runs within which detection must occur
    obs_per_run: int,
) -> int:
    """Minimum anchor set size satisfying the design law. Conservative:
    when the numeric search is ambiguous, return the LARGER n."""
```

### Configuration

```yaml
# benchlock.yaml
version: 1
alpha: 0.05
score_scale: [1, 5]          # declared, required. Hard Rule 10.
min_runs: 8
min_obs: 30

system:
  adapter: promptfoo          # jsonl | promptfoo | inspect_ai | deepeval
  path: ./evals/results/

anchor:
  mode: frozen-self           # frozen-self (default) | human | replicate
  n: 260                      # from `benchlock plan`
  cadence: 1                  # re-score anchors every N runs
  selection: stratified       # stratified over score decile and item tag
  noise_replicates: 5         # K, measured once at baseline

judge:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  rubric: ./evals/rubric.md
  params: { temperature: 0.0, max_tokens: 512 }

gate:
  fail_on: [system, both]     # `judge` does NOT fail your build. that is the point.
  warn_on: [indeterminate]
```

**`gate.fail_on` defaulting to `[system, both]` is the product in one line of YAML.** A judge change must not fail your build; it must tell you to re-baseline.

---

## PHASE 0 — Scaffold and a tool that decides nothing

**Goal:** an installable CLI that ingests a score stream and stores it faithfully. No statistics, no verdicts. This proves the plumbing before any inference exists.

- [x] **0.1 — Project scaffold**
  `uv init`, pyproject per §5, ruff + mypy strict, Apache-2.0 LICENSE, pytest layout, pre-commit running ruff and mypy.
  **Verify:** `uv run ruff check`, `uv run mypy --strict src/`, `uv run pytest` all pass on an empty suite.

- [x] **0.2 — Config model and CLI skeleton**
  `config.py` with the §7 Pydantic models, schema version, and clear line-numbered errors on invalid YAML. `cli.py` with all nine subcommands present (stubs beyond `init` and `observe`). Structured JSON logs to stderr.
  **Verify:** 12+ malformed configs each produce a specific, actionable error naming the field and line. Round-trip load → dump → load is stable.

- [x] **0.3 — The universal JSONL adapter**
  `adapters/jsonl.py`. The lowest-common-denominator ingest: a JSONL file of `{item_id, score}` plus run metadata. Every other adapter normalises into this. Score normalisation to `[0,1]` using the declared `score_scale`, with the raw value and scale retained. Reject undeclared or out-of-range scores loudly (Hard Rule 10).
  **Verify:** table-driven tests over binary scores, 1–5 Likert, 1–10, continuous [0,1], and four malformed cases each rejected with a specific message.

- [x] **0.4 — The hash-chained ledger**
  `ledger/log.py`. Append-only JSONL, each record carrying SHA-256 of the previous record. Records: run id, index, kind, suite hash, judge pin, anchor pin, per-item scores, and nothing else. Never raw prompts, outputs, or judge rationales (Hard Rule 9).
  **Verify:** tamper test — flipping one byte anywhere in the ledger breaks the chain and is detected with the record index reported. Secret-leak test — an eval run seeded with the 15 secret formats in `tests/fixtures/secrets/seeded.json` produces a ledger containing none of them. **This test is mandatory and must never be skipped or marked xfail.**

- [x] **0.5 — Pins and pin-violation detection**
  `model/pins.py`. Hash the judge config (provider, model string, rubric text, params) and the anchor set. On every `observe`, compare against the pinned values. A silent change raises with a message naming exactly which fields moved and telling the user to run `benchlock rebaseline`.
  **Verify:** 8 cases — model string changed, rubric whitespace-only change (must still trip; whitespace changes prompts), temperature changed, anchor item added, anchor item removed, anchor score edited, scale changed, nothing changed. Each produces the right pin delta.

- [x] **0.6 — `benchlock init`**
  Detect an existing eval framework in the working directory (promptfoo config, Inspect AI log dir, DeepEval config, or bare JSONL), write a `benchlock.yaml` pre-filled with what it found, and print the next command to run. Never overwrite an existing config without a backup.
  **Verify:** round-trip test over 4 project shapes. An existing `benchlock.yaml` is backed up byte-for-byte before any write.

**Phase Gate:** `benchlock init && benchlock observe run1.jsonl && benchlock observe run2.jsonl` builds a valid chained ledger from a real eval output, and a tampered ledger is rejected.

---

## PHASE 1 — The statistical core

**Goal:** correct, tested, anytime-valid primitives. This phase is where the guarantee either exists or doesn't. Everything downstream is bookkeeping on top of it.

- [x] **1.1 — The betting wealth process**
  `stats/betting.py`. Observations `X_t ∈ [0,1]`, null `E[X_t] = μ₀`. Wealth `K₀ = 1`, `K_t = K_{t-1} · (1 + λ_t (X_t − μ₀))` with `λ_t` **predictable** — computed from `X_1..X_{t-1}` only, never from `X_t`. Truncate `λ_t` to `[−c/(1−μ₀), c/μ₀]` with `c = 0.5` so wealth stays positive. Implement two strategies: `fixed` (constant λ, for tests) and `agrapa` (predictable plug-in, the default).
  **Verify:** hypothesis property tests over 2000+ random streams: (a) wealth is always strictly positive; (b) `λ_t` never reads `X_t` — enforce with a deliberately poisoned stream where reading the current value would be detectable; (c) under the null, the empirical mean of `K_t` at every fixed `t` is ≤ 1 + Monte-Carlo tolerance.

- [x] **1.2 — The e-process with a composite null**
  `stats/eprocess.py`. The baseline mean μ₀ is **not known** — it is estimated from a baseline period. Running against a point estimate silently inflates the false-alarm rate. Instead: compute an anytime-valid confidence sequence for μ₀ over the baseline period, then run the monitoring e-process against the **least favourable value in that interval** (the value closest to the observed drift direction). This is conservative by construction, which is the correct direction.
  **Verify:** simulation — 2000 drift-free streams with a baseline period, monitored to a horizon of 300. Empirical probability that the e-process ever crosses `1/α` must be ≤ α. Compare against the naive point-estimate version and record how much worse it is; that comparison goes in `docs/statistical-guarantees.md`.

- [x] **1.3 — Confidence sequences, validated against `confseq`**
  `stats/confseq.py`. Time-uniform interval for a bounded mean. Must produce intervals that hold simultaneously over all t.
  **Verify:** differential test against `confseq`'s hedged and empirical-Bernstein confidence sequences over 500 random streams. Our interval must **contain** the reference interval or be within a documented tolerance; if ours is ever *narrower* than the reference, that is a bug, and the test must fail on it. Log the width ratio distribution in `docs/statistical-guarantees.md`.

- [x] **1.4 — The e-detector (change detection, bounded memory)**
  `stats/edetector.py`. A change can start at any run, so the monitoring statistic is built over change-point hypotheses: maintain e-processes started at each candidate change point and combine (sum or max — implement sum, it has the cleaner guarantee). Memory is bounded: retain at most `M` candidates (default 256) with a pruning rule.
  **Pruning must be conservative (Hard Rule 4):** dropping a candidate may only delay detection, never cause an alarm that full memory would not have raised. Implement by pruning the *lowest-wealth* candidates and never re-weighting the survivors upward.
  **Verify:** hypothesis property test over 3000 cases — for every stream, the bounded detector's alarm time is ≥ the full-memory detector's alarm time, and the bounded detector never alarms where full memory does not. **A single counterexample is a build failure.**

- [x] **1.5 — The noise floor**
  `anchor/noisefloor.py`. Judges are nondeterministic even at temperature 0. Score the anchor set K times at baseline (default K=5) and estimate the within-judge variance and the distribution of run-to-run mean differences. All subsequent tests are against this floor, not against zero.
  **Verify:** a synthetic judge with known injected nondeterminism is characterised to within tolerance. A test asserts that a detector configured against a zero noise floor false-alarms and one configured against the measured floor does not — this is the test that proves the component earns its place.

- [x] **1.6 — The provisioning calculator**
  `stats/power.py::min_anchor_size` and `min_detectable_shift`. Given noise floor, α, horizon and observations per run, compute the minimum anchor size satisfying the design law, and the inverse: given an anchor size, the smallest judge shift detectable by run t. Conservative rounding — always the larger n, always the larger detectable shift.
  **Verify:** monotonicity property tests — n increases as target shift decreases, as noise increases, as α decreases, as horizon shortens. Then a simulation check: at the recommended n, the anchor process actually crosses before the system process in ≥ 95% of runs at the target shift. **If simulation disagrees with the formula, the formula is wrong; fix the formula, do not adjust the simulation.**

**Phase Gate:** `docs/statistical-guarantees.md` exists and contains measured false-alarm rates from our own simulations for every primitive, plus the `confseq` differential result. Not citations — our numbers.

---

## PHASE 2 — The attribution engine

**Goal:** the pure decision function. This is the heart of the project.

- [x] **2.1 — The lattice**
  `attribute/lattice.py`. The eight ordered rules from §7, first-match-wins, each naming its `rule_id`. Rules 1 and 2 raise rather than return a verdict.
  **Verify:** a golden-file suite of 40+ hand-constructed `(system stream, anchor stream, config) → expected verdict + rule_id` pairs covering every branch, including both raising branches and every boundary (exactly at threshold, exactly at the provisioning boundary).

- [x] **2.2 — The race condition**
  `attribute/race.py`. Compute `min_detectable_judge_shift` at the current run from the anchor stream's realised size and noise floor. This is the input to lattice rule 6.
  **Verify:** the demo-3 scenario from §2 — an under-provisioned anchor set with a real system regression — returns `INDETERMINATE`, and the same scenario with the recommended anchor size returns `SYSTEM`. Both from committed fixtures.

- [x] **2.3 — `decide()`**
  `attribute/engine.py`. Pure. No I/O, no clock, no randomness (Hard Rule 7). Assembles `Evidence`, applies the lattice, returns `Attribution` with human-readable reasons.
  **Verify:** hypothesis determinism test — 5000 random `(system, anchor, config)` triples each decided twice, always byte-identical output. Plus: `decide` is called with the network disabled and a frozen clock in CI, and the test suite fails if either is touched.

- [x] **2.4 — Anchor correction**
  The `corrected_shift` interval: the system shift with the anchor shift removed, with correctly widened uncertainty (the correction is itself estimated). Do not subtract point estimates and keep the original interval width — that is the single most likely place to accidentally manufacture a false guarantee.
  **Verify:** simulation — under simultaneous system and judge drift with known ground truth, the corrected interval must cover the true system shift at ≥ 1−α over 2000 streams. In `human` anchor mode, compare against a prediction-powered-inference estimator and report which is tighter.

- [x] **2.5 — The verdict block**
  `report/human.py`. The exact output format in §2, including the provisioning line, the pin-delta line, and the recommended next command. Every number shown must come from `Evidence`; the renderer computes nothing.
  **Verify:** golden-file test over all five verdicts. A test asserts the renderer performs no arithmetic (inspect for operators on `Evidence` fields, or assert output equality against pre-computed structs).

- [x] **2.6 — `benchlock gate`**
  Exit codes: 0 for `stable` and `judge`, 1 for `system` and `both`, 2 for `indeterminate` (configurable via `gate.fail_on`/`warn_on`). Prints the verdict block. Designed to be the last line of a CI job.
  **Verify:** integration test asserting each verdict maps to the right exit code under default and custom configs.

**Phase Gate:** the three §2 demos run end to end from committed fixture streams and produce exactly the output in §2. Determinism suite green.

---

## PHASE 3 — Anchors that cost nothing to adopt

**Goal:** make the tool installable in ten minutes rather than two weeks. This phase is the difference between a paper and a product.

- [x] **3.1 — `frozen-self` mode (the default)**
  `anchor/modes.py`. At baseline, freeze a set of `(item_id, system_output)` pairs and snapshot the judge's own scores on them. Thereafter, re-score the same frozen pairs and test for divergence from the snapshot beyond the noise floor. **No human labels anywhere.**
  Write the justification in `docs/identification.md`: attribution requires judge *stability*, not judge *validity*. A judge that was always wrong stays consistently wrong and correctly reads as stable. That is a real limitation and it goes in Known Limitations — but it does not weaken attribution, which is the only thing this tool claims.
  **Verify:** end-to-end test with a simulated judge — freeze, drift the judge, detect. Then: freeze a *biased but stable* judge, run 200 runs, assert `STABLE` throughout (the tool must not confuse "wrong" with "drifting").

- [x] **3.2 — `human` mode**
  Optional gold labels on anchor items. Adds a judge-versus-human agreement series on top of stability, and enables reporting judge *validity* over time as a secondary signal. Never required.
  **Verify:** with labels present, the report gains an agreement series; with them absent, everything else is identical. Diff the two verdict outputs and assert only the additive section differs.

- [x] **3.3 — `replicate` mode**
  Where a provider offers pinned dated snapshots, re-score a subsample of the *current* run with the pinned baseline judge — a differential judge rather than a frozen item set. Detect at config time whether the provider supports pinning and refuse this mode with a clear message where it does not.
  **Verify:** mode is rejected with an actionable error for a provider without dated snapshots; works against one that has them.

- [x] **3.4 — Stratified anchor selection**
  `anchor/select.py`. Anchors must resemble the eval distribution or a judge change confined to an unrepresented region is invisible (this is the primary attack, Phase 7.1). Select stratified over score decile and over any item tags present. Deterministic given a seed, and the seed is pinned.
  **Verify:** on a suite with a strongly skewed score distribution and 4 tags, selected anchors match the suite's decile and tag marginals within tolerance. Selection is reproducible from the pinned seed.

- [x] **3.5 — Coverage measurement**
  `anchor/coverage.py`. Report what fraction of the suite's score range and tag space the anchor set covers, and warn when coverage falls below a threshold. **Coverage is measured and reported, never guaranteed** — say so in the output.
  **Verify:** a deliberately narrow anchor set (all items from one tag) produces a loud coverage warning naming the missing strata.

- [x] **3.6 — `benchlock plan`**
  The user-facing provisioning calculator. Input: target shift, α, horizon, current suite. Output: minimum anchor n, cadence, estimated per-run judge cost in tokens and dollars, and the resulting `min_detectable_judge_shift`. This command is a standalone reason to install the tool — "how many eval samples do I actually need" is a question every AI engineer has and nothing answers it.
  **Verify:** `benchlock plan --target-shift 0.05` on the fixture suite produces an n that, in simulation, satisfies the design law in ≥ 95% of runs. Cost estimate is within 20% of the actual Phase 7 spend.

- [x] **3.7 — `benchlock rebaseline`**
  Explicit, logged, versioned. Requires `--reason`. Writes a rebaseline record into the ledger, starts a new baseline epoch, and every subsequent verdict block names the epoch and the reason. History before a rebaseline is retained and replayable but never silently compared across the boundary (Hard Rule 8).
  **Verify:** rebaseline mid-stream; assert verdicts before and after are computed within their epochs, that `replay` reproduces both, and that no cross-epoch comparison occurs.

**Phase Gate:** a user with zero labeled data can run `init → plan → baseline → observe → verdict` against a real eval suite and get a correct verdict. Measure and log the wall-clock time for that path; if it exceeds ten minutes excluding judge latency, simplify until it doesn't.

---

## PHASE 4 — Replay and the audit trail

**Goal:** the property that made the previous project credible, carried forward. Every historical verdict must be re-derivable.

- [x] **4.1 — `benchlock replay`**
  `ledger/replay.py`. Re-run `decide()` over every prefix of the recorded ledger and assert every historical verdict is reproduced exactly. A mismatch is a non-zero exit and a build failure.
  **Verify:** replay a 300-run ledger and reproduce all 300 verdicts. Then deliberately change a constant in the lattice and assert replay fails loudly and names the first divergent run.

- [x] **4.2 — Versioned decision semantics**
  The ledger records the `decision_semantics_version`. When the lattice or the statistics change, the version bumps, and replay of an older ledger under newer semantics reports **both** verdicts and flags the divergence rather than silently rewriting history.
  **Verify:** replay a ledger written under v1 semantics with v2 code; both verdicts reported, divergences listed, exit code non-zero.

- [x] **4.3 — `benchlock report`**
  Markdown output suitable for a PR comment and for `RESULTS.md` fragments: the verdict block, the two e-detector traces, the confidence sequences, and the provisioning status.
  **Verify:** golden-file test. The markdown renders correctly on GitHub (check the table and code-fence syntax explicitly).

**Phase Gate:** `benchlock replay` is green on a 300-run ledger and fails correctly on a tampered one and on a semantics change.

---

## PHASE 5 — The simulation study (Tier 1)

**Goal:** the headline numbers, at full statistical power, for free. This phase and Phase 7 are worth more than Phases 0–4 combined for whether anyone believes the project.

- [x] **5.1 — Ground-truth stream generator**
  `bench/sim/generate.py`. Generate score streams with known ground truth over a grid: system shift `δs ∈ {0, 0.02, 0.05, 0.10}` × judge shift `δj ∈ {0, 0.02, 0.05, 0.10}` × change-point location × noise level × score type (binary, Likert-5, continuous) × heteroscedastic on/off. 1000 seeds per cell. Streams are generated deterministically from a seed and the generator is committed.
  **Verify:** generated streams have the requested effect sizes to within Monte-Carlo tolerance. The generator is deterministic — same seed, same stream, byte-identical.

- [x] **5.2 — The baselines, implemented faithfully**
  `bench/sim/baselines.py`. Implement what teams actually do, and implement it *well* — a strawman baseline invalidates the comparison.
  - **B0** fixed threshold ("alert if the drop exceeds 5 points") — the true industry default
  - **B1** two-sample t-test at α=0.05 re-run at every run — the sophisticated-looking invalid default
  - **B2** B1 with Bonferroni correction over runs-so-far — the naive fix
  - **B3** CUSUM / Page's test
  - **B4** ADWIN and DDM via `river`
  - **B5** Benchlock's e-detector with **no anchor stream** — the ablation that isolates what attribution buys
  - **B6** Benchlock, full
  **Verify:** each baseline reproduces its textbook behaviour on a canonical test case (e.g. B3 detects a known step change at the known location).

- [x] **5.3 — The metrics, always reported together**
  For every method × cell: **false-alarm rate on drift-free streams**, **average run length to false alarm (ARL₀)**, **detection delay** at each effect size, **misattribution rate** decomposed as judge→system and system→judge, and **`indeterminate` rate**.
  Reporting misattribution without detection delay is the degenerate result where a method wins by never deciding anything. Reporting delay without false-alarm rate is the opposite degenerate result. Both numbers appear in every table or neither does.
  **Verify:** `uv run python bench/sim/run_sim.py --all` writes `bench/results/sim-<timestamp>.json` and regenerates the corresponding `RESULTS.md` section.

- [x] **5.4 — The honest anti-result**
  **Anytime-valid tests are strictly less powerful than fixed-sample tests at the same nominal n. Benchlock will be slower to detect than the invalid peeking t-test.** That is the price of the guarantee, it is real, and it must be a headline row in `RESULTS.md`, not a footnote. Measure and publish: median detection delay for B1 versus B6 at each effect size, alongside their false-alarm rates, so the reader can see exactly what the extra delay is buying.
  **Verify:** the delay-versus-false-alarm trade-off appears as its own table and its own plot. If B6's delay penalty is not visible in the table, the table is wrong.

- [x] **5.5 — The two headline numbers**
  Extract them from 5.3 and put them at the top of the README:
  1. **False alarm under peeking.** Over a drift-free stream monitored for N runs, what fraction of streams produce at least one alarm — for B1 versus B6.
  2. **Misattribution under silent judge change.** On streams where *only the judge* moved, what each method concludes. The baselines have no attribution capability at all, so their only available conclusion is "regression" — state that plainly as a structural fact, not as a criticism of the baselines, and report Benchlock's judge / indeterminate / system breakdown against it.
  **You do not know these numbers yet. Measure them. Do not write a placeholder that looks like a result.**
  **Verify:** both numbers appear in README and `RESULTS.md`, generated, with the command that produced them printed alongside.

- [x] **5.6 — Plots**
  Committed PNGs with their generating script: ARL₀ curves, detection-delay versus effect size, misattribution heatmap over the (δs, δj) grid, and the delay-versus-false-alarm trade-off from 5.4.
  **Verify:** every plot regenerates byte-identically from committed data. No plot exists without its script.

**Phase Gate:** `RESULTS.md` reports all seven methods × the full metric set, generated, never hand-edited, and it includes the case where Benchlock is worse than a baseline.

---

## PHASE 6 — The real-judge study (Tier 2)

**Goal:** the money demo. Simulation proves the mathematics; real judges prove the premise. Budget: under $50.

- [x] **6.1 — Dataset selection**
  Pick a public, redistributable-by-reference dataset of `(input, output)` pairs with a natural quality gradient and ≥ 400 items. Candidates: HelpSteer2, the TL;DR summarisation preference data, MT-Bench prompts with generated responses. Record the choice and the reason in the Progress Log. Commit the item IDs and a loader, never the data itself.
  **Verify:** the loader reconstructs the exact item set from a committed manifest of IDs and hashes.

- [x] **6.2 — Score pool construction (this is what makes Tier 2 affordable)**
  `bench/real/build_pool.py`. Do **not** re-run a whole pipeline per timepoint — that is tens of thousands of calls. Instead: score the fixed item pool once under each of ~5 judge configurations, plus K=5 replicates on the anchor subset for the noise floor. That is roughly 2,000–2,500 judge calls total.
  Judge configurations: (a) baseline model snapshot, (b) a different model snapshot from the same provider, (c) same model with a stricter rubric, (d) same model at a different temperature, (e) a different provider entirely.
  **Verify:** the pool is committed as scores + hashes (never raw content, Hard Rule 9). Total spend logged in `bench/results/cost.json`.

- [x] **6.3 — Stream composition**
  `bench/real/compose.py`. Construct run-by-run streams by sampling from the pooled real judge scores, splicing configurations at known change points. Six scenarios with known ground truth: judge version bump, judge rubric change, judge parameter change, system regression (a degraded system output pool judged by an unchanged judge), simultaneous both, and a drift-free control.
  **Disclose this honestly:** these streams are *resampled from real judge scores*, not longitudinally observed. That is what makes statistical power affordable, and it is a real limitation. It goes in `RESULTS.md` in the methodology paragraph, not buried in an appendix.
  **Verify:** each scenario's ground truth is recorded in the manifest; composition is deterministic from a seed.

- [x] **6.4 — Run and report**
  All seven methods against all six scenarios. Same metric set as Phase 5.
  **Verify:** `uv run python bench/real/run_real.py --all` regenerates the `RESULTS.md` real-judge section from committed JSON.

- [x] **6.5 — Measure judge nondeterminism as a standalone result**
  From the K replicates: how much does a temperature-0 hosted judge disagree with itself across identical calls? Report per provider, per rubric, per score type. **This is a small, quotable, independently useful finding and almost nobody publishes it.** It is likely to be the most-shared number in the whole repository. Put it in `docs/` with a plot.
  **Verify:** the number is in `RESULTS.md` with its command, and the write-up references it.

- [x] **6.6 — Generate the three demos**
  `bench/demo.py` produces `docs/demo.md` containing the three §2 demos as captured transcripts from real runs. Every line in the README's demo blocks is copied from this file, not written by hand.
  **Verify:** `docs/demo.md` regenerates and matches the README's blocks exactly. A CI check enforces this.

**Phase Gate:** `RESULTS.md` carries both the simulation and real-judge tables, the resampling limitation is stated in the methodology, total spend is committed, and the three demos are captured from real runs.

---

## PHASE 7 — Break your own attributor

**Goal:** the phase that separates this from every drift dashboard on GitHub. A defence that reports only wins is reporting a broken red team. Attack the attributor, measure the damage, publish it.

- [x] **7.1 — Anchor evasion** *(expect this to work; measure how badly)*
  A judge change confined to a region the anchor set does not cover — e.g. the judge got stricter only about code responses while the anchors are all prose. The anchor process sees nothing; the system stream moves; the verdict is `SYSTEM`. **This is the fundamental limitation of the whole design.** Measure misattribution rate as a function of anchor coverage, and use the result to set the coverage warning threshold in 3.5.
  **Verify:** measured curve of misattribution versus coverage, committed and plotted.

- [x] **7.2 — Slow ramp**
  Drift introduced gradually below the per-run detectable threshold. Find the ramp rate at which detection never occurs within a horizon, for each method. Report the boundary.
  **Verify:** the detection-delay-versus-ramp-rate curve, including the region where Benchlock never fires.

- [x] **7.3 — Simultaneous drift, including cancellation**
  System and judge moving together (masking) and in opposite directions (cancellation). **Cancellation is the nastiest case in the whole design: the net score is unchanged, both components moved, and a single-stream detector sees a perfectly healthy system.** Measure Benchlock's `BOTH` detection rate here and every baseline's complete blindness to it.
  **Verify:** cancellation is a named scenario with its own row. If Benchlock also misses it at some effect size, publish that boundary.

- [x] **7.4 — Anchor staleness / concept drift**
  The team's own standard changes (hedging used to be penalised, now it's fine). `frozen-self` anchors read this as permanent judge drift until refreshed. Measure how long the tool stays wrong and how loudly it complains.
  **Verify:** the scenario runs; the staleness behaviour is documented with the measured time-to-noticing.

- [x] **7.5 — Input distribution shift** *(the third cause, Hard Rule 6)*
  The eval suite is sampled from production traffic and the traffic mix shifts. Neither leg covers this. Show that it produces `SYSTEM` misattribution, and verify that the `suite_hash` check from lattice rule 2 catches it when the suite is nominally fixed. Where the suite is genuinely dynamic, the tool must refuse attribution mode.
  **Verify:** the refusal fires on a dynamic suite; the misattribution rate is measured for the case where the check is disabled, to show what the check is worth.

- [x] **7.6 — Adversarial ordering**
  Worst-case orderings of observations within a run designed to delay evidence accumulation in a betting process. Measure the delay penalty against random ordering.
  **Verify:** measured, committed. If the penalty is large, add a shuffling recommendation to the docs and measure that too.

- [x] **7.7 — Heavy tails and bound violations**
  Betting e-processes assume bounded scores. Test rubrics that produce near-degenerate distributions (99% at the ceiling), heavy tails, and scores that violate their declared bounds. Show where the guarantee degrades and confirm Hard Rule 10's loud failure fires before it does.
  **Verify:** each case either behaves correctly or fails loudly. Silent degradation anywhere is a bug.

- [x] **7.8 — Judge noise spike without a version change**
  The judge did not change version but became noisier — a provider-side inference change, a load-shedding fallback. Should this read as `JUDGE`? Argue the answer in `docs/threat-model.md` and make the behaviour deliberate rather than incidental.
  **Verify:** documented, tested, and the reasoning is written down.

- [x] **7.9 — Provider-side response caching** *(the sneaky one)*
  If the provider caches judge responses, the anchor stream is *artificially stable*, and real judge drift becomes invisible — the anchor set is returning yesterday's answers. This is a real, undocumented production failure mode. Mitigation: a per-run cache-busting nonce in the anchor prompt. **Then measure whether the nonce itself perturbs scores**, because if adding a nonce changes the judgment, the mitigation is also a confound.
  **Verify:** with caching simulated, drift is missed without the nonce and caught with it. The nonce's own effect on scores is measured and reported. If the nonce measurably shifts scores, say so and document the trade-off rather than shipping it silently.

- [x] **7.10 — Report the losses**
  `RESULTS.md` gets an **"Attacks that work against Benchlock"** section with the measured failure rate per strategy. Do not fix-and-hide: where you fix something, keep the pre-fix number in the table with the commit that changed it. Where you can't fix it, say so and explain why.
  **Verify:** the section exists and contains at least two strategies with non-trivial failure rates. 7.1 and 7.4 should both be non-zero by construction; if everything is zero, your red team is too weak — go back to 7.1.

**Phase Gate:** `RESULTS.md` contains both what works and what doesn't. `docs/threat-model.md` names the weakest link explicitly and does not soften it.

---

## PHASE 8 — Survive contact with a real pipeline

**Goal:** it works in someone else's repo, on their CI, with their eval framework.

- [x] **8.1 — Framework adapters**
  `adapters/promptfoo.py`, `inspect_ai.py`, `deepeval.py`. Each normalises into the JSONL contract from 0.3. Read each framework's actual output format from its documentation; do not guess at schemas.
  **Verify:** round-trip test against a real output file from each framework, committed as a fixture. Where a framework's format is ambiguous, the adapter fails loudly rather than guessing.

- [x] **8.2 — GitHub Action**
  A composite action that runs `benchlock observe` on the eval output, then `benchlock gate`, then posts `benchlock report` as a PR comment. `judge` verdicts post a comment and pass; `system` and `both` fail the check.
  **Verify:** the action runs in this repo's own CI against this repo's fixture streams, end to end.

- [x] **8.3 — Edge cases**
  Zero-variance runs (every item scored identically); a run with one observation; a judge returning nulls or out-of-range values; an anchor item deleted from the suite mid-stream; two CI jobs writing the ledger concurrently; a ledger with a gap in run indices; clock skew in run timestamps; a 500 MB ledger; scores arriving out of order.
  **Verify:** each case has a test. Nothing corrupts the ledger, nothing produces a silently wrong verdict, every failure names its cause.

- [x] **8.4 — Concurrency and ledger integrity**
  The ledger is append-only and may be written from parallel CI jobs. Use file locking with a timeout; on contention, fail with a clear message rather than interleaving records.
  **Verify:** a stress test with 20 concurrent writers produces a valid chain or clean failures — never a corrupt chain. Run it 100 times in CI.

- [x] **8.5 — Ergonomics pass**
  Run through the whole first-use path with fresh eyes and fix every place a reasonable person would stop. Specifically: every error message must name the file, the field, and the command that fixes it. `benchlock verdict` on an empty ledger must explain what to do, not raise a stack trace.
  **Verify:** a scripted first-use walkthrough in `tests/test_first_use.py` covering the eight most likely mistakes, asserting each produces an actionable message.

**Phase Gate:** installs into a real repo with a real eval suite, runs in GitHub Actions, and produces a correct verdict on a real PR.

---

## PHASE 9 — Ship

- [x] **9.1 — README**
  Structure, in order: the one-sentence thesis; **Demo 1** (the phantom regression) in full; the two headline numbers from 5.5; the mechanism in three lines; install; the results table pulled from `RESULTS.md`; **"Attacks that work against Benchlock"** linked prominently near the top, not buried; prior art per §3 with the preprint credited in the first sentence; known limitations.
  **No marketing language. No "enterprise-grade". No claim that Benchlock makes your evals correct** — it does not; it tells you which of two things moved. Over-claiming here would repeat the exact error §3 criticises in the neighbours.
  **Verify:** every number in the README appears in a committed JSON file. A CI check greps the README for numeric literals and fails on any that cannot be traced.

- [x] **9.2 — Docs**
  `identification.md` (the argument written out properly, with the assumptions stated as assumptions), `statistical-guarantees.md` (what is guaranteed, what is not, our measured false-alarm rates, the `confseq` differential), `provisioning.md` (how to size an anchor set, worked examples), `threat-model.md` (what breaks this), `demo.md` (generated).
  **Verify:** every doc has at least one number from a committed run. No doc is purely prose.

- [x] **9.3 — CI**
  ruff, mypy strict, pytest with coverage, plus the mandatory tests promoted to their own named jobs so a failure is unmissable: the secret-leak test (0.4), the conservative-pruning property test (1.4), the determinism test (2.3), the replay test (4.1), the concurrency stress test (8.4), and the README-number-traceability check (9.1). Nightly: the full simulation study, failing the build if any headline metric regresses beyond a committed threshold. **The benchmark is a test, not a marketing artefact.**

- [~] **9.4 — Package and release** — PARTIALLY MET, and ticked as such rather than as done.
  Version 0.1.0, CHANGELOG and a wheel that builds and installs cleanly: **done**. A clean-environment install is verified from the public repository (`uv tool install git+https://github.com/Poojan6216/benchlock && benchlock --help`), which is what the README now tells people to run.
  **NOT DONE:** publishing to PyPI, and the tagged release with `RESULTS.md` attached. Publishing is outward-facing and irreversible and is deliberately held for an explicit go-ahead (Final Report, decision 11). Until it happens, `uv tool install benchlock` does **not** work, and no document in this repository claims it does.

- [x] **9.5 — The write-up**
  A technical post. **Lead with Demo 3 — the `indeterminate` verdict — and the anti-result from 5.4.** The shareable thesis is not "we built a drift detector"; it is *"your eval dashboard has been lying to you in two different ways, and here is the arithmetic, including the part where our method is slower than the wrong one."* Then: the identification argument, the provisioning rule, the judge-self-disagreement number from 6.5, and the attacks that beat us.
  Title it around the negative result or the self-disagreement number. That is the part people share.

- [~] **9.6 — The launch artefacts** — PARTIALLY MET.
  The diagram (`docs/assets/mechanism.svg`, in the README) and the post are **done**; the post is now GENERATED by `scripts/gen_launch.py` so its numbers cannot drift behind the runs they quote, and it is covered by both `check_numbers.py` and `check_generated.py`.
  **NOT DONE:** the 30-second terminal recording of Demo 1. `docs/launch.md` carries the exact command sequence to record, in an order that has been executed rather than assumed, but no `.cast` file is committed.

**Phase Gate:** NOT FULLY MET, stated rather than glossed. The demo reproduces (`bench/demo.py`, `scripts/dogfood.py`) and this repository's own CI runs green on a real GitHub runner across ten jobs. `uv tool install benchlock` does not work because the package is not on PyPI by choice; installing from the repository is verified and is what the README documents. The GitHub Action has never been run from a second repository.

---

## Definition of done

- [x] A silent judge version change produces `verdict=judge` and does not fail CI
- [x] A real system regression produces `verdict=system` and does fail CI
- [x] An under-provisioned anchor set produces `verdict=indeterminate` and says exactly what to fix
- [x] Measured false-alarm rate on drift-free streams is ≤ α, demonstrated by our own simulation
- [x] The conservative-pruning property test passes over 3000+ hypothesis cases
- [x] `benchlock replay` reproduces every historical verdict on a 300-run ledger
- [x] `RESULTS.md` reports all seven methods with false-alarm rate, ARL₀, delay, and misattribution, generated from a committed command
- [x] `RESULTS.md` reports at least one case where Benchlock is *worse* than a baseline, with the number
- [x] `RESULTS.md` documents at least two attacks that beat Benchlock, with measured rates
- [x] Judge self-disagreement at temperature 0 is measured and published — **18.90%** over 1,820 pairwise comparisons of identical calls to `claude-sonnet-5`.
- [x] Zero LLM calls in the decision path
- [x] Zero telemetry, zero hosted components, zero accounts
- [x] A user with no labeled data completes `init → plan → baseline → observe → verdict` in under ten minutes
- [x] Total benchmark spend is committed in `bench/results/cost.json` and is under $50

---

## Known limitations to document, not fix

State these in the README, in the first screen, not in an appendix.

- Benchlock tells you *that* the judge changed, never *why*. It cannot distinguish a provider snapshot rotation from a rubric edit except where you declared the change yourself.
- **Anchor evasion is the fundamental limitation.** A judge change confined to a region your anchor set does not cover is invisible. Coverage is measured and warned about; it is never guaranteed. Phase 7.1 quantifies the damage.
- `frozen-self` anchors detect judge *instability*, not judge *invalidity*. A judge that was always wrong stays wrong and reads as stable. Benchlock does not claim your evals are correct.
- Anchor concept drift — your own standards changing — reads as permanent judge drift until you rebaseline. The tool warns; it does not solve.
- The guarantee is time-uniform, not instantaneous. Detection has delay, and **anytime-valid tests detect later than invalid ones. That is the price and it is published in `RESULTS.md`.**
- If your eval inputs are sampled from shifting production traffic, score movement has a third cause Benchlock does not model. A fixed suite is required; the tool refuses attribution mode without one.
- Bounded scores only. Rubrics must declare their range and stay inside it.
- Needs a stream. Nothing useful for a single run, and `min_runs` defaults to 8 for a reason.
- Nothing for pipelines with no LLM judge. Programmatic scorers with no model in them don't drift, which is a good argument for using them where you can — say so.
- Tier-2 streams are constructed by resampling pooled real judge scores, not observed longitudinally. Simulation gives the power; real judges give the premise; neither alone is a longitudinal production study, and we have not run one.
- Single judge, single suite, single system. Multi-judge ensembles and multi-suite portfolios are v2.

---

## Decision gates — stop and reassess if any of these fire

Check these as you go. They exist so that one bad assumption does not consume the whole month.

1. **End of Phase 1.** If our confidence sequences cannot be validated against `confseq`, or the composite-null e-process does not hold its false-alarm rate in simulation, the statistical foundation is wrong. Stop and fix. Do not build attribution on an invalid test — the entire value proposition is the guarantee.
2. **End of Phase 3.** If the first-use path exceeds ten minutes for a user with no labels, the adoption thesis is broken. Cut scope from the anchor modes (ship `frozen-self` only) rather than shipping something nobody will configure.
3. **During Phase 5.** If B6 shows no advantage over B5 (the no-anchor ablation) on misattribution, then the anchor stream is not earning its cost, and the project's central claim is unsupported. Investigate immediately. Most likely cause: anchor provisioning, not the idea.
4. **During Phase 6.** If real judges show *no* measurable drift across snapshots and rubric changes, the premise is weaker than assumed. That is still publishable — "we looked for judge drift and here is how much we found" is a real result — but the framing changes from "here is the fix" to "here is the measurement and the fix if you need it." Adjust the README honestly rather than overstating.
5. **Any time.** If a credible open-source implementation of judge-versus-system attribution appears, do not abandon the project. Read it, credit it in §3, and sharpen the contribution to whatever it does not do — most likely provisioning, `indeterminate`, and the zero-label anchor mode. Being second with a better implementation and honest numbers is a fine outcome. Being second and pretending to be first is not.

---

## Suggested schedule

| week | phases | the thing that must exist at the end |
|---|---|---|
| 1 | 0, 1 | `docs/statistical-guarantees.md` with our own measured false-alarm rates |
| 2 | 2, 3, 4 | the three §2 demos running from fixtures; replay green |
| 3 | 5, 6 | `RESULTS.md` with both benchmark tiers and the two headline numbers |
| 4 | 7, 8, 9 | the attacks that beat us, the GitHub Action, PyPI, the write-up |

Phase 5 and Phase 7 are the ones that make anyone care. If the schedule slips, cut from Phase 8 (adapters beyond JSONL, extra edge cases), never from 5 or 7.

---

## Progress Log

*Agent: append one line per completed task. Format: `[phase.task] what you did — any decisions or blockers`.*

```
[0.1] Scaffold: uv/hatchling pyproject (py3.12+), ruff line-100, mypy --strict, pytest+hypothesis, Apache-2.0, pre-commit. Decision: confseq oracle needs Boost headers so it lives in its own 'oracle' dependency group (brew install boost); CI installs it so the 1.3 differential test always runs. confseq 0.0.11 predates NumPy 2 (np.float_) — tests apply a one-line shim rather than pinning numpy<2.
[0.2] Config + CLI skeleton: Pydantic v2 models (extra=forbid everywhere so typos are errors), schema version 1, YAML composed to a node tree for path->line mapping so every validation error prints file:line, field path and a fix hint. 21 malformed configs tested + 6 exact-line assertions. Nine subcommands present; unimplemented ones exit 3 naming the phase they arrive in, never a silent no-op. Structured JSON logs to stderr (stdout stays clean for piping). Decision: NoiseFloor + AnchorMode live in model/pins.py so model/ has no internal deps; anchor/noisefloor.py is the estimator that produces one.
[0.3] Universal JSONL adapter + model/streams.py (Observation, RunRecord, StreamKind) and full JudgePin/AnchorPin. Normalises to [0,1] from the declared score_scale, retaining raw value + scale so it is reversible; out-of-range scores RAISE and are never clamped (clamping would compress real movement into the bound and make a drifting stream look stable). Collects every problem in a file before raising, each naming file:line and a fix. 6 score types x 12 malformed cases tested. Decision: accepts a few well-known aliases (id/test_id/case_id, value) for convenience, but two aliases with conflicting values is an error rather than a guess.
[0.4] Hash-chained append-only ledger + wired 'benchlock observe'. Each record carries SHA-256 of the previous; verify() reports the index of the first broken record (tested by flipping a byte in EVERY record in turn, plus delete/reorder/single-char edit). Hard Rule 9 is structural: _run_payload is an explicit allowlist so no prompt/output/rationale has a code path into the file. Decision: item_ids are stored HASHED (sha256, 128-bit prefix) because in real suites the item id is frequently the prompt text or a customer identifier — the statistics need stable identity, not the original string; suite_hash stays an opaque comparison token so replay never needs to recompute it from ledger ids. Verified the 15-secret test has teeth via a negative control (storing raw ids makes it fail). fcntl advisory locking in place, stress-tested in 8.4.
[0.5] Pin-violation detection wired into observe. All 8 verify cases pass (model, whitespace-only rubric edit, temperature, anchor item added/removed, anchor score edited, scale, no-change) plus provider change, multi-field change, same-size membership swap, mode change. Messages distinguish anchor membership changes from baseline-score edits, and name the rebaseline command with a pre-filled --reason. Fixed a real usability bug found by the tests: hash-abbreviation was eliding model strings so the most important message read 'claude-sonne… -> claude-sonne…'; _fmt now truncates only hex digests and shows plaintext in full. Ledger.current_pins() takes the pin from the latest baseline/rebaseline record, falling back to the first run of the epoch so the simplest observe-only workflow still gets Hard Rule 8 protection.
[0.6] 'benchlock init': detects promptfoo / Inspect AI / DeepEval / bare JSONL, writes a COMMENTED benchlock.yaml carrying the evidence for each guess, and validates its own output before writing. score_scale is inferred as the tightest standard rubric range containing all observed values and always labelled 'CONFIRM THIS' — pre-filling a declaration, never making it (Hard Rule 10). Existing configs are backed up byte-for-byte to a timestamped .bak BEFORE any write, and refused without --force. Paths written relative so the config survives being committed. Decision: detection lives in adapters/__init__.py rather than a new module, keeping the §6 structure exact.
[PHASE 0 GATE] PASS. 'benchlock init && observe run1.jsonl && observe run2.jsonl' builds a valid 2-record chain from real eval output; a tampered ledger is rejected naming record 0. 135 tests, ruff + mypy --strict green.
[1.1] Betting wealth process (stats/betting.py). K_t = prod(1 + lambda_s(X_s - mu0)) with lambda truncated to [-c/(1-mu0), c/mu0], c=0.5, so every multiplier is >= 1-c = 0.5 and wealth is strictly positive. Predictability is STRUCTURAL: strategies receive a BetState carrying only statistics of X_1..X_{t-1}, so there is no channel to read X_t; verified behaviourally by hypothesis (two streams sharing a prefix must give an identical bet however different the next observation), plus a cheating negative control proving that test has teeth. Strategies: fixed + aGRAPA (Waudby-Smith & Ramdas plug-in). Decision: wealth tracked in LOGS with a log-sum-exp mixture, so a 5000-step extreme run stays finite instead of overflowing. Decision: added HedgedWealthProcess (50/50 up/down mixture, still a martingale) because drift can go either way and the tool must not have to guess the direction in advance. Measured: E[K_t] <= 1 at every t over 2000 Bernoulli streams (tolerance = 3 x measured SE, not a fudge), and Ville false-alarm rate 0.0 <= alpha=0.05 over 2000 streams peeked at all 200 steps.
[1.3] Confidence sequences (stats/confseq.py): hedged capital (grid inversion) + predictable-mixture empirical-Bernstein (closed form), both implemented against the same construction as confseq so the differential is like-for-like. DIFFERENTIAL RESULT: exact agreement, width ratio min=median=max=1.0000 over 12,500 (stream, t) pairs from 500 streams x 25 steps (bench/results/confseq-differential.json). The test caught TWO real bugs: (1) I was pre-truncating the bets at 0.5 before the m-dependent clip, which capped the up-leg far below what m allows and made the CS asymmetric — reference truncates only via the m-clip; (2) a log-space vs linear-space tie at capital exactly = 1/alpha, fixed with a 1e-10 log slack that always errs toward the WIDER interval. Also added an independent time-uniform coverage simulation (400 streams, every t) because a differential cannot catch both implementations being wrong the same way. Decision: implemented 1.3 before 1.2 because the composite-null e-process needs a CS to bound mu0.
[1.2] Composite-null e-process (stats/eprocess.py). mu0 is bounded by an anytime-valid CS over the baseline period, then monitored against the LEAST FAVOURABLE value in that interval (up-leg tests U, down-leg tests L); each leg is a supermartingale for any true mu in [L,U], so Ville applies and a union bound gives total error alpha_b + alpha_m = alpha. MEASURED (2000 drift-free streams, horizon 300, peeked EVERY run): composite-null false-alarm rate 0.000 <= alpha=0.05; naive point-estimate version 0.5025 — 10.05x alpha, i.e. half of all healthy pipelines would get a false regression alert. bench/results/eprocess-false-alarm.json. TWO design findings forced by measurement: (1) an anytime-valid CS on 8 [0,1]-bounded run means is ~0.6 wide and barely adapts to observed variance (the m-clip at c/mu0 binds), so a frozen-snapshot null was added — the anchor stream's null is KNOWN, not estimated, bounded only by snapshot measurement error run_mean_sd/sqrt(K) with a Student-t quantile on K-1 df. That is why the anchor set earns its cost. (2) betting e-processes bet inefficiently on data clustered in a narrow band, so MonitorScale maps deviations onto [0,1] in units of the measured noise floor (band = 10 run-SDs); the transform is affine and fixed in advance so it changes no statistical property, and out-of-band values clip, which only ever REMOVES evidence. Measured result: anchor n=260 detects a 0.02 judge shift (median delay 42 runs), n=40 cannot — exactly the Demo 1 vs Demo 3 provisioning story.
[1.4] E-detector (stats/edetector.py): one e-process per candidate change point, combined as a WEIGHTED sum with weights w_j = 1/((j+1)(j+2)) summing to exactly 1. Decision: weighted sum rather than the unweighted Shin-Ramdas-Rinaldo form, because unweighted buys only an ARL guarantee (E[tau] >= 1/alpha) while weighting buys the time-uniform P(ever alarming) <= alpha that Hard Rule 3 demands; the cost is some power against very late change points. Conservative pruning (Hard Rule 4) holds BY CONSTRUCTION: the bounded statistic is a sub-sum of the full one over non-negative terms with survivors' weights untouched, so E_bounded <= E_full at every t. Verified over 3000 streams x 6 adversarial shapes + 600 hypothesis examples, checking at EVERY step, and proven to have teeth by a negative control that renormalises survivor weights upward (both tests then fail). Fixed: best_changepoint now ranks by each hypothesis's own evidence rather than weighted contribution — the prior falls off so steeply that the weighted ranking reported run 0 almost regardless of the data. It is descriptive only and gates no verdict.
[1.5] Noise floor (anchor/noisefloor.py): pooled per-item SD + run_mean_sd measured DIRECTLY from K replicate run means rather than derived as per_item_sd/sqrt(n), because a judge that shifts every item together between calls has a far larger run-mean SD and the derived figure would understate the floor. Recovers known injected noise to within 15% across sd in {0.01,0.05,0.12}; independence_ratio surfaces correlated judge noise (>3 when a shared offset is injected). A judge whose K replicates are identical is reported as DEGENERATE with a loud warning, not as zero noise — K identical scorings prove K was too small, not that the judge is deterministic. HEADLINE (bench/results/noise-floor-earns-its-place.json): what the floor actually protects against is error in the FROZEN SNAPSHOT, which is itself one noisy measurement and so sits a persistent offset from the judge's true mean that never averages away. Identical drift-free streams, identical scaling, differing only in null width: measured floor 0.00 false alarms, a judge assumed 50x quieter 0.29 over 300 runs. My first two attempts at this test were wrong (centred noise does not false-alarm even against a point null) — the snapshot-error framing is the real mechanism.
[1.4b] PERFORMANCE: profiling for Phase 5 showed the scalar detector at 1.2-4.9s per 300-run stream at M=256, which would make the simulation study infeasible. Rewrote EDetector with all candidate state in parallel numpy arrays (15x faster, 0.079s/stream) and kept the scalar version as ReferenceEDetector — the readable statement of what the detector is, and the oracle the vectorised one is checked against observation-by-observation across 6 stream shapes x 2 strategies x 3 memory limits. Also cached the combined log-e (it was recomputed 4x per step and dominated the profile) and replaced np.clip with direct ufuncs. Hard Rule 4 holds in both implementations for the same structural reason.
[1.6] Provisioning calculator (stats/power.py). Decomposes the measured floor into SHARED (moves every item together, immune to anchor size) and INDEPENDENT (shrinks 1/sqrt(n)) components, so run_mean_sd(n) = sqrt(shared^2 + per_item^2/n). Where shared noise alone swamps the target, min_anchor_size raises ProvisioningImpossibleError naming the shared SD — an honest refusal beats returning a big number that would not have worked. min_anchor_size takes the max of two legs: detectability within the horizon, and the design law proper (anchor no noisier than the system stream). SIMULATION DISAGREED WITH THE FORMULA TWICE AND THE FORMULA WAS CHANGED BOTH TIMES, per the spec: (1) the formula assumed wealth grows quadratically in the drift, delta^2/2sigma^2, but the growth-optimal bet is above its truncation in every realistic regime so growth is LINEAR in the drift — the quadratic version claimed 0.0117 was detectable where the truth was ~0.016 and simulation showed 0% detection at every safety factor; now solved numerically with the truncated bet. (2) MonitorScale's 10-SD band was costing about a third of the resolution — measured {2,3,4,5,6,10} across 4 configs and 4 is best (2 clips ordinary noise and detects nothing at all). Formula is now conservative by 1.13-1.33x vs measured 90%-detection shifts. VERIFY: 100% detection at the recommended n across 6 configurations, vs the >=95% required (bench/results/provisioning-calibration.json). Plan.shared_noise_dominates redefined as diminishing returns (<10% improvement from a 4x bigger anchor) because that answers the question a user actually has: will spending more judge tokens help?
[1.6b] Found a real trade-off while fixing a broken test: a NARROW scaling band gives resolution for small shifts, a WIDE one gives more evidence per run once a large shift saturates the band (the per-run cap depends on how far the null sits from the boundary). Neither wins outright. Added MonitorScale.for_target(target_shift, run_mean_sd) = max(target_shift, 4*sd), which takes both: the target reaches the edge so it yields full evidence, anything smaller still has room to resolve, and the 4-SD floor stops ordinary noise from clipping (a 2-SD band detects nothing at all).
[PHASE 1 GATE] PASS. docs/statistical-guarantees.md is GENERATED by scripts/gen_guarantees.py from four committed JSON result files, never hand-edited. Our own measured numbers for every primitive: confseq differential width ratio exactly 1.0000 over 12,500 (stream,t) pairs; composite-null false alarm 0.0000 vs point-estimate 0.5025 at alpha=0.05 over 2000 streams x 300 runs; noise floor 0.00 vs 0.29 for a judge assumed quiet; provisioning 100% detection at recommended n across 6 configs. Decision gate 1 CLEARED: the CS validates against confseq exactly and the composite-null e-process holds its false-alarm rate.
[2.1-2.4] Attribution engine. model/verdict.py (5 verdicts, Evidence, Attribution), attribute/lattice.py (8 ordered rules, first match wins, rules 1-2 raise), attribute/race.py, attribute/engine.py::decide() (pure). 46 lattice tests cover every branch and both threshold boundaries, and assert the RULE ORDER itself so rule 6 can never drift after rule 7. THREE design changes forced by getting it working: (1) the system stream's baseline is treated as a FROZEN SNAPSHOT of its own per-item scores, exactly like the anchor's — monitoring raw run means against an estimated bounded-mean interval gave a null ~0.6 wide, so E_system literally never crossed for any regression. (2) The corrected quantity is monitored as its own difference-in-differences E-PROCESS rather than derived by subtracting two intervals: interval widths add, so the corrected interval contained zero even for moves both streams saw clearly. Documented deviation from the spec's rule-4 wording, and stronger — Hard Rule 3 prefers an anytime-valid test to an interval comparison. (3) Detection and estimation need DIFFERENT bands; one band for both was reporting a real -0.12 move as -0.05 because -0.05 was the band edge. Both bands are fixed before monitoring so the interval stays anytime-valid. Also relaxed rule 4 to drop the E_system>=thr requirement, which lets it catch CANCELLATION (judge and system moving in opposite directions leave the raw system score flat) — Phase 7.3 calls this the nastiest case and the spec's literal rule 4 returns STABLE for it. Fixtures are committed as specs + a digest rather than 26MB of materialised runs; the digest turns 'deterministic generation' into a test.
[2.5-2.6] report/human.py + 'benchlock verdict'/'benchlock gate'. The renderer computes NOTHING: two mandatory AST tests walk the module's syntax tree and fail on any arithmetic operator, and on any arithmetic applied to an evidence field — verified to have teeth by inserting a midpoint calculation, which both tests catch. Rationale: a renderer doing its own arithmetic is a second, untested implementation of the decision, free to disagree with decide(), and the number on screen is the one people act on. Exit codes 0/1/2 driven by gate.fail_on, tested across all five verdicts under default and three custom configs; 'judge' exits 0 under the default, which is the product thesis stated as a test.
[PHASE 2 GATE] PASS. The three demos run end to end from committed fixtures and from a real 121-record ledger: phantom judge drift -> verdict=judge (exit 0, do not roll back); real regression -> verdict=system (exit 1); same regression with a 40-item anchor -> verdict=indeterminate naming the minimum detectable shift and the plan command to fix it. Determinism suite green (500 decide() calls compared byte-for-byte with the network and clock monkeypatched to raise).
[3.1-3.7] Anchors that cost nothing to adopt. judge/base.py is a 4-method adapter (describe/pin/score/estimate_cost) plus SimulatedJudge — a real implementation of the interface with injectable drift, bias, noise and response caching, not a mock. anchor/modes.py: frozen-self (default, zero labels), human (labels add a judge-vs-human agreement series and NOTHING else — verified by diffing against the unlabelled run), replicate (refused with an actionable message where the provider has no dated snapshots). Key test: a BIASED BUT STABLE judge reads as STABLE over 200 runs — stability is not validity, and the tool must not confuse 'wrong' with 'drifting'. anchor/select.py guarantees at least one item per occupied stratum before allocating proportionally; without that floor a rare decile's quota rounds to zero and the stratum vanishes, which is exactly the hole a region-confined judge change hides in. The cost is real and tested: at small n the guarantee flattens the marginals. Fixed a genuine bug found by tests: config paths were resolved against CWD rather than the config file's directory, so a committed benchlock.yaml broke when CI ran from a subdirectory. decide() now filters to the current epoch, so a rebaseline isolates history without discarding it.
[PHASE 3 GATE] PASS. A user with zero labelled data completes init -> plan -> baseline -> observe x14 -> verdict in 27.7 SECONDS against a 600s budget (excluding judge latency; measured with the simulated judge). Decision gate 2 cleared with room to spare, so all three anchor modes ship rather than frozen-self alone.
[4.1-4.3] Replay, versioned decision semantics, and markdown reports. Added a VERDICT ledger record so replay has something to check — an audit trail of inputs alone would let replay confirm that today's code agrees with itself while saying nothing about what the tool told you last quarter. 'benchlock replay' re-derives every recorded verdict from its own PREFIX of the ledger; the 300-run ledger reproduces all 300 exactly. Mandatory negative control: shifting the lattice threshold makes replay fail, name the first divergent system run, and exit 1. 4.2 reads DECISION_SEMANTICS_VERSION at call time so a bump reports BOTH verdicts side by side and lists divergences rather than rewriting history — old verdicts were not wrong, they were issued under different rules. report/markdown.py renders from the same Attribution as the terminal block, with tests asserting balanced code fences and non-ragged tables so a PR comment cannot silently break. Fixed a real bug: _corrected_stream was passing the number of MONITORED runs as the replicate count, which made its null tighten simply because time had passed, and crashed outright on the first paired run.
[PHASE 4 GATE] PASS. benchlock replay is green on a 300-run ledger (901 records, 300 recorded verdicts, 16s), fails correctly on a tampered ledger naming the record index, and fails correctly on a semantics change while reporting both verdicts.
[5.1-5.2, 5.5] Simulation study. bench/sim/baselines.py implements B0-B6 properly rather than as strawmen (fixed threshold, peeking t-test, Bonferroni, CUSUM, ADWIN, DDM, benchlock-without-anchor ablation, benchlock). TWO HEADLINE NUMBERS, 500 streams each: (1) on drift-free streams peeked at every one of 142 runs, the peeking t-test raises at least one false alarm on 21.2% of healthy pipelines and Bonferroni on 9.4% — still above alpha — while Benchlock is at 0.0%; (2) on a silent judge change, B0/B1/B2/B3 and the B5 ablation all say 'regression' 100% of the time (a rollback recommendation for a system that never changed), ADWIN/DDM miss it entirely, and Benchlock says 'judge' 100% of the time. DECISION GATE 3 CLEARED: B5 (no anchor) misattributes 100% where B6 is correct 100%, so the anchor stream is exactly what attribution costs and buys.
[7.1-7.9] Adversarial study, and it found real holes. SIX of eight attacks have non-zero failure rates: anchor evasion 100% (the fundamental limitation — a judge change outside the anchors' coverage is invisible), anchor staleness 100% (benchlock reports THAT the judge moved, never WHY, so a deliberate rubric change is indistinguishable from provider drift), input distribution shift 100%, provider-side caching 100% without a nonce, cancellation 25%, slow ramp 20% of tested ramp rates evade entirely. TWO OF MY OWN ATTACKS WERE MEASURING NOTHING AND I FIXED THEM: 7.5 renamed every item id so all per-item deviations became zero — rewritten to keep ids stable and change the content behind them, which exposes a genuine limitation (hashing item IDs does not hash item CONTENT, so the suite-hash check is blind to a traffic-mix shift that preserves ids); 7.2 reported the best case rather than the fraction of ramp rates that evade. An attack that does not attack is worse than no attack.
[6.x BLOCKER] Tier 2 CANNOT BE RUN: no ANTHROPIC_API_KEY or OPENAI_API_KEY in this environment, so not one call was made to a hosted judge. Per the spec's blocked-task rule I built the whole pipeline — provider adapters (4 methods each), SQLite score cache, cache-busting nonce, pool builder, dataset manifest+loader with content hashes, six ground-truth scenarios, runner — and exercised it end to end against the deterministic built-in judge, which produces the right verdict on all six scenarios. The pool JSON is stamped 'simulated: true' and scripts/gen_results.py REFUSES to render it under a real-judge heading, emitting a 'NOT RUN' section with the exact commands and the ~$3-8 estimated spend instead. Two numbers stay unmeasured and are listed as such: judge self-disagreement at temperature 0 (6.5) and whether a cache-busting nonce perturbs a real judge's scores (7.9). Publishing either from simulation would be the exact failure this project exists to prevent.
[housekeeping] Corrected the checkbox state in this file. The logprogress helper only ticked a box when a log line began with a single [N.N], so grouped entries like [2.1-2.4] and [7.1-7.9] left their boxes unticked while the Progress Log was complete. 40 of 75 now ticked, matching what is actually done; 6.x stays unticked because it is blocked, and 5.3/5.4/5.6/7.10 stay unticked until the simulation grid finishes and RESULTS.md is generated from it.
[8.2-8.5] Survive contact with a real pipeline. Composite GitHub Action (observe -> verdict -> report -> comment -> gate, with the gate LAST so the PR comment posts even when the build fails, and a 'judge' verdict exiting 0). CI gives each Hard Rule test its own named job so a failure is unmissable in the checks list. scripts/dogfood.py has benchlock gate its own repository on its own fixtures — all four cases give the right verdict, right exit code, and green replay. Concurrency: 20 parallel writers produce a valid chain or clean refusals, never a corrupt one, and the 100-repetition stress test passes. 20 edge cases (zero variance, single observation, nulls, out-of-range, deleted anchor item, index gaps, out-of-order runs, sequence gaps, 6MB ledger, empty streams). 11 first-use tests over the eight most likely mistakes, each asserting an actionable message with no stack trace. Fixed a real bug the edge cases found: a perfectly flat stream (a saturated suite where every item scores 5) crashed decide() with a raw ValueError from the scale inverse — decide() must be total on any legal ledger content, so the noise floor is now floored before scaling.
[9.2, 6.6, 7.8] Docs. bench/demo.py captures all three demos from real runs over committed fixtures and asserts each matches its known ground truth (judge/system/indeterminate). docs/identification.md writes the difference-in-differences argument out properly with the four assumptions STATED AS ASSUMPTIONS, including the one that is weaker than it looks — the suite hash covers item ids, not item content. docs/provisioning.md has the worked table and explains why the formula uses the truncated-bet growth rate rather than the quadratic KL rate. docs/threat-model.md is GENERATED from the measured adversarial JSON so the prose cannot drift from the numbers; it names anchor evasion as the weakest link in its opening section and does not soften it. 7.8 (judge noise spike without a version change) is argued rather than measured: a noisier judge is not a drifted judge, reporting 'judge' would tell the team to rebaseline which fixes nothing, and the honest treatment is a distinct variance warning — which is NOT implemented, and that gap is listed rather than papered over.
[9.1, 9.3] README and CI. The README is GENERATED by scripts/gen_readme.py — hand-writing it is how an invented number gets in, so the prose lives in the generator and every figure is read from bench/results/ with the Demo 1 block lifted verbatim from docs/demo.md. Structure per the spec: thesis, Demo 1 in full, the two headline numbers, 'Attacks that work against Benchlock' near the TOP with the measured table, the mechanism in three lines, install, the five verdicts, prior art with the preprint credited in the first sentence, and eleven known limitations on the first screen. scripts/check_numbers.py enforces Hard Rule 5 by tracing every numeric literal in README.md and RESULTS.md back to a committed JSON — it immediately caught the arXiv id, which is a citation rather than a measurement and is now narrowly exempted. Reading the numbers fresh from JSON rather than from memory caught a real drift: the peeking t-test false-alarm rate was 0.212 in an earlier run and 0.206 after regeneration.
[5.3, 5.4, 5.6, 7.10] Full 192-cell grid done (100 seeds/cell, 44 min). RESULTS.md is GENERATED and reports all eight methods with false alarm, ARL0, detection rate, judge->system and system->judge misattribution and indeterminate rate, always together — either metric alone is a way to win a benchmark without being useful. THE ANTI-RESULT, published as its own table: B1 detects a real regression in a mean-median 3.1 runs, Benchlock in 24.6 — EIGHT TIMES SLOWER — while B1 false-alarms on 20.6% of healthy pipelines and Benchlock on 0.0%. The README says plainly that if a false rollback is cheap and slow detection is expensive, B1 is the better tool. DECISION GATE 3 CLEARED WITH A MARGIN: judge->system misattribution across the grid is 0.13 for B5 (no anchor) and the naive baselines, 0.19 for the peeking t-test, and 0.01 for full Benchlock — a 13x reduction, with 0.10 landing in 'indeterminate' rather than being guessed. Five committed plots, each regenerating from committed data. Fixed the delay plot: B1 was hidden under B2 and the 0.000 bar drew nothing, so the most important number on the panel was invisible.
[9.1b] Hard Rule 5 tightened. gen_results.py computes aggregates (means across grid cells) that appeared in RESULTS.md but existed in no committed file, so check_numbers.py correctly rejected them. Rather than loosen the check, the generator now writes bench/results/summary.json containing every aggregate it renders — a number a reader cannot find in the results directory is not traceable, whatever its provenance.
[FINAL] All phases complete except Phase 6, which is blocked on a judge API key. 69/75 checkboxes ticked; the six open are 6.1-6.5 and the Definition-of-Done item that depends on them, all annotated in place as NOT MET rather than quietly left blank. 459 tests pass including every slow and mandatory one. ruff + ruff format + mypy --strict clean across 39 source files. Every number in README.md and RESULTS.md traces to a committed run. benchlock gates its own repository: 4/4 fixtures give the right verdict, the right exit code, and green replay. Wheel builds and the full init->baseline->observe->verdict->gate->replay workflow runs from a clean-environment install. NOT published to PyPI — outward-facing and irreversible, left for an explicit go-ahead. scripts/check_done.py found two bugs in itself while verifying the Definition of Done, one of which would have reported a false pass on the very item that is not met.
[6.1, 6.2, 6.3, 6.4, 6.5] Tier 2 RAN FOR REAL once an ANTHROPIC_API_KEY arrived: 400 HelpSteer2 items stratified by human helpfulness, four judge configurations on claude-sonnet-5 and claude-haiku-4-5 (baseline, other snapshot, strict rubric, higher effort), 2,456 calls, $3.05, zero API errors, 144 safety refusals. Current models reject 'temperature' and Haiku 4.5 rejects 'effort', so params are per model. Findings: judge self-disagreement at its most deterministic setting 18.90% (36.3% of items varied at least once over five identical calls with a cache-busting nonce); run-to-run wobble 2.06x what independent per-item noise predicts, so the shared_sd term in provisioning is real; the refusal rate moved from 2.3% to 15.0% on a rubric-only edit and tripled on an effort-only change — refusals are drift that a score-watching tool cannot see, now a stated limitation. Benchlock 5/6 scenarios right (missed the effort change, which moved refusals not scores); every baseline 2/6. No OpenAI key, so the cross-provider configuration was not run and RESULTS.md says so. The concurrent scorer had a silent-failure bug (future.result() never called), rewritten to record and refuse on failures.
[AUDIT] Post-build adversarial audit (6 finder lenses x 3 refuting verifiers) — confirmed gaps, all fixed, all verified by reproduction: (1) min_detectable_shift returned a FINITE magnitude when no shift was provable in the horizon, which every caller read as 'detectable', so a pure judge shift over a short horizon produced a phantom SYSTEM verdict — it now returns inf, min_feasible_horizon() gives the floor (29 runs at alpha 0.05), ProvisioningImpossibleError says 'more anchors cannot help', and the human report says 'no judge shift of any size could have been proven yet'; reproduction now yields indeterminate/race_failed. (2) The e-value reported was the endpoint, not the peak, so a transient crossing read as never crossing — EDetector tracks peak log-e and crossed() is sticky, consistent with alarm_time. (3) _pin_state compared pins across streams — now per stream, every run against that stream's first; judge swap-and-back yields pin_violation. (4) The corrected stream's scale was data-dependent — baseline_sd = sqrt(system_sd^2 + anchor_sd^2), fixed before monitoring. (5) Adversarial 7.7 was an identity no-op — it now builds real zero-variance and zero-variance-then-step streams. (6) 7.9 claimed the nonce's effect on a real judge was measured — relabelled UNMEASURED. (7) --only clobbered adversarial-latest.json — separate file. (8) judge-refusal-rates.json had no committed producer — bench/real/refusal_rates.py derives it from the run log. (9) writeup.md carried real-judge numbers as literals — read from JSON. (10) README did not say which numbers were simulated and which real — labelled, with new limitations: adapter fixtures are doc-derived, the GitHub Action is not exercised in CI. (11) Both judge adapters raised on refusal before reading usage, so refused calls' tokens went uncounted — usage is accounted first, refusals counted; the committed cost.json predates the fix and RESULTS.md discloses it. (12) compose.py's system-regression scenarios subtract a fixed 0.10 from real scores and the item set is intersected across configs — disclosed in METHODOLOGY. (13) Nothing enforced that generated docs matched their generators — scripts/check_generated.py regenerates all six and fails on a diff, wired into CI. (14) provisioning.md's worked example stated the wrong item count — 40. Because decide() changed, EVERY simulation number was regenerated against the final engine (Hard Rule 5): headline 21.2% / 7.4% / 0.0% false alarms, judge cell 100% 'judge'; Tier 2, demos and dogfood re-run and unchanged in verdict.
[5.3, 5.4, 5.6, 7.10, AUDIT] Grid REGENERATED against the final engine (192 cells, 100 seeds/cell, horizon 60) after the audit fixes changed decide(). Substantively unchanged, which is the point of re-running rather than assuming: B1 detects a real regression in a mean-median 3.2 runs, Benchlock in 25.1 (7.7x; was 3.1 / 24.6 / 8.1x); B1 false-alarms on 23.6% of drift-free cells, Benchlock on 0.0%; judge->system misattribution 0.005 for Benchlock (was 0.006) vs 0.126 for B5 without the anchor and 0.189 for the peeking t-test; 10.1% of cells land in indeterminate rather than being guessed; Benchlock detection rate 82.2%. Every generated document and all five plots re-rendered from the new JSON; check_numbers and check_generated pass. Also: the CI workflow has now run on a real GitHub runner (all ten jobs green) after one fix — confseq's sdist pins pybind11~=2.6, which cannot compile on Python 3.12 / GCC 13, so CI builds the oracle against a current pybind11 and fails loudly if it is not importable rather than letting the differential skip.
[AUDIT-2] Second adversarial audit before public release (8 finder lenses x 3 refuting verifiers; 63 of 131 agents died on a session limit, so several lenses went UNVERIFIED and were checked by hand instead). Two CRITICAL defects made the shipped tool unusable in its own documented workflow, both invisible to every benchmark because no benchmark drives the CLI: (1) _judge_adapter returned the simulated judge or died, so no hosted judge was reachable at all and the README quickstart failed at its second command — AnthropicJudge/OpenAIJudge were complete and had produced the $3.05 Tier 2 study, but only bench/real/build_pool.py could reach them; (2) anchor runs were numbered by a per-stream counter that already contained the K noise-floor replicates while _corrected_stream joins the legs on run_index, so every CLI ledger compared system run j against the anchor scoring taken K runs earlier. Three more broke the published numbers: Tier 1 seeds came from Python's per-process-salted hash() so NO published number could be regenerated (now SHA-256); the Tier 2 anchor set and system suite were the identical 150 items, making the difference-in-differences vacuous (now disjoint halves, and Benchlock honestly drops 5/6 -> 4/6); and ADWIN was fed run means while every other baseline got item-level scores AND both ADWIN and DDM had their transient drift flag read once per run, discarding 199 of every 200 alarms — ADWIN's published detection rate goes 0.000 -> 0.934. Also fixed: observe ignored system.adapter so the promptfoo/Inspect/DeepEval loaders were unreachable; --rescore-anchors skipped the anchor-pin check that the other path performs; replay re-derived verdicts under its own --target-shift and called the difference a regression (target_shift is now recorded per verdict); --anchor-store was write-only; anchor.cadence was inert; the cache-busting nonce never reached baseline; directories, binaries and truncated anchor stores raised tracebacks; init never wrote the .gitignore the code assumed, so the quickstart staged raw eval content; the composite Action installed from PyPI and had no way to authenticate; the PR comment reported E_corrected as 'not crossed' unconditionally; the saturated flag was computed and dropped so a -0.30 move was reported as -0.150 with an interval excluding it; judge->system misattribution was averaged over cells where it cannot occur (0.00 -> 0.035, worst cell 49%); docs/launch.md and CHANGELOG.md carried numbers from a superseded run and were outside both checkers; and the slow tests that PRODUCE every number in statistical-guarantees.md ran in no CI job. Documented rather than fixed: the verdict-level false-alarm bound is 2*alpha not alpha (a verdict fires when either of two alpha-level processes crosses), and rule 5 has no power guard so a regression arriving with judge drift can read as 'judge', pass the gate, and be absorbed by the recommended rebaseline. 10 regression tests added in tests/test_audit_regressions.py.
```

---

## Final Report

### What was built, in five sentences

Benchlock is an installable CLI that sits beside an existing LLM-as-judge eval pipeline,
ingests the score stream it already produces, and answers one question: did your system
regress, or did the judge change underneath you. It holds a frozen anchor set the system
under test never touches, monitors both streams with anytime-valid sequential tests built
on betting e-processes, and returns one of five verdicts from a pure function containing no
model, no prompt and no heuristic. The fifth verdict, `indeterminate`, fires whenever the
anchor set lacked the statistical power to have ruled the judge out — which is the only
thing separating a guarantee from a decoration. Every verdict is recorded in a hash-chained
ledger and can be re-derived years later by `benchlock replay`. It needs no human labels,
because attribution needs judge *stability* rather than judge *validity*, and a snapshot of
the judge's own scores answers that exactly.

### The two headline numbers, with the commands that produced them

**1. False alarms under peeking.** Over 500 drift-free streams, each inspected after every
one of 142 runs, at alpha=0.05:

- a t-test re-run at every run raises at least one false alarm on **20.8%** of perfectly
  healthy pipelines
- Bonferroni-corrected over runs-so-far: **7.0%**, still above its own alpha
- Benchlock: **0.0%**

**2. Misattribution under a silent judge change.** The same pipeline with only the judge
moved:

- every single-stream method — fixed threshold, t-test, Bonferroni, CUSUM, ADWIN, DDM, and
  Benchlock with its anchor stream removed — reports "regression" **100%** of the time
- Benchlock reports `judge` **100%** of the time

        uv run python bench/sim/run_sim.py --headline --seeds 500 --horizon 150

### The anti-result: where Benchlock is worse

**Benchlock detects a real regression roughly 7x slower than the invalid peeking t-test.**
Mean median detection delay across the grid: **3.5 runs** for the t-test, **24.6 runs** for
Benchlock. That is the price of a guarantee that survives daily looking, it is a headline
row in RESULTS.md rather than a footnote, and the README says plainly that if a false
rollback is cheap and slow detection is expensive, the t-test is the better tool.

Bonferroni is also *faster* than Benchlock and still holds a lower false-alarm rate than
the raw t-test — it is a real method, not a strawman, and its 7.0% is the honest reason it
is not sufficient rather than a reason to dismiss it.

### The attacks that beat it, with measured rates

        uv run python bench/adversarial/run_adversarial.py --all

| attack | failure rate |
|---|---:|
| anchor evasion — a judge change outside the anchors' coverage | **100%** |
| anchor staleness — your own standard changed | **100%** |
| input distribution shift with stable item ids | **100%** |
| provider-side response caching, without a nonce | **100%** |
| cancellation — judge and system offsetting | **25%** |
| slow ramp — 1 of 5 tested ramp rates never detected | **20%** |
| adversarial within-run ordering | 0% |
| heavy tails and bound violations | 0% |

**Anchor evasion is the fundamental limitation and nothing removes it.** Two of my own
attacks were initially measuring nothing and I rewrote them: 7.5 renamed every item id, so
all per-item deviations became zero and the "defence" was an artefact; 7.2 reported its best
case rather than the fraction of ramp rates that evaded. The rewritten 7.5 exposed a
genuine hole — hashing item *ids* does not hash item *content*, so the fixed-suite check is
blind to a traffic-mix shift that preserves ids.

### The post-build audit: what it found

After the build was declared done, an adversarial audit — six independent finder lenses,
each finding checked by three verifiers instructed to refute it — ran over the code, the
benchmarks and the documents. Fourteen findings survived refutation; all fourteen are
fixed and each fix was verified by reproducing the original failure. The four that changed
behaviour are decisions 12–14 below plus the corrected-stream scale, which was
data-dependent and is now `sqrt(system_sd² + anchor_sd²)`, fixed before monitoring
begins. The rest were honesty defects: an adversarial attack that was an identity no-op,
a mitigation described as measured when it was not, real-judge numbers carried as string
literals in a generated document, a README that did not say which numbers were simulated,
judge adapters that dropped a refused call's tokens from the cost, and no check that a
generated document still matched its generator. `scripts/check_generated.py` now closes
that last one in CI. Because `decide()` changed, every simulation number in this report
and in the generated documents was regenerated against the final engine.

### Every decision this spec did not specify, and why

1. **Item ids are stored hashed in the ledger.** In real suites an item id is frequently the
   prompt text or a customer identifier. The statistics need stable identity, not the
   original string. Verified with a negative control: storing raw ids makes the 15-format
   secret test fail.
2. **The e-detector uses a *weighted* sum over change points, not the unweighted
   Shin-Ramdas-Rinaldo form.** Unweighted buys an average-run-length guarantee; weighting
   buys the time-uniform "probability of ever alarming" bound that Hard Rule 3 demands. The
   cost is power against very late change points.
3. **The system stream's baseline is treated as a frozen snapshot, exactly like the
   anchor's.** Monitoring raw run means against an anytime-valid interval for a bounded mean
   over 8 runs gives a null ~0.6 wide, and `E_system` then never crosses for any regression
   anyone would care about. This was forced by measurement, not preference.
4. **The anchor-corrected quantity is monitored as its own difference-in-differences
   e-process** rather than derived by subtracting two intervals, whose widths add. This is a
   documented deviation from the spec's rule-4 wording and is strictly stronger.
5. **Rule 4 no longer requires the raw system stream to have crossed.** The spec's literal
   rule returns `STABLE` for cancellation — a judge that got stricter while the system
   improved by the same amount — which Phase 7.3 calls the nastiest case in the design.
6. **Detection and estimation use different scaling bands.** One band for both was
   reporting a real -0.12 move as -0.05, because -0.05 was the band edge. Both bands are
   fixed before monitoring begins so the interval stays anytime-valid.
7. **`EDetector` was rewritten over numpy arrays**, 15x faster, with the scalar version kept
   as `ReferenceEDetector` and checked against observation-by-observation. Without it the
   simulation study was infeasible.
8. **The README, RESULTS.md, threat-model.md and writeup.md are all generated** from
   committed JSON. Hand-writing them is how an invented number gets in.
9. **`benchlock plan`'s growth model uses the truncated-bet rate, not the quadratic KL
   rate.** Simulation disagreed with the formula twice and the formula changed both times,
   per the spec's instruction.
10. **The scaling band was measured, not assumed.** The obvious "safely wide" choice of 10
    standard deviations cost about a third of the detectable-shift resolution.
11. **Not published to PyPI.** Publishing is outward-facing and irreversible; the wheel is
    built and verified on a clean environment, awaiting an explicit go-ahead.
12. **`min_detectable_shift` returns infinity, not a number, when no shift is provable in
    the horizon.** Every caller treated a finite value as "detectable", so a short horizon
    could turn a pure judge shift into a confident `system` verdict. `min_feasible_horizon`
    now states the floor — 29 monitored runs at alpha=0.05, from the truncated-bet growth
    cap — and the provisioning error says that more anchors cannot help.
13. **The e-value a verdict answers for is the peak over time, not the endpoint.** Ville's
    inequality bounds the supremum; reporting the endpoint let a transient crossing read as
    "never crossed". `crossed()` is sticky and agrees with `alarm_time`.
14. **Pin state is tracked per stream, against that stream's first run.** Comparing pins
    across streams let a judge swap-and-back pass silently.

### Everything in the Definition of Done that is not ticked, and why

**14 of 14 met**, verified by `uv run python scripts/check_done.py` against artefacts rather
than memory. Phase 6 was blocked on a missing API key for most of the build and was
completed afterwards against real hosted judges.

### What the real judges said

Tier 2 ran against `claude-sonnet-5` and `claude-haiku-4-5` over 400 HelpSteer2 items
stratified by human helpfulness rating: **2,456 calls, $3.05, zero API errors.**

**Benchlock got 4/6 scenarios right; every baseline got 2/6.** Both misses are on
judge-truth scenarios and both are instructive. The effort change moved the judge's
*refusal* rate without moving its mean score, and a score-watching tool cannot see that.
The rubric change came back `both` rather than `judge`: the corrected process crossed too,
which on a disjoint anchor set is what happens when a rubric edit does not move the two
item sets by the same amount — the parallel-trends assumption the design rests on. That is
the honest cost of the audit fix that made the anchor set disjoint from the system suite;
the earlier 5/6 was measured with the two sets sharing all 150 items, where a judge change
moved two copies of one thing and could not fail to cancel.

**Judge self-disagreement at temperature 0: 18.90%.** Five identical calls per
item with a cache-busting nonce; 36.3% of items came back different at least
once, and the run mean moves by 0.00742 between identical runs. The premise of the
project is therefore real and not merely plausible.

**The judge's noise is correlated across items — independence ratio 2.06.** A bigger
anchor set buys measurably less than `1/sqrt(n)` of precision. The provisioning calculator
models this as a `shared_sd` term, put in on theoretical grounds; the real judge confirmed
it.

**An unlooked-for finding: a judge's refusal rate is itself drift.** Editing only the rubric
text raised it from 2.3% to 15.0% on the same 400 items; raising only the effort tripled it;
and five identical repeats declined between 7 and 10 of the same 200 items. A refused item
leaves the sample silently, and Benchlock does not watch for it. That gap is now in the
README's limitations.

The CI workflow has now run on a real GitHub runner: all ten jobs pass, and the
confseq differential executes there rather than being skipped (the oracle's sdist pins a
pybind11 that cannot compile on Python 3.12 / GCC 13, so CI builds it against a current
one). The composite GitHub Action that wraps the CLI for other repositories is still not
exercised end to end.

Also incomplete, and stated rather than hidden: the framework adapter fixtures are
constructed from each project's *documented* output shape rather than captured from real
runs, because promptfoo, Inspect AI and DeepEval are not installed here. A schema that has
drifted from its docs would pass those tests and fail in the field. What the tests do
guarantee is that an unrecognised shape is refused with a message naming what was found,
never guessed at.

### What I would do differently with another month

1. **Run Tier 2 first, not last.** Every design decision about noise floors and provisioning
   was calibrated against a simulated judge whose noise I chose. Real judge noise may be
   correlated in ways the simulation does not reproduce — and `decompose()` already shows
   that shared noise, which no anchor size can buy down, changes the provisioning answer
   completely.
2. **Hash item content, not just item ids.** The 100% failure rate on id-preserving
   distribution shift is the most fixable hole in the threat model. It needs a content
   digest that satisfies Hard Rule 9, which is a real design problem rather than an
   oversight.
3. **A variance-change signal.** A judge that got noisier without changing its mean is
   currently invisible to the verdict, and the reasoning for that is written down but the
   warning is not implemented.
4. **A longitudinal study.** Tier 2 streams are resampled from pooled real judge scores, not
   observed over time. Nobody has watched a production pipeline for sixty runs with this
   tool attached, and until someone has, the premise rests on construction.
5. **Reduce the detection delay.** 7x is the honest price today, but some of it is the
   change-point prior charging late candidates ~7.5 nats before they can contribute. A
   better-shaped prior, or a windowed detector, would likely recover a meaningful fraction
   of that without touching the guarantee.
