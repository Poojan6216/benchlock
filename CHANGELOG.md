# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

First release.

### Added

- **Attribution.** `decide()` separates eval-score movement into judge, system, both,
  neither, or `indeterminate`, using a difference-in-differences design where a frozen
  anchor set is the control group. Pure arithmetic; no model in the decision path.
- **Anytime-valid statistics.** Betting e-processes, hedged-capital and
  empirical-Bernstein confidence sequences, and a bounded-memory e-detector whose pruning
  can only delay an alarm, never create one. Validated against `confseq` to exact
  agreement over 12,500 (stream, time) pairs.
- **`frozen-self` anchor mode**, which needs no human labels, plus optional `human` and
  `replicate` modes.
- **Provisioning.** `benchlock plan` sizes the anchor set to satisfy the design law, and
  refuses outright when the judge's shared noise makes a target unreachable.
- **Tamper-evident ledger.** Append-only, hash-chained, holding scores and hashes only —
  item ids are stored hashed, because in real suites an item id is often the prompt text.
- **`benchlock replay`**, which re-derives every historical verdict from its own prefix
  and reports both verdicts rather than rewriting history when the semantics version bumps.
- **CLI**: `init`, `plan`, `baseline`, `observe`, `verdict`, `gate`, `replay`,
  `rebaseline`, `report`.
- **Adapters** for promptfoo, Inspect AI, DeepEval and a universal JSONL contract. Each
  refuses an unrecognised schema rather than guessing at it.
- **A GitHub Action** that gates CI and posts the verdict as a PR comment.

### Measured

- False alarms on drift-free streams inspected after every run: **0.0%**, against
  **20.8%** for a t-test re-run at every run.
- Attribution under a silent judge change: **100%** correct, where every single-stream
  method reports a regression.
- Detection delay: **8× slower** than the invalid peeking t-test. That is the price of the
  guarantee and it is published in `RESULTS.md`, not hidden.
- Six of eight adversarial strategies beat Benchlock at some rate. They are in
  `docs/threat-model.md`.
- Against real judges (2,456 calls, $3.05): 5 of 6 scenarios correct, where every
  single-stream baseline gets 2. A temperature-0 judge disagrees with itself on
  **18.9%** of identical calls.

### Not in this release

- A cross-provider judge configuration. The real-judge study ran four Anthropic
  configurations; no OpenAI key was available for the fifth.
- A refusal-rate monitor. The study found a judge's *refusal* rate moves with its
  configuration (6.7x on a rubric edit) and Benchlock does not watch for it.
- Multi-judge ensembles and multi-suite portfolios.
