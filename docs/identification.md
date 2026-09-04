# The identification argument

What Benchlock claims, what it assumes, and where the argument stops.

## The claim

Given a **fixed eval suite** and a stream of judge scores, Benchlock separates movement
caused by the system under test from movement caused by the judge.

That is the whole claim. It is narrower than "we solved judge drift", and deliberately so.

## The construction

Two streams are monitored:

| stream | what it scores | who can move it |
|---|---|---|
| **system** | the output of the system under test, judged each run | the system, or the judge |
| **anchor** | a frozen set of `(item_id, system_output)` pairs, re-judged each run | **only the judge** |

The anchor set's outputs are frozen at baseline and never regenerated. The system under
test does not touch them. So:

- the anchor score moved ⇒ **only the judge can have moved it**
- the system score moved and the anchor did not ⇒ **the system moved**
- both moved ⇒ **confounded**, and Benchlock says so rather than picking

This is a difference-in-differences design. The anchor set is the control group; the judge
is the shared time effect. Formally, writing `S_t` for the system stream's mean and `A_t`
for the anchor's, with `σ_t` the judge's state at run `t` and `π_t` the system's:

```
S_t = f(π_t, σ_t) + ε_t
A_t = f(π_0, σ_t) + η_t        ← π is pinned at its baseline value by construction
```

`A_t` varies only through `σ_t`, so it identifies the judge's contribution. `S_t − A_t`
removes it, leaving the system's. That difference is monitored directly as its own
e-process rather than derived by subtracting two intervals, because interval widths add.

It is not clever. It works precisely because it is deterministic and requires no model to
adjudicate it (Hard Rule 1).

## Why this needs no human labels

The observation that makes `frozen-self` mode work:

> **Attribution needs judge *stability*, not judge *validity*.**

We are not asking whether the judge is right. We are asking whether it is the same judge it
was last week. A frozen snapshot of the judge's *own* scores answers that exactly, and it
costs nothing to produce — no labelling project, no gold set, no multi-week adoption cost.

The consequence, stated plainly because it is a real limitation:

> A judge that was always wrong stays consistently wrong, and correctly reads as **stable**.
> Benchlock does not claim your evals are correct. It claims to tell you which of two
> things moved.

`tests/test_anchor.py::test_a_biased_but_stable_judge_reads_as_stable` asserts exactly
this over 200 runs. The tool must not confuse "wrong" with "drifting".

## The assumptions, stated as assumptions

**1. The eval suite is fixed.**
If the suite's items change between runs, score movement has a third cause that neither
stream covers, and the identification collapses. Lattice rule 2 compares `suite_hash`
between runs and refuses to attribute when it moves.

*Where this is weaker than it looks:* the suite hash is a hash of item **ids**, not item
**content**. A suite whose 200 test slots are regenerated monthly from production traffic
keeps its ids and its hash while changing entirely underneath. Phase 7.5 measures this and
it is undefended — see `threat-model.md`.

**2. The anchor set is representative of the eval distribution.**
A judge change confined to a region the anchors do not cover is invisible to the anchor
stream while still moving the system stream, and the verdict is then `SYSTEM` — confidently
wrong. This is the **fundamental limitation of the design**. Stratified selection makes it
less likely, coverage is measured and warned about, and Phase 7.1 quantifies the damage.
Coverage is measured; it is never guaranteed.

**3. The judge's noise floor is stationary during the baseline period.**
The K replicates that establish the floor must be taken under conditions representative of
what follows. A judge that was unusually quiet during baseline yields a floor that is too
low and a detector that false-alarms.

**4. Scores are bounded and the bound is declared.**
The betting e-processes require observations in `[0, 1]`. An undeclared or violated range
raises rather than clamping (Hard Rule 10) — clamping would compress real movement into the
bound and make a drifting stream look stable.

## What the argument does not license

- **It says nothing about *why* the judge moved.** A provider snapshot rotation and a
  teammate's rubric edit are indistinguishable unless you declared one of them. Phase 7.4
  measures this; the answer is `benchlock rebaseline --reason`.
- **It says nothing about whether your evals measure the right thing.** See "stability, not
  validity" above.
- **It does not extend to multiple judges or multiple suites.** One judge, one suite, one
  system. Ensembles and portfolios are v2.
- **It is not a causal claim about your product.** It is a claim about which of two
  measurement components moved, given a fixed instrument and a fixed set of questions.
