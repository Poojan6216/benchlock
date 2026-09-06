"""`decide()` — the pure decision function. The heart of the project.

Given the system stream and the anchor stream, work out which of the two moved. There is
no model in here. The judge produces numbers; this consumes numbers and returns a verdict
using arithmetic (Hard Rule 1).

**The identification argument, in three lines.** The system under test never touches the
anchor items, so:

* the anchor score moved  =>  only the judge can have moved it
* the system score moved and the anchor did not  =>  the system moved
* both moved  =>  confounded, and we say so rather than picking

That is a difference-in-differences design where the anchor set is the control group and
the judge is the shared time effect. It is not clever. It works because it is deterministic
and needs no model to adjudicate it.

**Purity (Hard Rule 7).** No I/O, no clock, no randomness, no network. The same
``(system, anchor, config)`` yields byte-identical output forever, which is what makes
`benchlock replay` possible and what makes a historical verdict auditable rather than
merely remembered.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace

from benchlock.attribute.lattice import apply_lattice
from benchlock.attribute.race import check_race
from benchlock.config import AttributionConfig
from benchlock.model.pins import NoiseFloor
from benchlock.model.streams import RunRecord, StreamKind
from benchlock.model.verdict import Attribution, AttributionRefusedError, Evidence, Provisioning
from benchlock.stats.confseq import Interval, empirical_bernstein_cs
from benchlock.stats.edetector import EDetector
from benchlock.stats.eprocess import (
    BaselineNull,
    MonitorScale,
    frozen_baseline_null,
    split_alpha,
)

#: Smallest run-to-run SD we will scale by. Below this a stream is effectively constant and
#: any movement at all is enormous in scaled units.
_MIN_SD = 1e-6


def subtract_intervals(minuend: Interval, subtrahend: Interval) -> Interval:
    """Interval difference, with the uncertainty of *both* sides carried through.

    The correction is itself an estimate, so::

        corrected = [ sys.lower - anchor.upper ,  sys.upper - anchor.lower ]

    and the width is the sum of the two widths. Subtracting point estimates and keeping
    the original width is the single most likely place to accidentally manufacture a false
    guarantee: it would produce a corrected interval that looks as precise as the system
    measurement alone, while actually depending on a second measurement that has its own
    error. By a union bound this construction covers the true difference with at least the
    combined confidence of its inputs.
    """
    return Interval(
        minuend.lower - subtrahend.upper,
        minuend.upper - subtrahend.lower,
    )


@dataclass(frozen=True, slots=True)
class StreamAnalysis:
    """One stream's monitoring result. Internal to `decide`."""

    e_value: float
    crossed_at: int | None
    shift: Interval
    n_monitored: int
    #: True when a deviation ran past the estimation band, so the magnitude is a lower
    #: bound rather than a measurement.
    saturated: bool = False


def _run_deviations(runs: Sequence[RunRecord], baseline: dict[str, float]) -> list[float]:
    """Per-run mean deviation from a frozen per-item baseline.

    Items missing from a run are skipped rather than imputed: a run that scored a
    different item set is caught by the suite-hash check, and silently substituting a
    value here would hide it.
    """
    out: list[float] = []
    for run in runs:
        deltas = [o.score - baseline[o.item_id] for o in run.observations if o.item_id in baseline]
        out.append(statistics.fmean(deltas) if deltas else 0.0)
    return out


def _per_item_baseline(runs: Sequence[RunRecord]) -> dict[str, float]:
    """Mean score per item across the baseline replicate runs."""
    totals: dict[str, list[float]] = {}
    for run in runs:
        for obs in run.observations:
            totals.setdefault(obs.item_id, []).append(obs.score)
    return {item: statistics.fmean(scores) for item, scores in totals.items()}


def estimation_scale(target_shift: float, baseline_sd: float) -> MonitorScale:
    """The scale the *shift interval* is computed on, as opposed to the detector's.

    Detection and estimation want different bands, and using one band for both gets one of
    them wrong:

    * **Detection** wants the band at the target shift, so the target reaches the edge and
      contributes full evidence per run.
    * **Estimation** must not clip. A clipped observation saturates, and the reported
      magnitude is then pinned to the band edge rather than measured — a stream that
      really moved -0.12 was being reported as -0.05, because -0.05 was the band.

    So estimation uses a deliberately generous band. Its width matters: a betting interval
    on [0,1] has a width driven by the range rather than by the observed variance, so the
    raw width comes out roughly proportional to the band. Four times the target shift
    keeps real regressions well inside it while holding the interval tight enough to
    exclude zero when the movement is real.

    Both bands are fixed **before monitoring begins**, from the declared target and the
    baseline period, so neither is a data-dependent transform of the stream being
    monitored and the interval stays anytime-valid.
    """
    return MonitorScale(center=0.5, half_width=max(4.0 * target_shift, 20.0 * baseline_sd))


def _monitor_stream(
    deviations: Sequence[float],
    null: BaselineNull,
    scale: MonitorScale,
    estimate_scale: MonitorScale,
    alpha_monitor: float,
    max_candidates: int,
) -> StreamAnalysis:
    """Run the detector and the shift interval over one stream's deviations."""
    if not deviations:
        return StreamAnalysis(
            e_value=0.0,
            crossed_at=None,
            shift=Interval(0.0, 0.0),
            n_monitored=0,
            saturated=False,
        )
    scaled = [scale.to_unit(d) for d in deviations]
    detector = EDetector(null, alpha_monitor, max_candidates=max_candidates)
    detector.update_many(scaled)

    # Empirical-Bernstein rather than the hedged CS: closed form, so `replay` over a long
    # ledger stays linear, and slightly wider, which is the conservative direction.
    for_estimate = [estimate_scale.to_unit(d) for d in deviations]
    saturated = any(z <= 0.0 or z >= 1.0 for z in for_estimate)
    unit_interval = empirical_bernstein_cs(for_estimate, alpha_monitor)[-1]
    return StreamAnalysis(
        # The PEAK, not the endpoint. `crossed_at` is already sticky, so pairing it with
        # an endpoint e-value let the report say "crossed at run 27" while the lattice
        # read the same stream as never having crossed.
        e_value=detector.peak_e_value,
        crossed_at=detector.alarm_time,
        shift=estimate_scale.raw_interval(unit_interval),
        n_monitored=len(deviations),
        saturated=saturated,
    )


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return _MIN_SD
    return max(statistics.stdev(values), _MIN_SD)


def decide(
    system: Sequence[RunRecord],
    anchor: Sequence[RunRecord],
    config: AttributionConfig,
) -> Attribution:
    """Pure function. No I/O, no clock, no randomness, no network, no LLM.

    Hard Rule 7: same ``(system, anchor, config)`` MUST yield the same `Attribution`,
    forever. This is what makes `benchlock replay` possible.

    Hard Rule 2: if the anchor process could not have detected a judge shift of the
    magnitude observed in the system stream, the verdict is `INDETERMINATE`. Never
    `SYSTEM`.
    """
    alpha = config.alpha
    _, alpha_monitor = split_alpha(alpha)

    # Hard Rule 8: a rebaseline starts a new epoch, and verdicts never compare across the
    # boundary. History before it is retained and replayable — it is simply not mixed in
    # with measurements taken under a different judge or a different anchor set.
    system_runs = [r for r in system if r.kind is StreamKind.SYSTEM]
    anchor_runs = [r for r in anchor if r.kind is StreamKind.ANCHOR]
    current_epoch = max(
        [r.epoch for r in (*system_runs, *anchor_runs)],
        default=0,
    )
    system_runs = [r for r in system_runs if r.epoch == current_epoch]
    anchor_runs = [r for r in anchor_runs if r.epoch == current_epoch]

    pin_delta, pin_rebaselined = _pin_state(system_runs, anchor_runs)
    suite_ok = _suite_hashes_agree(system_runs)

    # --- the anchor stream: a frozen snapshot, so its null is known -----------------------
    anchor_pin = anchor_runs[-1].anchor_pin if anchor_runs else None
    noise_floor = anchor_pin.noise_floor if anchor_pin is not None else _placeholder_floor()
    # A perfectly flat stream is legal — a saturated suite where every item scores 5 has
    # genuinely zero measured variance — but a zero scale has no inverse. Flooring keeps
    # `decide` total on any ledger content, which matters because a ledger can be written
    # by an older version and must still be replayable.
    noise_floor = replace(noise_floor, run_mean_sd=max(noise_floor.run_mean_sd, _MIN_SD))
    anchor_n = anchor_pin.n if anchor_pin is not None else 0
    replicates = noise_floor.replicates

    anchor_analysis = StreamAnalysis(0.0, None, Interval(0.0, 0.0), 0)
    anchor_mds = float("inf")
    if len(anchor_runs) > replicates:
        baseline_scores = _per_item_baseline(anchor_runs[:replicates])
        deviations = _run_deviations(anchor_runs[replicates:], baseline_scores)
        anchor_scale = MonitorScale.for_target(config.target_shift, noise_floor.run_mean_sd)
        anchor_null = frozen_baseline_null(
            noise_floor.run_mean_sd, replicates, alpha, scale=anchor_scale
        )
        anchor_analysis = _monitor_stream(
            deviations,
            anchor_null,
            anchor_scale,
            estimation_scale(config.target_shift, noise_floor.run_mean_sd),
            alpha_monitor,
            config.max_candidates,
        )

    # --- the system stream: its own baseline is a frozen snapshot too ---------------------
    # The eval suite is fixed (rule 2 guarantees it), so the system's baseline runs are a
    # snapshot of the system's own per-item scores, exactly as the anchor's are of the
    # judge's. Monitoring per-item deviations from that snapshot rather than raw run means
    # is what makes the system stream detectable at all: a worst-case anytime-valid
    # interval for a bounded mean over eight run means is roughly 0.6 wide, which is wider
    # than any regression anyone would care about, so nothing could ever be rejected.
    # Against a snapshot the only uncertainty is how precisely it was measured,
    # `sd / sqrt(baseline_runs)`, which is two orders of magnitude smaller.
    system_analysis = StreamAnalysis(0.0, None, Interval(0.0, 0.0), 0)
    system_sd = _MIN_SD
    if len(system_runs) > config.baseline_runs:
        baseline_runs = system_runs[: config.baseline_runs]
        system_baseline = _per_item_baseline(baseline_runs)
        baseline_devs = _run_deviations(baseline_runs, system_baseline)
        monitor_devs = _run_deviations(system_runs[config.baseline_runs :], system_baseline)
        system_sd = _sd(baseline_devs)
        system_scale = MonitorScale.for_target(config.target_shift, system_sd)
        system_null = frozen_baseline_null(system_sd, len(baseline_runs), alpha, scale=system_scale)
        system_analysis = _monitor_stream(
            monitor_devs,
            system_null,
            system_scale,
            estimation_scale(config.target_shift, system_sd),
            alpha_monitor,
            config.max_candidates,
        )

    # --- the corrected stream: difference-in-differences, monitored directly -------------
    # The spec's rule 4 asks whether the anchor-corrected system interval excludes zero.
    # Subtracting two intervals answers that but throws away most of the power: the widths
    # add, so the corrected interval is wider than either input and contains zero even for
    # movements both streams saw clearly. The quantity of interest is per-run and directly
    # observable — (system deviation) - (anchor deviation) at the same run — so we monitor
    # it with its own e-process. That is a stronger test, and an anytime-valid one, which
    # Hard Rule 3 prefers to a comparison of intervals.
    corrected_analysis = _corrected_stream(
        system_runs,
        anchor_runs,
        config,
        alpha,
        alpha_monitor,
        replicates,
        system_sd=system_sd if len(system_runs) > config.baseline_runs else _MIN_SD,
        anchor_sd=noise_floor.run_mean_sd,
    )
    corrected = (
        corrected_analysis.shift
        if corrected_analysis.n_monitored > 0
        else subtract_intervals(system_analysis.shift, anchor_analysis.shift)
    )

    # --- the race: could the anchor have seen a judge shift this big? ---------------------
    observed_system_shift = system_analysis.shift.midpoint
    if anchor_n > 0 and anchor_analysis.n_monitored > 0:
        race = check_race(
            observed_system_shift,
            noise_floor,
            anchor_n,
            anchor_analysis.n_monitored,
            alpha,
            target_shift=config.target_shift,
        )
        anchor_mds = race.min_detectable_judge_shift
        provisioning = race.provisioning
    else:
        # With no anchor stream at all, nothing can rule the judge out.
        provisioning = Provisioning.UNDER_PROVISIONED
        anchor_mds = float("inf")

    evidence = Evidence(
        e_system=system_analysis.e_value,
        e_anchor=anchor_analysis.e_value,
        e_corrected=corrected_analysis.e_value,
        threshold=1.0 / alpha_monitor,
        system_shift=system_analysis.shift,
        anchor_shift=anchor_analysis.shift,
        corrected_shift=corrected,
        crossed_at_system=system_analysis.crossed_at,
        crossed_at_anchor=anchor_analysis.crossed_at,
        min_detectable_judge_shift=anchor_mds,
        provisioning=provisioning,
        judge_pin_delta=tuple(pin_delta),
        n_runs=len(system_runs),
        n_anchor_runs=anchor_analysis.n_monitored,
        anchor_n=anchor_n,
        epoch=current_epoch,
        baseline_judge_model=system_runs[0].judge_pin.model if system_runs else "",
        current_judge_model=system_runs[-1].judge_pin.model if system_runs else "",
        judge_pin_changed_at=_pin_change_index(system_runs),
    )

    return apply_lattice(
        evidence,
        alpha,
        min_runs=config.min_runs,
        min_obs=config.min_obs,
        observations_in_last_run=system_runs[-1].n if system_runs else 0,
        pin_delta=pin_delta,
        pin_rebaselined=pin_rebaselined,
        suite_hashes_agree=suite_ok,
    )


def _corrected_stream(
    system_runs: Sequence[RunRecord],
    anchor_runs: Sequence[RunRecord],
    config: AttributionConfig,
    alpha: float,
    alpha_monitor: float,
    replicates: int,
    *,
    system_sd: float,
    anchor_sd: float,
) -> StreamAnalysis:
    """Monitor ``system deviation - anchor deviation``, run by run.

    This is the difference-in-differences estimator stated as a stream. A judge shift moves
    both legs by the same amount and cancels here; a system change survives. Runs are
    paired by ``run_index``, so a cadence that scores anchors less often simply yields
    fewer paired runs rather than a misaligned comparison.
    """
    empty = StreamAnalysis(0.0, None, Interval(0.0, 0.0), 0, False)
    if len(system_runs) <= config.baseline_runs or len(anchor_runs) <= replicates:
        return empty

    system_baseline = _per_item_baseline(system_runs[: config.baseline_runs])
    anchor_baseline = _per_item_baseline(anchor_runs[:replicates])

    system_devs = {
        r.run_index: d
        for r, d in zip(
            system_runs[config.baseline_runs :],
            _run_deviations(system_runs[config.baseline_runs :], system_baseline),
            strict=True,
        )
    }
    anchor_devs = {
        r.run_index: d
        for r, d in zip(
            anchor_runs[replicates:],
            _run_deviations(anchor_runs[replicates:], anchor_baseline),
            strict=True,
        )
    }
    shared = sorted(set(system_devs) & set(anchor_devs))
    if len(shared) < 2:
        return empty

    paired = [system_devs[i] - anchor_devs[i] for i in shared]
    # The scale and null come from the two HELD-OUT baselines, not from the monitored
    # stream itself. The system and anchor legs each hold out a baseline period; fitting
    # this leg's scale on the first few monitored pairs made it the one data-dependent
    # transform in the engine. The difference of two independent quantities has variance
    # equal to the sum of theirs; if the judge's noise is partly shared between the legs
    # the true SD is smaller, so this over-estimates — a wider band and a wider null, which
    # is the conservative direction.
    baseline_sd = max(math.sqrt(system_sd**2 + anchor_sd**2), _MIN_SD)
    scale = MonitorScale.for_target(config.target_shift, baseline_sd)
    # The corrected stream's null is centred on zero by construction: under "the judge
    # explains the whole move", the two deviations are equal and their difference is zero.
    # Its uncertainty is that of the two *baselines* it was built from — the anchor's K
    # replicates and the system's baseline runs — not the number of runs monitored since.
    # Using the monitored count would make the null tighten simply because time passed,
    # which is not a thing that happens.
    effective_k = max(2, min(replicates, config.baseline_runs))
    null = frozen_baseline_null(baseline_sd, effective_k, alpha, scale=scale)
    return _monitor_stream(
        paired,
        null,
        scale,
        estimation_scale(config.target_shift, baseline_sd),
        alpha_monitor,
        config.max_candidates,
    )


def _placeholder_floor() -> NoiseFloor:
    """Used only when there is no anchor stream; every verdict path then refuses anyway."""
    return NoiseFloor(per_item_sd=1e-6, run_mean_sd=1e-6, replicates=2, n_items=1)


def _pin_state(
    system_runs: Sequence[RunRecord], anchor_runs: Sequence[RunRecord]
) -> tuple[tuple[str, ...], bool]:
    """Which judge fields moved within the current epoch, and whether it was declared.

    A change accompanied by an epoch bump is a rebaseline and is legitimate; a change
    inside one epoch is a Hard Rule 8 violation.
    """
    # Compare within each stream, and against EVERY run rather than just the last.
    #
    # Concatenating the two streams and comparing first-to-last was wrong twice over: it
    # pitted the first system run against the last anchor run, which is a cross-stream
    # comparison rather than a drift check; and comparing only the endpoints missed a pin
    # that changed at run 10 and changed back by run 40 — a judge swapped out and back is
    # exactly as invalidating as one that stayed swapped.
    changed: set[str] = set()
    for stream in (system_runs, anchor_runs):
        if len(stream) < 2:
            continue
        reference = stream[0].judge_pin
        for run in stream[1:]:
            changed.update(run.judge_pin.differs_from(reference))
    if not changed:
        return (), True
    # The pin moved without the epoch moving with it.
    return tuple(sorted(changed)), False


def _pin_change_index(runs: Sequence[RunRecord]) -> int | None:
    """First run index whose judge pin differs from the stream's first run."""
    if not runs:
        return None
    first = runs[0].judge_pin
    for run in runs:
        if run.judge_pin.differs_from(first):
            return run.run_index
    return None


def _suite_hashes_agree(runs: Sequence[RunRecord]) -> bool:
    """Hard Rule 6: attribution requires a fixed suite."""
    if len(runs) < 2:
        return True
    epoch = runs[-1].epoch
    hashes = {r.suite_hash for r in runs if r.epoch == epoch}
    return len(hashes) <= 1


__all__ = ["AttributionRefusedError", "decide", "subtract_intervals"]
