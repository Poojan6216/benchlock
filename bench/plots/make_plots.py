"""Every plot in the repository, and nothing that is not regenerated from committed data.

    uv run python bench/plots/make_plots.py

matplotlib only — no seaborn, no plotly. Each figure is written as a PNG beside this
script, and no PNG exists without the code that made it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # deterministic, headless, no display required
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "bench" / "results"
PLOTS = ROOT / "bench" / "plots"

#: One colour per method, stable across every figure so a reader learns them once.
COLOURS = {
    "B0-fixed-threshold": "#9e9e9e",
    "B1-peeking-t-test": "#d62728",
    "B2-bonferroni-t-test": "#ff7f0e",
    "B3-cusum": "#8c564b",
    "B4-adwin": "#7f7f7f",
    "B4-ddm": "#c7c7c7",
    "B5-benchlock-no-anchor": "#1f77b4",
    "B6-benchlock": "#2ca02c",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": True,
        }
    )


def load(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)} — run the simulation study first:\n"
            "  uv run python bench/sim/run_sim.py --all"
        )
    return json.loads(path.read_text())


def plot_false_alarm(headline: dict[str, Any]) -> Path:
    """The first headline number: false alarms on a drift-free stream, under peeking."""
    _style()
    stable = headline["stable"]
    names = list(stable)
    rates = [stable[n]["alarm_rate"] for n in names]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(range(len(names)), rates, color=[COLOURS[n] for n in names])
    ax.axhline(headline["alpha"], color="black", ls="--", lw=1)
    ax.text(
        len(names) - 0.4,
        headline["alpha"],
        f"  alpha = {headline['alpha']}",
        va="bottom",
        ha="right",
        fontsize=8,
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split("-", 1)[0] for n in names])
    ax.set_ylabel("P(at least one false alarm)")
    ax.set_title(
        f"False alarms on a drift-free stream, inspected after every one of "
        f"{headline['monitored_runs']} runs\n({headline['seeds']} streams per method)",
        fontsize=9,
    )
    for i, rate in enumerate(rates):
        ax.text(i, rate, f" {rate:.3f}", ha="center", va="bottom", fontsize=8)
    path = PLOTS / "false-alarm-under-peeking.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_misattribution(headline: dict[str, Any]) -> Path:
    """The second headline number: what each method concludes when only the judge moved."""
    _style()
    judge = headline["judge"]
    names = list(judge)
    categories = ["judge", "indeterminate", "regression", "system", "both", "stable"]
    palette = {
        "judge": "#2ca02c",
        "indeterminate": "#ff7f0e",
        "regression": "#d62728",
        "system": "#d62728",
        "both": "#e377c2",
        "stable": "#9e9e9e",
    }

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bottoms = [0.0] * len(names)
    for category in categories:
        values = [judge[n]["verdicts"].get(category, 0) / headline["seeds"] for n in names]
        if not any(values):
            continue
        ax.bar(
            range(len(names)),
            values,
            bottom=bottoms,
            color=palette[category],
            label=category,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms = [b + v for b, v in zip(bottoms, values, strict=True)]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split("-", 1)[0] for n in names])
    ax.set_ylabel("fraction of streams")
    ax.set_ylim(0, 1)
    ax.set_title(
        "What each method concludes when ONLY the judge moved\n"
        "single-stream methods have no attribution available: 'regression' is their only "
        "possible answer",
        fontsize=9,
    )
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    path = PLOTS / "misattribution-under-judge-drift.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_delay_vs_false_alarm(sim: dict[str, Any], headline: dict[str, Any]) -> Path:
    """Phase 5.4's anti-result: what the guarantee costs, in runs."""
    _style()
    by_method: dict[str, dict[float, float]] = defaultdict(dict)
    for row in sim["rows"]:
        if row["truth"] != "system" or row["median_delay"] is None:
            continue
        shift = row["system_shift"]
        prior = by_method[row["method"]].get(shift)
        by_method[row["method"]][shift] = (
            row["median_delay"] if prior is None else min(prior, row["median_delay"])
        )

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    # B1 and B2 both sit near zero and overlap exactly, so distinct markers and dashes are
    # what keep the faster method visible rather than hidden under the slower one.
    styles = {
        "B1-peeking-t-test": {"marker": "o", "ls": "-", "ms": 7},
        "B2-bonferroni-t-test": {"marker": "s", "ls": "--", "ms": 5},
        "B6-benchlock": {"marker": "D", "ls": "-", "ms": 6},
    }
    for name, style in styles.items():
        points = sorted(by_method.get(name, {}).items())
        if not points:
            continue
        left.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color=COLOURS[name],
            label=name,
            **style,
        )
    left.set_xlabel("true system shift")
    left.set_ylabel("median detection delay (runs)")
    left.set_title("Detection delay", fontsize=9)
    left.legend(fontsize=8)

    stable = headline["stable"]
    names = ["B1-peeking-t-test", "B2-bonferroni-t-test", "B6-benchlock"]
    rates = [stable[n]["alarm_rate"] for n in names]
    right.bar(range(len(names)), rates, color=[COLOURS[n] for n in names])
    right.axhline(headline["alpha"], color="black", ls="--", lw=1)
    right.text(
        len(names) - 0.5,
        headline["alpha"],
        f" alpha={headline['alpha']}",
        va="bottom",
        ha="right",
        fontsize=8,
    )
    # A zero bar draws nothing, so the value is written above it — otherwise the most
    # important number on the panel is the one the reader cannot see.
    for i, rate in enumerate(rates):
        right.text(i, rate, f" {rate:.3f}", ha="center", va="bottom", fontsize=8)
    right.set_xticks(range(len(names)))
    right.set_xticklabels([n.split("-", 1)[0] for n in names])
    right.set_ylabel("false-alarm rate")
    right.set_ylim(0, max(rates) * 1.25)
    right.set_title("What that delay buys", fontsize=9)

    fig.suptitle(
        "The price of the guarantee: anytime-valid tests detect later than invalid ones",
        fontsize=10,
    )
    path = PLOTS / "delay-vs-false-alarm.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_misattribution_heatmap(sim: dict[str, Any]) -> Path:
    """Misattribution over the (judge shift, system shift) grid, for Benchlock."""
    _style()
    import numpy as np

    judge_shifts = sorted({row["judge_shift"] for row in sim["rows"]})
    system_shifts = sorted({row["system_shift"] for row in sim["rows"]})

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for ax, method in zip(axes, ("B5-benchlock-no-anchor", "B6-benchlock"), strict=True):
        grid = np.full((len(judge_shifts), len(system_shifts)), np.nan)
        for row in sim["rows"]:
            if row["method"] != method:
                continue
            i = judge_shifts.index(row["judge_shift"])
            j = system_shifts.index(row["system_shift"])
            wrong = row["judge_as_system"] + row["system_as_judge"]
            grid[i, j] = wrong if np.isnan(grid[i, j]) else max(grid[i, j], wrong)
        image = ax.imshow(grid, vmin=0, vmax=1, cmap="Reds", origin="lower")
        ax.set_xticks(range(len(system_shifts)))
        ax.set_xticklabels([f"{s:g}" for s in system_shifts])
        ax.set_yticks(range(len(judge_shifts)))
        ax.set_yticklabels([f"{s:g}" for s in judge_shifts])
        ax.set_xlabel("system shift")
        ax.set_ylabel("judge shift")
        ax.set_title(method, fontsize=9)
        ax.grid(False)
        for i in range(len(judge_shifts)):
            for j in range(len(system_shifts)):
                if not np.isnan(grid[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if grid[i, j] < 0.6 else "white",
                    )
    fig.colorbar(image, ax=axes, shrink=0.8, label="misattribution rate")
    fig.suptitle("Misattribution over the grid: what the anchor stream buys", fontsize=10)
    path = PLOTS / "misattribution-heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_arl(sim: dict[str, Any]) -> Path:
    """ARL0 on drift-free streams: how long each method runs before crying wolf."""
    _style()
    values: dict[str, list[float]] = defaultdict(list)
    for row in sim["rows"]:
        if row["truth"] == "stable" and row["arl0"] is not None:
            values[row["method"]].append(row["arl0"])
    names = sorted(values)
    means = [sum(values[n]) / len(values[n]) for n in names]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(range(len(names)), means, color=[COLOURS[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split("-", 1)[0] for n in names])
    ax.set_ylabel("runs before a false alarm (censored at the horizon)")
    ax.set_title(
        "Average run length to a false alarm on drift-free streams\n"
        "censored at the horizon, so every bar is a lower bound",
        fontsize=9,
    )
    path = PLOTS / "arl0.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    headline = load("sim-headline.json")
    sim = load("sim-latest.json")
    written = [
        plot_false_alarm(headline),
        plot_misattribution(headline),
        plot_delay_vs_false_alarm(sim, headline),
        plot_misattribution_heatmap(sim),
        plot_arl(sim),
    ]
    for path in written:
        print(f"  wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
