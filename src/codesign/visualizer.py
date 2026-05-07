"""Matplotlib helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codesign.attacker import AttackTrace
    from codesign.interpreter import CircuitReport

log = logging.getLogger(__name__)


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib not installed (pip install codesign[notebook])") from e
    return plt


def plot_attack_trace(trace: AttackTrace, out_path: str | Path) -> Path:
    plt = _mpl()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not trace.steps:
        log.warning("empty trace; skipping plot")
        return out

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    xs = list(range(len(trace.steps)))
    ys = [s.score_after for s in trace.steps]
    ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label="vuln. score")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="decision boundary")
    ax.set_xlabel("MAB iteration")
    ax.set_ylabel("SVD vulnerability confidence")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Attack trace")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_head_heatmap(report: CircuitReport, out_path: str | Path) -> Path:
    plt = _mpl()
    import numpy as np

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    grid = np.zeros((report.n_layers, report.n_heads), dtype=float)
    for h in report.head_logit_diffs:
        if 0 <= h.layer < report.n_layers and 0 <= h.head < report.n_heads:
            grid[h.layer, h.head] = abs(h.logit_diff)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    im = ax.imshow(grid, aspect="auto", cmap="magma")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"|logit_diff| per head: {report.model_name}")
    fig.colorbar(im, ax=ax, label="|logit_diff|")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_diff_heatmap(
    clean: CircuitReport,
    adv: CircuitReport,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Render a diff heatmap: |logit_diff_clean| - |logit_diff_adv|.

    Red cells: head was important on clean code, less so on adversarial
    (attack suppressed it). Blue cells: head was more important on the
    adversarial input than on the clean one (attack activated it).
    """
    plt = _mpl()
    import numpy as np

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if (clean.n_layers, clean.n_heads) != (adv.n_layers, adv.n_heads):
        raise ValueError("clean and adv reports must share grid shape")

    L, H = clean.n_layers, clean.n_heads
    g_clean = np.zeros((L, H))
    g_adv = np.zeros((L, H))
    for h in clean.head_logit_diffs:
        g_clean[h.layer, h.head] = abs(h.logit_diff)
    for h in adv.head_logit_diffs:
        g_adv[h.layer, h.head] = abs(h.logit_diff)
    diff = g_clean - g_adv

    vmax = float(max(abs(diff.min()), abs(diff.max()), 1e-9))
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title or f"clean - adv |logit_diff|, model={clean.model_name}")
    fig.colorbar(im, ax=ax, label="Δ|logit_diff|  (red=suppressed, blue=activated)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_qtable_evolution(trace: AttackTrace, out_path: str | Path) -> Path:
    plt = _mpl()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not trace.steps:
        return out

    strategies = sorted({s.strategy for s in trace.steps})
    series: dict[str, list[float]] = {s: [0.0] for s in strategies}
    counts = dict.fromkeys(strategies, 0)
    q = dict.fromkeys(strategies, 0.0)

    for step in trace.steps:
        counts[step.strategy] += 1
        a = 1.0 / counts[step.strategy]
        q[step.strategy] += a * (step.reward - q[step.strategy])
        for s in strategies:
            series[s].append(q[s])

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=140)
    for s in strategies:
        ax.plot(series[s], label=s, linewidth=1.4)
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xlabel("MAB iteration")
    ax.set_ylabel("Q")
    ax.set_title("Strategy Q-values")
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
