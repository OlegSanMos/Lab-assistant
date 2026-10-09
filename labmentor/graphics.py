from __future__ import annotations

import textwrap
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .data_analysis import FitResult


def _new_path(out_dir: Path, prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{prefix}_{uuid.uuid4().hex[:8]}.png"


def plot_fit(df: pd.DataFrame, fit: FitResult, out_dir: Path, title: str | None = None) -> str:
    labels = df.attrs.get("labels", {})
    data = df[[fit.x, fit.y]].apply(pd.to_numeric, errors="coerce").dropna()
    x, y = data[fit.x].to_numpy(float), data[fit.y].to_numpy(float)
    x_label, y_label = labels.get(fit.x, fit.x), labels.get(fit.y, fit.y)

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=130)
    ax.plot(x, y, "o", color="#1f6feb", markersize=6, label="Экспериментальные точки")
    xs = np.linspace(x.min(), x.max(), 300)
    ax.plot(
        xs,
        fit.predict(xs),
        "-",
        color="#d1242f",
        linewidth=1.8,
        label=f"МНК: {fit.equation()}, R² = {fit.r2:.5f}",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or f"Зависимость {y_label.split(',')[0]}({x_label.split(',')[0]})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path = _new_path(out_dir, "plot")
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def draw_flowchart(title: str, steps: list[str], out_dir: Path) -> str:
    blocks = [("terminal", "Начало")]
    blocks += [("step", f"{i}. {step}") for i, step in enumerate(steps, start=1)]
    blocks += [("terminal", "Конец")]
    wrapped = [(kind, textwrap.fill(text, 52)) for kind, text in blocks]
    heights = [0.22 * (text.count("\n") + 1) + 0.32 for _, text in wrapped]
    gap, top = 0.38, 0.9
    total = top + sum(heights) + gap * (len(heights) - 1) + 0.3

    fig = plt.figure(figsize=(7, total), dpi=130)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7)
    ax.set_ylim(total, 0)
    ax.axis("off")
    ax.text(
        3.5,
        0.45,
        textwrap.fill(title, 60),
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    y = top
    for index, ((kind, text), height) in enumerate(zip(wrapped, heights, strict=True)):
        width = 2.4 if kind == "terminal" else 5.8
        style = (
            "round,pad=0.02,rounding_size=0.25"
            if kind == "terminal"
            else "round,pad=0.02,rounding_size=0.06"
        )
        face, edge = ("#e6f4ea", "#1a7f37") if kind == "terminal" else ("#eaf2fe", "#1f6feb")
        ax.add_patch(
            FancyBboxPatch(
                (3.5 - width / 2, y),
                width,
                height,
                boxstyle=style,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.4,
            )
        )
        ax.text(3.5, y + height / 2, text, ha="center", va="center", fontsize=10)
        if index < len(wrapped) - 1:
            ax.annotate(
                "",
                xy=(3.5, y + height + gap - 0.02),
                xytext=(3.5, y + height + 0.02),
                arrowprops={"arrowstyle": "-|>", "color": "#57606a", "linewidth": 1.4},
            )
        y += height + gap

    path = _new_path(out_dir, "flowchart")
    fig.savefig(path)
    plt.close(fig)
    return str(path)
