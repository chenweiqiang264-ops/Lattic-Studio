"""Plot the spatial Ramp weights used by the implicit transition module."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lattice_studio.engine.implicit.transition import TransitionSpec, transition_weights


OUTPUT_STEM = Path(__file__).with_name("transition_weight_functions")
WIDTH_MM = 10.0

COLORS = {
    "linear": "#9AA0A6",
    "smoothstep": "#69A7A0",
    "smootherstep": "#0F4D92",
    "cosine": "#D18B36",
    "sigmoid": "#A24B6B",
    "first": "#7884B4",
    "second": "#B64342",
    "band": "#EAF1F8",
    "neutral": "#4D4D4D",
}

LABELS = {
    "linear": "线性",
    "smoothstep": "Smoothstep（C1）",
    "smootherstep": "Smootherstep（C2，自动）",
    "cosine": "余弦",
    "sigmoid": "Sigmoid（锐度 = 1）",
}


def weights(distance_mm: np.ndarray, kind: str, sharpness: float = 1.0) -> np.ndarray:
    """Return the second-operand weight through the production evaluator."""

    spec = TransitionSpec(
        width_mm=WIDTH_MM,
        weight_kind=kind,
        sigmoid_sharpness=sharpness,
        automatic_registration=False,
        topology_correction=False,
    )
    _, second = transition_weights(distance_mm, spec)
    return second.astype(np.float64)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "stixsans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def style_weight_axis(ax: plt.Axes) -> None:
    ax.axvspan(-WIDTH_MM / 2, WIDTH_MM / 2, color=COLORS["band"], zorder=0)
    ax.axvline(-WIDTH_MM / 2, color="#9BA7B4", linewidth=0.9, linestyle="--")
    ax.axvline(WIDTH_MM / 2, color="#9BA7B4", linewidth=0.9, linestyle="--")
    ax.axvline(0.0, color="#B9B9B9", linewidth=0.8, linestyle=":")
    ax.set_xlim(-0.8 * WIDTH_MM, 0.8 * WIDTH_MM)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks((-WIDTH_MM / 2, 0.0, WIDTH_MM / 2))
    ax.set_xticklabels((r"$-W/2$", "0", r"$+W/2$"))
    ax.set_yticks((0.0, 0.5, 1.0))
    ax.set_xlabel(r"相对过渡中心的有符号距离  $d_c$（mm）")
    ax.set_ylabel(r"晶胞 B 权重  $w_B$")
    ax.text(
        0.5,
        0.97,
        "物理过渡带 W",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="#526579",
        fontsize=9,
    )


def build_figure() -> plt.Figure:
    distance = np.linspace(-0.8 * WIDTH_MM, 0.8 * WIDTH_MM, 1_601)
    fig = plt.figure(figsize=(13.333, 7.5), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.28, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.07,
        right=0.97,
        bottom=0.11,
        top=0.82,
        wspace=0.24,
        hspace=0.42,
    )
    ax_all = fig.add_subplot(grid[:, 0])
    ax_sharpness = fig.add_subplot(grid[0, 1])
    ax_formula = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "晶格过渡的空间 Ramp 权重与隐式场融合",
        x=0.07,
        y=0.94,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#272727",
    )
    fig.text(
        0.07,
        0.875,
        "过渡宽度限定作用范围；权重曲线控制两类晶胞在带内的连续交接方式",
        ha="left",
        fontsize=10.5,
        color="#60656B",
    )

    curve_order = ("linear", "smoothstep", "smootherstep", "cosine", "sigmoid")
    for kind in curve_order:
        line_width = 3.0 if kind == "smootherstep" else 1.9
        zorder = 5 if kind == "smootherstep" else 3
        ax_all.plot(
            distance,
            weights(distance, kind),
            color=COLORS[kind],
            linewidth=line_width,
            label=LABELS[kind],
            zorder=zorder,
        )
    style_weight_axis(ax_all)
    ax_all.set_title("a  项目支持的 Ramp 权重函数", loc="left", fontweight="bold")
    ax_all.legend(loc="upper left", bbox_to_anchor=(0.015, 0.91), fontsize=9)
    ax_all.annotate(
        "带外严格保持晶胞 A",
        xy=(-0.62 * WIDTH_MM, 0.0),
        xytext=(-0.75 * WIDTH_MM, 0.18),
        arrowprops={"arrowstyle": "->", "color": COLORS["neutral"], "lw": 0.9},
        color=COLORS["neutral"],
        fontsize=9,
    )
    ax_all.annotate(
        "带外严格保持晶胞 B",
        xy=(0.62 * WIDTH_MM, 1.0),
        xytext=(0.18 * WIDTH_MM, 0.82),
        arrowprops={"arrowstyle": "->", "color": COLORS["neutral"], "lw": 0.9},
        color=COLORS["neutral"],
        fontsize=9,
    )

    sharpness_values = (0.5, 1.0, 2.0, 4.0)
    sharpness_colors = ("#9BB7D4", "#0F4D92", "#A36A8D", "#B64342")
    for sharpness, color in zip(sharpness_values, sharpness_colors):
        ax_sharpness.plot(
            distance,
            weights(distance, "sigmoid", sharpness),
            color=color,
            linewidth=2.0 if sharpness == 1.0 else 1.6,
            label=fr"$k={sharpness:g}$",
        )
    style_weight_axis(ax_sharpness)
    ax_sharpness.set_title("b  Sigmoid 锐度（不改变 W）", loc="left", fontweight="bold")
    ax_sharpness.legend(ncol=2, loc="upper left", fontsize=8.5)

    ax_formula.axis("off")
    ax_formula.set_title("c  项目权重与基础场融合公式", loc="left", fontweight="bold")
    formula_box = dict(
        boxstyle="round,pad=0.35",
        facecolor="#F5F7F9",
        edgecolor="#D4DAE1",
    )
    ax_formula.text(
        0.02,
        0.84,
        r"$d_c=(\mathbf{x}-\mathbf{p})\cdot\hat{\mathbf{n}}-\delta,\quad "
        r"t=\mathrm{clip}\!\left(\dfrac{d_c+W/2}{W},0,1\right)$",
        fontsize=12,
        color="#272727",
        bbox=formula_box,
    )
    ax_formula.text(
        0.02,
        0.61,
        r"$w_B=R(t),\quad w_A=1-w_B,\quad "
        r"R_{\mathrm{auto}}(t)=6t^5-15t^4+10t^3$",
        fontsize=12,
        color="#272727",
        bbox=formula_box,
    )
    ax_formula.text(
        0.02,
        0.38,
        r"$R_{\mathrm{sig}}(t)=\dfrac{\sigma[\kappa(2t-1)]-\sigma(-\kappa)}"
        r"{\sigma(\kappa)-\sigma(-\kappa)},\quad \kappa=\ln(9)\,k$",
        fontsize=11.5,
        color="#272727",
        bbox=formula_box,
    )
    ax_formula.text(
        0.02,
        0.15,
        r"$F_{\mathrm{base}}=(1-w_B)F_A+w_BF_B,\quad "
        r"F_{\mathrm{final}}=\max(F_{\mathrm{domain}},F_{\mathrm{tr}})$",
        fontsize=11.5,
        color="#272727",
        bbox=formula_box,
    )
    ax_formula.text(
        0.98,
        -0.05,
        "Ftr 为基础融合场经可选局部配准与拓扑约束修正后的过渡场",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#60656B",
    )
    return fig


def main() -> None:
    configure_matplotlib()
    figure = build_figure()
    save_options = {"bbox_inches": "tight", "pad_inches": 0.18}
    figure.savefig(OUTPUT_STEM.with_suffix(".svg"), **save_options)
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), **save_options)
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, **save_options)
    plt.close(figure)


if __name__ == "__main__":
    main()
