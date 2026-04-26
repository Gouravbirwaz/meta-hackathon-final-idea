"""
KisanAgent — Results Plotter
================================
Generates training progress plots and evaluation comparison charts.

Usage:
  python eval/plot_results.py --input eval/training_log.json
  python eval/plot_results.py --baseline eval/baseline_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional


def plot_training_progress(
    training_log: List[Dict[str, Any]],
    output_path: str = "eval/training_progress.png",
) -> None:
    """
    Plot income and composite score over training episodes.
    Includes baseline and optimal reference lines.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    episodes = [l["episode"] for l in training_log]
    incomes = [l.get("net_income_inr", 0) for l in training_log]
    scores = [l.get("composite_score", 0) for l in training_log]

    # Rolling average
    def rolling_avg(data: List[float], w: int = 10) -> List[float]:
        result = []
        for i in range(len(data)):
            window = data[max(0, i - w): i + 1]
            result.append(sum(window) / len(window))
        return result

    income_roll = rolling_avg(incomes, w=10)
    score_roll = rolling_avg(scores, w=10)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "KisanAgent — GRPO Training Progress",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # ── Income plot ──────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(episodes, incomes, alpha=0.3, s=12, color="#4CAF50", label="Episode income")
    ax.plot(episodes, income_roll, color="#1B5E20", linewidth=2.5, label="Rolling avg (10 ep)")
    ax.axhline(11_200, color="#E53935", linestyle="--", linewidth=1.5,
               label="Random baseline ₹11,200")
    ax.axhline(40_000, color="#FFD600", linestyle="--", linewidth=1.5,
               label="Optimal ₹40,000")
    ax.fill_between(episodes, 15_000, 25_000, alpha=0.07, color="#4CAF50",
                    label="Karnataka avg range (₹15k–₹25k)")

    # Curriculum phase markers
    for ep, label in [(0, "Easy"), (150, "Medium"), (350, "Hard")]:
        ax.axvline(ep, color="#9E9E9E", linestyle=":", linewidth=1)
        if ep < max(episodes, default=0):
            ax.text(ep + 5, 2000, label, fontsize=8, color="#757575")

    ax.set_xlabel("Training Episodes", fontsize=11)
    ax.set_ylabel("Net Income (INR ₹)", fontsize=11)
    ax.set_title("Income vs Training", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── Score plot ───────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(episodes, scores, alpha=0.3, s=12, color="#1565C0", label="Episode score")
    ax.plot(episodes, score_roll, color="#0D47A1", linewidth=2.5, label="Rolling avg (10 ep)")
    ax.axhline(0.3, color="#E53935", linestyle="--", linewidth=1.5,
               label="Random baseline ~0.30")
    ax.axhline(1.0, color="#FFD600", linestyle="--", linewidth=1.5,
               label="Perfect score 1.0")

    for ep, label in [(0, "Easy"), (150, "Medium"), (350, "Hard")]:
        ax.axvline(ep, color="#9E9E9E", linestyle=":", linewidth=1)
        if ep < max(episodes, default=0):
            ax.text(ep + 5, 0.02, label, fontsize=8, color="#757575")

    ax.set_xlabel("Training Episodes", fontsize=11)
    ax.set_ylabel("Composite Score (0–1)", fontsize=11)
    ax.set_title("Grader Score vs Training", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    if incomes:
        final_income = income_roll[-1]
        print(f"\n✅ Training progress plot saved: {output_path}")
        print(f"   Final trained agent income (rolling avg): ₹{final_income:,.0f}")
        print(f"   vs Random baseline:                       ₹11,200")
        improvement = (final_income / 11_200 - 1) * 100
        print(f"   Improvement:                              +{improvement:.0f}%")


def plot_score_breakdown(
    final_scores_list: List[Dict[str, Any]],
    output_path: str = "eval/score_breakdown.png",
) -> None:
    """Radar / bar chart of multi-dimensional grader scores."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not final_scores_list:
        print("No scores to plot.")
        return

    dims = [
        "income_score",
        "tool_use_quality",
        "pest_response_accuracy",
        "scheme_capture_rate",
        "sustainability_score",
    ]
    labels = [
        "Income\n(40%)",
        "Tool Use\n(20%)",
        "Pest\nResponse\n(20%)",
        "Scheme\nCapture\n(10%)",
        "Sustainability\n(10%)",
    ]
    weights = [0.40, 0.20, 0.20, 0.10, 0.10]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

    # Average across episodes
    avg_scores = {
        d: sum(s.get(d, 0) for s in final_scores_list) / len(final_scores_list)
        for d in dims
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dims))
    bars = ax.bar(x, [avg_scores[d] for d in dims], color=colors, width=0.5, alpha=0.85)

    for bar, val in zip(bars, [avg_scores[d] for d in dims]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score (0–1)", fontsize=11)
    ax.set_title(
        "KisanAgent — Multi-Dimensional Grader Scores\n"
        f"(avg over {len(final_scores_list)} episodes)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="#E53935", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Score breakdown plot saved: {output_path}")


def plot_comparison(
    baseline_summary: Dict[str, Any],
    trained_income: float,
    output_path: str = "eval/agent_comparison.png",
) -> None:
    """Bar chart comparing random, heuristic, trained, and optimal agents."""
    import matplotlib.pyplot as plt

    agents = ["Random\nAgent", "Heuristic\nAgent", "KisanAgent\n(Trained)", "Optimal\n(Theoretical)"]
    incomes = [
        baseline_summary.get("random_agent", {}).get("mean", 11_200),
        baseline_summary.get("heuristic_agent", {}).get("mean", 18_000),
        trained_income,
        40_000,
    ]
    colors = ["#EF9A9A", "#FFCC80", "#A5D6A7", "#FFD54F"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agents, incomes, color=colors, width=0.5, edgecolor="white", linewidth=2)

    for bar, val in zip(bars, incomes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 300,
            f"₹{val:,.0f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Net Income (INR ₹)", fontsize=11)
    ax.set_title(
        "KisanAgent — Agent Comparison\n(Net Income per 90-day Season)",
        fontsize=12, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.set_ylim(0, 46_000)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(15_000, color="#9E9E9E", linestyle=":", linewidth=1,
               label="Karnataka farmer avg (₹15,000)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Comparison chart saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KisanAgent Results Plotter")
    parser.add_argument("--training-log", type=str, default="eval/training_log.json",
                        help="JSON file with training log")
    parser.add_argument("--baseline", type=str, default="eval/baseline_summary.json",
                        help="Baseline summary JSON")
    parser.add_argument("--trained-income", type=float, default=28_000.0,
                        help="Final trained agent average income")
    parser.add_argument("--output-dir", type=str, default="eval/")
    args = parser.parse_args()

    # Load training log if available
    if os.path.exists(args.training_log):
        with open(args.training_log) as f:
            training_log = json.load(f)
        plot_training_progress(
            training_log,
            os.path.join(args.output_dir, "training_progress.png"),
        )
    else:
        print(f"Training log not found: {args.training_log}")

    # Load baseline if available
    if os.path.exists(args.baseline):
        with open(args.baseline) as f:
            baseline_summary = json.load(f)
        plot_comparison(
            baseline_summary,
            args.trained_income,
            os.path.join(args.output_dir, "agent_comparison.png"),
        )
    else:
        print(f"Baseline summary not found: {args.baseline}")


if __name__ == "__main__":
    main()
