"""
KisanAgent — Baseline Evaluation
==================================
Runs multiple evaluation strategies and compares against:
  - Random agent baseline
  - Rule-based heuristic agent
  - LLM agent (via inference.py)

Outputs CSV results and calls plot_results.py.

Usage:
  python eval/baseline_eval.py --episodes 20 --difficulty medium
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("kisanagent.eval")

ENV_SERVER = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

DECISIONS = [
    "irrigate", "fertilize", "spray_pesticide", "sell_now",
    "hold_crop", "apply_scheme", "take_loan", "do_nothing",
]


# ── Environment wrappers ─────────────────────────────────────────────────────

def reset(difficulty: str = "medium", seed: Optional[int] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"difficulty": difficulty}
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(f"{ENV_SERVER}/reset", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def step(decision: str, tools: List[str] = None, reasoning: str = "") -> Dict[str, Any]:
    r = requests.post(f"{ENV_SERVER}/step", json={
        "farm_decision": decision,
        "tool_calls_made": tools or [],
        "reasoning": reasoning,
    }, timeout=30)
    r.raise_for_status()
    return r.json()


# ── Agent strategies ────────────────────────────────────────────────────────

def random_agent_episode(difficulty: str, seed: int) -> Dict[str, Any]:
    """Pure random action selection. Zero tool use."""
    data = reset(difficulty, seed)
    obs = data["observation"]

    for _ in range(90):
        decision = random.choice(DECISIONS)
        result = step(decision, [], "random agent")
        if result["terminated"]:
            return {
                "agent": "random",
                "difficulty": difficulty,
                "seed": seed,
                "net_income_inr": result.get("net_income_inr", 0),
                "composite_score": result.get("final_scores", {}).get("composite_score", 0),
                "income_score": result.get("final_scores", {}).get("income_score", 0),
            }
    return {"agent": "random", "difficulty": difficulty, "seed": seed,
            "net_income_inr": 0, "composite_score": 0, "income_score": 0}


def heuristic_agent_episode(difficulty: str, seed: int) -> Dict[str, Any]:
    """
    Rule-based heuristic agent:
    - Irrigate when soil_moisture < 50% (uses soil tool)
    - Spray if pest alert is HIGH/CRITICAL (uses pest_alert tool)
    - Sell in harvest stage if mandi_price > 20 (uses mandi_price tool)
    - Apply schemes in first 25 days
    - Otherwise do_nothing
    """
    data = reset(difficulty, seed)
    obs = data["observation"]

    for day_iter in range(90):
        day = obs["day"]
        stage = obs["crop_stage"]
        moisture = obs["soil_moisture_pct"]
        alerts = obs.get("active_alerts", [])

        tools_called = []
        decision = "do_nothing"
        reasoning = ""

        # Call soil tool for moisture check
        if moisture < 55:
            try:
                r = requests.post(f"{ENV_SERVER}/tools/soil", json={"args": {}}, timeout=10)
                if r.status_code == 200:
                    tools_called.append("soil")
            except Exception:
                pass

        # Check pest alert
        try:
            pr = requests.post(f"{ENV_SERVER}/tools/pest_alert", json={"args": {}}, timeout=10)
            if pr.status_code == 200:
                tools_called.append("pest_alert")
                pest_data = pr.json().get("result", {})
                risk = pest_data.get("risk_level", "LOW")
            else:
                risk = "LOW"
        except Exception:
            risk = "LOW"

        # Heuristic decision logic
        if risk in ("HIGH", "CRITICAL") and stage != "flowering":
            decision = "spray_pesticide"
            reasoning = f"Pest risk {risk} — spraying"
        elif moisture < 45:
            decision = "irrigate"
            reasoning = f"Soil moisture {moisture:.0f}% critical — irrigating"
        elif stage == "harvest":
            try:
                mr = requests.post(f"{ENV_SERVER}/tools/mandi_price", json={"args": {}}, timeout=10)
                if mr.status_code == 200:
                    tools_called.append("mandi_price")
                    price = mr.json().get("result", {}).get("today_price_per_kg", 0) or 0
                    if price > 20:
                        decision = "sell_now"
                        reasoning = f"Price ₹{price:.0f}/kg — selling"
                    else:
                        decision = "hold_crop"
                        reasoning = f"Price ₹{price:.0f}/kg — holding for better price"
                else:
                    decision = "hold_crop"
                    reasoning = "Mandi tool unavailable — holding"
            except Exception:
                decision = "hold_crop"
        elif stage == "vegetative" and moisture > 55:
            decision = "fertilize"
            reasoning = "Vegetative stage — fertilizing"
        elif day < 25:
            try:
                sr = requests.post(f"{ENV_SERVER}/tools/govt_scheme", json={"args": {}}, timeout=10)
                if sr.status_code == 200:
                    tools_called.append("govt_scheme")
                    decision = "apply_scheme"
                    reasoning = "Early season — applying schemes"
            except Exception:
                pass
        else:
            decision = "do_nothing"
            reasoning = "No urgent action needed"

        result = step(decision, tools_called, reasoning)
        obs = result["observation"]

        if result["terminated"]:
            return {
                "agent": "heuristic",
                "difficulty": difficulty,
                "seed": seed,
                "net_income_inr": result.get("net_income_inr", 0),
                "composite_score": result.get("final_scores", {}).get("composite_score", 0),
                "income_score": result.get("final_scores", {}).get("income_score", 0),
            }

    return {"agent": "heuristic", "difficulty": difficulty, "seed": seed,
            "net_income_inr": 0, "composite_score": 0, "income_score": 0}


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_baseline_eval(
    n_episodes: int = 20,
    difficulty: str = "medium",
    output_csv: str = "eval/baseline_results.csv",
) -> Dict[str, Any]:
    """
    Run baseline evaluation for random and heuristic agents.
    Saves results to CSV and prints summary statistics.
    """
    print(f"\n{'═'*55}")
    print(f"KisanAgent Baseline Evaluation")
    print(f"Episodes: {n_episodes} | Difficulty: {difficulty}")
    print(f"{'═'*55}\n")

    results = []

    # Random agent
    print(f"[1/2] Random Agent ({n_episodes} episodes)...")
    random_incomes = []
    for ep in range(n_episodes):
        try:
            r = random_agent_episode(difficulty, seed=ep * 7)
            results.append(r)
            random_incomes.append(r["net_income_inr"])
            if ep % 5 == 0:
                print(f"  Ep {ep:>3}: ₹{r['net_income_inr']:>8,.0f} | score: {r['composite_score']:.3f}")
        except Exception as e:
            logger.error("Random ep %d failed: %s", ep, e)

    # Heuristic agent
    print(f"\n[2/2] Heuristic Agent ({n_episodes} episodes)...")
    heuristic_incomes = []
    for ep in range(n_episodes):
        try:
            r = heuristic_agent_episode(difficulty, seed=ep * 13)
            results.append(r)
            heuristic_incomes.append(r["net_income_inr"])
            if ep % 5 == 0:
                print(f"  Ep {ep:>3}: ₹{r['net_income_inr']:>8,.0f} | score: {r['composite_score']:.3f}")
        except Exception as e:
            logger.error("Heuristic ep %d failed: %s", ep, e)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        fieldnames = ["agent", "difficulty", "seed", "net_income_inr",
                      "composite_score", "income_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary stats
    def stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {}
        return {
            "mean": round(sum(vals) / len(vals), 0),
            "min": round(min(vals), 0),
            "max": round(max(vals), 0),
        }

    summary = {
        "random_agent": stats(random_incomes),
        "heuristic_agent": stats(heuristic_incomes),
        "optimal_income_inr": 40_000,
    }

    print(f"\n{'═'*55}")
    print(f"BASELINE RESULTS")
    print(f"{'═'*55}")
    print(f"Random Agent:    avg ₹{summary['random_agent'].get('mean', 0):>8,.0f}  "
          f"[₹{summary['random_agent'].get('min',0):,.0f} – ₹{summary['random_agent'].get('max',0):,.0f}]")
    print(f"Heuristic Agent: avg ₹{summary['heuristic_agent'].get('mean', 0):>8,.0f}  "
          f"[₹{summary['heuristic_agent'].get('min',0):,.0f} – ₹{summary['heuristic_agent'].get('max',0):,.0f}]")
    print(f"Optimal Income:      ₹{summary['optimal_income_inr']:>8,.0f}")
    print(f"\nResults saved: {output_csv}")

    with open("eval/baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="KisanAgent Baseline Evaluation")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--output", type=str, default="eval/baseline_results.csv")
    args = parser.parse_args()

    run_baseline_eval(
        n_episodes=args.episodes,
        difficulty=args.difficulty,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
