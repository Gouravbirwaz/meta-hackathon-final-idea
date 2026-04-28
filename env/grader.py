"""
KisanAgent — Multi-Dimensional Grader
=======================================
All scores strictly in (0.0, 1.0).
Mirrors Round 1 Multi-Dimensional Grader pattern.

Dimensions:
  income_score           (40%) — net income vs optimal
  tool_use_quality       (20%) — right tool, right decision
  pest_response_accuracy (20%) — treated in 4-day window
  scheme_capture_rate    (10%) — applied before deadline
  sustainability_score   (10%) — water + chemical efficiency
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from env.models import GraderScores

logger = logging.getLogger("kisanagent.grader")


class KisanGrader:
    """
    Multi-dimensional grader for KisanAgent.

    Usage:
        grader = KisanGrader()
        # Per step:
        step_sc = grader.compute_step_scores(action, state, tool_calls)
        grader.log_step({...})
        # At terminal:
        final_sc = grader.compute_final_scores(final_state, grader.episode_log)
    """

    WEIGHTS: Dict[str, float] = {
        "income_score": 0.40,
        "tool_use_quality": 0.20,
        "pest_response_accuracy": 0.20,
        "scheme_capture_rate": 0.10,
        "sustainability_score": 0.10,
    }

    # Karnataka 2024-25 benchmarks (INR)
    BASELINE_INCOME: float = 15_000.0   # avg unadvised 2-acre farmer
    OPTIMAL_INCOME: float = 40_000.0    # theoretical maximum for the season

    # Tool → decision mapping for quality scoring
    DECISION_TOOL_MAP: Dict[str, List[str]] = {
        "sell_now":        ["mandi_price"],
        "irrigate":        ["soil", "weather"],
        "spray_pesticide": ["pest_alert"],
        "apply_scheme":    ["govt_scheme"],
        "take_loan":       ["credit"],
        "fertilize":       ["soil"],
        "hold_crop":       ["mandi_price"],
        "do_nothing":      [],
    }

    # Sustainability thresholds
    MAX_WATER_LITERS: float = 180_000.0   # optimal season total
    MAX_CHEMICALS: int = 4                # applications per season

    def __init__(self) -> None:
        self.episode_log: List[Dict[str, Any]] = []
        self.reset()

    # ──────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear episode state — call after /reset."""
        self.episode_log = []
        logger.debug("KisanGrader reset.")

    def log_step(self, step_data: Dict[str, Any]) -> None:
        """
        Append per-step data to episode log.
        Expected keys:
          day, action, tool_calls, tool_score, pest_score,
          pest_risk_level, pest_treated_today (bool)
        """
        self.episode_log.append(step_data)

    # ──────────────────────────────────────────────────────────
    # Per-step scoring (reward shaping)
    # ──────────────────────────────────────────────────────────

    def compute_step_scores(
        self,
        action: str,
        state: Dict[str, Any],
        tool_calls: List[str],
    ) -> GraderScores:
        """
        Lightweight per-step scoring for reward shaping.
        income_score = 0.5 (neutral) until terminal.
        Returns partial GraderScores.
        """
        tool_quality = self._score_tool_use(action, tool_calls, state)
        pest_score = self._score_pest_response(action, state)
        composite = (tool_quality * self.WEIGHTS["tool_use_quality"] +
                     pest_score * self.WEIGHTS["pest_response_accuracy"]) / (
                        self.WEIGHTS["tool_use_quality"] +
                        self.WEIGHTS["pest_response_accuracy"]
                    )

        return GraderScores(
            income_score=0.5,
            tool_use_quality=tool_quality,
            pest_response_accuracy=pest_score,
            scheme_capture_rate=0.5,
            sustainability_score=0.5,
            composite_score=float(min(max(composite, 0.0), 1.0)),
        )

    # ──────────────────────────────────────────────────────────
    # Terminal scoring (primary GRPO reward)
    # ──────────────────────────────────────────────────────────

    def compute_final_scores(
        self,
        final_state: Dict[str, Any],
        episode_log: Optional[List[Dict[str, Any]]] = None,
    ) -> GraderScores:
        """
        Terminal scoring — called at day 90.
        This is the primary reward signal for GRPO training.

        final_state expected keys:
          net_income_inr, pest_events (list of dicts with 'treated_in_window'),
          schemes_available (list[str]), schemes_captured (list[str]),
          water_used_liters (float), chemical_applications (int)
        """
        log = episode_log if episode_log is not None else self.episode_log

        income_score = self._score_income(
            float(final_state.get("net_income_inr", self.BASELINE_INCOME))
        )
        tool_quality = self._score_episode_tool_quality(log)
        pest_score = self._score_episode_pest_response(
            log, final_state.get("pest_events", [])
        )
        scheme_score = self._score_scheme_capture(
            final_state.get("schemes_available", []),
            final_state.get("schemes_captured", []),
        )
        sustain_score = self._score_sustainability(
            float(final_state.get("water_used_liters", 0.0)),
            int(final_state.get("chemical_applications", 0)),
        )

        composite = (
            income_score  * self.WEIGHTS["income_score"]
            + tool_quality  * self.WEIGHTS["tool_use_quality"]
            + pest_score    * self.WEIGHTS["pest_response_accuracy"]
            + scheme_score  * self.WEIGHTS["scheme_capture_rate"]
            + sustain_score * self.WEIGHTS["sustainability_score"]
        )

        logger.info(
            "Final scores — income=%.3f tool=%.3f pest=%.3f scheme=%.3f "
            "sustain=%.3f composite=%.3f",
            income_score, tool_quality, pest_score,
            scheme_score, sustain_score, composite,
        )

        return GraderScores(
            income_score=income_score,
            tool_use_quality=tool_quality,
            pest_response_accuracy=pest_score,
            scheme_capture_rate=scheme_score,
            sustainability_score=sustain_score,
            composite_score=float(min(max(composite, 0.0), 1.0)),
        )

    # ──────────────────────────────────────────────────────────
    # Individual scoring helpers
    # ──────────────────────────────────────────────────────────

    def _score_income(self, net_income_inr: float) -> float:
        """
        Normalise net income to (0, 1).
          0.0 = baseline unadvised farmer (₹15,000)
          1.0 = theoretical optimal (₹40,000)
        Clipped so negative income yields 0.0, super-optimal yields 1.0.
        """
        if self.OPTIMAL_INCOME <= self.BASELINE_INCOME:
            return 0.5
        score = (net_income_inr - self.BASELINE_INCOME) / (
            self.OPTIMAL_INCOME - self.BASELINE_INCOME
        )
        return float(min(max(score, 0.0), 1.0))

    def _score_tool_use(
        self,
        action: str,
        tool_calls: List[str],
        state: Dict[str, Any],
    ) -> float:
        """
        Score quality of tool use before this decision.

        Rules:
          1.0 — called the decision-relevant tool (or do_nothing with no tools)
          0.5 — called tools but not the most relevant one
          0.0 — blind decision (no tools called when tools were needed)
        """
        expected = self.DECISION_TOOL_MAP.get(action, [])

        # do_nothing correctly requires no tools, but is neutral (0.5) to prevent reward hacking
        if action == "do_nothing":
            return 0.5 if not tool_calls else 0.2  # penalty for wasting budget on nothing

        if not tool_calls:
            return 0.0  # blind decision

        if expected and any(t in tool_calls for t in expected):
            return 1.0  # called the right tool

        return 0.5  # called tools but not decision-relevant ones

    def _score_pest_response(
        self,
        action: str,
        state: Dict[str, Any],
    ) -> float:
        """
        Score pest response correctness at this step.

        CRITICAL pest + no spray → 0.0
        LOW pest + spray → 0.0 (pollinator kill)
        MEDIUM/HIGH + spray → 1.0
        Neutral otherwise → 0.7
        """
        pest_level = state.get("pest_risk_level", "LOW")

        if pest_level == "CRITICAL" and action != "spray_pesticide":
            return 0.0
        if pest_level == "LOW" and action == "spray_pesticide":
            return 0.0  # unnecessary chemical use
        if pest_level in ("MEDIUM", "HIGH") and action == "spray_pesticide":
            return 1.0
        if pest_level == "CRITICAL" and action == "spray_pesticide":
            return 1.0

        return 0.5  # neutral / uninvolved

    def _score_episode_tool_quality(
        self, episode_log: List[Dict[str, Any]]
    ) -> float:
        """Average tool quality score across all 90 steps."""
        if not episode_log:
            return 0.5
        scores = [s.get("tool_score", 0.5) for s in episode_log]
        return float(sum(scores) / len(scores))

    def _score_episode_pest_response(
        self,
        episode_log: List[Dict[str, Any]],
        pest_events: List[Dict[str, Any]],
    ) -> float:
        """
        For each pest event: was it treated within the 4-day window?
        Score = treated_count / total_pest_events.
        No pest events → perfect score (1.0).
        """
        if not pest_events:
            return 1.0

        treated = sum(
            1 for e in pest_events if e.get("treated_in_window", False)
        )
        return float(treated / len(pest_events))

    def _score_scheme_capture(
        self,
        available: List[str],
        captured: List[str],
    ) -> float:
        """
        Fraction of available government schemes captured before deadline.
        No schemes available → 1.0 (agent can't do better).
        """
        if not available:
            return 1.0
        captured_set = set(captured)
        captured_count = sum(1 for s in available if s in captured_set)
        return float(captured_count / len(available))

    def _score_sustainability(
        self,
        water_liters: float,
        chemical_apps: int,
    ) -> float:
        """
        Penalise over-irrigation and excess pesticide application.

        water_score: 1.0 at or below MAX_WATER_LITERS, degrades linearly above.
        chem_score:  1.0 up to MAX_CHEMICALS, -0.2 per extra application.
        """
        # Water efficiency
        if water_liters <= self.MAX_WATER_LITERS:
            water_score = 1.0
        else:
            excess_ratio = water_liters / self.MAX_WATER_LITERS - 1.0
            water_score = max(0.0, 1.0 - excess_ratio)

        # Chemical efficiency
        excess_chem = max(0, chemical_apps - self.MAX_CHEMICALS)
        chem_score = max(0.0, 1.0 - excess_chem * 0.2)

        return float((water_score + chem_score) / 2.0)

    # ──────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Return grader state for /state endpoint debugging."""
        return {
            "episode_steps_logged": len(self.episode_log),
            "weights": self.WEIGHTS,
            "baseline_income_inr": self.BASELINE_INCOME,
            "optimal_income_inr": self.OPTIMAL_INCOME,
            "max_water_liters": self.MAX_WATER_LITERS,
            "max_chemical_apps": self.MAX_CHEMICALS,
        }
