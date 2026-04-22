"""
KisanAgent — Pest Alert Tool
==============================
Real-world analog: TNAU / Karnataka Dept of Agriculture
pest surveillance network. Reports aggregated from
field scouts and neighboring farm self-reports.

Escalation schedule (days from detection):
  Day 0 → LOW     ("Monitor closely")
  Day 2 → MEDIUM  ("Consider treatment")
  Day 4 → HIGH    ("Treat within 24 hours")
  Day 6 → CRITICAL ("Emergency — 40% yield loss")

Treatment window: 4 days from onset.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("kisanagent.tools.pest_alert")

PEST_TREATMENTS = {
    "aphids": {
        "recommended": "Spray imidacloprid 0.5ml/L or neem oil 5ml/L",
        "organic_alternative": "Neem oil 5ml/L + 2ml dish soap per litre",
        "cost_estimate_inr": 650,
    },
    "whitefly": {
        "recommended": "Yellow sticky traps + thiamethoxam 25WG 0.3g/L",
        "organic_alternative": "Reflective mulch + neem oil spray",
        "cost_estimate_inr": 720,
    },
    "leaf_curl_virus": {
        "recommended": "Remove infected plants. Spray vector (whitefly) with imidacloprid",
        "organic_alternative": "Strict whitefly control with neem + reflective mulch",
        "cost_estimate_inr": 800,
    },
}

ESCALATION_MESSAGES = {
    "LOW":      "Monitor field daily. Scout neighboring farms.",
    "MEDIUM":   "Increasing infestation detected. Consider treatment within 2 days.",
    "HIGH":     "High infestation — treat within 24 hours to prevent significant loss.",
    "CRITICAL": "EMERGENCY: Critical infestation. Treat immediately — 40% yield loss if delayed.",
}


class PestAlertTool:
    """
    Karnataka Dept of Agriculture pest surveillance network analog.

    Aggregates field scout reports and neighboring farm data.
    Risk level escalates over time if untreated.
    """

    def __init__(
        self,
        event_engine_ref: Any,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.event_engine = event_engine_ref
        self.rng = rng or np.random.default_rng(44)

    def call(
        self,
        region: str = "Kolar",
        crop: str = "tomato",
        current_day: int = 0,
    ) -> Dict[str, Any]:
        """
        Return current pest risk status for Kolar tomato region.

        Returns:
            {
              "risk_level": "LOW"|"MEDIUM"|"HIGH"|"CRITICAL",
              "detected_pests": list[str],
              "recommended_action": str,
              "treatment_cost_estimate": int,
              "days_to_critical": int | None,
              "neighboring_farm_reports": int,
              "last_field_scout": str,
              "organic_alternative": str | None,
              "data_quality": str
            }
        """
        # Get event state from event engine
        event_state = self.event_engine.get_event_state(current_day)
        pest_active = event_state.get("pest_active", False)
        pest_name = event_state.get("pest_name", None)
        risk_level = event_state.get("pest_risk_level", "LOW")
        days_since_onset = event_state.get("days_since_pest_onset", None)

        detected_pests = [pest_name] if pest_active and pest_name else []

        # Days to critical
        if pest_active and risk_level != "CRITICAL" and days_since_onset is not None:
            days_to_critical = max(0, 6 - days_since_onset)
        else:
            days_to_critical = None

        # Neighbouring farm reports — positively correlated with risk
        risk_multiplier = {"LOW": 1, "MEDIUM": 4, "HIGH": 9, "CRITICAL": 20}.get(risk_level, 1)
        neighboring_reports = int(self.rng.poisson(risk_multiplier))

        # Scout recency
        scout_days_ago = int(self.rng.integers(1, 4))
        last_field_scout = f"{scout_days_ago} day{'s' if scout_days_ago > 1 else ''} ago"

        # Treatment info
        treatment_cost = 0
        recommended_action = ESCALATION_MESSAGES.get(risk_level, "No action needed.")
        organic_alt = None
        if pest_name and pest_name in PEST_TREATMENTS:
            info = PEST_TREATMENTS[pest_name]
            treatment_cost = info["cost_estimate_inr"]
            organic_alt = info["organic_alternative"]
            if risk_level in ("HIGH", "CRITICAL"):
                recommended_action = (
                    f"{ESCALATION_MESSAGES[risk_level]} "
                    f"Recommended: {info['recommended']}"
                )

        # Slight noise on risk level (10% chance of one level off)
        reported_risk = _add_noise_to_risk(risk_level, self.rng)

        data_quality = "degraded" if self.rng.random() < 0.05 else "good"

        return {
            "risk_level": reported_risk,
            "detected_pests": detected_pests,
            "recommended_action": recommended_action,
            "treatment_cost_estimate": treatment_cost,
            "days_to_critical": days_to_critical,
            "neighboring_farm_reports": neighboring_reports,
            "last_field_scout": last_field_scout,
            "organic_alternative": organic_alt,
            "region": region,
            "crop": crop,
            "data_quality": data_quality,
        }


def _add_noise_to_risk(true_level: str, rng: np.random.Generator) -> str:
    """10% chance of reporting one level off (realistic surveillance noise)."""
    levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if rng.random() < 0.10:
        idx = levels.index(true_level)
        delta = int(rng.choice([-1, 1]))
        noisy_idx = max(0, min(len(levels) - 1, idx + delta))
        return levels[noisy_idx]
    return true_level
