"""
KisanAgent — Event Scheduler
==============================
Deterministic event calendar generator.
Ensures each episode has the right density of:
  - Pest outbreaks (with escalation triggers)
  - Government scheme windows (open/closing/expired)
  - Mandi price spikes
  - Tool API failures

Works in conjunction with EventEngine (runtime state machine)
and ScenarioConfig (difficulty parameters).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tasks.season_scenarios import ScenarioConfig


class EventScheduler:
    """
    Pre-generates the complete 90-day event calendar for a season.
    Output is consumed by EventEngine.reset() during /reset.

    Each event is a dict with:
      type:      "pest" | "scheme" | "price_spike" | "tool_failure"
      day:       onset day (0-89)
      ...        event-specific fields
    """

    def __init__(self, config: ScenarioConfig, seed: int = 42) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed + 9999)

    def generate(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate full event calendar for the season.

        Returns:
            {
              "pest_events": [...],
              "scheme_events": [...],
              "price_spikes": [...],
              "tool_failures": [...]
            }
        """
        return {
            "pest_events": self._generate_pest_events(),
            "scheme_events": self._generate_scheme_events(),
            "price_spikes": self._generate_price_spikes(),
            "tool_failures": self._generate_tool_failures(),
        }

    def _generate_pest_events(self) -> List[Dict[str, Any]]:
        """Generate pest outbreak events."""
        cfg = self.config
        pests = ["aphids", "whitefly", "leaf_curl_virus"]
        events = []

        onset_days = sorted(
            int(d)
            for d in self.rng.integers(
                cfg.pest_onset_range[0],
                cfg.pest_onset_range[1],
                size=cfg.n_pest_events,
            )
        )

        for i, onset in enumerate(onset_days):
            events.append({
                "type": "pest",
                "pest_name": pests[i % len(pests)],
                "onset_day": onset,
                "escalation_schedule": {
                    0: "LOW",
                    2: "MEDIUM",
                    4: "HIGH",
                    6: "CRITICAL",
                },
                "treatment_window_days": cfg.pest_treatment_window_days,
                "yield_loss_per_day_if_untreated_kg": 100,
                "critical_yield_loss_pct": 40,
                "treated_in_window": False,
                "treatment_day": None,
            })

        return events

    def _generate_scheme_events(self) -> List[Dict[str, Any]]:
        """
        Generate government scheme windows.
        Deadlines adjusted by difficulty (hard = tighter).
        """
        adjustment = self.config.scheme_deadline_adjustment_days
        base_schemes = [
            {
                "type": "scheme",
                "name": "PM-KISAN Input Supplement",
                "benefit_inr": 2000,
                "open_day": 0,
                "deadline_day": max(10, 30 + adjustment),
                "required_action": "apply_scheme",
            },
            {
                "type": "scheme",
                "name": "Crop Insurance PMFBY",
                "benefit_inr": 3000,
                "open_day": 0,
                "deadline_day": max(5, 20 + adjustment),
                "required_action": "apply_scheme",
            },
            {
                "type": "scheme",
                "name": "Drip Irrigation Subsidy",
                "benefit_inr": 5000,
                "open_day": 10,
                "deadline_day": max(20, 50 + adjustment),
                "required_action": "apply_scheme",
            },
            {
                "type": "scheme",
                "name": "Tomato Special Support Price",
                "benefit_inr": 1500,
                "open_day": 40,
                "deadline_day": max(50, 70 + adjustment),
                "required_action": "apply_scheme",
            },
        ]
        return base_schemes

    def _generate_price_spikes(self) -> List[Dict[str, Any]]:
        """Generate mandi price spike events during harvest window."""
        cfg = self.config
        spikes = []
        for _ in range(cfg.n_price_spikes):
            spike_day = int(self.rng.integers(60, 88))
            duration = int(self.rng.integers(3, 7))
            price = round(float(self.rng.uniform(22.0, cfg.price_spike_magnitude_max)), 2)
            spikes.append({
                "type": "price_spike",
                "onset_day": spike_day,
                "duration_days": duration,
                "price_per_kg": price,
                "market": "KR Puram Bangalore",
            })
        return spikes

    def _generate_tool_failures(self) -> List[Dict[str, Any]]:
        """Generate random tool API failure days."""
        cfg = self.config
        tools = ["weather", "soil", "mandi_price", "govt_scheme", "pest_alert", "credit"]
        failures = []
        for _ in range(cfg.n_tool_failures):
            tool = tools[int(self.rng.integers(0, len(tools)))]
            fail_day = int(self.rng.integers(0, 90))
            failures.append({
                "type": "tool_failure",
                "tool_name": tool,
                "day": fail_day,
                "duration_days": cfg.tool_failure_duration_days,
                "reason": "scheduled_maintenance" if self.rng.random() < 0.5 else "connectivity_lost",
            })
        return failures
