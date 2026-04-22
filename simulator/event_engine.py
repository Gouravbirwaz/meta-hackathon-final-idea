"""
KisanAgent — Event Engine (State Machine)
==========================================
Schedules and manages discrete events across the 90-day season:
  - Pest outbreaks (aphids, whitefly, leaf_curl_virus)
  - Government scheme windows and deadlines
  - Mandi price spikes
  - Tool failure days (sensor offline, IMD outage)

Events are scheduled at reset() and injected into simulate_day() via
an event_state dict — the agent sees them only through noisy tool APIs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("kisanagent.event_engine")


# ──────────────────────────────────────────────────────────────
# Canonical scheme definitions (Karnataka 2024-25)
# ──────────────────────────────────────────────────────────────

KARNATAKA_SCHEMES = [
    {
        "name": "PM-KISAN Input Supplement",
        "benefit_inr": 2000,
        "eligibility": "2-acre tomato farmer, Kolar district",
        "deadline_day": 30,
        "required_documents": ["Aadhaar", "Land records", "Bank passbook"],
        "application_url": "https://pmkisan.gov.in",
    },
    {
        "name": "Crop Insurance PMFBY",
        "benefit_inr": 3000,
        "eligibility": "All insured crop farmers, premium ₹500",
        "deadline_day": 20,
        "required_documents": ["Aadhaar", "Land records", "Crop sowing certificate"],
        "application_url": "https://pmfby.gov.in",
    },
    {
        "name": "Drip Irrigation Subsidy",
        "benefit_inr": 5000,
        "eligibility": "Farmers with drip system installed, Karnataka",
        "deadline_day": 50,
        "required_documents": ["Drip system invoice", "Aadhaar", "Land records"],
        "application_url": "https://raitamitra.karnataka.gov.in",
    },
    {
        "name": "Tomato Special Support Price",
        "benefit_inr": 1500,
        "eligibility": "Kolar/Chikkaballapur tomato growers",
        "deadline_day": 70,
        "required_documents": ["Mandi receipt", "Aadhaar"],
        "application_url": "https://krishimela.karnataka.gov.in",
    },
]


class EventEngine:
    """
    Discrete event state machine for the 90-day farming season.

    Events managed:
      1. Pest events  — onset day, escalation, treatment window
      2. Scheme events — open/closing/expired windows
      3. Price spikes  — mandi price surge events
      4. Tool failures — sensor offline, API outage days

    The engine exposes get_event_state(day) which the server merges
    into simulate_day() via the event_state dict.
    """

    # ── Pest escalation schedule (days from onset) ────────────
    PEST_ESCALATION = {
        0: "LOW",
        2: "MEDIUM",
        4: "HIGH",
        6: "CRITICAL",
    }
    TREATMENT_WINDOW_DAYS = 4  # days from onset to treat before CRITICAL

    # ── Difficulty multipliers ────────────────────────────────
    DIFFICULTY_PARAMS = {
        "easy":   {"n_pest_events": 1, "n_tool_failures": 2,  "price_spike_magnitude": 20},
        "medium": {"n_pest_events": 2, "n_tool_failures": 4,  "price_spike_magnitude": 25},
        "hard":   {"n_pest_events": 3, "n_tool_failures": 8,  "price_spike_magnitude": 32},
    }

    def __init__(self) -> None:
        self.rng = np.random.default_rng(42)
        self.difficulty = "medium"
        self._reset()

    # ──────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────

    def reset(self, difficulty: str = "medium", seed: int = 42) -> None:
        """
        Schedule all events for a new episode.
        Called by the server after FarmSimulator.reset().
        """
        self.rng = np.random.default_rng(seed + 1000)  # offset from weather seed
        self.difficulty = difficulty
        self._reset()
        self._schedule_pest_events()
        self._schedule_scheme_events()
        self._schedule_tool_failures()
        logger.info(
            "EventEngine reset: difficulty=%s seed=%d pests=%d",
            difficulty, seed, len(self.pest_schedule),
        )

    def _reset(self) -> None:
        """Clear all event state."""
        self.pest_schedule: List[Dict[str, Any]] = []
        self.scheme_schedule: List[Dict[str, Any]] = KARNATAKA_SCHEMES.copy()
        self.tool_failure_days: Dict[str, List[int]] = {
            "weather": [], "soil": [], "mandi_price": [],
            "govt_scheme": [], "pest_alert": [], "credit": [],
        }
        self.price_spike_days: Dict[int, float] = {}

        # Runtime state
        self.active_pest_events: List[Dict[str, Any]] = []
        self.captured_schemes: List[str] = []
        self.pest_events_log: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────
    # Event Scheduling
    # ──────────────────────────────────────────────────────────

    def _schedule_pest_events(self) -> None:
        """Schedule 1-3 pest outbreaks depending on difficulty."""
        params = self.DIFFICULTY_PARAMS[self.difficulty]
        n = params["n_pest_events"]
        pests = ["aphids", "whitefly", "leaf_curl_virus"]

        # Spread outbreaks across the season, prefer vegetative/fruiting
        onset_days = sorted(
            int(d) for d in self.rng.integers(15, 82, size=n)
        )
        for i, onset in enumerate(onset_days):
            pest_name = pests[i % len(pests)]
            self.pest_schedule.append({
                "onset_day": onset,
                "pest_name": pest_name,
                "treated_in_window": False,
                "treatment_day": None,
                "escalation_day": onset + 6,
                "critical_day": onset + 6,
            })
        logger.debug("Scheduled %d pest events: %s", n, onset_days)

    def _schedule_scheme_events(self) -> None:
        """Use the canonical KARNATAKA_SCHEMES as the schedule."""
        # Already set in _reset via KARNATAKA_SCHEMES
        pass

    def _schedule_tool_failures(self) -> None:
        """Randomly assign tool failure days."""
        params = self.DIFFICULTY_PARAMS[self.difficulty]
        n_failures = params["n_tool_failures"]
        all_tools = list(self.tool_failure_days.keys())

        for _ in range(n_failures):
            tool = all_tools[int(self.rng.integers(0, len(all_tools)))]
            fail_day = int(self.rng.integers(0, 90))
            self.tool_failure_days[tool].append(fail_day)

    # ──────────────────────────────────────────────────────────
    # Runtime Event State
    # ──────────────────────────────────────────────────────────

    def get_event_state(self, day: int) -> Dict[str, Any]:
        """
        Return the merged event state dict for the current day.
        Passed into FarmSimulator.simulate_day() as event_state.
        """
        # ── Activate new pests ──────────────────────────────────
        newly_active = [
            p for p in self.pest_schedule
            if p["onset_day"] == day and not any(
                a["pest_name"] == p["pest_name"] and a["onset_day"] == p["onset_day"]
                for a in self.active_pest_events
            )
        ]
        self.active_pest_events.extend(newly_active)

        # ── Compute current pest risk level ────────────────────
        pest_active = False
        pest_name = None
        pest_risk_level = "LOW"
        days_since_onset = None

        for event in self.active_pest_events:
            if event.get("treated_in_window") or event.get("treatment_day") is not None:
                continue
            onset = event["onset_day"]
            age = day - onset
            if age >= 0:
                pest_active = True
                pest_name = event["pest_name"]
                days_since_onset = age
                pest_risk_level = self._escalation_level(age)

        # ── Active government scheme ────────────────────────────
        active_scheme = None
        scheme_benefit_inr = 0
        for scheme in self.scheme_schedule:
            if scheme["deadline_day"] >= day and scheme["name"] not in self.captured_schemes:
                active_scheme = scheme["name"]
                scheme_benefit_inr = scheme["benefit_inr"]
                break

        # ── Tool failures today ─────────────────────────────────
        tool_failures_today = {
            tool: day in days
            for tool, days in self.tool_failure_days.items()
        }

        return {
            "pest_active": pest_active,
            "pest_name": pest_name,
            "pest_risk_level": pest_risk_level,
            "days_since_pest_onset": days_since_onset,
            "pest_suppressed": False,  # will be set by simulator after spray
            "active_scheme_name": active_scheme,
            "scheme_benefit_inr": scheme_benefit_inr,
            "tool_failures_today": tool_failures_today,
            "loan_amount_requested": 10_000,
        }

    def _escalation_level(self, days_since_onset: int) -> str:
        """Return pest risk level based on days since detection."""
        level = "LOW"
        for threshold, lv in sorted(self.PEST_ESCALATION.items()):
            if days_since_onset >= threshold:
                level = lv
        return level

    # ──────────────────────────────────────────────────────────
    # Treatment Recording
    # ──────────────────────────────────────────────────────────

    def record_treatment(self, day: int) -> bool:
        """
        Called when agent sprays pesticide.
        Returns True if treatment was within the 4-day window.
        """
        treated_any = False
        for event in self.active_pest_events:
            if event.get("treatment_day") is not None:
                continue
            onset = event["onset_day"]
            age = day - onset
            if 0 <= age <= self.TREATMENT_WINDOW_DAYS:
                event["treated_in_window"] = True
                event["treatment_day"] = day
                treated_any = True
                logger.info(
                    "Pest '%s' treated on day %d (age=%d, within window)",
                    event["pest_name"], day, age,
                )
            elif age > 0:
                # Late treatment — past window
                event["treatment_day"] = day
                logger.warning(
                    "Late treatment for '%s' on day %d (age=%d, window was %d)",
                    event["pest_name"], day, age, self.TREATMENT_WINDOW_DAYS,
                )
        return treated_any

    def record_scheme_capture(self, scheme_name: str) -> bool:
        """Mark a government scheme as captured."""
        if scheme_name and scheme_name not in self.captured_schemes:
            self.captured_schemes.append(scheme_name)
            logger.info("Scheme captured: %s", scheme_name)
            return True
        return False

    # ──────────────────────────────────────────────────────────
    # Alerts for Agent Observation
    # ──────────────────────────────────────────────────────────

    def get_active_alerts(self, day: int) -> List[str]:
        """Generate advisory alert strings for FarmerObservation."""
        alerts: List[str] = []

        # Pest alerts
        for event in self.active_pest_events:
            if event.get("treatment_day") is not None:
                continue
            onset = event["onset_day"]
            age = day - onset
            if age >= 0:
                level = self._escalation_level(age)
                alerts.append(
                    f"PEST_ALERT: {event['pest_name']} {level} "
                    f"(day {age} since onset)"
                )

        # Scheme deadline alerts
        for scheme in self.scheme_schedule:
            if scheme["name"] in self.captured_schemes:
                continue
            days_left = scheme["deadline_day"] - day
            if 0 < days_left <= 10:
                urgency = "CLOSING_SOON" if days_left <= 5 else "DEADLINE_ALERT"
                alerts.append(
                    f"{urgency}: {scheme['name']} — {days_left} days left "
                    f"(₹{scheme['benefit_inr']})"
                )
            elif days_left < 0:
                alerts.append(f"EXPIRED: {scheme['name']} deadline passed")

        return alerts

    # ──────────────────────────────────────────────────────────
    # Terminal State
    # ──────────────────────────────────────────────────────────

    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Return event summary for final grader scoring.
        Passed as part of final_state to KisanGrader.compute_final_scores().
        """
        all_schemes = [s["name"] for s in self.scheme_schedule]
        return {
            "pest_events": self.pest_schedule,   # includes treated_in_window
            "schemes_available": all_schemes,
            "schemes_captured": self.captured_schemes,
        }

    def get_state(self) -> Dict[str, Any]:
        """Return event engine state for /state endpoint."""
        return {
            "difficulty": self.difficulty,
            "pest_schedule": self.pest_schedule,
            "active_pest_events": self.active_pest_events,
            "captured_schemes": self.captured_schemes,
            "tool_failure_days": {
                tool: days for tool, days in self.tool_failure_days.items()
            },
        }

    # ──────────────────────────────────────────────────────────
    # Tool Failure Check
    # ──────────────────────────────────────────────────────────

    def is_tool_failing_today(self, tool_name: str, day: int) -> bool:
        """Returns True if the given tool is scheduled to fail today."""
        return day in self.tool_failure_days.get(tool_name, [])
