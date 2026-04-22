"""
KisanAgent — Season Scenario Definitions
==========================================
3 canonical scenario presets: easy / medium / hard.
Each scenario defines difficulty multipliers for:
  - Weather variance
  - Pest event count and timing
  - Tool failure frequency
  - Price spike magnitude and count
  - Scheme deadline tightness

Used by the server /reset endpoint to configure the episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ScenarioConfig:
    """
    Full configuration for one season difficulty level.
    """

    name: str
    description: str

    # Weather
    weather_variance_multiplier: float = 1.0
    unseasonal_rain_prob: float = 0.15

    # Pest events
    n_pest_events: int = 2
    pest_onset_range: tuple = (15, 82)   # (min_day, max_day)
    pest_treatment_window_days: int = 4

    # Tool reliability
    n_tool_failures: int = 4
    tool_failure_duration_days: int = 1

    # Price
    n_price_spikes: int = 1
    price_spike_magnitude_max: float = 25.0  # INR/kg

    # Scheme deadlines (days tighter = harder)
    scheme_deadline_adjustment_days: int = 0  # negative = tighter

    # Initial conditions
    initial_balance_inr: float = 15_000.0
    initial_soil_moisture: float = 65.0

    # Optimal income reference for this difficulty
    optimal_income_inr: float = 40_000.0
    baseline_income_inr: float = 15_000.0


# ── Canonical Scenarios ──────────────────────────────────────────────────────


EASY_SEASON = ScenarioConfig(
    name="easy",
    description=(
        "Favourable season: predictable monsoon, single pest outbreak, "
        "minimal tool failures, long scheme windows. "
        "Target: ₹28,000+ net income."
    ),
    weather_variance_multiplier=0.7,
    unseasonal_rain_prob=0.05,
    n_pest_events=1,
    pest_onset_range=(25, 70),
    n_tool_failures=2,
    n_price_spikes=2,
    price_spike_magnitude_max=22.0,
    scheme_deadline_adjustment_days=10,   # more time for schemes
    initial_balance_inr=20_000.0,         # slightly more starting cash
    initial_soil_moisture=70.0,
    optimal_income_inr=40_000.0,
)

MEDIUM_SEASON = ScenarioConfig(
    name="medium",
    description=(
        "Typical Kolar season: moderate monsoon variance, 2 pest events, "
        "occasional sensor outages. "
        "Target: ₹22,000–₹30,000 net income."
    ),
    weather_variance_multiplier=1.0,
    unseasonal_rain_prob=0.15,
    n_pest_events=2,
    pest_onset_range=(15, 82),
    n_tool_failures=4,
    n_price_spikes=1,
    price_spike_magnitude_max=28.0,
    scheme_deadline_adjustment_days=0,
    initial_balance_inr=15_000.0,
    initial_soil_moisture=65.0,
    optimal_income_inr=40_000.0,
)

HARD_SEASON = ScenarioConfig(
    name="hard",
    description=(
        "Challenging season: erratic NE monsoon failure, 3 pest events "
        "with tight treatment windows, frequent tool outages, "
        "early scheme deadlines. "
        "Target: ₹18,000+ net income — requires expert decisions."
    ),
    weather_variance_multiplier=1.5,
    unseasonal_rain_prob=0.25,
    n_pest_events=3,
    pest_onset_range=(10, 85),
    pest_treatment_window_days=3,          # 1 day tighter
    n_tool_failures=8,
    n_price_spikes=1,
    price_spike_magnitude_max=32.0,
    scheme_deadline_adjustment_days=-7,   # tighter deadlines
    initial_balance_inr=12_000.0,         # starting cash harder
    initial_soil_moisture=55.0,           # drier start
    optimal_income_inr=35_000.0,          # harder to reach optimal
)


SCENARIOS = {
    "easy": EASY_SEASON,
    "medium": MEDIUM_SEASON,
    "hard": HARD_SEASON,
}


def get_scenario(difficulty: str) -> ScenarioConfig:
    """Return the canonical scenario config for the given difficulty."""
    if difficulty not in SCENARIOS:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. Choose from: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[difficulty]
