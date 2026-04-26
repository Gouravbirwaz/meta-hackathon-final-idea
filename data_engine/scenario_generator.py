"""
KisanAgent — Scenario Generator
=================================
Generates randomized season configurations for training diversity.
Used during GRPO training to prevent overfitting to fixed scenarios.

Produces varied:
  - Monsoon onset/offset timing
  - Pest species and onset days
  - Price spike timing and magnitude
  - Scheme deadline windows
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List

import numpy as np


class ScenarioGenerator:
    """
    Randomized season generator for training data diversity.

    Each generated scenario is a complete configuration dict
    that can seed FarmSimulator and EventEngine.
    """

    PEST_SPECIES = ["aphids", "whitefly", "leaf_curl_virus"]
    MARKETS = ["KR Puram Bangalore", "Kolar Town", "Chikkaballapur"]

    def __init__(self, master_seed: int = 0) -> None:
        self.master_seed = master_seed
        self.rng = np.random.default_rng(master_seed)

    def generate_batch(self, n: int = 100) -> List[Dict[str, Any]]:
        """Generate a batch of n randomized season configs."""
        return [self.generate_one(seed=self.master_seed + i) for i in range(n)]

    def generate_one(self, seed: int = 42) -> Dict[str, Any]:
        """Generate a single randomized season configuration."""
        rng = np.random.default_rng(seed)

        # Random monsoon timing (±5 days around typical)
        sw_monsoon_start = int(rng.integers(18, 26))   # typically day 21
        ne_monsoon_start = int(rng.integers(42, 50))   # typically day 46
        post_monsoon_start = int(rng.integers(72, 80))  # typically day 76

        # Random initial conditions
        initial_moisture = float(rng.uniform(55.0, 75.0))
        initial_balance = float(rng.uniform(12_000, 20_000))

        # Pest events
        n_pests = int(rng.integers(1, 4))
        pest_events = []
        for i in range(n_pests):
            onset = int(rng.integers(10, 85))
            pest_events.append({
                "pest_name": self.PEST_SPECIES[i % len(self.PEST_SPECIES)],
                "onset_day": onset,
                "treatment_window_days": int(rng.integers(3, 5)),
            })

        # Price spikes
        n_spikes = int(rng.integers(1, 3))
        price_spikes = []
        for _ in range(n_spikes):
            price_spikes.append({
                "day": int(rng.integers(55, 89)),
                "price_per_kg": float(round(rng.uniform(20.0, 35.0), 2)),
                "duration_days": int(rng.integers(2, 6)),
            })

        return {
            "seed": seed,
            "monsoon_profile": {
                "sw_start": sw_monsoon_start,
                "ne_start": ne_monsoon_start,
                "post_monsoon_start": post_monsoon_start,
            },
            "initial_conditions": {
                "soil_moisture_pct": round(initial_moisture, 1),
                "bank_balance_inr": round(initial_balance, 0),
            },
            "pest_events": pest_events,
            "price_spikes": price_spikes,
        }

    def save_batch(self, n: int = 100, path: str = "assets/training_scenarios.json") -> None:
        """Generate and save a batch of scenarios to JSON."""
        batch = self.generate_batch(n)
        with open(path, "w") as f:
            json.dump(batch, f, indent=2)
        print(f"Saved {n} scenarios to {path}")


if __name__ == "__main__":
    gen = ScenarioGenerator(master_seed=2025)
    gen.save_batch(100)
