"""
KisanAgent — Soil Sensor Tool
================================
Real-world analog: IoT soil moisture sensor network
deployed under PM-KISAN scheme in Karnataka.

4-8 hour reading delay simulates real sensor latency.
Readings include realistic ±5% noise.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("kisanagent.tools.soil")


class SoilTool:
    """
    IoT soil sensor analog.

    Returns soil moisture, pH, nitrogen, phosphorus with
    calibration-dependent noise. Sensor can go offline.

    Default Kolar district tomato soil parameters:
      pH:         6.2 – 7.1 (slightly acidic, ideal)
      Nitrogen:   20 – 60 kg/acre
      Phosphorus: 15 – 45 kg/acre
    """

    # Calibration drift — sensor degrades over time
    CALIBRATION_DRIFT_RATE = 0.1  # % per day uncalibrated

    def __init__(
        self,
        simulator_ref: Any,
        failure_days: Optional[List[int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.simulator = simulator_ref
        self.failure_days = set(failure_days or [])
        self.rng = rng or np.random.default_rng(77)
        self._last_calibrated_day: int = 0

    def call(
        self,
        farm_id: str = "farm_001",
        current_day: int = 0,
    ) -> Dict[str, Any]:
        """
        Return soil sensor reading for Harish's farm.

        Returns:
            {
              "moisture_pct": float | null,
              "ph": float | null,
              "nitrogen_kg_per_acre": float | null,
              "phosphorus_kg_per_acre": float | null,
              "reading_delay_hours": int,
              "sensor_status": "online"|"offline"|"low_battery",
              "last_calibrated_days_ago": int,
              "data_quality": str,
              "advisory": str
            }
        """
        # Failure check
        if current_day in self.failure_days:
            return self._offline_response(current_day, reason="scheduled_maintenance")

        # Random sensor failure (5% chance)
        if self.rng.random() < 0.05:
            return self._offline_response(current_day, reason="connectivity_lost")

        # Low battery (8% chance)
        low_battery = self.rng.random() < 0.08

        # Calibration age
        days_since_calibration = current_day - self._last_calibrated_day
        calibration_noise = days_since_calibration * self.CALIBRATION_DRIFT_RATE

        # True moisture from simulator
        true_moisture = getattr(self.simulator, "soil_moisture_pct", 65.0)
        noise = float(self.rng.normal(0, 5.0 + calibration_noise))
        noisy_moisture = float(np.clip(true_moisture + noise, 0.0, 100.0))

        # Soil chemistry (Kolar tomato typical ranges)
        ph = round(float(self.rng.uniform(6.2, 7.1) + self.rng.normal(0, 0.1)), 2)
        nitrogen = round(float(self.rng.uniform(20, 60) + self.rng.normal(0, 3)), 1)
        phosphorus = round(float(self.rng.uniform(15, 45) + self.rng.normal(0, 2)), 1)

        reading_delay = int(self.rng.integers(4, 9))

        data_quality = "degraded" if (low_battery or calibration_noise > 5) else "good"
        advisory = _soil_advisory(noisy_moisture, ph, nitrogen)

        return {
            "moisture_pct": round(noisy_moisture, 1),
            "ph": ph,
            "nitrogen_kg_per_acre": nitrogen,
            "phosphorus_kg_per_acre": phosphorus,
            "reading_delay_hours": reading_delay,
            "sensor_status": "low_battery" if low_battery else "online",
            "last_calibrated_days_ago": days_since_calibration,
            "farm_id": farm_id,
            "data_quality": data_quality,
            "advisory": advisory,
        }

    def _offline_response(self, current_day: int, reason: str) -> Dict[str, Any]:
        return {
            "moisture_pct": None,
            "ph": None,
            "nitrogen_kg_per_acre": None,
            "phosphorus_kg_per_acre": None,
            "reading_delay_hours": 0,
            "sensor_status": "offline",
            "last_calibrated_days_ago": current_day - self._last_calibrated_day,
            "farm_id": "farm_001",
            "data_quality": "unavailable",
            "advisory": f"Sensor offline ({reason}). Estimate from weather data or visual inspection.",
        }


def _soil_advisory(moisture: float, ph: float, nitrogen: float) -> str:
    if moisture < 40:
        return f"CRITICAL: Soil moisture {moisture:.0f}% — irrigate immediately. Risk of fruit drop."
    elif moisture > 80:
        return f"WARNING: Soil moisture {moisture:.0f}% — waterlogging risk. Skip next irrigation."
    elif moisture < 55:
        return f"Moisture {moisture:.0f}% — consider irrigating within 2 days."
    if ph < 6.0:
        return f"Soil pH {ph:.1f} — too acidic for tomatoes. Apply lime."
    if nitrogen < 25:
        return f"Low nitrogen ({nitrogen:.0f} kg/acre) — consider top-dressing fertiliser."
    return f"Soil conditions optimal. Moisture: {moisture:.0f}%, pH: {ph:.1f}."
