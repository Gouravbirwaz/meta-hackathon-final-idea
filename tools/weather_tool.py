"""
KisanAgent — Weather Tool
==========================
Real-world analog: India Meteorological Department (IMD)
Karnataka district weather API.

Provides probabilistic 3-day forecast with realistic
monsoon patterns for Kolar district.
Noise model degrades with forecast horizon.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("kisanagent.tools.weather")


class WeatherTool:
    """
    IMD-analog weather forecast tool.

    Noise model:
      Day+1: ±10% rain_prob, ±1°C temp
      Day+2: ±20% rain_prob, ±2°C temp
      Day+3: ±30% rain_prob, ±3°C temp
      10% chance data_quality = "degraded"
       2% chance data_quality = "unavailable"
    During tool failure day: force "unavailable".
    """

    SOURCE = "IMD Karnataka District Bulletin"

    def __init__(
        self,
        weather_sequence: List[Dict[str, Any]],
        failure_days: Optional[List[int]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.weather_sequence = weather_sequence
        self.failure_days = set(failure_days or [])
        self.rng = rng or np.random.default_rng(99)

    def call(
        self,
        current_day: int,
        days_ahead: int = 3,
    ) -> Dict[str, Any]:
        """
        Return noisy 3-day weather forecast for Kolar district.

        Returns:
            {
              "forecast": [...],
              "data_quality": "good"|"degraded"|"unavailable",
              "source": str,
              "advisory": str,
              "latency_ms": float
            }
        """
        start = time.perf_counter()

        # Tool failure check
        if current_day in self.failure_days:
            return {
                "forecast": [],
                "data_quality": "unavailable",
                "source": self.SOURCE,
                "advisory": "IMD Karnataka API offline. Retry after 2 hours.",
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }

        # Random quality degradation
        rand_q = self.rng.random()
        if rand_q < 0.02:
            quality = "unavailable"
            return {
                "forecast": [],
                "data_quality": quality,
                "source": self.SOURCE,
                "advisory": "Data feed temporarily unavailable.",
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }
        elif rand_q < 0.12:
            quality = "degraded"
        else:
            quality = "good"

        # Build noisy forecast
        forecast: List[Dict[str, Any]] = []
        for offset in range(1, days_ahead + 1):
            d = current_day + offset
            if d >= len(self.weather_sequence):
                break

            truth = self.weather_sequence[d]
            noise_scale = offset * 0.10  # 10% per day ahead

            noisy_rain_prob = float(np.clip(
                truth["rain_prob"] + self.rng.normal(0, noise_scale),
                0.0, 1.0
            ))
            noisy_temp = round(
                float(truth["temp_c"]) + float(self.rng.normal(0, offset)), 1
            )
            noisy_rain_mm = max(0.0, round(
                float(truth["rainfall_mm"]) + float(self.rng.normal(0, max(1, truth["rainfall_mm"] * noise_scale))),
                1
            ))
            confidence = round(max(0.3, 1.0 - offset * 0.15), 2)

            forecast.append({
                "day": d,
                "rain_prob": round(noisy_rain_prob, 2),
                "temp_c": noisy_temp,
                "rainfall_mm": noisy_rain_mm,
                "confidence": confidence,
                "summary": _weather_summary(noisy_rain_mm, noisy_temp),
            })

        # Generate advisory
        advisory = _generate_advisory(forecast)

        latency_ms = round((time.perf_counter() - start) * 1000 + float(self.rng.uniform(80, 400)), 1)

        return {
            "forecast": forecast,
            "data_quality": quality,
            "source": self.SOURCE,
            "advisory": advisory,
            "latency_ms": latency_ms,
        }


def _weather_summary(rain_mm: float, temp_c: float) -> str:
    if rain_mm >= 15:
        return "heavy_rain"
    elif rain_mm > 2:
        return "light_rain"
    elif temp_c > 34:
        return "dry"
    return "cloudy"


def _generate_advisory(forecast: List[Dict[str, Any]]) -> str:
    if not forecast:
        return "No forecast available."
    max_rain = max(f["rainfall_mm"] for f in forecast)
    max_rain_day = next(
        (f["day"] for f in forecast if f["rainfall_mm"] == max_rain), None
    )
    if max_rain >= 20:
        return f"Heavy rain expected day {max_rain_day} ({max_rain:.0f}mm). Delay irrigation. Check drainage."
    elif max_rain >= 5:
        return f"Light rain forecast day {max_rain_day}. Monitor soil moisture before irrigating."
    else:
        return "Dry weather expected next 3 days. Monitor soil moisture closely."
