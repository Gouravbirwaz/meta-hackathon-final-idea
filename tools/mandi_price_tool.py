"""
KisanAgent — Mandi Price Tool
================================
Real-world analog: Agmarknet / eNAM national agricultural
market price feed.

Prices updated daily. Reflects real Bangalore KR Puram
mandi tomato price range (₹12–35/kg).
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("kisanagent.tools.mandi_price")

# Public holidays / market closure days (typical Karnataka calendar)
MARKET_HOLIDAYS = {14, 15, 46, 47, 73}  # Sankranti, Independence Day, Dasara


class MandiPriceTool:
    """
    Agmarknet / eNAM price feed analog for KR Puram Bangalore mandi.

    Base price range: ₹12–18/kg
    During PriceSpikeEvent: ₹22–35/kg
    Price inversely correlated with market_arrivals_kg.
    Forecast has ±15% noise.
    """

    MARKET_NAME = "KR Puram Wholesale Market, Bangalore"
    BASE_MARKET_ARRIVALS = 45_000  # kg per day typical

    def __init__(
        self,
        price_sequence: List[float],
        spike_events: Optional[List[Dict[str, Any]]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.price_sequence = price_sequence
        self.spike_events = spike_events or []
        self.rng = rng or np.random.default_rng(55)
        # Season start date (approximate planting date)
        self._season_start = date(2025, 6, 15)

    def call(
        self,
        crop: str = "tomato",
        market: str = "KR Puram Bangalore",
        current_day: int = 0,
    ) -> Dict[str, Any]:
        """
        Return today's mandi price data and 3-day forecast.

        Returns:
            {
              "today_price_per_kg": float,
              "7day_trend": list[float],
              "market_arrivals_kg": int,
              "price_forecast_3day": list[float],
              "price_change_pct_vs_yesterday": float,
              "last_updated": str,
              "market_status": "open"|"closed"|"holiday",
              "data_quality": str,
              "advisory": str
            }
        """
        today_date = self._season_start + timedelta(days=current_day)

        # Market status
        if today_date.weekday() == 6 or current_day in MARKET_HOLIDAYS:
            return {
                "today_price_per_kg": None,
                "7day_trend": self._get_trend(current_day),
                "market_arrivals_kg": 0,
                "price_forecast_3day": self._get_forecast(current_day),
                "price_change_pct_vs_yesterday": 0.0,
                "last_updated": today_date.isoformat(),
                "market_status": "holiday" if current_day in MARKET_HOLIDAYS else "closed",
                "data_quality": "good",
                "advisory": "Market closed today. Use yesterday's price for planning.",
            }

        # Today's price
        today_price = self._get_price(current_day)
        yesterday_price = self._get_price(current_day - 1) if current_day > 0 else today_price
        price_change_pct = round(
            ((today_price - yesterday_price) / max(yesterday_price, 1.0)) * 100, 1
        )

        # Market arrivals (inversely correlated with price)
        arrivals_noise = int(self.rng.normal(0, 5000))
        price_ratio = today_price / 15.0  # normalised against typical
        arrivals = int(max(5000, self.BASE_MARKET_ARRIVALS / price_ratio + arrivals_noise))

        # Forecast
        forecast_3day = self._get_forecast(current_day)

        # Advisory
        advisory = _price_advisory(today_price, price_change_pct, forecast_3day)

        return {
            "today_price_per_kg": today_price,
            "7day_trend": self._get_trend(current_day),
            "market_arrivals_kg": arrivals,
            "price_forecast_3day": forecast_3day,
            "price_change_pct_vs_yesterday": price_change_pct,
            "last_updated": today_date.isoformat(),
            "market_status": "open",
            "crop": crop,
            "market": self.MARKET_NAME,
            "data_quality": "good",
            "advisory": advisory,
        }

    def _get_price(self, day: int) -> float:
        if day < 0 or day >= len(self.price_sequence):
            return 14.0
        price = self.price_sequence[day]
        noise = float(self.rng.normal(0, 0.3))
        return round(max(8.0, price + noise), 2)

    def _get_trend(self, current_day: int) -> List[float]:
        trend = []
        for d in range(max(0, current_day - 6), current_day + 1):
            trend.append(self._get_price(d))
        return trend

    def _get_forecast(self, current_day: int) -> List[float]:
        forecast = []
        for offset in range(1, 4):
            d = current_day + offset
            price = self._get_price(d)
            noise = float(self.rng.normal(0, price * 0.15))
            forecast.append(round(max(8.0, price + noise), 2))
        return forecast


def _price_advisory(price: float, change_pct: float, forecast: List[float]) -> str:
    max_forecast = max(forecast) if forecast else price
    if price > 25:
        return f"SELL ALERT: Price at ₹{price:.0f}/kg — premium window. Consider selling now."
    elif change_pct > 10:
        return f"Price rising ({change_pct:+.1f}% today). Forecast peak: ₹{max_forecast:.0f}/kg in 3 days."
    elif change_pct < -10:
        return f"Price falling ({change_pct:+.1f}% today). Hold if forecast improves."
    elif max_forecast > price * 1.15:
        return f"Hold recommendation: forecast ₹{max_forecast:.0f}/kg in 3 days vs today ₹{price:.0f}."
    return f"Stable prices. Today ₹{price:.0f}/kg. No major movement expected."
