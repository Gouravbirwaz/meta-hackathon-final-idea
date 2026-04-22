"""
KisanAgent — Farm Simulator (Ground Truth Engine)
==================================================
The agent NEVER sees this directly.
All agent observations arrive through noisy tool APIs.
This engine determines the actual outcomes.

Karnataka / Kolar district tomato farm — 2 acres.
90-day season from planting to harvest.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("kisanagent.simulator")


class FarmSimulator:
    """
    Ground-truth physical simulation of Harish's 2-acre tomato farm.

    Key state:
      weather_truth      — 90-day sequence of actual weather
      soil_moisture_pct  — continuously updated
      yield_multiplier   — starts 1.0, degrades on stress
      bank_balance_inr   — real economic state
    """

    # ── Farm constants ──────────────────────────────────────────
    FARM_AREA_ACRES: float = 2.0
    BASE_YIELD_KG: float = 8_000.0          # optimal 2-acre season
    INITIAL_BALANCE_INR: float = 15_000.0

    # ── Action costs (INR) ──────────────────────────────────────
    IRRIGATION_COST: float = 200.0
    FERTILIZER_COST: float = 600.0
    PESTICIDE_COST: float = 800.0

    # ── Soil physics ────────────────────────────────────────────
    DAILY_EVAPORATION_PCT: float = 3.0      # moisture lost per dry day
    RAIN_ABSORPTION_FACTOR: float = 0.6     # fraction of rainfall absorbed
    MAX_SOIL_MOISTURE: float = 95.0
    MIN_SOIL_MOISTURE: float = 5.0

    # ── Yield stress thresholds ─────────────────────────────────
    DROUGHT_MOISTURE_THRESHOLD: float = 40.0   # below this → yield loss
    WATERLOG_MOISTURE_THRESHOLD: float = 85.0  # above this → root rot
    DROUGHT_YIELD_LOSS_KG_PER_DAY: float = 50.0
    PEST_YIELD_LOSS_KG_PER_DAY: float = 100.0
    ROOT_ROT_YIELD_LOSS_KG_PER_DAY: float = 30.0

    # ── Loan parameters (Kisan Credit Card) ─────────────────────
    KCC_MAX_LOAN_INR: int = 25_000
    KCC_INTEREST_ANNUAL: float = 0.07
    KCC_REPAYMENT_DAYS: int = 180

    # ── Distress sale price (force-sell on day 90) ───────────────
    DISTRESS_PRICE_PER_KG: float = 10.0

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._reset_state()
        self._generate_season()

    # ──────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        """Clear all mutable simulation state."""
        self.day: int = 0
        self.soil_moisture_pct: float = 65.0
        self.yield_kg: float = 0.0
        self.yield_multiplier: float = 1.0
        self.root_rot_risk_days: int = 0

        # Economics
        self.bank_balance_inr: float = self.INITIAL_BALANCE_INR
        self.total_costs_inr: float = 0.0
        self.gross_revenue_inr: float = 0.0
        self.active_debt_inr: float = 0.0
        self.sell_decisions: List[Dict[str, Any]] = []

        # Pest state (managed by EventEngine but tracked here)
        self.pest_active: bool = False
        self.pest_name: Optional[str] = None
        self.pest_onset_day: Optional[int] = None
        self.pest_risk_level: str = "LOW"
        self.pest_events_log: List[Dict[str, Any]] = []

        # Scheme state
        self.schemes_available: List[str] = []
        self.schemes_captured: List[str] = []
        self.applied_schemes: set = set()

        # Resource accounting
        self.water_used_liters: float = 0.0
        self.chemical_applications: int = 0

        # Loan tracking
        self.has_active_loan: bool = False

        # Daily alerts
        self.active_alerts: List[str] = []

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Full reset for a new episode.
        Regenerates weather/price sequences with optional new seed.
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        self._generate_season()
        logger.info("FarmSimulator reset with seed=%d", self.seed)

    # ──────────────────────────────────────────────────────────
    # Season Generation
    # ──────────────────────────────────────────────────────────

    def _generate_season(self) -> None:
        """
        Generate 90-day weather and price sequences.

        Kolar district, Karnataka monsoon pattern:
          Days  0-20:  pre-monsoon dry     rain_prob=0.05, temp=32-38°C
          Days 21-45:  SW monsoon active   rain_prob=0.65, temp=24-28°C, 15-40mm
          Days 46-75:  NE monsoon trans.   rain_prob=0.35, temp=26-32°C
          Days 76-90:  post-monsoon dry    rain_prob=0.08, temp=28-34°C

        +15% chance of unseasonal rain any dry day.
        """
        self.weather_truth: List[Dict[str, Any]] = []
        self.price_truth: List[float] = []

        # Weather phases: (start_day, end_day, rain_prob, temp_min, temp_max, rain_mm_min, rain_mm_max)
        phases = [
            (0,  20,  0.05, 32, 38,  2,  8),
            (21, 45,  0.65, 24, 28, 15, 40),
            (46, 75,  0.35, 26, 32,  5, 20),
            (76, 90,  0.08, 28, 34,  2,  6),
        ]

        for day in range(90):
            phase = self._get_phase(day, phases)
            rain_prob, t_min, t_max, r_min, r_max = (
                phase[2], phase[3], phase[4], phase[5], phase[6]
            )

            # Unseasonal rain chance on dry days
            effective_rain_prob = rain_prob + (
                0.15 if self.rng.random() < 0.15 and rain_prob < 0.2 else 0
            )
            effective_rain_prob = min(effective_rain_prob, 1.0)

            is_rain = bool(self.rng.random() < effective_rain_prob)
            rainfall_mm = float(self.rng.uniform(r_min, r_max)) if is_rain else 0.0
            temp_c = float(self.rng.uniform(t_min, t_max))

            if rainfall_mm >= 15:
                summary = "heavy_rain"
            elif rainfall_mm > 0:
                summary = "light_rain"
            elif temp_c > 34:
                summary = "dry"
            else:
                summary = "cloudy" if self.rng.random() < 0.2 else "dry"

            self.weather_truth.append({
                "day": day,
                "rainfall_mm": round(rainfall_mm, 1),
                "temp_c": round(temp_c, 1),
                "rain_prob": round(effective_rain_prob, 2),
                "summary": summary,
            })

        # Mandi price sequence — Bangalore KR Puram tomato
        # Base: ₹12-18/kg, spikes during shortage periods
        base_price = float(self.rng.uniform(12, 16))
        for day in range(90):
            # Gentle daily drift
            drift = float(self.rng.normal(0, 0.5))
            base_price = float(np.clip(base_price + drift, 10.0, 40.0))
            self.price_truth.append(round(base_price, 2))

        # Inject 1-2 price spikes (harvest season demand)
        n_spikes = int(self.rng.integers(1, 3))
        for _ in range(n_spikes):
            spike_day = int(self.rng.integers(60, 88))
            spike_height = float(self.rng.uniform(22, 35))
            for offset in range(-2, 5):
                d = spike_day + offset
                if 0 <= d < 90:
                    self.price_truth[d] = round(spike_height, 2)

        logger.debug("Generated 90-day season (seed=%d)", self.seed)

    @staticmethod
    def _get_phase(day: int, phases: list) -> tuple:
        """Return the monsoon phase tuple for a given day."""
        for phase in phases:
            if phase[0] <= day <= phase[1]:
                return phase
        return phases[-1]

    # ──────────────────────────────────────────────────────────
    # Stage Calculation
    # ──────────────────────────────────────────────────────────

    def get_crop_stage(self, day: int) -> str:
        """
        Return crop stage string for a given day.
          seedling:   days  0-15
          vegetative: days 16-40
          flowering:  days 41-60
          fruiting:   days 61-80
          harvest:    days 81-90
        """
        if day <= 15:
            return "seedling"
        if day <= 40:
            return "vegetative"
        if day <= 60:
            return "flowering"
        if day <= 80:
            return "fruiting"
        return "harvest"

    # ──────────────────────────────────────────────────────────
    # Day Simulation
    # ──────────────────────────────────────────────────────────

    def simulate_day(
        self,
        day: int,
        action: str,
        current_state: Dict[str, Any],
        event_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float, float]:
        """
        Apply one day of simulation.

        Args:
            day:            Current day (0-89).
            action:         FarmDecision string.
            current_state:  Mutable state dict (modified in place).
            event_state:    Injected event info from EventEngine.

        Returns:
            (new_state, yield_delta_kg, cost_incurred_inr)
        """
        weather = self.weather_truth[day] if day < len(self.weather_truth) else self.weather_truth[-1]
        price_today = self.price_truth[day] if day < len(self.price_truth) else 12.0
        stage = self.get_crop_stage(day)
        cost = 0.0
        yield_delta = 0.0
        event_state = event_state or {}

        # ── Natural soil moisture dynamics ──────────────────────
        rain_mm = weather["rainfall_mm"]
        # Evaporation
        moisture = current_state["soil_moisture_pct"]
        moisture -= self.DAILY_EVAPORATION_PCT
        # Rain absorption
        if rain_mm > 0:
            absorbed = rain_mm * self.RAIN_ABSORPTION_FACTOR
            moisture += absorbed
        moisture = float(np.clip(moisture, self.MIN_SOIL_MOISTURE, self.MAX_SOIL_MOISTURE))
        current_state["soil_moisture_pct"] = moisture

        # ── Action effects ──────────────────────────────────────
        action_info = {}

        if action == "irrigate":
            moisture += 20.0
            moisture = float(np.clip(moisture, 0, self.MAX_SOIL_MOISTURE))
            current_state["soil_moisture_pct"] = moisture
            cost += self.IRRIGATION_COST
            # Water usage accounting: ~2000L per irrigation event for 2 acres
            current_state["water_used_liters"] = (
                current_state.get("water_used_liters", 0.0) + 2000.0
            )
            if moisture > 75.0:
                current_state["root_rot_risk_days"] = (
                    current_state.get("root_rot_risk_days", 0) + 1
                )
            action_info = {"irrigated": True, "moisture_after": round(moisture, 1)}

        elif action == "fertilize":
            if stage in ("vegetative", "fruiting"):
                current_state["yield_multiplier"] = min(
                    current_state.get("yield_multiplier", 1.0) + 0.05, 2.0
                )
            else:
                # Wrong stage — wasteful
                current_state["yield_multiplier"] = max(
                    current_state.get("yield_multiplier", 1.0) - 0.02, 0.0
                )
            cost += self.FERTILIZER_COST
            action_info = {"fertilized": True, "stage": stage}

        elif action == "spray_pesticide":
            pest_active = event_state.get("pest_active", False)
            is_flowering = stage == "flowering"

            if pest_active and not (is_flowering and not pest_active):
                current_state["pest_suppressed"] = True
                event_state["pest_suppressed"] = True
            if is_flowering and not pest_active:
                # Pollinator kill — yield penalty
                current_state["yield_multiplier"] = max(
                    current_state.get("yield_multiplier", 1.0) - 0.15, 0.0
                )
            cost += self.PESTICIDE_COST
            current_state["chemical_applications"] = (
                current_state.get("chemical_applications", 0) + 1
            )
            action_info = {
                "sprayed": True,
                "pest_active_at_spray": pest_active,
                "pollinator_risk": is_flowering and not pest_active,
            }

        elif action == "sell_now":
            if stage == "harvest":
                harvestable_kg = current_state.get("estimated_yield_kg", 0.0)
                revenue = harvestable_kg * price_today
                current_state["gross_revenue_inr"] = (
                    current_state.get("gross_revenue_inr", 0.0) + revenue
                )
                current_state["bank_balance_inr"] = (
                    current_state.get("bank_balance_inr", 0.0) + revenue
                )
                sell_record = {
                    "day": day,
                    "kg": harvestable_kg,
                    "price_per_kg": price_today,
                    "revenue": revenue,
                }
                current_state.setdefault("sell_decisions", []).append(sell_record)
                current_state["estimated_yield_kg"] = 0.0  # sold
                action_info = sell_record
            else:
                action_info = {"sell_now": False, "reason": f"Not harvest stage (day {day}, stage={stage})"}

        elif action == "apply_scheme":
            scheme_benefit = event_state.get("scheme_benefit_inr", 0)
            scheme_name = event_state.get("active_scheme_name", None)
            if scheme_benefit > 0 and scheme_name:
                already = current_state.get("schemes_captured", [])
                if scheme_name not in already:
                    current_state["bank_balance_inr"] = (
                        current_state.get("bank_balance_inr", 0.0) + scheme_benefit
                    )
                    current_state.setdefault("schemes_captured", []).append(scheme_name)
            action_info = {"scheme_applied": scheme_name, "benefit_inr": scheme_benefit}

        elif action == "take_loan":
            balance = current_state.get("bank_balance_inr", 0.0)
            existing_debt = current_state.get("active_debt_inr", 0.0)
            if existing_debt == 0 and balance >= 2000:
                loan_amount = min(
                    event_state.get("loan_amount_requested", 10000),
                    self.KCC_MAX_LOAN_INR,
                )
                repayable = loan_amount * (1 + self.KCC_INTEREST_ANNUAL * self.KCC_REPAYMENT_DAYS / 365)
                current_state["bank_balance_inr"] = balance + loan_amount
                current_state["active_debt_inr"] = repayable
                action_info = {
                    "loan_approved": True,
                    "amount": loan_amount,
                    "repayable": round(repayable, 2),
                }
            else:
                action_info = {
                    "loan_approved": False,
                    "reason": "Existing loan unpaid" if existing_debt > 0 else "Balance below ₹2,000",
                }

        elif action == "do_nothing":
            action_info = {"do_nothing": True}

        elif action == "hold_crop":
            action_info = {"hold_crop": True, "price_today": price_today}

        # ── Deduct costs ────────────────────────────────────────
        if cost > 0:
            current_state["bank_balance_inr"] = (
                current_state.get("bank_balance_inr", 0.0) - cost
            )
            current_state["total_costs_inr"] = (
                current_state.get("total_costs_inr", 0.0) + cost
            )

        # ── Natural yield accumulation ───────────────────────────
        # Yield builds during fruiting stage
        if stage == "fruiting":
            daily_yield = (self.BASE_YIELD_KG / 20.0) * current_state.get("yield_multiplier", 1.0)
            yield_delta += daily_yield
        elif stage == "harvest":
            daily_yield = 0.0  # already harvested / stable

        # ── Stress damage ────────────────────────────────────────
        if moisture < self.DROUGHT_MOISTURE_THRESHOLD and stage in ("fruiting", "flowering"):
            stress_loss = self.DROUGHT_YIELD_LOSS_KG_PER_DAY
            yield_delta -= stress_loss
            current_state.setdefault("active_alerts", [])
            if "WATER_STRESS: soil moisture critically low" not in current_state["active_alerts"]:
                current_state["active_alerts"].append(
                    "WATER_STRESS: soil moisture critically low"
                )

        pest_active = event_state.get("pest_active", False)
        pest_suppressed = current_state.get("pest_suppressed", False) or event_state.get("pest_suppressed", False)
        if pest_active and not pest_suppressed:
            yield_delta -= self.PEST_YIELD_LOSS_KG_PER_DAY

        if moisture > self.WATERLOG_MOISTURE_THRESHOLD:
            yield_delta -= self.ROOT_ROT_YIELD_LOSS_KG_PER_DAY

        # ── Update running yield estimate ─────────────────────────
        current_yield = current_state.get("estimated_yield_kg", 0.0)
        new_yield = max(0.0, current_yield + yield_delta)
        current_state["estimated_yield_kg"] = round(new_yield, 1)
        current_state["stage"] = stage
        current_state["day"] = day + 1

        # ── Weather summary ────────────────────────────────────────
        current_state["weather_summary"] = weather["summary"]
        current_state["last_action_info"] = action_info

        logger.debug(
            "Day %d | action=%s | cost=%.0f | yield_delta=%.1f | moisture=%.1f",
            day, action, cost, yield_delta, moisture,
        )

        return current_state, yield_delta, cost

    # ──────────────────────────────────────────────────────────
    # Income Calculation
    # ──────────────────────────────────────────────────────────

    def calculate_final_income(
        self,
        yield_kg: float,
        sell_decisions: List[Dict[str, Any]],
        total_costs: float,
        active_debt: float,
        day_90_price: Optional[float] = None,
    ) -> float:
        """
        Compute net income at season end.

        If agent never called sell_now during harvest, force-sell all
        remaining yield at distress price (₹10/kg).

        net_income = gross_revenue - total_costs - active_debt
        """
        gross_revenue = sum(s.get("revenue", 0.0) for s in sell_decisions)

        # Force-sell unsold yield at distress price
        sold_kg = sum(s.get("kg", 0.0) for s in sell_decisions)
        remaining_kg = max(0.0, yield_kg - sold_kg)
        if remaining_kg > 0:
            distress_price = day_90_price or self.DISTRESS_PRICE_PER_KG
            # Distress sale is always the worse of distress cap and market
            effective_price = min(distress_price, self.DISTRESS_PRICE_PER_KG)
            distress_revenue = remaining_kg * effective_price
            gross_revenue += distress_revenue
            logger.info(
                "Force-sell distress: %.0f kg × ₹%.1f = ₹%.0f",
                remaining_kg, effective_price, distress_revenue,
            )

        net_income = gross_revenue - total_costs - active_debt
        return round(net_income, 2)

    def get_optimal_income(self) -> float:
        """
        Theoretical maximum income for this season.
        Assumes a perfect agent: no stress, max price, all schemes.
        Used for income_score normalisation.
        """
        return 40_000.0

    # ──────────────────────────────────────────────────────────
    # Noisy Observation Generation
    # ──────────────────────────────────────────────────────────

    def get_noisy_moisture(self) -> float:
        """Return soil moisture with ±5% sensor noise."""
        noise = float(self.rng.normal(0.0, 5.0))
        noisy = self.soil_moisture_pct + noise
        return round(float(np.clip(noisy, 0.0, 100.0)), 1)

    def get_weather_forecast(
        self, current_day: int, days_ahead: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Return a noisy weather forecast starting from current_day.
        Noise increases with forecast horizon.
        """
        forecast = []
        for offset in range(1, days_ahead + 1):
            d = current_day + offset
            if d >= len(self.weather_truth):
                break
            truth = self.weather_truth[d]
            noise_factor = offset * 0.10   # ±10% per day ahead
            confidence = round(1.0 - offset * 0.12, 2)

            noisy_rain_prob = float(np.clip(
                truth["rain_prob"] + self.rng.normal(0, noise_factor), 0.0, 1.0
            ))
            noisy_temp = float(truth["temp_c"] + self.rng.normal(0, offset))
            noisy_rain_mm = max(0.0, float(
                truth["rainfall_mm"] + self.rng.normal(0, noise_factor * truth.get("rainfall_mm", 5))
            ))

            forecast.append({
                "day": d,
                "rain_prob": round(noisy_rain_prob, 2),
                "temp_c": round(noisy_temp, 1),
                "rainfall_mm": round(noisy_rain_mm, 1),
                "confidence": confidence,
            })
        return forecast

    def get_state(self) -> Dict[str, Any]:
        """Return simulator internals for /state endpoint."""
        return {
            "day": self.day,
            "seed": self.seed,
            "soil_moisture_pct": self.soil_moisture_pct,
            "yield_multiplier": self.yield_multiplier,
            "root_rot_risk_days": self.root_rot_risk_days,
            "pest_active": self.pest_active,
            "pest_risk_level": self.pest_risk_level,
            "water_used_liters": self.water_used_liters,
            "chemical_applications": self.chemical_applications,
            "weather_today": self.weather_truth[self.day] if self.day < 90 else None,
        }
