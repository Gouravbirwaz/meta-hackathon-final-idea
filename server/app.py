"""
KisanAgent — OpenEnv Framework Server
=====================================
Production-grade RL environment using official OpenEnv architecture.

IMPORTANT ARCHITECTURE NOTE:
The OpenEnv HTTP server calls self._env_factory() to create a BRAND NEW
environment instance for every single /reset and /step request, then destroys
it after. This means instance-level state (self._season_active, etc.) cannot
survive between requests.

FIX: All mutable season state is stored in a module-level singleton dict
(_SHARED_STATE). Every KisanEnvironment instance reads/writes this shared
state, making the HTTP endpoint stateless-friendly.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server import Environment, create_web_interface_app

from env.grader import KisanGrader
from env.models import (
    CropStage,
    FarmerObservation,
    KisanAction,
    KisanState,
)
from simulator.event_engine import EventEngine
from simulator.farm_simulator import FarmSimulator
from tools.credit_tool import CreditTool
from tools.govt_scheme_tool import GovtSchemeTool
from tools.mandi_price_tool import MandiPriceTool
from tools.pest_alert_tool import PestAlertTool
from tools.soil_tool import SoilTool
from tools.weather_tool import WeatherTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("kisanagent.server")

# ── Module-level singleton state ─────────────────────────────────────────────
# Because OpenEnv HTTP server creates a new env instance per request, all
# persistent state must live here at the module level.
_SHARED_STATE: Dict[str, Any] = {
    "season_active": False,
    "season_state": {},
    "simulator": FarmSimulator(seed=42),
    "event_engine": EventEngine(),
    "grader": KisanGrader(),
    "tools": {},
}


def _init_tools(simulator: FarmSimulator, event_engine: EventEngine) -> Dict:
    farmer_profile = {
        "name": "Harish",
        "district": "Kolar",
        "state": "Karnataka",
        "land_acres": 2.0,
        "crop": "tomato",
    }
    return {
        "weather": WeatherTool(
            weather_sequence=simulator.weather_truth,
            failure_days=event_engine.tool_failure_days.get("weather", []),
            rng=simulator.rng,
        ),
        "soil": SoilTool(
            simulator_ref=simulator,
            failure_days=event_engine.tool_failure_days.get("soil", []),
            rng=simulator.rng,
        ),
        "mandi_price": MandiPriceTool(
            price_sequence=simulator.price_truth,
            rng=simulator.rng,
        ),
        "govt_scheme": GovtSchemeTool(
            captured_schemes_ref=event_engine.captured_schemes,
            rng=simulator.rng,
        ),
        "pest_alert": PestAlertTool(
            event_engine_ref=event_engine,
            rng=simulator.rng,
        ),
        "credit": CreditTool(
            farmer_profile=farmer_profile,
            simulator_ref=simulator,
            rng=simulator.rng,
        ),
    }


class KisanEnvironment(Environment[KisanAction, FarmerObservation, KisanState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__()
        # Each instance uses the shared module-level singletons
        self._shared = _SHARED_STATE

    @property
    def _simulator(self) -> FarmSimulator:
        return self._shared["simulator"]

    @property
    def _event_engine(self) -> EventEngine:
        return self._shared["event_engine"]

    @property
    def _grader(self) -> KisanGrader:
        return self._shared["grader"]

    @property
    def _season_state(self) -> Dict:
        return self._shared["season_state"]

    @property
    def _season_active(self) -> bool:
        return self._shared["season_active"]

    @property
    def _tools(self) -> Dict:
        return self._shared["tools"]

    def _build_observation(self) -> FarmerObservation:
        s = self._season_state
        day = s.get("day", 0)
        stage_str = self._simulator.get_crop_stage(day)
        stage = CropStage(stage_str)
        used_today = s.get("tool_calls_today", 0)
        noisy_moisture = self._simulator.get_noisy_moisture()

        obs = FarmerObservation(
            day=day,
            crop_stage=stage,
            soil_moisture_pct=noisy_moisture,
            estimated_yield_kg=round(s.get("estimated_yield_kg", 0.0), 1),
            bank_balance_inr=round(s.get("bank_balance_inr", 15000.0), 2),
            last_tool_result=s.get("last_tool_result"),
            active_alerts=self._event_engine.get_active_alerts(day),
            days_to_harvest=max(0, 89 - day),
            weather_summary=s.get("weather_summary", "dry"),
            tool_calls_used_today=used_today,
            tool_calls_remaining=max(0, 3 - used_today),
        )
        return obs

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> FarmerObservation:
        seed = seed if seed is not None else int(time.time()) % 100_000
        difficulty = kwargs.get("difficulty", "medium")

        sim = self._shared["simulator"]
        ee = self._shared["event_engine"]
        grader = self._shared["grader"]

        sim.reset(seed=seed)
        ee.reset(difficulty=difficulty, seed=seed)
        grader.reset()

        season_id = episode_id or str(uuid.uuid4())

        self._shared["season_state"] = {
            "season_id": season_id,
            "day": 0,
            "difficulty": difficulty,
            "seed": seed,
            "bank_balance_inr": FarmSimulator.INITIAL_BALANCE_INR,
            "total_costs_inr": 0.0,
            "gross_revenue_inr": 0.0,
            "active_debt_inr": 0.0,
            "estimated_yield_kg": 0.0,
            "soil_moisture_pct": 65.0,
            "yield_multiplier": 1.0,
            "root_rot_risk_days": 0,
            "chemical_applications": 0,
            "water_used_liters": 0.0,
            "sell_decisions": [],
            "schemes_captured": [],
            "pest_suppressed": False,
            "tool_calls_today": 0,
            "last_tool_result": None,
            "weather_summary": "dry",
            "active_alerts": [],
        }
        self._shared["season_active"] = True

        sim.soil_moisture_pct = 65.0
        sim.bank_balance_inr = FarmSimulator.INITIAL_BALANCE_INR
        sim.active_debt_inr = 0.0

        self._shared["tools"] = _init_tools(sim, ee)

        s = self._shared["season_state"]
        if sim.weather_truth:
            s["weather_summary"] = sim.weather_truth[0]["summary"]

        obs = self._build_observation()

        schemes_available = [sc["name"] for sc in ee.scheme_schedule]
        total_scheme_value = sum(sc["benefit_inr"] for sc in ee.scheme_schedule)

        obs.metadata = {
            "season_id": season_id,
            "difficulty": difficulty,
            "seed": seed,
            "optimal_income_inr": sim.get_optimal_income(),
            "baseline_income_inr": KisanGrader.BASELINE_INCOME,
            "schemes_available": schemes_available,
            "total_scheme_value_inr": total_scheme_value,
            "pest_events_scheduled": len(ee.pest_schedule),
            "tool_failure_days_total": sum(len(v) for v in ee.tool_failure_days.values()),
            "message": "New season started. 90 days to harvest.",
        }

        logger.info("Season reset: difficulty=%s seed=%d season_id=%s", difficulty, seed, season_id)
        return obs

    def step(self, action: KisanAction, timeout_s: Optional[float] = None, **kwargs: Any) -> FarmerObservation:
        if not self._shared["season_active"]:
            raise ValueError("No active season. Call /reset first.")

        if action.tool_name:
            tool_name = action.tool_name.value
            raw_args = action.tool_args
            args: Dict[str, Any] = {}
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str):
                import json
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    pass
            return self._call_tool(tool_name, args)

        if action.farm_decision:
            return self._apply_farm_decision(action.farm_decision.value, action.reasoning)

        raise ValueError("Either tool_name or farm_decision must be provided in KisanAction.")

    def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> FarmerObservation:
        s = self._shared["season_state"]
        calls_used = s.get("tool_calls_today", 0)
        if calls_used >= 3:
            obs = self._build_observation()
            obs.metadata["error"] = "Tool budget exceeded (3/3 calls used today)."
            return obs

        s["tool_calls_today"] = calls_used + 1
        day = s.get("day", 0)

        ee = self._shared["event_engine"]
        is_failing = ee.is_tool_failing_today(tool_name, day)
        tool = self._shared["tools"].get(tool_name)

        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not initialised.")

        start_time = time.perf_counter()
        try:
            if tool_name == "weather":
                result = tool.call(current_day=day, days_ahead=args.get("days_ahead", 3))
                if is_failing:
                    result["data_quality"] = "unavailable"
                    result["forecast"] = []
                    result["advisory"] = "IMD API offline."
            elif tool_name == "soil":
                result = tool.call(farm_id=args.get("farm_id", "farm_001"), current_day=day)
                if is_failing:
                    result = {"data_quality": "unavailable", "sensor_status": "offline"}
            elif tool_name == "mandi_price":
                result = tool.call(crop=args.get("crop", "tomato"), market=args.get("market", "KR Puram Bangalore"), current_day=day)
            elif tool_name == "govt_scheme":
                result = tool.call(state=args.get("state", "Karnataka"), crop=args.get("crop", "tomato"), current_day=day)
            elif tool_name == "pest_alert":
                result = tool.call(region=args.get("region", "Kolar"), crop=args.get("crop", "tomato"), current_day=day)
            elif tool_name == "credit":
                result = tool.call(amount_inr=int(args.get("amount_inr", 10_000)), purpose=args.get("purpose", "farming_input"))
            else:
                result = {}
        except Exception as exc:
            result = {"error": str(exc)}

        latency_ms = round((time.perf_counter() - start_time) * 1000 + 100.0, 1)
        s["last_tool_result"] = {"tool": tool_name, "latency_ms": latency_ms, **result}

        obs = self._build_observation()
        obs.metadata["tool_called"] = tool_name
        return obs

    def _apply_farm_decision(self, farm_decision: str, reasoning: Optional[str]) -> FarmerObservation:
        s = self._shared["season_state"]
        day = s["day"]
        ee = self._shared["event_engine"]
        sim = self._shared["simulator"]
        grader = self._shared["grader"]

        event_state = ee.get_event_state(day)

        if farm_decision == "spray_pesticide":
            ee.record_treatment(day)
            s["pest_suppressed"] = True
            event_state["pest_suppressed"] = True

        if farm_decision == "apply_scheme":
            active_scheme = event_state.get("active_scheme_name")
            if active_scheme:
                ee.record_scheme_capture(active_scheme)
                if active_scheme not in s.get("schemes_captured", []):
                    s.setdefault("schemes_captured", []).append(active_scheme)

        new_state, yield_delta, cost_incurred = sim.simulate_day(
            day=day,
            action=farm_decision,
            current_state=s,
            event_state=event_state,
        )
        self._shared["season_state"] = new_state
        s = new_state

        sim.soil_moisture_pct = s["soil_moisture_pct"]
        sim.bank_balance_inr = s.get("bank_balance_inr", 15000.0)
        sim.active_debt_inr = s.get("active_debt_inr", 0.0)

        step_state_for_grader = {
            "pest_risk_level": event_state.get("pest_risk_level", "LOW"),
            "soil_moisture_pct": s["soil_moisture_pct"],
            "day": day,
        }
        step_scores = grader.compute_step_scores(farm_decision, step_state_for_grader, [])

        new_day = day + 1
        s["day"] = new_day
        s["tool_calls_today"] = 0
        s["pest_suppressed"] = False

        terminated = new_day >= 90
        final_scores = None
        net_income_inr = None

        if terminated:
            self._shared["season_active"] = False
            final_yield = s.get("estimated_yield_kg", 0.0)
            sell_decisions = s.get("sell_decisions", [])
            day90_price = sim.price_truth[89] if len(sim.price_truth) > 89 else 10.0

            net_income_inr = sim.calculate_final_income(
                yield_kg=final_yield,
                sell_decisions=sell_decisions,
                total_costs=s.get("total_costs_inr", 0.0),
                active_debt=s.get("active_debt_inr", 0.0),
                day_90_price=day90_price,
            )

            episode_summary = ee.get_episode_summary()
            final_state = {
                "net_income_inr": net_income_inr,
                "pest_events": episode_summary["pest_events"],
                "schemes_available": episode_summary["schemes_available"],
                "schemes_captured": episode_summary["schemes_captured"],
                "water_used_liters": s.get("water_used_liters", 0.0),
                "chemical_applications": s.get("chemical_applications", 0),
            }

            final_scores = grader.compute_final_scores(final_state, grader.episode_log)
            reward = float(final_scores.composite_score)
        else:
            reward = (float(step_scores.composite_score) - 0.5) * 0.2

        obs = self._build_observation()
        obs.done = terminated
        obs.reward = round(reward, 4)
        obs.metadata = {
            "day_completed": day,
            "action_applied": farm_decision,
            "cost_incurred_inr": round(cost_incurred, 2),
            "yield_delta_kg": round(yield_delta, 1),
            "pest_risk_level": event_state.get("pest_risk_level", "LOW"),
            "net_income_inr": net_income_inr,
        }

        if final_scores:
            obs.metadata["final_scores"] = final_scores.model_dump()

        return obs

    @property
    def state(self) -> KisanState:
        s = self._shared["season_state"]
        return KisanState(
            episode_id=s.get("season_id", "N/A"),
            step_count=s.get("day", 0),
            season_state=dict(s),
            event_schedule=self._event_engine.get_state() if self._event_engine else {},
            simulator_state=self._simulator.get_state() if self._simulator else {},
            grader_state=self._grader.get_state() if self._grader else {},
        )


import os
os.environ["ENABLE_WEB_INTERFACE"] = "true"

app = create_web_interface_app(KisanEnvironment, KisanAction, FarmerObservation)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)
