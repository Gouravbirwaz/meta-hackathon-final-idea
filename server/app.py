"""
KisanAgent — OpenEnv + MCP Server
=================================
Production-grade RL environment + MCP tool interface.

FIXES APPLIED:
1. MCP tools route through env._call_tool() → tool budget enforced, grader logs correctly
2. Session isolation via _SHARED_SESSIONS dict keyed by session_id
3. step_env has explicit farm_decision param (not opaque dict)
4. observe() tool added — get current state without spending a tool call
5. data_quality degraded flag only set when tool failure is active
6. notifications/initialized handshake supported
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from fastmcp import FastMCP
from openenv.core.env_server import Environment, create_web_interface_app

from env.grader import KisanGrader
from env.models import CropStage, FarmerObservation, KisanAction, KisanState
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


# =====================================================
# SESSION STORE (replaces single global env instance)
# Keyed by session_id so eval + training don't collide
# =====================================================

_SHARED_SESSIONS: Dict[str, Dict[str, Any]] = {}
_DEFAULT_SESSION = "default"


def _make_session(seed: int, difficulty: str = "medium") -> Dict[str, Any]:
    simulator = FarmSimulator(seed=seed)
    event_engine = EventEngine()
    grader = KisanGrader()

    simulator.reset(seed=seed)
    event_engine.reset(difficulty=difficulty, seed=seed)
    grader.reset()

    farmer_profile = {
        "name": "Harish",
        "district": "Kolar",
        "state": "Karnataka",
        "land_acres": 2.0,
        "crop": "tomato",
    }

    tools = {
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

    season_state = {
        "season_id": str(uuid.uuid4()),
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
        "weather_summary": simulator.weather_truth[0]["summary"] if simulator.weather_truth else "dry",
        "active_alerts": [],
    }

    simulator.soil_moisture_pct = 65.0
    simulator.bank_balance_inr = FarmSimulator.INITIAL_BALANCE_INR
    simulator.active_debt_inr = 0.0

    return {
        "season_active": True,
        "season_state": season_state,
        "simulator": simulator,
        "event_engine": event_engine,
        "grader": grader,
        "tools": tools,
    }


def _get_session(session_id: str = _DEFAULT_SESSION) -> Dict[str, Any]:
    if session_id not in _SHARED_SESSIONS:
        raise ValueError(f"No active session '{session_id}'. Call reset_env first.")
    return _SHARED_SESSIONS[session_id]


# =====================================================
# OPENENV ENVIRONMENT (uses session store)
# =====================================================

class KisanEnvironment(Environment[KisanAction, FarmerObservation, KisanState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, session_id: str = _DEFAULT_SESSION):
        super().__init__()
        self._session_id = session_id

    @property
    def _sess(self) -> Dict[str, Any]:
        return _get_session(self._session_id)

    def _build_observation(self, session_id: str = None) -> FarmerObservation:
        sess = _get_session(session_id or self._session_id)
        s = sess["season_state"]
        sim = sess["simulator"]
        ee = sess["event_engine"]

        day = s.get("day", 0)
        stage = CropStage(sim.get_crop_stage(day))
        used_today = s.get("tool_calls_today", 0)
        noisy_moisture = sim.get_noisy_moisture()

        obs = FarmerObservation(
            day=day,
            crop_stage=stage,
            soil_moisture_pct=noisy_moisture,
            estimated_yield_kg=round(s.get("estimated_yield_kg", 0.0), 1),
            bank_balance_inr=round(s.get("bank_balance_inr", 15000.0), 2),
            last_tool_result=s.get("last_tool_result"),
            active_alerts=ee.get_active_alerts(day),
            days_to_harvest=max(0, 89 - day),
            weather_summary=s.get("weather_summary", "dry"),
            tool_calls_used_today=used_today,
            tool_calls_remaining=max(0, 3 - used_today),
        )
        obs.metadata = {}
        return obs

    def _call_tool(self, tool_name: str, args: Dict[str, Any], session_id: str = None) -> FarmerObservation:
        sid = session_id or self._session_id
        sess = _get_session(sid)
        s = sess["season_state"]
        sim = sess["simulator"]
        ee = sess["event_engine"]

        calls_used = s.get("tool_calls_today", 0)
        if calls_used >= 3:
            obs = self._build_observation(sid)
            obs.metadata["error"] = "Tool budget exceeded (3/3 calls used today)."
            return obs

        s["tool_calls_today"] = calls_used + 1
        day = s.get("day", 0)
        is_failing = ee.is_tool_failing_today(tool_name, day)
        tool = sess["tools"].get(tool_name)

        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not initialised.")

        start = time.perf_counter()
        try:
            if tool_name == "weather":
                result = tool.call(current_day=day, days_ahead=args.get("days_ahead", 3))
                if is_failing:
                    result["data_quality"] = "unavailable"
                    result["forecast"] = []
                    result["advisory"] = "IMD API offline."
                # Only set degraded if the tool itself returned degraded — don't override good data
                elif result.get("data_quality") not in ("good", "degraded", "unavailable"):
                    result["data_quality"] = "good"
            elif tool_name == "soil":
                result = tool.call(farm_id=args.get("farm_id", "farm_001"), current_day=day)
                if is_failing:
                    result = {"data_quality": "unavailable", "sensor_status": "offline"}
            elif tool_name == "mandi_price":
                result = tool.call(
                    crop=args.get("crop", "tomato"),
                    market=args.get("market", "KR Puram Bangalore"),
                    current_day=day,
                )
            elif tool_name == "govt_scheme":
                result = tool.call(
                    state=args.get("state", "Karnataka"),
                    crop=args.get("crop", "tomato"),
                    current_day=day,
                )
            elif tool_name == "pest_alert":
                result = tool.call(
                    region=args.get("region", "Kolar"),
                    crop=args.get("crop", "tomato"),
                    current_day=day,
                )
            elif tool_name == "credit":
                result = tool.call(
                    amount_inr=int(args.get("amount_inr", 10_000)),
                    purpose=args.get("purpose", "farming_input"),
                )
            else:
                result = {}
        except Exception as exc:
            result = {"error": str(exc)}

        latency_ms = round((time.perf_counter() - start) * 1000 + 100.0, 1)
        result["latency_ms"] = latency_ms
        s["last_tool_result"] = {"tool": tool_name, **result}

        obs = self._build_observation(sid)
        obs.metadata["tool_called"] = tool_name
        obs.metadata["tool_result"] = result
        return obs

    def _apply_farm_decision(self, farm_decision: str, reasoning: str = "", session_id: str = None) -> FarmerObservation:
        sid = session_id or self._session_id
        sess = _get_session(sid)
        s = sess["season_state"]
        sim = sess["simulator"]
        ee = sess["event_engine"]
        grader = sess["grader"]

        day = s["day"]
        event_state = ee.get_event_state(day)

        if farm_decision == "spray_pesticide":
            ee.record_treatment(day)
            s["pest_suppressed"] = True
            event_state["pest_suppressed"] = True

        if farm_decision == "apply_scheme":
            active_scheme = event_state.get("active_scheme_name")
            if active_scheme and active_scheme not in s.get("schemes_captured", []):
                ee.record_scheme_capture(active_scheme)
                s.setdefault("schemes_captured", []).append(active_scheme)

        new_state, yield_delta, cost_incurred = sim.simulate_day(
            day=day,
            action=farm_decision,
            current_state=s,
            event_state=event_state,
        )
        sess["season_state"] = new_state
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
            sess["season_active"] = False
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

        obs = self._build_observation(sid)
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

    def reset(self, seed=None, episode_id=None, **kwargs):
        seed = seed or int(time.time()) % 100_000
        difficulty = kwargs.get("difficulty", "medium")
        session_id = episode_id or self._session_id

        _SHARED_SESSIONS[session_id] = _make_session(seed=seed, difficulty=difficulty)
        logger.info("Season reset: session=%s difficulty=%s seed=%d", session_id, difficulty, seed)

        obs = self._build_observation(session_id)
        sess = _get_session(session_id)
        sim = sess["simulator"]
        ee = sess["event_engine"]

        obs.metadata = {
            "season_id": sess["season_state"]["season_id"],
            "difficulty": difficulty,
            "seed": seed,
            "optimal_income_inr": sim.get_optimal_income(),
            "baseline_income_inr": KisanGrader.BASELINE_INCOME,
            "schemes_available": [sc["name"] for sc in ee.scheme_schedule],
            "total_scheme_value_inr": sum(sc["benefit_inr"] for sc in ee.scheme_schedule),
            "pest_events_scheduled": len(ee.pest_schedule),
            "tool_failure_days_total": sum(len(v) for v in ee.tool_failure_days.values()),
            "message": "New season started. 90 days to harvest.",
        }
        return obs

    def step(self, action: KisanAction, timeout_s=None, **kwargs):
        if not _SHARED_SESSIONS.get(self._session_id, {}).get("season_active", False):
            raise ValueError("No active season. Call reset_env first.")

        if action.tool_name:
            tool_name = action.tool_name.value
            args = action.tool_args or {}
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            return self._call_tool(tool_name, args)

        if action.farm_decision:
            return self._apply_farm_decision(
                action.farm_decision.value,
                reasoning=action.reasoning or "",
            )

        raise ValueError("Either tool_name or farm_decision must be set in KisanAction.")

    @property
    def state(self) -> KisanState:
        sess = _get_session(self._session_id)
        s = sess["season_state"]
        return KisanState(
            episode_id=s.get("season_id", "N/A"),
            step_count=s.get("day", 0),
            season_state=dict(s),
            event_schedule=sess["event_engine"].get_state(),
            simulator_state=sess["simulator"].get_state(),
            grader_state=sess["grader"].get_state(),
        )


# =====================================================
# OPENENV HTTP APP
# =====================================================

os.environ["ENABLE_WEB_INTERFACE"] = "true"
app = create_web_interface_app(KisanEnvironment, KisanAction, FarmerObservation)

# Seed a default session so MCP tools work immediately without calling reset_env
_SHARED_SESSIONS[_DEFAULT_SESSION] = _make_session(seed=42, difficulty="medium")


# =====================================================
# MCP SERVER
# =====================================================

mcp = FastMCP("KisanAgent")

VALID_FARM_DECISIONS = [
    "irrigate",
    "fertilize",
    "spray_pesticide",
    "sell_now",
    "hold_crop",
    "apply_scheme",
    "take_loan",
    "do_nothing",
]


# ── Observation ────────────────────────────────────────────────────────────────

@mcp.tool()
def observe(session_id: str = _DEFAULT_SESSION) -> dict:
    """
    Get current farm state without spending a tool call budget.
    Returns day, crop_stage, soil_moisture, balance, yield estimate, alerts, and tool budget.
    """
    sess = _get_session(session_id)
    s = sess["season_state"]
    sim = sess["simulator"]
    ee = sess["event_engine"]
    day = s["day"]
    return {
        "day": day,
        "days_to_harvest": max(0, 89 - day),
        "crop_stage": sim.get_crop_stage(day),
        "soil_moisture_pct": round(sim.get_noisy_moisture(), 1),
        "estimated_yield_kg": round(s.get("estimated_yield_kg", 0.0), 1),
        "bank_balance_inr": round(s.get("bank_balance_inr", 15000.0), 2),
        "active_debt_inr": round(s.get("active_debt_inr", 0.0), 2),
        "tool_calls_used_today": s.get("tool_calls_today", 0),
        "tool_calls_remaining": max(0, 3 - s.get("tool_calls_today", 0)),
        "active_alerts": ee.get_active_alerts(day),
        "weather_summary": s.get("weather_summary", "dry"),
        "season_active": sess["season_active"],
    }


# ── Farm tool calls (routed through env._call_tool for budget + grader) ────────

@mcp.tool()
def weather(days_ahead: int, session_id: str = _DEFAULT_SESSION) -> dict:
    """Weather forecast for next N days (costs 1 tool call from today's budget of 3)."""
    env = KisanEnvironment(session_id)
    obs = env._call_tool("weather", {"days_ahead": days_ahead}, session_id)
    return {"tool": "weather", "data": obs.metadata.get("tool_result", {}), "observation": _obs_summary(obs)}


@mcp.tool()
def soil(session_id: str = _DEFAULT_SESSION) -> dict:
    """Soil moisture sensor data (costs 1 tool call)."""
    env = KisanEnvironment(session_id)
    obs = env._call_tool("soil", {}, session_id)
    return {"tool": "soil", "data": obs.metadata.get("tool_result", {}), "observation": _obs_summary(obs)}


@mcp.tool()
def mandi_price(session_id: str = _DEFAULT_SESSION) -> dict:
    """Tomato mandi price at KR Puram Bangalore (costs 1 tool call)."""
    env = KisanEnvironment(session_id)
    obs = env._call_tool("mandi_price", {}, session_id)
    return {"tool": "mandi_price", "data": obs.metadata.get("tool_result", {}), "observation": _obs_summary(obs)}


@mcp.tool()
def govt_scheme(session_id: str = _DEFAULT_SESSION) -> dict:
    """Active Karnataka government schemes for tomato farmers (costs 1 tool call)."""
    env = KisanEnvironment(session_id)
    obs = env._call_tool("govt_scheme", {}, session_id)
    return {"tool": "govt_scheme", "data": obs.metadata.get("tool_result", {}), "observation": _obs_summary(obs)}


@mcp.tool()
def pest_alert(session_id: str = _DEFAULT_SESSION) -> dict:
    """Pest surveillance alerts for Kolar district (costs 1 tool call)."""
    env = KisanEnvironment(session_id)
    obs = env._call_tool("pest_alert", {}, session_id)
    return {"tool": "pest_alert", "data": obs.metadata.get("tool_result", {}), "observation": _obs_summary(obs)}


@mcp.tool()
def credit(amount_inr: int, session_id: str = _DEFAULT_SESSION) -> dict:
    """Request a KCC/NABARD microfinance loan (costs 1 tool call)."""
    env = KisanEnvironment(session_id)
    obs = env._call_tool("credit", {"amount_inr": amount_inr}, session_id)
    return {"tool": "credit", "data": obs.metadata.get("tool_result", {}), "observation": _obs_summary(obs)}


# ── Control tools ──────────────────────────────────────────────────────────────

@mcp.tool()
def reset_env(
    difficulty: str = "medium",
    seed: int = 0,
    session_id: str = _DEFAULT_SESSION,
) -> dict:
    """
    Reset the farming season and start day 0.
    difficulty: 'easy' | 'medium' | 'hard'
    seed: 0 = random seed
    """
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError("difficulty must be 'easy', 'medium', or 'hard'")
    actual_seed = seed or int(time.time()) % 100_000
    _SHARED_SESSIONS[session_id] = _make_session(seed=actual_seed, difficulty=difficulty)
    sess = _SHARED_SESSIONS[session_id]
    s = sess["season_state"]
    sim = sess["simulator"]
    ee = sess["event_engine"]
    logger.info("reset_env: session=%s difficulty=%s seed=%d", session_id, difficulty, actual_seed)
    return {
        "status": "reset",
        "session_id": session_id,
        "difficulty": difficulty,
        "seed": actual_seed,
        "season_id": s["season_id"],
        "starting_balance_inr": s["bank_balance_inr"],
        "optimal_income_inr": sim.get_optimal_income(),
        "baseline_income_inr": KisanGrader.BASELINE_INCOME,
        "schemes_available": [sc["name"] for sc in ee.scheme_schedule],
        "pest_events_scheduled": len(ee.pest_schedule),
        "message": "Season started. 90 days to harvest. Use observe() to check state.",
    }


@mcp.tool()
def step_env(
    farm_decision: str,
    reasoning: str = "",
    session_id: str = _DEFAULT_SESSION,
) -> dict:
    """
    Apply a daily farm decision and advance to the next day.

    farm_decision must be one of:
      irrigate         — Water the crop (₹200). Raises soil moisture +20%.
      fertilize        — Apply NPK fertilizer (₹600). Boosts yield in vegetative/fruiting stage.
      spray_pesticide  — Apply pesticide (₹800). Stops pest damage. Penalised in flowering stage.
      sell_now         — Sell at today's mandi price (harvest stage only).
      hold_crop        — Wait for better mandi price (harvest stage).
      apply_scheme     — Claim active government subsidy (if available today).
      take_loan        — Draw KCC/NABARD loan up to ₹25,000 at 7% p.a.
      do_nothing       — No action. Natural simulation step.

    reasoning: optional explanation of why you chose this action (used for grader logging).
    """
    if farm_decision not in VALID_FARM_DECISIONS:
        raise ValueError(
            f"Invalid farm_decision '{farm_decision}'. "
            f"Must be one of: {', '.join(VALID_FARM_DECISIONS)}"
        )

    sess = _get_session(session_id)
    if not sess.get("season_active", False):
        raise ValueError("No active season. Call reset_env first.")

    env = KisanEnvironment(session_id)
    obs = env._apply_farm_decision(farm_decision, reasoning=reasoning, session_id=session_id)

    result = {
        "status": "stepped",
        "action_applied": farm_decision,
        "day_completed": obs.metadata.get("day_completed"),
        "cost_incurred_inr": obs.metadata.get("cost_incurred_inr"),
        "yield_delta_kg": obs.metadata.get("yield_delta_kg"),
        "pest_risk_level": obs.metadata.get("pest_risk_level"),
        "reward": obs.reward,
        "done": obs.done,
        "observation": _obs_summary(obs),
    }
    if obs.done:
        result["net_income_inr"] = obs.metadata.get("net_income_inr")
        result["final_scores"] = obs.metadata.get("final_scores")
        result["message"] = "Season complete! Check final_scores for performance breakdown."

    return result


# ── Helper ─────────────────────────────────────────────────────────────────────

def _obs_summary(obs: FarmerObservation) -> dict:
    """Compact observation dict to include alongside tool results."""
    return {
        "day": obs.day,
        "crop_stage": obs.crop_stage.value if hasattr(obs.crop_stage, "value") else str(obs.crop_stage),
        "soil_moisture_pct": obs.soil_moisture_pct,
        "estimated_yield_kg": obs.estimated_yield_kg,
        "bank_balance_inr": obs.bank_balance_inr,
        "tool_calls_remaining": obs.tool_calls_remaining,
        "days_to_harvest": obs.days_to_harvest,
        "active_alerts": obs.active_alerts,
    }


# =====================================================
# ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=7860)