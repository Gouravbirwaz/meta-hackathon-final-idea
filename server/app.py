"""
KisanAgent — FastAPI OpenEnv Server
=====================================
Production-grade RL environment server.
OpenEnv-compatible: /reset /step /health /state /tools/{tool_name}

Endpoints:
  POST /reset              → ResetResponse
  POST /step               → StepResult
  GET  /health             → HealthResponse
  GET  /state              → StateResponse
  POST /tools/{tool_name}  → ToolResponse  (max 3/step enforced)
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware

from env.grader import KisanGrader
from env.models import (
    CropStage,
    FarmerObservation,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResult,
    ToolRequest,
    ToolResponse,
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

# ── Global State ───────────────────────────────────────────────────────────────
# Mirrors Round 1 pattern — single-season global state
_simulator: FarmSimulator = None
_event_engine: EventEngine = None
_grader: KisanGrader = None
_season_state: Dict[str, Any] = {}
_tools: Dict[str, Any] = {}
_season_active: bool = False


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise singletons on startup."""
    global _simulator, _event_engine, _grader
    _simulator = FarmSimulator(seed=42)
    _event_engine = EventEngine()
    _grader = KisanGrader()
    logger.info("KisanAgent environment initialised and ready.")
    yield
    logger.info("KisanAgent environment shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="KisanAgent — Farm Advisory RL Environment",
    description=(
        "OpenEnv-compatible reinforcement learning environment for training "
        "LLM agents to advise Indian smallholder farmers across a 90-day "
        "tomato season in Kolar district, Karnataka.\n\n"
        "**Theme 3: World Modeling** — Meta-PyTorch OpenEnv Hackathon Finale 2026"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_season() -> None:
    """Raise 400 if no season is active."""
    if not _season_active:
        raise HTTPException(
            status_code=400,
            detail="No active season. Call POST /reset first.",
        )


def _build_observation() -> FarmerObservation:
    """Construct a FarmerObservation from current global state."""
    s = _season_state
    day = s.get("day", 0)
    stage_str = _simulator.get_crop_stage(day)
    stage = CropStage(stage_str)
    used_today = s.get("tool_calls_today", 0)

    # Build noisy moisture reading
    noisy_moisture = _simulator.get_noisy_moisture()

    return FarmerObservation(
        day=day,
        crop_stage=stage,
        soil_moisture_pct=noisy_moisture,
        estimated_yield_kg=round(s.get("estimated_yield_kg", 0.0), 1),
        bank_balance_inr=round(s.get("bank_balance_inr", 15000.0), 2),
        last_tool_result=s.get("last_tool_result"),
        active_alerts=_event_engine.get_active_alerts(day),
        days_to_harvest=max(0, 89 - day),
        weather_summary=s.get("weather_summary", "dry"),
        tool_calls_used_today=used_today,
        tool_calls_remaining=max(0, 3 - used_today),
    )


def _init_tools() -> None:
    """Instantiate tool objects bound to current simulator/event-engine."""
    global _tools
    farmer_profile = {
        "name": "Harish",
        "district": "Kolar",
        "state": "Karnataka",
        "land_acres": 2.0,
        "crop": "tomato",
    }
    _tools = {
        "weather": WeatherTool(
            weather_sequence=_simulator.weather_truth,
            failure_days=_event_engine.tool_failure_days.get("weather", []),
            rng=_simulator.rng,
        ),
        "soil": SoilTool(
            simulator_ref=_simulator,
            failure_days=_event_engine.tool_failure_days.get("soil", []),
            rng=_simulator.rng,
        ),
        "mandi_price": MandiPriceTool(
            price_sequence=_simulator.price_truth,
            rng=_simulator.rng,
        ),
        "govt_scheme": GovtSchemeTool(
            captured_schemes_ref=_event_engine.captured_schemes,
            rng=_simulator.rng,
        ),
        "pest_alert": PestAlertTool(
            event_engine_ref=_event_engine,
            rng=_simulator.rng,
        ),
        "credit": CreditTool(
            farmer_profile=farmer_profile,
            simulator_ref=_simulator,
            rng=_simulator.rng,
        ),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.post("/reset", response_model=ResetResponse, tags=["Environment"])
async def reset(request: ResetRequest) -> ResetResponse:
    """
    Start a new 90-day farming season.

    Initialises:
    - day = 0, crop_stage = seedling
    - soil_moisture = 65%, bank_balance = ₹15,000
    - yield_kg = 0, yield_multiplier = 1.0
    - Schedules pest/scheme/tool-failure events

    Returns initial FarmerObservation + season metadata.
    """
    global _simulator, _event_engine, _grader, _season_state, _season_active

    seed = request.seed if request.seed is not None else int(time.time()) % 100_000
    difficulty = request.difficulty.value

    # Reset all components
    _simulator.reset(seed=seed)
    _event_engine.reset(difficulty=difficulty, seed=seed)
    _grader.reset()

    # Initialise mutable season state
    _season_state = {
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

    season_id = str(uuid.uuid4())
    _season_state["season_id"] = season_id
    _season_active = True

    # Sync simulator state
    _simulator.soil_moisture_pct = 65.0
    _simulator.bank_balance_inr = FarmSimulator.INITIAL_BALANCE_INR
    _simulator.active_debt_inr = 0.0

    # Bind tools to new simulator/event-engine instances
    _init_tools()

    # Get initial weather summary
    if _simulator.weather_truth:
        _season_state["weather_summary"] = _simulator.weather_truth[0]["summary"]

    observation = _build_observation()

    # Compute available schemes for info block
    schemes_available = [s["name"] for s in _event_engine.scheme_schedule]
    total_scheme_value = sum(s["benefit_inr"] for s in _event_engine.scheme_schedule)

    logger.info(
        "Season started | id=%s | difficulty=%s | seed=%d",
        season_id, difficulty, seed,
    )

    return ResetResponse(
        observation=observation,
        info={
            "season_id": season_id,
            "difficulty": difficulty,
            "seed": seed,
            "optimal_income_inr": _simulator.get_optimal_income(),
            "baseline_income_inr": KisanGrader.BASELINE_INCOME,
            "schemes_available": schemes_available,
            "total_scheme_value_inr": total_scheme_value,
            "pest_events_scheduled": len(_event_engine.pest_schedule),
            "tool_failure_days_total": sum(
                len(v) for v in _event_engine.tool_failure_days.values()
            ),
            "farmer": "Harish — 2-acre tomato farm, Kolar, Karnataka",
        },
        season_id=season_id,
        difficulty=request.difficulty,
        message="New season started. 90 days to harvest. Good luck, Harish!",
    )


@app.post("/step", response_model=StepResult, tags=["Environment"])
async def step(request: StepRequest) -> StepResult:
    """
    Advance one calendar day.

    Process:
    1. Apply farm_decision to simulator
    2. Advance weather/soil/events by 1 day
    3. Compute step reward via grader
    4. Check termination (day == 89 → day 90)
    5. Return new observation + reward + grader scores

    On day 90 (terminated=True): final_scores and net_income_inr populated.
    """
    global _season_state, _season_active

    _require_season()

    day = _season_state["day"]
    action = request.farm_decision.value
    tool_calls = request.tool_calls_made or []

    # Get event state for this day
    event_state = _event_engine.get_event_state(day)

    # Record treatment if spraying
    if action == "spray_pesticide":
        _event_engine.record_treatment(day)
        _season_state["pest_suppressed"] = True
        event_state["pest_suppressed"] = True

    # Record scheme capture if applying
    if action == "apply_scheme":
        active_scheme = event_state.get("active_scheme_name")
        if active_scheme:
            _event_engine.record_scheme_capture(active_scheme)
            if active_scheme not in _season_state.get("schemes_captured", []):
                _season_state.setdefault("schemes_captured", []).append(active_scheme)

    # Run one day of simulation
    new_state, yield_delta, cost_incurred = _simulator.simulate_day(
        day=day,
        action=action,
        current_state=_season_state,
        event_state=event_state,
    )
    _season_state = new_state

    # Sync simulator attributes for tool access
    _simulator.soil_moisture_pct = _season_state["soil_moisture_pct"]
    _simulator.bank_balance_inr = _season_state.get("bank_balance_inr", 15000.0)
    _simulator.active_debt_inr = _season_state.get("active_debt_inr", 0.0)

    # Compute per-step scores for reward shaping
    step_state_for_grader = {
        "pest_risk_level": event_state.get("pest_risk_level", "LOW"),
        "soil_moisture_pct": _season_state["soil_moisture_pct"],
        "day": day,
    }
    step_scores = _grader.compute_step_scores(action, step_state_for_grader, tool_calls)

    # Log step to grader
    _grader.log_step({
        "day": day,
        "action": action,
        "tool_calls": tool_calls,
        "tool_score": step_scores.tool_use_quality,
        "pest_score": step_scores.pest_response_accuracy,
        "pest_risk_level": event_state.get("pest_risk_level", "LOW"),
        "cost_incurred": cost_incurred,
        "yield_delta": yield_delta,
    })

    # Advance day
    new_day = day + 1
    _season_state["day"] = new_day
    _season_state["tool_calls_today"] = 0  # reset tool budget for next day

    # Reset pest suppression flag each day
    _season_state["pest_suppressed"] = False

    # ── Terminal check ─────────────────────────────────────────
    terminated = new_day >= 90
    final_scores = None
    net_income_inr = None

    if terminated:
        _season_active = False

        # Force-sell any remaining yield at distress price
        final_yield = _season_state.get("estimated_yield_kg", 0.0)
        sell_decisions = _season_state.get("sell_decisions", [])
        day90_price = (
            _simulator.price_truth[89]
            if len(_simulator.price_truth) > 89 else 10.0
        )

        net_income_inr = _simulator.calculate_final_income(
            yield_kg=final_yield,
            sell_decisions=sell_decisions,
            total_costs=_season_state.get("total_costs_inr", 0.0),
            active_debt=_season_state.get("active_debt_inr", 0.0),
            day_90_price=day90_price,
        )

        # Get episode summary from event engine
        episode_summary = _event_engine.get_episode_summary()

        final_state = {
            "net_income_inr": net_income_inr,
            "pest_events": episode_summary["pest_events"],
            "schemes_available": episode_summary["schemes_available"],
            "schemes_captured": episode_summary["schemes_captured"],
            "water_used_liters": _season_state.get("water_used_liters", 0.0),
            "chemical_applications": _season_state.get("chemical_applications", 0),
        }

        final_scores = _grader.compute_final_scores(final_state, _grader.episode_log)
        reward = float(final_scores.composite_score)

        logger.info(
            "Season complete | income=₹%.0f | composite=%.3f",
            net_income_inr, final_scores.composite_score,
        )
    else:
        # Reward shaping for non-terminal steps
        reward = float(step_scores.composite_score) * 0.1  # scaled shaping signal

    # Build updated observation
    observation = _build_observation()

    return StepResult(
        observation=observation,
        reward=round(reward, 4),
        terminated=terminated,
        truncated=False,
        info={
            "day_completed": day,
            "action_applied": action,
            "cost_incurred_inr": round(cost_incurred, 2),
            "yield_delta_kg": round(yield_delta, 1),
            "pest_risk_level": event_state.get("pest_risk_level", "LOW"),
            "weather_today": _simulator.weather_truth[day] if day < 90 else None,
            "tool_calls_made": tool_calls,
            "reasoning_logged": bool(request.reasoning),
        },
        step_scores=step_scores,
        final_scores=final_scores,
        net_income_inr=net_income_inr,
    )


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health() -> HealthResponse:
    """Liveness and readiness probe. Returns current season day."""
    return HealthResponse(
        status="healthy",
        current_day=_season_state.get("day", 0),
        season_active=_season_active,
        environment="KisanAgent v1.0",
    )


@app.get("/state", response_model=StateResponse, tags=["Meta"])
async def state() -> StateResponse:
    """
    Full internal state dump — for debugging and evaluation.
    Not exposed to the LLM agent during normal operation.
    """
    return StateResponse(
        season_state=dict(_season_state),
        event_schedule=_event_engine.get_state() if _event_engine else {},
        simulator_state=_simulator.get_state() if _simulator else {},
        grader_state=_grader.get_state() if _grader else {},
    )


@app.post(
    "/tools/{tool_name}",
    response_model=ToolResponse,
    tags=["Tools"],
)
async def call_tool(
    tool_name: str = Path(
        ...,
        description="One of: weather | soil | mandi_price | govt_scheme | pest_alert | credit",
    ),
    request: ToolRequest = ...,
) -> ToolResponse:
    """
    Unified tool endpoint — agent calls real-world-style tools here.

    **Tool budget: max 3 calls per step.**
    Exceeding the limit returns HTTP 429 with advisory to make decision.

    Available tools:
    - **weather** — IMD Karnataka 3-day forecast (noisy)
    - **soil** — IoT soil moisture sensor reading (±5% noise)
    - **mandi_price** — Agmarknet KR Puram tomato prices
    - **govt_scheme** — Karnataka Raitha Seva Kendra portal
    - **pest_alert** — Dept of Agriculture pest surveillance
    - **credit** — KCC / NABARD microfinance assessment
    """
    _require_season()

    # Validate tool name
    valid_tools = {"weather", "soil", "mandi_price", "govt_scheme", "pest_alert", "credit"}
    if tool_name not in valid_tools:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown tool '{tool_name}'. Valid tools: {sorted(valid_tools)}",
        )

    # Enforce tool budget (max 3 per step)
    calls_used = _season_state.get("tool_calls_today", 0)
    if calls_used >= 3:
        raise HTTPException(
            status_code=429,
            detail=(
                "Tool budget exceeded (3/3 calls used today). "
                "Make your farm_decision now. POST /step with your decision."
            ),
        )

    # Increment budget counter
    _season_state["tool_calls_today"] = calls_used + 1
    call_number = calls_used + 1

    day = _season_state.get("day", 0)
    args = request.args or {}

    # Check if this tool is failing today
    is_failing = _event_engine.is_tool_failing_today(tool_name, day)
    start_time = time.perf_counter()

    # Route to tool
    tool = _tools.get(tool_name)
    if tool is None:
        raise HTTPException(status_code=500, detail=f"Tool '{tool_name}' not initialised.")

    try:
        if tool_name == "weather":
            result = tool.call(
                current_day=day,
                days_ahead=args.get("days_ahead", 3),
            )
            if is_failing:
                result["data_quality"] = "unavailable"
                result["forecast"] = []
                result["advisory"] = "IMD API offline. Scheduled maintenance."

        elif tool_name == "soil":
            result = tool.call(
                farm_id=args.get("farm_id", "farm_001"),
                current_day=day,
            )
            if is_failing:
                result = {
                    "moisture_pct": None, "ph": None,
                    "nitrogen_kg_per_acre": None, "phosphorus_kg_per_acre": None,
                    "reading_delay_hours": 0,
                    "sensor_status": "offline",
                    "last_calibrated_days_ago": 0,
                    "data_quality": "unavailable",
                    "advisory": "Sensor offline. Estimate from weather data.",
                }

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
        logger.error("Tool '%s' raised: %s", tool_name, exc, exc_info=True)
        result = {"error": str(exc)}

    latency_ms = round((time.perf_counter() - start_time) * 1000 + 100.0, 1)
    data_quality = result.get("data_quality", "good")

    # Store last tool result in session for observation
    _season_state["last_tool_result"] = {"tool": tool_name, **result}

    logger.info(
        "Tool call | %s | day=%d | call=%d/3 | quality=%s | latency=%.0fms",
        tool_name, day, call_number, data_quality, latency_ms,
    )

    return ToolResponse(
        tool_name=tool_name,
        result=result,
        latency_ms=latency_ms,
        data_quality=data_quality,
        call_number=call_number,
    )


# ── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
