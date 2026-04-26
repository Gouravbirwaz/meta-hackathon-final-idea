"""
KisanAgent — All Pydantic Models
=================================
Defines every data shape used across the environment:
  Observation, Action, StepResult, SeasonState,
  ToolRequest, ToolResponse, GraderScores
All scores strictly in (0, 1) range.
Mirrors Round 1 multi-dimensional grader pattern.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from openenv.core.env_server import Action, Observation, State


# ─────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────


class Difficulty(str, Enum):
    """Season difficulty controls weather variance and event density."""

    easy = "easy"
    medium = "medium"
    hard = "hard"


class CropStage(str, Enum):
    """
    Tomato crop growth stages for Kolar district, Karnataka.
    Stage boundaries drive irrigation, fertilizer, and sell decisions.
    """

    seedling = "seedling"      # days  0–15
    vegetative = "vegetative"  # days 16–40
    flowering = "flowering"    # days 41–60
    fruiting = "fruiting"      # days 61–80
    harvest = "harvest"        # days 81–90


class FarmDecision(str, Enum):
    """
    The 8 atomic actions the agent can take each day.
    Exactly one must be submitted per /step call.
    """

    irrigate = "irrigate"
    fertilize = "fertilize"
    spray_pesticide = "spray_pesticide"
    sell_now = "sell_now"
    hold_crop = "hold_crop"
    apply_scheme = "apply_scheme"
    take_loan = "take_loan"
    do_nothing = "do_nothing"


class ToolName(str, Enum):
    """Available real-world-style tool APIs."""

    weather = "weather"
    soil = "soil"
    mandi_price = "mandi_price"
    govt_scheme = "govt_scheme"
    pest_alert = "pest_alert"
    credit = "credit"


# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────


class ResetRequest(BaseModel):
    """
    POST /reset — start a fresh 90-day farming season.
    """

    difficulty: Difficulty = Field(
        default=Difficulty.medium,
        description="Season difficulty: easy | medium | hard",
    )
    seed: Optional[int] = Field(
        default=None,
        description="RNG seed for reproducible seasons. "
                    "None → random seed each episode.",
    )

    model_config = {"json_schema_extra": {"example": {"difficulty": "medium", "seed": 42}}}


class StepRequest(BaseModel):
    """
    POST /step — advance one calendar day.
    Submit exactly one farm_decision alongside any tool calls made.
    """

    farm_decision: Optional[FarmDecision] = Field(
        default=None,
        description="One of 8 atomic farm decisions for today.",
    )
    tool_calls_made: List[str] = Field(
        default_factory=list,
        description="Ordered list of tool names called before this decision. "
                    "Max 3. Tracked by server; additional calls return 429.",
    )
    reasoning: str = Field(
        default="",
        max_length=2048,
        description="Agent's chain-of-thought. Logged for analysis and GRPO.",
    )

    @field_validator("tool_calls_made")
    @classmethod
    def validate_tool_names(cls, v: List[str]) -> List[str]:
        valid = {t.value for t in ToolName}
        for name in v:
            if name not in valid:
                raise ValueError(
                    f"Unknown tool '{name}'. Valid tools: {sorted(valid)}"
                )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "farm_decision": "irrigate",
                "tool_calls_made": ["soil", "weather"],
                "reasoning": "Soil moisture at 38%, rain forecast day+2. Irrigate now.",
            }
        }
    }


class ToolRequest(BaseModel):
    """
    POST /tools/{tool_name} — invoke a named real-world tool API.
    """

    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific keyword arguments. "
                    "Optional — each tool has sensible defaults.",
    )

    model_config = {"json_schema_extra": {"example": {"args": {"days_ahead": 3}}}}

class KisanAction(Action):
    """Unified OpenEnv action for KisanAgent."""
    farm_decision: Optional[FarmDecision] = Field(
        default=None,
        description="One of 8 atomic farm decisions to execute this day.",
    )
    tool_name: Optional[ToolName] = Field(
        default=None,
        description="If calling a tool, specify the tool name here.",
    )
    tool_args: Optional[Union[Dict[str, Any], str]] = Field(
        default=None,
        description="Arguments for the tool (JSON string or dict).",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's chain-of-thought.",
    )

class KisanState(State):
    """Internal state for debugging."""
    season_state: Dict[str, Any]
    event_schedule: Dict[str, Any]
    simulator_state: Dict[str, Any]
    grader_state: Dict[str, Any]


# ─────────────────────────────────────────────────────────────
# OBSERVATION MODEL
# ─────────────────────────────────────────────────────────────


class FarmerObservation(Observation):
    """
    The agent's window onto the farm world.
    All sensor readings are noisy — ground truth lives in FarmSimulator.
    This is what the LLM agent sees each day.
    """

    day: int = Field(
        ge=0, le=90,
        description="Current season day (0-indexed). Day 0 = planting day.",
    )
    crop_stage: CropStage = Field(
        description="Current crop growth stage driven by day count.",
    )
    soil_moisture_pct: float = Field(
        ge=0.0, le=100.0,
        description="Noisy soil moisture reading (±5%). "
                    "Below 40% during fruiting → permanent yield loss.",
    )
    estimated_yield_kg: float = Field(
        ge=0.0,
        description="Running yield estimate in kg. "
                    "Drops on pest damage, water stress, chemical misuse.",
    )
    bank_balance_inr: float = Field(
        description="Harish's current bank balance in INR. "
                    "Starts at ₹15,000. Costs deducted per action.",
    )
    last_tool_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Result dict from the most recent tool call, or null.",
    )
    active_alerts: List[str] = Field(
        default_factory=list,
        description="Active advisory strings e.g. "
                    "['PEST_ALERT: aphids MEDIUM', 'SCHEME_DEADLINE: day 30']",
    )
    days_to_harvest: int = Field(
        ge=0, le=90,
        description="Calendar days remaining until day 90 (mandatory harvest).",
    )
    weather_summary: str = Field(
        description="Human-readable weather summary: "
                    "'dry' | 'light_rain' | 'heavy_rain' | 'cloudy'",
    )
    tool_calls_used_today: int = Field(
        ge=0, le=3,
        description="Number of tool calls consumed this step (server-tracked).",
    )
    tool_calls_remaining: int = Field(
        ge=0, le=3,
        description="Tool calls left in today's budget (max 3 per day).",
    )

    @model_validator(mode="after")
    def validate_tool_budget(self) -> "FarmerObservation":
        if self.tool_calls_used_today + self.tool_calls_remaining > 3:
            raise ValueError(
                "tool_calls_used_today + tool_calls_remaining must not exceed 3"
            )
        return self


# ─────────────────────────────────────────────────────────────
# GRADER SCORES
# ─────────────────────────────────────────────────────────────


class GraderScores(BaseModel):
    """
    Multi-dimensional grader output.
    ALL individual scores and composite_score are in (0.0, 1.0).
    Matches Round 1 Multi-Dimensional Grader pattern.

    Weights:
      income_score           40%  — primary economic signal
      tool_use_quality       20%  — right tools, right timing
      pest_response_accuracy 20%  — caught + treated in window
      scheme_capture_rate    10%  — applied before deadline
      sustainability_score   10%  — water + chemical efficiency
    """

    income_score: float = Field(
        ge=0.0, le=1.0,
        description="Net income normalised vs optimal (40% weight). "
                    "0 = baseline unadvised farmer. 1 = theoretical max.",
    )
    tool_use_quality: float = Field(
        ge=0.0, le=1.0,
        description="Quality of tool use before each decision (20% weight). "
                    "1.0 = always called the decision-relevant tool.",
    )
    pest_response_accuracy: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of pest events treated within 4-day window (20%).",
    )
    scheme_capture_rate: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of available govt schemes captured before expiry (10%).",
    )
    sustainability_score: float = Field(
        ge=0.0, le=1.0,
        description="Water and chemical efficiency score (10%). "
                    "Penalises over-irrigation and excess pesticide use.",
    )
    composite_score: float = Field(
        ge=0.0, le=1.0,
        description="Weighted composite of all 5 dimensions. "
                    "Primary GRPO reward signal.",
    )

    @field_validator(
        "income_score",
        "tool_use_quality",
        "pest_response_accuracy",
        "scheme_capture_rate",
        "sustainability_score",
        "composite_score",
    )
    @classmethod
    def clamp_to_unit_interval(cls, v: float) -> float:
        return float(min(max(v, 0.0), 1.0))


# ─────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────


class ResetResponse(BaseModel):
    """Returned by POST /reset on season initialisation."""

    observation: FarmerObservation = Field(
        description="Day-0 observation: planting day with initial soil/weather state.",
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata: optimal_income, schemes_available, event_count, etc.",
    )
    season_id: str = Field(
        description="UUID for this season — links grader logs to episodes.",
    )
    difficulty: Difficulty = Field(
        description="Season difficulty used for this episode.",
    )
    message: str = Field(
        default="New season started. 90 days to harvest.",
        description="Human-readable session message.",
    )


class StepResult(BaseModel):
    """
    Returned by POST /step after each daily decision.
    On day 90 (terminated=True): final_scores and net_income_inr are populated.
    """

    observation: FarmerObservation = Field(
        description="Updated farm observation after applying the decision.",
    )
    reward: float = Field(
        description="Step reward in (0, 1) range for reward shaping. "
                    "Terminal step uses composite_score as reward.",
    )
    terminated: bool = Field(
        description="True when day == 90. Season is over. Compute final scores.",
    )
    truncated: bool = Field(
        default=False,
        description="True if episode ends before day 90 (e.g. bankruptcy). "
                    "Not used in v1 — always False.",
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-step diagnostics: action applied, costs, weather truth, etc.",
    )
    step_scores: Optional[GraderScores] = Field(
        default=None,
        description="Intermediate grader scores for this step (reward shaping). "
                    "income_score=0.5 (neutral) until episode end.",
    )
    final_scores: Optional[GraderScores] = Field(
        default=None,
        description="Terminal grader scores — populated only when terminated=True.",
    )
    net_income_inr: Optional[float] = Field(
        default=None,
        description="Harish's final net income in INR — populated only when terminated=True.",
    )


class HealthResponse(BaseModel):
    """GET /health — liveness and readiness probe."""

    status: str = Field(default="healthy")
    current_day: int = Field(
        ge=0, le=90,
        description="Current day in the active season. 0 if no season active.",
    )
    season_active: bool = Field(
        description="True if a season is currently running.",
    )
    environment: str = Field(default="KisanAgent v1.0")


class StateResponse(BaseModel):
    """
    GET /state — full internal state dump for debugging and logging.
    Not exposed to the LLM agent during normal operation.
    """

    season_state: Dict[str, Any] = Field(
        description="Top-level season variables: day, balance, yield, stage, etc.",
    )
    event_schedule: Dict[str, Any] = Field(
        description="Scheduled pest/scheme/price-spike events from EventEngine.",
    )
    simulator_state: Dict[str, Any] = Field(
        description="FarmSimulator internals: weather truth, moisture, yield multiplier.",
    )
    grader_state: Dict[str, Any] = Field(
        description="KisanGrader episode log summary.",
    )


class ToolResponse(BaseModel):
    """
    POST /tools/{tool_name} — unified tool response envelope.
    result contains tool-specific data matching each tool's schema.
    """

    tool_name: str = Field(description="Name of the tool invoked.")
    result: Dict[str, Any] = Field(
        description="Tool-specific result dict. Schema varies per tool.",
    )
    latency_ms: float = Field(
        ge=0.0,
        description="Simulated API latency in milliseconds (realistic noise).",
    )
    data_quality: str = Field(
        description="'good' | 'degraded' | 'unavailable' — "
                    "reflects real-world API reliability.",
    )
    call_number: int = Field(
        ge=1, le=3,
        description="This call's position in today's 3-call budget (1, 2, or 3).",
    )

    @field_validator("data_quality")
    @classmethod
    def validate_quality(cls, v: str) -> str:
        allowed = {"good", "degraded", "unavailable"}
        if v not in allowed:
            raise ValueError(f"data_quality must be one of {allowed}")
        return v


# ─────────────────────────────────────────────────────────────
# INTERNAL STATE MODEL (used between simulator and server)
# ─────────────────────────────────────────────────────────────


class SeasonState(BaseModel):
    """
    Internal mutable state passed between simulator, event engine, and server.
    Not serialised to the agent — reconstructed each step.
    """

    day: int = Field(default=0, ge=0, le=90)
    crop_stage: CropStage = Field(default=CropStage.seedling)
    difficulty: Difficulty = Field(default=Difficulty.medium)
    seed: int = Field(default=42)
    season_id: str = Field(default="")

    # Farm economics
    bank_balance_inr: float = Field(default=15000.0)
    total_costs_inr: float = Field(default=0.0)
    gross_revenue_inr: float = Field(default=0.0)
    active_debt_inr: float = Field(default=0.0)

    # Crop health
    soil_moisture_pct: float = Field(default=65.0, ge=0.0, le=100.0)
    yield_kg: float = Field(default=0.0, ge=0.0)
    yield_multiplier: float = Field(default=1.0, ge=0.0, le=2.0)
    root_rot_risk_days: int = Field(default=0, ge=0)

    # Pest tracking
    pest_active: bool = Field(default=False)
    pest_name: Optional[str] = Field(default=None)
    pest_onset_day: Optional[int] = Field(default=None)
    pest_risk_level: str = Field(default="LOW")
    pest_events_log: List[Dict[str, Any]] = Field(default_factory=list)

    # Scheme tracking
    schemes_available: List[str] = Field(default_factory=list)
    schemes_captured: List[str] = Field(default_factory=list)

    # Resource usage
    water_used_liters: float = Field(default=0.0)
    chemical_applications: int = Field(default=0)

    # Sell tracking
    sell_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls_today: int = Field(default=0, ge=0, le=3)
    active_alerts: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
