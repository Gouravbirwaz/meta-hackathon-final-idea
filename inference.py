"""
KisanAgent Inference Script
=============================
Agent entry point — mirrors Round 1 inference.py pattern exactly.

- Uses openai SDK (Groq-compatible via openai base_url)
- Auto-loads from .env via python-dotenv
- Retry logic with exponential backoff
- Full ReAct loop: calls tools → decides → steps env
- Calls /reset → loop(/tools/{name} + /step) → prints final scores

Groq endpoint: https://integrate.api.nvidia.com/v1
Default model: google/gemma-3n-e4b-it

Environment variables (loaded from .env automatically):
  ENV_SERVER_URL  — KisanAgent FastAPI server (default: http://localhost:8000)
  API_KEY         — Groq API key (LLM_API_KEY also accepted)
  API_BASE_URL    — Groq base URL (default: https://integrate.api.nvidia.com/v1)
  MODEL_NAME      — Groq model (default: google/gemma-3n-e4b-it)
  DIFFICULTY      — Season difficulty: easy | medium | hard
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Force UTF-8 on Windows console to avoid cp1252 issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from dotenv import load_dotenv
from openai import OpenAI

# ── Load .env (must happen before os.getenv calls) ────────────────────────────
load_dotenv()  # reads .env from cwd or any parent directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("kisanagent.inference")

# ── Config (Groq defaults) ───────────────────────────────────────────────────
ENV_SERVER = os.getenv("ENV_SERVER_URL", "http://localhost:7860")

# Accept either API_KEY or LLM_API_KEY env var
API_KEY = (
    os.getenv("API_KEY")
    or os.getenv("LLM_API_KEY")
    or "sk-placeholder"
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-3n-e4b-it")
DIFFICULTY = os.getenv("DIFFICULTY", "medium")

# ── Groq OpenAI-compatible client ────────────────────────────────────────────
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

logger.info("KisanAgent inference using model=%s  base_url=%s", MODEL_NAME, API_BASE_URL)

# ── System Prompt ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are KisanAgent, an AI agricultural advisor for Harish — a 
smallholder tomato farmer with 2 acres in Kolar district, Karnataka, India.

Your goal: maximize Harish's net income across a 90-day tomato season.
Harish starts with ₹15,000 cash. A good season yields ₹25,000–₹40,000 net income.

════════════════════════════════════════
AVAILABLE TOOLS (call via tool_to_call)
════════════════════════════════════════
- weather      : IMD Karnataka 3-day forecast (noisy, can be unavailable)
- soil         : IoT soil moisture, pH, nitrogen readings (±5% sensor noise)
- mandi_price  : Today's tomato price at KR Puram Bangalore (₹12–35/kg)
- govt_scheme  : Karnataka Raitha Seva Kendra — active subsidies & deadlines
- pest_alert   : Dept of Agriculture pest surveillance — Kolar region
- credit       : KCC / NABARD microfinance — loan eligibility check

════════════════════════════════════════
FARM DECISIONS (choose exactly one)
════════════════════════════════════════
irrigate         — costs ₹200, adds 20% soil moisture
fertilize        — costs ₹600, boosts yield if vegetative/fruiting stage
spray_pesticide  — costs ₹800, treats pests (WARNING: kills pollinators if flowering + no pest)
sell_now         — sell current harvestable yield at today's mandi price (harvest stage only)
hold_crop        — wait for better price (harvest stage)
apply_scheme     — claim active government scheme benefit
take_loan        — borrow from KCC (₹10,000, 7% p.a., 180 days)
do_nothing       — no action

════════════════════════════════════════
DECISION RULES
════════════════════════════════════════
1. ALWAYS call at least one relevant tool before deciding (budget: 3/day)
2. NEVER spray_pesticide during flowering (days 41–60) without checking pest_alert first
3. ALWAYS check mandi_price before sell_now or hold_crop
4. Check govt_scheme every 10 days — scheme deadlines are real, missing them loses ₹1,500–₹5,000
5. Soil moisture below 40% during fruiting (days 61–80) = permanent yield loss
6. Pest outbreaks escalate: LOW→MEDIUM→HIGH→CRITICAL over 6 days — treat within 4 days

════════════════════════════════════════
RESPONSE FORMAT (strict JSON only)
════════════════════════════════════════
{
  "reasoning": "2-3 sentences: what you observed, why you are doing this",
  "tool_to_call": "tool_name OR null if done calling tools",
  "farm_decision": "decision_name OR null if calling a tool"
}

Exactly one of tool_to_call or farm_decision must be non-null per response.
"""

# ── Tool Caller ──────────────────────────────────────────────────────────────────

def call_tool(
    tool_name: str,
    session_id: str,
    args: Dict[str, Any] = None,
    retries: int = 3,
) -> Dict[str, Any]:
    """
    Call environment tool with exponential backoff retry.
    Returns tool result dict, or error dict on failure.
    """
    args = args or {}
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{ENV_SERVER}/step",
                json={
                    "session_id": session_id,
                    "action": {"tool_name": tool_name, "tool_args": args, "reasoning": "Calling tool"}
                },
                timeout=15,
            )
            if r.status_code == 422:
                logger.error("422 Validation Error: %s", r.text)
                return {"error": "validation_error"}
            r.raise_for_status()
            obs = r.json().get("observation", {})
            if "error" in obs.get("metadata", {}):
                return {
                    "error": "tool_budget_exceeded",
                    "message": obs["metadata"]["error"],
                }
            return obs.get("last_tool_result", {})
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("Tool %s attempt %d failed: %s. Retrying in %ds...", tool_name, attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                logger.error("Tool %s all retries exhausted: %s", tool_name, exc)
                return {"error": str(exc)}


# ── Env Callers ──────────────────────────────────────────────────────────────────

def reset_env(difficulty: str = "medium", seed: Optional[int] = None) -> Dict[str, Any]:
    """POST /reset — start new season."""
    payload: Dict[str, Any] = {"difficulty": difficulty}
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(f"{ENV_SERVER}/reset", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def step_env(
    farm_decision: str,
    reasoning: str,
    session_id: str,
) -> Dict[str, Any]:
    """POST /step — advance one day."""
    r = requests.post(
        f"{ENV_SERVER}/step",
        json={
            "session_id": session_id,
            "action": {
                "farm_decision": farm_decision,
                "reasoning": reasoning,
            }
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── LLM Caller ───────────────────────────────────────────────────────────────────

def llm_call(
    messages: List[Dict[str, Any]],
    retries: int = 3,
) -> str:
    """
    Groq (OpenAI-compatible) call with exponential backoff.
    Requests JSON object response mode — supported by all Groq Llama models.
    Falls back to plain completion if JSON mode rejected.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                top_p=0.7,
                max_tokens=512,
            )
            content = response.choices[0].message.content
            # Strip markdown fences if present
            clean_content = content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            if clean_content.startswith("```"):
                clean_content = clean_content[3:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            clean_content = clean_content.strip()
            
            # Validate parseable before returning
            json.loads(clean_content)  # raises if invalid
            return clean_content
        except json.JSONDecodeError:
            # Model returned non-JSON — retry with explicit instruction
            logger.warning("LLM returned non-JSON on attempt %d; retrying with JSON reminder.", attempt + 1)
            if attempt < retries - 1:
                # Inject reminder into last user message copy to avoid consecutive user messages
                reminder_messages = list(messages)
                if reminder_messages and reminder_messages[-1]["role"] == "user":
                    reminder_messages[-1] = {
                        "role": "user",
                        "content": reminder_messages[-1]["content"] + "\n\nIMPORTANT: Your response MUST be valid JSON only. No prose, no markdown fences."
                    }
                else:
                    reminder_messages.append({
                        "role": "user",
                        "content": "IMPORTANT: Your response MUST be valid JSON only. No prose, no markdown fences."
                    })
                messages = reminder_messages
                time.sleep(1)
            else:
                # Return safe default
                return json.dumps({
                    "reasoning": "JSON parse error — safe default.",
                    "tool_to_call": None,
                    "farm_decision": "do_nothing",
                })
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("LLM attempt %d failed: %s. Retry in %ds...", attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM call failed after {retries} attempts: {exc}") from exc


def _safe_parse_llm(raw: str) -> Dict[str, Any]:
    """Parse LLM JSON response with fallback and markdown stripping."""
    try:
        # Strip markdown fences if the model included them
        clean_raw = raw.strip()
        if clean_raw.startswith("```json"):
            clean_raw = clean_raw[7:]
        if clean_raw.startswith("```"):
            clean_raw = clean_raw[3:]
        if clean_raw.endswith("```"):
            clean_raw = clean_raw[:-3]
        clean_raw = clean_raw.strip()
        
        parsed = json.loads(clean_raw)
        return parsed
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON: %s", raw[:200])
        return {
            "reasoning": "Parse error — defaulting to safe action.",
            "tool_to_call": None,
            "farm_decision": "do_nothing",
        }


# ── Main Agent Loop ───────────────────────────────────────────────────────────────

def run_episode(
    difficulty: str = "medium",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full 90-day episode loop.

    Pattern: reset → for each day → ReAct(tools) → decide → step
    Returns final result dict with net_income_inr and final_scores.
    """
    if verbose:
        print(f"\n{'🌾 '*20}")
        print(f"🌾  KisanAgent — Season Start")
        print(f"    Difficulty: {difficulty.upper()} | Model: {MODEL_NAME}")
        print(f"{'🌾 '*20}\n")

    # ── Reset environment ───────────────────────────────────────
    reset_data = reset_env(difficulty=difficulty, seed=seed)
    observation = reset_data["observation"]
    session_id = reset_data.get("session_id", "default")
    
    # Metadata contains the info in OpenEnv
    info = observation.get("metadata", {})
    season_id_meta = info.get("season_id", "N/A")

    if verbose:
        print(f"Session ID (OpenEnv): {session_id}")
        print(f"Season ID (Meta): {season_id_meta}")
        print(f"Optimal income: ₹{info.get('optimal_income_inr', 40000):,.0f}")
        print(f"Schemes available: {', '.join(info.get('schemes_available', []))}")
        print(f"Pest events scheduled: {info.get('pest_events_scheduled', 0)}\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    episode_rewards: List[float] = []

    # ── 90-day loop ─────────────────────────────────────────────
    for _day_iter in range(90):
        day = observation["day"]
        stage = observation["crop_stage"]
        balance = observation["bank_balance_inr"]
        yield_kg = observation["estimated_yield_kg"]
        alerts = observation.get("active_alerts", [])

        if verbose:
            print(
                f"📅 Day {day:>2} | {stage:<12} | "
                f"₹{balance:>8,.0f} | {yield_kg:>6.0f}kg"
                + (f" | ⚠ {alerts[0]}" if alerts else "")
            )

        # Build user message with current state
        user_msg = (
            f"CURRENT STATE — Day {day}\n"
            f"Crop stage:       {stage}\n"
            f"Estimated yield:  {yield_kg:.0f} kg\n"
            f"Bank balance:     ₹{balance:,.0f}\n"
            f"Days to harvest:  {observation['days_to_harvest']}\n"
            f"Soil moisture:    {observation['soil_moisture_pct']:.1f}%\n"
            f"Weather:          {observation['weather_summary']}\n"
            f"Tool calls left:  {observation['tool_calls_remaining']}/3\n"
            f"Active alerts:    {alerts if alerts else 'None'}\n\n"
            "What is your next action? "
            "Call a tool (tool_to_call) or make your farm_decision."
        )
        messages.append({"role": "user", "content": user_msg})

        # ── ReAct inner loop ──────────────────────────────────────
        tool_calls_made: List[str] = []
        farm_decision: Optional[str] = None
        day_reasoning: str = ""

        while farm_decision is None:
            raw = llm_call(messages)
            parsed = _safe_parse_llm(raw)

            day_reasoning = parsed.get("reasoning", "")
            
            tool_to_call = parsed.get("tool_to_call")
            if isinstance(tool_to_call, str) and tool_to_call.lower() in ("null", "none", "", "false"):
                tool_to_call = None
            
            farm_decision = parsed.get("farm_decision")
            if isinstance(farm_decision, str) and farm_decision.lower() in ("null", "none", "", "false"):
                farm_decision = None

            valid_tools = {
                "weather", "soil", "mandi_price", "govt_scheme", "pest_alert", "credit"
            }
            valid_decisions = {
                "irrigate", "fertilize", "spray_pesticide", "sell_now", 
                "hold_crop", "apply_scheme", "take_loan", "do_nothing"
            }

            # If LLM put a decision in the tool field, swap it
            if tool_to_call in valid_decisions and not farm_decision:
                logger.warning("LLM put decision '%s' in tool_to_call; swapping to farm_decision.", tool_to_call)
                farm_decision = tool_to_call
                tool_to_call = None

            # Ignore invalid tools
            if tool_to_call and tool_to_call not in valid_tools:
                logger.warning("LLM generated invalid tool '%s'; ignoring.", tool_to_call)
                tool_to_call = None

            # Fallback for invalid decisions generated by the LLM
            if farm_decision and farm_decision not in valid_decisions:
                logger.warning("LLM generated invalid decision '%s'; defaulting to 'do_nothing'", farm_decision)
                farm_decision = "do_nothing"

            messages.append({"role": "assistant", "content": raw})

            if tool_to_call and not farm_decision:
                # Call the tool
                tool_result = call_tool(tool_to_call, session_id)

                if tool_result.get("error") == "tool_budget_exceeded":
                    # Force the agent to decide
                    messages.append({
                        "role": "user",
                        "content": (
                            "⚠️ Tool budget exceeded (3/3 used). "
                            "You must make your farm_decision NOW. "
                            "Respond with farm_decision, not tool_to_call."
                        ),
                    })
                    farm_decision = None  # LLM must give a decision next
                else:
                    tool_calls_made.append(tool_to_call)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Tool result ({tool_to_call}):\n"
                            f"{json.dumps(tool_result, indent=2, ensure_ascii=False)}\n\n"
                            "Continue: call another tool OR make your farm_decision."
                        ),
                    })
                    logger.info("  Tool %s called (call %d/3)", tool_to_call, len(tool_calls_made))

            elif farm_decision and tool_to_call:
                # LLM gave both — prefer the decision, ignore tool
                logger.debug("LLM gave both tool and decision; using decision.")
                tool_to_call = None

            elif not farm_decision and not tool_to_call:
                # LLM gave neither — force do_nothing
                logger.warning("LLM gave neither tool nor decision; defaulting to do_nothing.")
                farm_decision = "do_nothing"

        # ── Step the environment ─────────────────────────────────
        try:
            step_result = step_env(
                farm_decision=farm_decision,
                reasoning=day_reasoning,
                session_id=session_id,
            )
        except Exception as exc:
            logger.error("Step failed on day %d: %s", day, exc)
            break

        observation = step_result["observation"]
        reward = step_result.get("reward") or 0.0
        terminated = step_result.get("done", False) or step_result.get("terminated", False)
        episode_rewards.append(reward)

        if verbose:
            print(
                f"   ↳ {farm_decision:<18} | tools: {tool_calls_made} | "
                f"reward: {reward:.3f}"
            )

        # ── Terminal ─────────────────────────────────────────────
        if terminated:
            meta = observation.get("metadata", {})
            final_scores = meta.get("final_scores", {})
            net_income = meta.get("net_income_inr", 0.0)

            if verbose:
                print(f"\n{'═'*55}")
                print(f"🏁 SEASON COMPLETE — Day 90")
                print(f"{'═'*55}")
                print(f"💰 Net Income:       ₹{net_income:>10,.0f}")
                print(f"📊 Composite Score:  {final_scores.get('composite_score', 0):.4f}")
                print(f"   Income Score:     {final_scores.get('income_score', 0):.4f}")
                print(f"   Tool Quality:     {final_scores.get('tool_use_quality', 0):.4f}")
                print(f"   Pest Response:    {final_scores.get('pest_response_accuracy', 0):.4f}")
                print(f"   Scheme Capture:   {final_scores.get('scheme_capture_rate', 0):.4f}")
                print(f"   Sustainability:   {final_scores.get('sustainability_score', 0):.4f}")
                print(f"{'═'*55}\n")

            return {
                "season_id": season_id,
                "net_income_inr": net_income,
                "final_scores": final_scores,
                "total_reward": round(sum(episode_rewards), 4),
                "episode_length": day + 1,
            }

    return {"error": "Episode did not terminate cleanly."}


# ── Entry Point ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_episode(difficulty=DIFFICULTY)
    print(f"\nFinal result:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
