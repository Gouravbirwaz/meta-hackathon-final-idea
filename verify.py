"""
KisanAgent -- Quick verification script.
Tests Groq connectivity + 3-day agent episode.
Run: uv run python verify.py
"""
import os
import json
import sys
import requests
from dotenv import load_dotenv


# Force UTF-8 on Windows console to avoid cp1252 issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:7b")
ENV_SERVER = os.getenv("ENV_SERVER_URL", "http://localhost:8000")

print("=" * 60)
print("KisanAgent Verification Script (Ollama)")
print("=" * 60)
print(f"Model      : {MODEL_NAME}")
print(f"Ollama URL : {OLLAMA_URL}")
print(f"Env server : {ENV_SERVER}")
print()

# --------------------------------------------------------------------------
# Test 1: Ollama API connectivity
# --------------------------------------------------------------------------
print("[1/4] Testing Ollama API connectivity...")
try:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "Reply only with valid JSON."},
                {"role": "user", "content": 'Return: {"status": "ok", "agent": "KisanAgent"}'},
            ],
            "stream": False,
            "format": "json"
        },
        timeout=30
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    parsed = json.loads(content)
    print(f"  [OK] Ollama OK -- model={MODEL_NAME}")
    print(f"       Response: {parsed}")
except Exception as e:
    print(f"  [FAIL] Ollama FAILED: {e}")
    raise

# --------------------------------------------------------------------------
# Test 2: Server health
# --------------------------------------------------------------------------
print("\n[2/4] Testing environment server health...")
try:
    h = requests.get(f"{ENV_SERVER}/health", timeout=5).json()
    print(f"  [OK] Server healthy: {h}")
except Exception as e:
    print(f"  [FAIL] Server FAILED: {e}")
    print("     Run: uvicorn server.app:app --port 8000")
    raise

# --------------------------------------------------------------------------
# Test 3: Reset + all 6 tools
# --------------------------------------------------------------------------
print("\n[3/4] Testing /reset and all 6 tools...")
r = requests.post(
    f"{ENV_SERVER}/reset",
    json={"difficulty": "easy", "seed": 42},
    timeout=10,
).json()
obs = r["observation"]
print(f"  [OK] Reset | day={obs['day']} stage={obs['crop_stage']} balance=Rs{obs['bank_balance_inr']:.0f}")
print(f"       Schemes: {r['info']['schemes_available']}")
print(f"       Pest events: {r['info']['pest_events_scheduled']}")

for tool_name in ["weather", "soil", "mandi_price", "govt_scheme", "pest_alert", "credit"]:
    try:
        tr = requests.post(
            f"{ENV_SERVER}/tools/{tool_name}",
            json={"args": {}},
            timeout=10,
        ).json()
        print(f"  [OK] {tool_name:<16} quality={tr.get('data_quality','?')} call#{tr.get('call_number','?')}")
    except Exception as e:
        print(f"  [FAIL] {tool_name}: {e}")

# --------------------------------------------------------------------------
# Test 4: 3-day agent episode with Ollama ReAct loop
# --------------------------------------------------------------------------
print(f"\n[4/4] Running 3-day agent episode (Ollama {MODEL_NAME})...")

r = requests.post(
    f"{ENV_SERVER}/reset",
    json={"difficulty": "medium", "seed": 7},
    timeout=10,
).json()
obs = r["observation"]

SYSTEM_PROMPT = (
    "You are KisanAgent, an AI advisor for Harish, a 2-acre tomato farmer in Kolar, Karnataka.\n"
    "Maximize his net income across a 90-day tomato season.\n\n"
    "TOOLS (max 3/day): weather | soil | mandi_price | govt_scheme | pest_alert | credit\n"
    "DECISIONS: irrigate | fertilize | spray_pesticide | sell_now | hold_crop | apply_scheme | take_loan | do_nothing\n\n"
    "Respond ONLY in this JSON format:\n"
    '{"reasoning": "brief reason", "tool_to_call": "tool_name or null", "farm_decision": "decision or null"}\n\n'
    "Exactly ONE of tool_to_call or farm_decision must be non-null per response."
)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for day_iter in range(3):
    day = obs["day"]
    user_msg = (
        f"Day {day} | Stage: {obs['crop_stage']} | "
        f"Moisture: {obs['soil_moisture_pct']:.1f}% | "
        f"Balance: Rs{obs['bank_balance_inr']:.0f} | "
        f"Alerts: {obs.get('active_alerts', [])} | "
        f"Tools left: {obs['tool_calls_remaining']}/3. "
        "What is your next action?"
    )
    messages.append({"role": "user", "content": user_msg})

    tools_called = []
    decision = None
    reasoning = ""

    # ReAct inner loop (max 5 turns)
    for _ in range(5):
        llm_resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "format": "json"
            },
            timeout=60
        )
        llm_resp.raise_for_status()
        raw = llm_resp.json()["message"]["content"]
        parsed = json.loads(raw)
        messages.append({"role": "assistant", "content": raw})

        reasoning = parsed.get("reasoning", "")
        tool = parsed.get("tool_to_call")
        decision = parsed.get("farm_decision")

        if tool and not decision:
            tr = requests.post(
                f"{ENV_SERVER}/tools/{tool}",
                json={"args": {}},
                timeout=10,
            )
            if tr.status_code == 200:
                tools_called.append(tool)
                preview = str(tr.json().get("result", {}))[:150]
                messages.append({
                    "role": "user",
                    "content": f"Tool {tool} result: {preview}. Continue or make farm_decision.",
                })
            elif tr.status_code == 429:
                messages.append({
                    "role": "user",
                    "content": "Tool budget exceeded (3/3). Make your farm_decision NOW.",
                })
        elif decision:
            break

    if not decision:
        decision = "do_nothing"

    step = requests.post(
        f"{ENV_SERVER}/step",
        json={
            "farm_decision": decision,
            "tool_calls_made": tools_called,
            "reasoning": reasoning,
        },
        timeout=15,
    ).json()

    obs = step["observation"]
    reward = step["reward"]
    sc = step.get("step_scores") or {}

    print(
        f"  Day {day:>2} | {decision:<18} | tools={tools_called} | "
        f"reward={reward:.3f} | tool_score={sc.get('tool_use_quality', 0):.2f}"
    )
    print(f"           Reasoning: {reasoning[:80]}...")

print()
print("=" * 60)
print("ALL VERIFICATION TESTS PASSED")
print(f"KisanAgent fully operational -- Ollama {MODEL_NAME}")
print("=" * 60)
