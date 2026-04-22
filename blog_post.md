# KisanAgent: Teaching LLMs to Farm

## Helping 600 Million Indian Farmers Through Reinforcement Learning

*Meta-PyTorch OpenEnv Hackathon Finale 2026 — Theme 3: World Modeling*

---

### The Problem

Harish grows tomatoes on 2 acres in Kolar district, Karnataka. Every day across a 90-day season, 
he makes decisions with incomplete, noisy, often contradictory information.

Should he irrigate today — or is rain coming tomorrow? Should he spray for the aphids he spotted 
yesterday — or will that kill the pollinators during flowering? Is today's ₹14/kg mandi price 
the bottom, or should he hold for the price spike that sometimes comes in week 11?

One wrong decision compounds. Miss the drip irrigation subsidy deadline by three days and ₹5,000 
disappears. Spray pesticide during flowering and lose 15% of your yield to pollinator death. 
Miss a pest outbreak treatment window by 48 hours and lose 40% of the crop.

These are the daily decisions of 600 million Indian smallholder farmers. Most make them alone, 
with no advisory support, in a data vacuum.

**KisanAgent** is our answer.

---

### What We Built

KisanAgent is a production-grade RL environment where an LLM agent acts as Harish's AI advisor.

The environment runs as a FastAPI server with five OpenEnv-compatible endpoints:

```
POST /reset              → Start a new 90-day season
POST /step               → Submit a daily farm decision
GET  /health             → Liveness probe
GET  /state              → Debug: full internal state
POST /tools/{tool_name}  → Call one of 6 real-world tool APIs
```

The agent calls tools before making decisions — just like a real advisor would:

| Tool | Real-World Analog | What It Returns |
|------|-------------------|-----------------|
| `weather` | IMD Karnataka district bulletin | Noisy 3-day forecast, rain probability |
| `soil` | PM-KISAN IoT sensor network | Soil moisture ±5%, pH, nitrogen |
| `mandi_price` | Agmarknet / eNAM price feed | Today's KR Puram tomato price, 7-day trend |
| `govt_scheme` | Karnataka Raitha Seva Kendra | Active schemes, deadlines, benefits |
| `pest_alert` | Dept of Agriculture surveillance | Risk level: LOW→MEDIUM→HIGH→CRITICAL |
| `credit` | Kisan Credit Card / NABARD | Loan eligibility, KCC terms |

Every tool is **noisy** (±5–30% readings), occasionally **degraded**, and sometimes **unavailable** — 
because that's what real-world Karnataka agricultural APIs are like.

---

### The World Model

The simulator runs a 90-day Kolar district tomato season under the hood. The agent *never* sees 
this ground truth — it only accesses it through noisy tool APIs, exactly as a real farmer would 
navigate government portals and IoT sensors.

**Kolar monsoon pattern modelled:**
- Days 0–20: Pre-monsoon dry (rain prob 5%, temp 32–38°C)
- Days 21–45: SW monsoon active (rain prob 65%, 15–40mm events)
- Days 46–75: NE monsoon transition (rain prob 35%, intermittent)
- Days 76–90: Post-monsoon dry — the harvest window

**Discrete events injected each episode:**
- 1–3 pest outbreaks (aphids, whitefly, leaf curl virus) with 4-day treatment windows
- 4 government scheme deadlines (PMFBY by day 20, drip subsidy by day 50...)
- 1–2 mandi price spikes (₹22–35/kg) during harvest
- 2–8 random tool failure days (sensor offline, API outage)

---

### The Grader

KisanAgent uses a **multi-dimensional grader** that returns scores strictly in (0, 1):

```python
GraderScores(
    income_score           = 0.82,  # 40% weight — ₹28k vs ₹40k optimal
    tool_use_quality       = 0.91,  # 20% — called relevant tool before deciding
    pest_response_accuracy = 1.00,  # 20% — treated both pest events in window
    scheme_capture_rate    = 0.75,  # 10% — got 3/4 schemes before deadline
    sustainability_score   = 0.88,  # 10% — water efficient, minimal chemicals
    composite_score        = 0.874  # weighted sum — primary GRPO reward
)
```

The income dimension normalises against Karnataka 2024-25 benchmarks:
- 0.0 = baseline unadvised farmer (₹15,000 avg)
- 1.0 = theoretical optimal (₹40,000)

This gives the GRPO training signal a rich, shaped reward surface rather than sparse end-of-season binary.

---

### The Agent

The LLM agent uses a ReAct (Reasoning + Action) loop:

```
For each of 90 days:
  1. Receive FarmerObservation (noisy state)
  2. ReAct loop (max 3 tool calls):
     → LLM decides: call tool OR make decision
     → If tool: call /tools/{name}, add result to context
     → Repeat until farm_decision is made
  3. POST /step with decision + tool_calls_made + reasoning
  4. Receive new observation + step reward
```

The 3-tool-per-day budget forces the agent to prioritise: don't check pest_alert, soil, 
AND weather AND mandi_price every day. Learn what matters when.

---

### Training with GRPO

We fine-tune **Qwen2.5-7B-Instruct** (4-bit, Unsloth) using GRPO with curriculum learning:

| Stage | Episodes | Difficulty |
|-------|----------|------------|
| Warm-up | 0–150 | easy |
| Main | 150–350 | medium |
| Hard | 350–500 | hard |

The verifiable reward signal is `composite_score` from KisanGrader — deterministic, 
no judge model needed. This makes KisanAgent a perfect fit for GRPO's verifiable reward paradigm.

---

### Why This Matters

The gap between random agent (₹11,200 avg) and optimal (₹40,000) represents real rupees for 
real farmers. In Karnataka alone, 3.2 million tomato farmers make these decisions every season.

A trained KisanAgent that closes even 50% of that gap — reaching ₹25,000+ net income — 
represents ₹42 billion in aggregate farmer income gains per season in Karnataka alone.

World modeling for agriculture isn't an academic exercise. It's economic justice at scale.

---

### Technical Stack

```
FastAPI server (OpenEnv)  ←→  LLM Agent (LiteLLM proxy)
       ↓                              ↓
FarmSimulator              inference.py (ReAct loop)
EventEngine                    openai SDK
KisanGrader                    Qwen2.5-7B (GRPO)
```

- **Runtime**: FastAPI + Uvicorn on HuggingFace Spaces (Docker SDK)
- **Package manager**: uv with locked dependencies
- **Training**: Unsloth + TRL GRPO on Google Colab A100
- **Deployment**: HuggingFace Spaces, port 7860

---

### Reproduce

```bash
# Install
pip install uv
uv sync

# Start environment server
uvicorn server.app:app --port 8000

# Run smoke test
curl -X POST localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"difficulty": "easy"}'

# Run agent (set your API keys first)
export API_KEY=your_key
export API_BASE_URL=https://your-litellm-proxy.com/v1
export MODEL_NAME=qwen2.5-7b-instruct
python inference.py
```

---

*KisanAgent — Because every ₹ matters to Harish.*
