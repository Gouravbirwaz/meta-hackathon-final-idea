# 🌾 KisanAgent

**RL Environment for Indian Smallholder Farm Advisory**  
*Meta-PyTorch OpenEnv Hackathon Finale 2026 — Theme 3: World Modeling*

---

KisanAgent is a production-grade reinforcement learning environment where an LLM agent acts as 
an AI advisor for **Harish**, a 2-acre tomato farmer in Kolar district, Karnataka, India. The 
agent makes daily farm decisions across a 90-day growing season using 6 real-world-style tool 
APIs, and is rewarded by actual rupee income at harvest.

## 📁 Project Structure

```
KisanAgent/
├── server/app.py               # FastAPI OpenEnv server (5 endpoints)
├── env/
│   ├── models.py               # All Pydantic models
│   └── grader.py               # Multi-dimensional grader (0–1 scores)
├── simulator/
│   ├── farm_simulator.py       # Ground truth crop/soil/weather engine
│   └── event_engine.py         # Pest/scheme/price-spike event state machine
├── tools/
│   ├── weather_tool.py         # IMD Karnataka forecast analog
│   ├── soil_tool.py            # IoT soil sensor analog
│   ├── mandi_price_tool.py     # Agmarknet price feed analog
│   ├── govt_scheme_tool.py     # Karnataka Raitha Seva Kendra analog
│   ├── pest_alert_tool.py      # Dept of Agriculture surveillance analog
│   └── credit_tool.py          # KCC / NABARD microfinance analog
├── tasks/
│   ├── season_scenarios.py     # easy / medium / hard presets
│   └── event_scheduler.py      # Event calendar generator
├── data_engine/
│   └── scenario_generator.py   # Randomized season generator
├── assets/
│   ├── kolar_weather_profile.json
│   ├── tomato_growth_model.json
│   └── karnataka_schemes_2025.json
├── training/
│   ├── train_grpo_unsloth.ipynb
│   └── grpo_config.yaml
├── eval/
│   ├── baseline_eval.py
│   └── plot_results.py
├── inference.py                # Agent entry point (ReAct loop)
├── pyproject.toml
├── Dockerfile
└── blog_post.md
```

## 🚀 Quick Start

### Requirements
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
git clone https://github.com/YOUR_USERNAME/kisanagent.git
cd kisanagent
pip install uv
uv sync
```

### Run the Environment Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Smoke Test

```bash
# Health check
curl http://localhost:8000/health

# Start a new season
curl -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"difficulty": "medium", "seed": 42}'

# Call the weather tool
curl -X POST http://localhost:8000/tools/weather \
     -H "Content-Type: application/json" \
     -d '{"args": {"days_ahead": 3}}'

# Make a farm decision
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"farm_decision": "irrigate", "tool_calls_made": ["soil", "weather"], "reasoning": "Soil moisture low, no rain forecast"}'
```

### Run the Agent

```bash
export ENV_SERVER_URL=http://localhost:8000
export API_KEY=your_api_key
export API_BASE_URL=https://your-litellm-proxy.com/v1
export MODEL_NAME=gpt-4o
export DIFFICULTY=medium

python inference.py
```

## 🌱 Environment Details

### The Farm
- **Farmer**: Harish, 2-acre tomato farm, Kolar district, Karnataka
- **Season**: 90 days, planting to harvest
- **Starting capital**: ₹15,000
- **Baseline income**: ₹15,000 (unadvised farmer average)
- **Optimal income**: ₹40,000 (theoretical maximum)

### Crop Stages

| Stage | Days | Key Decisions |
|-------|------|---------------|
| Seedling | 0–15 | Irrigation, establishment |
| Vegetative | 16–40 | Nitrogen fertilizer, pest scouting |
| Flowering | 41–60 | ⚠️ NO pesticide unless CRITICAL pest |
| Fruiting | 61–80 | Irrigation mandatory, potassium feed |
| Harvest | 81–90 | Time sell_now to mandi price spike |

### 8 Farm Decisions

| Decision | Cost | Effect |
|----------|------|--------|
| `irrigate` | ₹200 | +20% soil moisture |
| `fertilize` | ₹600 | +5% yield (vegetative/fruiting only) |
| `spray_pesticide` | ₹800 | Halts pest damage; −15% yield if no pest + flowering |
| `sell_now` | — | Sell at today's mandi price (harvest stage) |
| `hold_crop` | — | Wait for better price |
| `apply_scheme` | — | Claim active govt subsidy |
| `take_loan` | — | KCC loan up to ₹25,000 at 7% p.a. |
| `do_nothing` | — | Natural simulation step |

### Grader Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| `income_score` | 40% | Net income vs optimal |
| `tool_use_quality` | 20% | Right tools before each decision |
| `pest_response_accuracy` | 20% | Treated in 4-day window |
| `scheme_capture_rate` | 10% | Applied before deadline |
| `sustainability_score` | 10% | Water and chemical efficiency |

All scores are strictly in **[0.0, 1.0]**.

## 🏋️ Training

Open `training/train_grpo_unsloth.ipynb` in Google Colab (A100 recommended).

**Model**: Qwen2.5-7B-Instruct-bnb-4bit (Unsloth)  
**Algorithm**: GRPO (Group Relative Policy Optimization)  
**Curriculum**: easy (0–150) → medium (150–350) → hard (350–500)  
**Reward**: `composite_score` from KisanGrader — verifiable, no judge model needed

## 📊 Evaluation

```bash
# Run baseline evaluation (random + heuristic agents)
python eval/baseline_eval.py --episodes 20 --difficulty medium

# Plot training progress
python eval/plot_results.py --training-log eval/training_log.json
```

## 🐳 Deploy to HuggingFace Spaces

The Dockerfile is configured for HuggingFace Spaces Docker SDK (port 7860):

```bash
# Build locally
docker build -t kisanagent .
docker run -p 7860:7860 kisanagent

# Deploy: Push to HuggingFace Spaces with Docker SDK selected
```

## 🔧 Difficulty Levels

| Level | Pest Events | Tool Failures | Starting Balance | Description |
|-------|-------------|---------------|-----------------|-------------|
| `easy` | 1 | 2 | ₹20,000 | Predictable season, forgiving |
| `medium` | 2 | 4 | ₹15,000 | Typical Kolar conditions |
| `hard` | 3 | 8 | ₹12,000 | NE monsoon failure, tight deadlines |

## 📄 API Reference

See live docs at `http://localhost:8000/docs` after starting the server.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new 90-day season |
| `/step` | POST | Submit daily farm decision |
| `/tools/{name}` | POST | Call real-world tool API (max 3/day) |
| `/health` | GET | Liveness probe |
| `/state` | GET | Full internal state dump |

## 📝 License

MIT License — See LICENSE file.

---

*Built for the Meta-PyTorch OpenEnv Hackathon Finale 2026.*  
*Because every ₹ matters to Harish.*
