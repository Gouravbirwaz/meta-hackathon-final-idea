---
title: Neuroninjas KisanAgent
emoji: 🍅
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🍅 KisanAgent: Empowering 600M Farmers via Verifiable RL
[![Meta-PyTorch Hackathon](https://img.shields.io/badge/Hackathon-Meta--PyTorch-blueviolet)](https://meta-pytorch.devpost.com/)
[![OpenEnv Framework](https://img.shields.io/badge/Framework-OpenEnv-green)](https://github.com/meta-pytorch/OpenEnv)
[![GRPO Optimized](https://img.shields.io/badge/RL-GRPO-orange)](https://unsloth.ai/)

## ⚡ Quick Navigation
- [🚨 The Problem](#-the-problem-the-invisible-weight) & [🚀 Solution](#-the-solution-kisanagent-co-pilot)
- [🌍 KisanEnv World Model](#-1-the-unique-environment-kisanenv)
- [🧠 GRPO Reward Engine](#-3-the-brain-3-tier-verifiable-reward-grpo)
- [📈 **THE PROOFS (Evidence of Learning)**](#-4-the-evidence-proofs-of-learning)
- [🔍 **Audit Trail (Raw Logs)**](#-5-audit--reproducibility)

## 🚨 The Problem: The "Invisible Weight"
In India alone, 600 million farmers like Harish make 90+ critical decisions a season (irrigation, pests, harvest) with **incomplete, noisy, and contradictory data**. Portals crash, sensors fail, and weather forecasts are often "vibes-based."
- **The Gap**: The difference between an unadvised farmer (₹15k/season) and a theoretical optimal (₹40k/season) is massive.
- **The Risk**: One wrong guess on Day 23 compounds into a 40% yield loss by Day 90.

## 🚀 The Solution: KisanAgent Co-Pilot
KisanAgent is a production-grade **World Model** and **RL Agent** built on the **OpenEnv Framework**. It acts as a farm advisor that learns to extract "Signal from Noise" to close the ₹25,000 income gap.

---

## 🌍 1. The Unique Environment: KisanEnv
Unlike "toy" simulators, KisanEnv (built on OpenAI Gymnasium) models the **Causal Complexity** of agriculture:
- **Kolar Monsoon Engine**: Models the 3-phase monsoon of Karnataka (Pre-monsoon, SW Monsoon, NE Transition) with stochastic rainfall.
- **Causal Growth Cycles**: 5 stages (Seedling → Harvest) where today's irrigation impacts tomorrow's pest resilience.
- **API Realism**: 6 tool APIs (`weather`, `soil`, `mandi`, etc.) that are **noisy (±10%)**, **degraded**, or **unavailable**—mirroring real-world rural infrastructure.

## 📊 2. Data Strategy: Diverse Scenario Generation
To train a robust agent, we built a **Synthetic Scenario Engine** that generated 500+ unique 90-day episodes:
- **Difficulty Scaling**: Easy (Stable weather) → Hard (Overlapping crises, failing sensors).
- **Chaos Injection**: Every episode features 1-3 pest outbreaks and 1-2 mandi price spikes at random intervals.
- **Authenticity**: Soil and weather profiles are based on real Kolar District 2024 benchmarks.

## 🧠 3. The Brain: 3-Tier Verifiable Reward (GRPO)
We used **Group Relative Policy Optimization (GRPO)** to train Qwen-2.5-7B. Our unique innovation is the **3-Tier Verifiable Reward System**:

| Tier | Name | Goal | Signal |
|------|------|------|--------|
| **1** | **Format Reward** | valid JSON | Ensures the LLM strictly follows the ReAct structure. |
| **2** | **Interaction Reward** | Tool-Use | Bonus for checking `soil` or `weather` before a decision. |
| **3** | **Strategy Reward** | Economic | Scored by **KisanGrader** based on Net Income vs Optimal. |

**No "Judge Model" Vibes**: All rewards are deterministic and mathematically verifiable, preventing reward hacking and ensuring stable policy convergence.

---

## 📈 4. The Evidence (Proofs of Learning)
The agent successfully learned to prioritize crop health and economic stability.

### Analytical Performance Gallery
<p align="center"> 
  <img src="https://huggingface.co/gouravbirwaz/kisanagent-trained-model/resolve/main/training_analysis.png" width="45%" alt="Reward Convergence" />
  <img src="https://huggingface.co/gouravbirwaz/kisanagent-trained-model/resolve/main/training_deep_insight.png" width="45%" alt="Deep Insight" />
</p>
<p align="center">
  <img src="https://huggingface.co/gouravbirwaz/kisanagent-trained-model/resolve/main/training_progress.png" width="90%" alt="Training Progress" />
</p>

### Final Winning Run Results
| Epoch | Avg Kisan Reward | Total Reward | Performance |
|-------|------------------|--------------|-------------|
| 0.02 | 0.1812 | 0.3312 | Exploration |
| 0.33 | 0.2187 | 0.3937 | Resilience |
| **0.58** | **0.2562** | **0.4562** | **Full Mastery** |

---

## 📂 5. Audit & Reproducibility
> [!IMPORTANT]
> **Training Infrastructure**: While we provide a [Colab Notebook](training/train_grpo_unsloth.ipynb) for easy experimentation, the final production run was executed via **Hugging Face Jobs** (using NVIDIA A10G) after Colab GPU limits were reached.
> 
> **Source of Truth**: The definitive training logic used for the winning model is in the Python script: [training/train_grpo_unsloth.py](training/train_grpo_unsloth.py).

- **📜 Raw Training Logs**: [View agent_traing_log.log](agent_traing_log.log)
- **📝 Summary**: [Full Analytical Breakdown](eval/training_summary.md)
- **🎥 Interactive Demo**: [Launch Hugging Face Space](https://huggingface.co/spaces/gouravbirwaz/neuroninjas)

---

## 🎮 Playground User Guide
The **Live Playground** tab allows you to manually act as the KisanAgent. Here is how to use the tools:

### 🚜 The Playground Controls
The playground is a manual interface to the **KisanAction** model. Here is how to use every field:

#### 1. Configuration
- **Difficulty**: 
    - `easy`: Stable weather, high sensor reliability.
    - `medium`: Variable monsoon, occasional sensor noise.
    - `hard`: High crisis density (pests + droughts), frequent tool failures.

#### 2. Farm Decisions (The Economics)
| Decision | Impact | Cost (INR) |
|----------|--------|------------|
| `irrigate` | Boosts soil moisture +20%. Prevents drought stress. | ₹200 |
| `fertilize` | Increases yield multiplier by 5% during growth. | ₹600 |
| `spray_pesticide` | Suppresses active pest outbreaks. | ₹800 |
| `sell_now` | Harvests crop and converts to cash (only in `harvest` stage). | ₹0 |
| `hold_crop` | Wait for better prices. Risky if post-monsoon rain hits. | ₹0 |
| `apply_scheme` | Claims government subsidy (must check `govt_scheme` first). | +₹Benefit |
| `take_loan` | Injects cash if balance > ₹2,000 and no active debt. | +₹10,000 |
| `do_nothing` | Advance to next day without spending. | ₹0 |

#### 3. Reasoning (CoT)
Use this field to explain **why** you are taking an action. In training, this is where the agent's **Chain-of-Thought** is logged. 
- *Example*: "Soil is 38%, IMD predicts rain in 2 days, but fruiting stage requires immediate water. Irrigating now to prevent stress."

### 🛠️ Tool Parameters (JSON)
| Tool Name | Example Args (JSON) | Note |
|-----------|---------------------|------|
| `weather` | `{"days_ahead": 3}` | Look into the future (noisy). |
| `soil` | `{"farm_id": "farm_001"}` | Check real-time moisture. |
| `mandi_price` | `{"market": "KR Puram"}` | Check current crop market value. |

## 🛠️ 6. Technical Stack
- **Server**: FastAPI (OpenEnv Core)
- **Training**: Unsloth + TRL (GRPO)
- **Deployment**: Docker on HF Spaces (Port 7860)

---
*Developed for the Meta-PyTorch OpenEnv Hackathon Finale. Closing the gap, one decision at a time.*
