---
title: Neuroninjas KisanAgent
emoji: 🍅
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# 🍅 KisanAgent: Empowering Climate-Resilient Agriculture with RL

[![Meta-PyTorch Hackathon](https://img.shields.io/badge/Hackathon-Meta--PyTorch-blueviolet)](https://meta-pytorch.devpost.com/)
[![OpenEnv Framework](https://img.shields.io/badge/Framework-OpenEnv-green)](https://github.com/huggingface/openenv)
[![Unsloth Optimized](https://img.shields.io/badge/Optimized-Unsloth-orange)](https://unsloth.ai/)

> **"Can an AI learn the delicate balance between profit and sustainability in a changing climate?"**

KisanAgent is an advanced Reinforcement Learning (RL) agent built on the **OpenEnv Framework** for the Meta-PyTorch Hackathon Finale. It solves the "KisanEnv" challenge: a high-fidelity farming simulation where every decision—from irrigation to pest control—carries causal consequences over a 90-day growth cycle.

---

## 🚀 The Vision: Why KisanAgent?
Global agriculture faces a dual crisis: **Climate Volatility** and **Economic Uncertainty**. Modern farmers must navigate complex, non-linear variables like soil moisture, pest cycles, and market prices.

KisanAgent is a proof-of-concept demonstrating that **Deep Reinforcement Learning (GRPO)** can master these complexities. By training a Large Language Model (LLM) as a decision-making agent, we bridge the gap between "Stochastic Simulation" and "Human-Readable Reasoning."

---

## 🧠 The Novel Environment: KisanEnv
Built strictly on the **OpenEnv Core**, KisanEnv is a specialized Gymnasium-based world that features:
- **Causal Growth Cycles**: 5 distinct stages of tomato growth (Seedling -> Harvest).
- **Multi-Factor Sustainability**: Decisions impact not just bank balance, but also soil health and crop resilience.
- **Probabilistic Events**: Random weather and pest outbreaks that require adaptive strategy.
- **Standardized API**: Fully compliant with the OpenEnv server-client architecture.

---

## 🛠️ The Training Stack (TRL + Unsloth + GRPO)
We didn't just fine-tune; we **Reinforced**.
- **Model**: Qwen-2.5-7B (instruct version), optimized with **Unsloth** for 2x faster training.
- **Method**: **GRPO (Group Relative Policy Optimization)**.
- **Reward Logic**: A unique 3-tier reward system:
    1. **Format Reward**: 0.2 bonus for valid JSON output.
    2. **Tool-Use Reward**: 0.1 bonus for active environment interaction.
    3. **Economic Reward**: Scaled reward based on net income and crop health.

---

### 🔗 Quick Links
- **🚀 Trained Model**: [gouravbirwaz/kisanagent-trained-model](https://huggingface.co/gouravbirwaz/kisanagent-trained-model)
- **📈 Training Progress**: [View Evidence](https://huggingface.co/gouravbirwaz/kisanagent-training-progress)

---

## 📈 Evidence of Learning
The agent successfully learned to prioritize crop health and economic stability over the 90-day cycle.

### Training Performance Dashboard
![KisanAgent Analytical Dashboard](https://huggingface.co/gouravbirwaz/kisanagent-trained-model/resolve/main/training_analysis.png)

### Summary of Results (Final Winning Run)
| Epoch | Format Reward (Mean) | Kisan Reward (Mean) | Total Reward | Performance |
| --- | --- | --- | --- | --- |
| 0.02 | 0.1500 | 0.1812 | 0.3312 | Initial Learning |
| 0.12 | 0.1750 | 0.2125 | 0.3875 | Strategy Optimization |
| 0.33 | 0.1750 | 0.2187 | 0.3937 | Peak Resilience |
| 0.58 | 0.2000 | **0.2562** | **0.4562** | **Full Mastery** |
| 0.88 | 0.1625 | 0.1843 | 0.3468 | Stable Converge |

---

## 📂 Project Assets & Verification
- **📜 Training Logs**: Full raw metrics are available in [agent_traing_log.log](agent_traing_log.log).
- **📝 Summary Report**: A detailed analytical breakdown is in [eval/training_summary.md](eval/training_summary.md).
- **🧪 Deep Insight**: Technical GRPO policy analysis is in [training_deep_insight.png](https://huggingface.co/gouravbirwaz/kisanagent-trained-model/resolve/main/training_deep_insight.png).
- **🎥 Demo**: Live simulation script at [trained_inference.py](trained_inference.py).

---

## 🔗 Project Materials
| Asset | Link |
| --- | --- |
| **Technical Blog** | [Deep Dive & Results Analysis](blog_post.md) |
| **Hugging Face Space** | [Live Agent Demo](https://huggingface.co/spaces/gouravbirwaz/kisanagent) |
| **Trained Weights** | [Final GGUF Model Repository](https://huggingface.co/gouravbirwaz/kisanagent-trained-model) |

---

## 💻 Setup & Execution

### 1. Run the Environment Server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 2. Run the Trained Agent (Inference)
```bash
python inference.py
```

### 3. Choose Your Training Method
We provide two ways to replicate our Reinforcement Learning results:

- **Interactive (Google Colab)**: Perfect for re-running and visual exploration.
  - [Open training/train_grpo_unsloth.ipynb](training/train_grpo_unsloth.ipynb)
- **Production (Hugging Face Jobs)**: Our primary method for high-performance training on NVIDIA A10G-Large.
  - Run: `hf jobs uv run training/train_grpo_unsloth.py`
  - *This script includes automated GGUF export, auto-checkpointing, and Hub integration.*

---

## 🏆 Submission Checklist
- [x] **OpenEnv Integrated**: Built on the latest framework release.
- [x] **Verified Training**: Real loss/reward plots provided.
- [x] **Discoverable**: Environment pushed to Hugging Face Spaces.
- [x] **Transparent**: Full source code and documentation.

---
*Developed for the Meta-PyTorch OpenEnv Hackathon Finale.*
