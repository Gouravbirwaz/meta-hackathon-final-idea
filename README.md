# SirenWorld RL: World Modeling & Disaster Response

**SirenWorld** is a high-performance, hackathon-winning RL environment for training LLM agents (Mistral/Llama) in dynamic, partially observable disaster response scenarios.

## 🚀 Key Features
- **Deterministic World Simulator**: A standalone physics/stochastic engine (`WorldSimulator`) that separates world dynamics from the RL wrapper.
- **Partial Observability**: Implements realistic constraints like GPS noise, severity masking, and weather-dependent visibility.
- **Advanced Reward System**: Centralized logic featuring immediate rewards for accuracy and delayed rewards for safety outcomes.
- **Mistral-Ready Pipeline**: Integration with **HuggingFace TRL** and **Unsloth** for 4-bit PPO training.
- **Live Demo**: Interactive simulation showing agent "Thought" processes and real-time world evolution.

## 📂 Structure
```text
sirennet_world_model/
├── env/
│   └── siren_env.py          # Gymnasium/OpenEnv Wrapper
├── world/
│   └── world_simulator.py    # stochastic world engine
├── reward/
│   └── reward_function.py    # centralized compute_reward()
├── agents/
│   └── policy_model.py       # Mistral Prompting & Parsing
├── training/
│   └── train_ppo.py          # TRL PPO Training Loop
├── evaluation/
│   └── metrics.py            # Latency & Success tracking
└── demo/
    └── run_simulation.py     # Interactive Hackathon Demo
```

## 🛠 Usage

### 1. Run the Demo
```bash
python -m demo.run_simulation
```

### 2. Start Training (GPU Required)
```bash
python -m training.train_ppo
```

### 3. Generate Data
```bash
python -m data.scenario_generator
```

## 🧠 LLM Training Signal
The environment is designed to reward logic-first agents. The +20/-15 accuracy reward for classification and +30/-30 reward for safety resolution ensures that the agent learns to prioritize high-risk events and allocate resources efficiently.
