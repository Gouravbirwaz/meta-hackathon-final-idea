# 1. The Environment

KisanAgent operates in a custom-built, production-grade reinforcement learning environment that simulates a 90-day tomato farming season in Kolar district, Karnataka. This environment adheres to the Meta-PyTorch OpenEnv standards.

## Core Components
* **OpenEnv API Server (`server/app.py`)**: A FastAPI server exposing standard RL endpoints (`/reset`, `/step`, `/tools/{name}`, `/health`, `/state`). It acts as the intermediary between the LLM agent and the simulation.
* **Farm Simulator (`simulator/farm_simulator.py`)**: Maintains the ground truth of the farm. It simulates crop growth stages (seedling to harvest), soil moisture decay based on weather and irrigation, and cumulative financial ledgers (costs, debt, revenue).
* **Event Engine (`simulator/event_engine.py`)**: A state machine that injects real-world chaos. It schedules pest outbreaks, government scheme deadlines, market price spikes, and random IoT/API tool failures to force the agent to adapt dynamically.

## Configuration & Difficulty
The environment supports curriculum learning through three difficulty tiers:
* **Easy**: Predictable weather, 1 pest event, 2 tool failures, ₹20,000 starting balance. Highly forgiving to mistakes.
* **Medium**: Typical Kolar conditions, 2 pest events, 4 tool failures, ₹15,000 starting balance. Represents the baseline farmer experience.
* **Hard**: Erratic monsoons, 3 pest outbreaks, 8 tool failures, ₹12,000 starting balance. Demands precise timing and resource management.
