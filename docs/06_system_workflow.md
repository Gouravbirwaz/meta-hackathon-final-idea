# 6. Complete System Workflow

This section ties the entire KisanAgent system together, explaining how the agent learns and functions continuously.

## How It All Comes Together
1. **Initialization**: The execution script calls `/reset`. The environment silently generates a hidden 90-day weather pattern, schedules random pest outbreaks, sets tool failure dates, and places ₹15,000 in Harish's bank.
2. **Execution Phase**: The agent runs its ReAct loop continuously for 90 days. It associates symptoms (e.g., `active_alerts: ["PEST_ALERT"]`) with necessary information gathering (calling the `pest_alert` tool) and the subsequent optimal action (`spray_pesticide`).
3. **World Modeling Strategy**: Because the environment is partially observable and deeply causal, the agent is forced to build an internal representation of the world. 
   * It must learn that soil moisture decays over time without rain.
   * It must learn that pests escalate from LOW to CRITICAL over 6 days.
   * It must learn that applying pesticides during the `flowering` stage drops the final yield due to pollinator death.

## Continuous Learning via GRPO
* **Data Generation**: The logs of the agent's prompts, reasoning traces, tool choices, and final actions are saved to an episode log.
* **Optimization**: These logs are fed into a training pipeline (`train_grpo_unsloth.ipynb`). Using **Group Relative Policy Optimization (GRPO)**, the model generates multiple trajectory variations for a given state.
* **Verifiable Reward**: The environment's `composite_score` serves as a verifiable, programmatic reward, eliminating the need for an LLM-as-a-judge.
* **Reinforcement**: Trajectories that yield high composite scores (efficient tool use, perfect pest response, high net income) are reinforced. Trajectories that cause bankruptcy or crop failure are heavily penalized.
* **Mastery**: Through curriculum learning (Easy → Medium → Hard), the LLM gradually masters the complex, dynamic causal relationships of the Kolar agricultural ecosystem.
