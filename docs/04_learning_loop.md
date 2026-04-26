# 4. The Learning Loop

The agent interacts with the environment through a **ReAct (Reasoning + Action) loop** that spans exactly 90 discrete calendar days. Each day constitutes a single multi-step interaction between the LLM and the environment server.

## Daily Execution Cycle

1. **Observe**: The agent receives a `FarmerObservation` object. This includes the current day, crop stage, noisy soil moisture, bank balance, and any active alerts (like pest warnings).
2. **Think & Gather (Inner Loop)**:
   * The LLM generates its reasoning and decides to call an information-gathering tool (e.g., `soil`).
   * The `inference.py` script hits `POST /tools/soil` and appends the JSON result to the LLM's context window.
   * This repeats up to 3 times per day, allowing the agent to piece together a picture of the partially observable world.
3. **Act**: Once sufficient information is gathered, the LLM outputs a `farm_decision` from the 8 atomic choices (e.g., `irrigate`, `spray_pesticide`, `sell_now`, `do_nothing`).
4. **Step Environment**: The environment advances 1 day (`POST /step`), applies the action, updates the farm state, calculates costs, and returns the next observation along with a shaped intermediate reward.
5. **Terminate**: On Day 90, the crop is force-sold, and the final `composite_score` is returned as the terminal reward.
