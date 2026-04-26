# 3. The Agent & Configuration

The agent (`inference.py`) acts as the "brain" of the advisor, running entirely on a Large Language Model. It connects to the environment remotely via HTTP.

## Configuration
* **Model Selection**: Defaults to `llama-3.3-70b-versatile` via the Groq API for rapid, high-quality reasoning.
* **Response Format**: Forced `json_object` mode ensures structured, programmatic outputs that can be safely parsed by the OpenEnv interface.
* **Hyperparameters**: Low temperature (`0.3`) is used to prioritize logical consistency, deterministic action sequences, and strict adherence to rules over creative variation.

## Persona & Prompting
The agent is prompted as a dedicated AI agricultural advisor. Its system prompt provides the rules of the world and imposes strict constraints:
* **Tool Budget**: It has a strict budget of maximum **3 tool calls per day**.
* **Output Structure**: It must output exactly one of two things at a time: a `tool_to_call` OR a `farm_decision`.
* **Chain-of-Thought**: It must generate a short `reasoning` string for every step. This captures the agent's logic for debugging and builds a structured dataset of reasoning traces for downstream GRPO training.
