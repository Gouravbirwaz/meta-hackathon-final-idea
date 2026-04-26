# 2. The Reward Function

To provide a rich, continuous training signal for the LLM, the environment uses a **Multi-Dimensional Grader** (`env/grader.py`) rather than a simple binary pass/fail at the end of the episode. All scores are strictly bounded between `0.0` and `1.0`.

The final reward (the `composite_score`) is a weighted sum of five distinct dimensions:

1. **Income Score (40%)**: The primary economic signal. Net income is normalized against a baseline unadvised farmer (₹15,000) and the theoretical optimum (₹40,000). A score of 0 means the farmer barely broke even, while 1.0 represents perfect market timing and yield.
2. **Tool Use Quality (20%)**: Evaluates whether the agent queried the correct APIs before making a decision (e.g., checking `mandi_price` before deciding to sell, or checking `soil` before irrigating).
3. **Pest Response Accuracy (20%)**: Measures how effectively the agent treated pest outbreaks within the critical 4-day treatment window before permanent yield loss occurred.
4. **Scheme Capture Rate (10%)**: The fraction of available government schemes the agent successfully applied for before their respective deadlines expired.
5. **Sustainability Score (10%)**: Penalizes over-irrigation and unnecessary pesticide applications. Crucially, applying pesticide during the `flowering` stage when no pest is present results in pollinator death and a heavy penalty here.
