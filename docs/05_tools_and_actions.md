# 5. Tools and Actions

To succeed, the agent must leverage information-gathering tools and execute precise actions.

## The 6 Real-World Tools
The agent cannot see the ground truth. It navigates the environment using APIs that mimic real-world Indian agricultural infrastructure. Crucially, these tools are **noisy**, **can degrade**, and **experience outages**:
1. **`weather`**: IMD Karnataka 3-day forecast (provides noisy rain probability).
2. **`soil`**: PM-KISAN IoT sensor network (returns moisture ±5% error, pH, nitrogen).
3. **`mandi_price`**: Agmarknet price feed for KR Puram (current price and 7-day trend).
4. **`govt_scheme`**: Karnataka Raitha Seva Kendra portal (active subsidies and deadlines).
5. **`pest_alert`**: Dept of Agriculture regional surveillance (LOW to CRITICAL risk levels).
6. **`credit`**: KCC / NABARD microfinance eligibility check.

## The 8 Farm Decisions
After consulting the tools, the agent must execute exactly one action per day:
* `irrigate` (costs ₹200) - Boosts soil moisture by 20%.
* `fertilize` (costs ₹600) - Boosts yield during vegetative/fruiting stages.
* `spray_pesticide` (costs ₹800) - Cures pests, but damages yield if used unnecessarily during flowering.
* `sell_now` - Liquidates the crop at today's mandi price.
* `hold_crop` - Waits for better prices during the harvest window.
* `apply_scheme` - Captures available government subsidies.
* `take_loan` - Adds ₹10,000 debt at 7% p.a. to survive cashflow crunches.
* `do_nothing` - Allows the farm to progress naturally.
