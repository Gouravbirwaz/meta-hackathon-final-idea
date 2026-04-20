from typing import Dict, Any

def compute_reward(state: Dict[str, Any], action: Dict[str, Any], next_state: Dict[str, Any]) -> float:
    """
    Centralized reward function for SirenWorld.
    Calculates rewards based on accuracy, response, and outcomes.
    """
    reward = 0.0
    
    event_id = action.get("event_id")
    if not event_id:
        return -5.0 # Penalty for no action
    
    # 1. State extraction
    prev_event = next((e for e in state["events"] if e["id"] == event_id), None)
    next_event = next((e for e in next_state["events"] if e["id"] == event_id), None)
    
    if not prev_event:
        return -10.0 # Action on non-existent event
    
    # 2. Classification Accuracy (+20 / -15)
    classification = action.get("classification")
    if classification == prev_event["category"]:
        reward += 20.0
    else:
        reward -= 15.0
        
    # 3. Dispatch Correctness (+25 / -20)
    dispatch_ids = action.get("dispatch", [])
    correct_dispatch = False
    for rid in dispatch_ids:
        res = next((r for r in state["resources"] if r["id"] == rid), None)
        if res and res["status"] == "free":
            # Match type
            match = (
                (prev_event["category"] == "fire" and res["type"] == "fire_truck") or
                (prev_event["category"] == "medical" and res["type"] == "ambulance") or
                (prev_event["category"] == "disaster" and res["type"] in ["fire_truck", "rescue_team"])
            )
            if match:
                reward += 25.0
                correct_dispatch = True
            else:
                reward -= 20.0
        else:
            reward -= 10.0 # Dispatching busy/invalid resource

    # 4. Success / Safety Outcome (Delayed reward) (+30 / -30)
    if prev_event["status"] != "resolved" and next_event["status"] == "resolved":
        reward += 30.0
    elif next_event["status"] == "failed":
        reward -= 30.0

    # 5. Resource Optimization (+5 / -8)
    if correct_dispatch and len(dispatch_ids) == 1:
        reward += 5.0
    elif len(dispatch_ids) > 2:
        reward -= 8.0

    # 6. Penalty for Ignored SOS
    for e in next_state["events"]:
        if e["status"] == "pending" and (next_state["time"] - e["creation_time"]) > 20:
            reward -= 2.0 # Cumulative penalty for slow response
            
    if next_event["status"] == "failed":
        reward -= 25.0 # Big penalty for ignoring until failure

    return reward
