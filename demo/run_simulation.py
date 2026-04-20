import time
import json
from env.siren_env import SirenWorldEnv
from agents.policy_model import SirenAgentPolicy
from evaluation.metrics import MetricsTracker

def run_live_simulation(steps=20):
    print("\n" + "="*60)
    print("SIRENNET WORLD MODEL: HACKATHON DEMO".center(60))
    print("="*60)
    
    env = SirenWorldEnv()
    policy = SirenAgentPolicy()
    tracker = MetricsTracker()
    
    obs, _ = env.reset()
    
    for i in range(steps):
        # 1. Agent Reasoning
        prompt = policy.format_obs_for_prompt(obs)
        print(f"\n[STEP {i+1}] World State Summary:")
        print(prompt)
        
        # 2. Mock Agent Decision (Simulating LLM for Demo)
        # In a real run, this would be model.generate()
        sos_list = obs["sos_requests"]
        if sos_list:
            target = sos_list[0]
            # Simple heuristic mock
            if "fire" in target["description"].lower() or target["id"] == "SOS_1":
                decision = {
                    "thought": f"Prioritizing {target['id']} due to potential severity. Dispatching fire truck.",
                    "event_id": target["id"],
                    "classification": "fire",
                    "dispatch": ["FIRE_TRUCK_0"],
                    "routing": "shortest"
                }
            else:
                decision = {
                    "thought": f"Responding to {target['id']}. Sending medical unit.",
                    "event_id": target["id"],
                    "classification": "medical",
                    "dispatch": ["AMBULANCE_0"],
                    "routing": "efficient"
                }
        else:
            decision = {"thought": "Monitoring world for new alerts.", "event_id": None}

        print(f"AGENT THOUGHT: {decision.get('thought')}")
        print(f"ACTION: Dispatch {decision.get('dispatch')} to {decision.get('event_id')}")
        
        # 3. Environment Step
        action_json = json.dumps(decision)
        obs, reward, done, _, info = env.step(action_json)
        tracker.log_step(reward, info["full_state"])
        
        print(f"STEP REWARD: {reward:+.1f}")
        
        # 4. World Dynamics (Visuals)
        full_state = info["full_state"]
        active_ids = [e["id"] for e in full_state["events"] if e["status"] in ["active", "pending"]]
        print(f"WORLD: Time={full_state['time']} | Weather={full_state['weather']} | Active SOS: {active_ids}")
        
        time.sleep(0.5)
        if done:
            print("\n[EPISODE TERMINATED]")
            break

    tracker.print_report()

if __name__ == "__main__":
    run_live_simulation()
