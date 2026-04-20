import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, Any, Optional, List
from world.world_simulator import WorldSimulator
from reward.reward_function import compute_reward
from utils.helpers import extract_json

class SirenWorldEnv(gym.Env):
    """
    OpenEnv compatible wrapper for the SirenWorldSimulator.
    Provides partial observability and integrated reward computation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.simulator = WorldSimulator(grid_size=self.config.get("grid_size", 100.0))
        
        # Action space: String for LLM JSON output
        self.action_space = spaces.Text(min_length=0, max_length=2048)
        
        # Observation space: Nested structure for agent observability
        self.observation_space = spaces.Dict({
            "sos_requests": spaces.Sequence(spaces.Dict({
                "id": spaces.Text(min_length=0, max_length=10),
                "description": spaces.Text(min_length=0, max_length=256),
                "location": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
                "severity_estimate": spaces.Discrete(3)
            })),
            "resource_status": spaces.Sequence(spaces.Dict({
                "id": spaces.Text(min_length=0, max_length=15),
                "type": spaces.Text(min_length=0, max_length=20),
                "status": spaces.Text(min_length=0, max_length=15),
                "location": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
            })),
            "env_conditions": spaces.Dict({
                "weather": spaces.Text(min_length=0, max_length=20),
                "time": spaces.Discrete(10000)
            })
        })

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        state = self.simulator.reset()
        self.last_state = state
        return self._get_partial_obs(state), {}

    def _get_partial_obs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implements partial observability:
        - Add location noise to SOS.
        - Mask true severity into 3 levels.
        - Only show active/pending SOS.
        """
        noise = 2.0 if state["weather"] == "clear" else 5.0
        
        # Process Events
        obs_sos = []
        for e in state["events"]:
            if e["status"] in ["pending", "active"]:
                noisy_loc = [
                    e["location"][0] + random.gauss(0, noise),
                    e["location"][1] + random.gauss(0, noise)
                ]
                # Severity estimate: Low (0), Med (1), High (2)
                sev = 0 if e["true_severity"] < 4 else (1 if e["true_severity"] < 8 else 2)
                
                obs_sos.append({
                    "id": e["id"],
                    "description": e["description"],
                    "location": noisy_loc,
                    "severity_estimate": sev
                })
        
        # Process Resources
        obs_res = []
        for r in state["resources"]:
            obs_res.append({
                "id": r["id"],
                "type": r["type"],
                "status": r["status"],
                "location": r["location"]
            })
            
        return {
            "sos_requests": obs_sos,
            "resource_status": obs_res,
            "env_conditions": {
                "weather": state["weather"],
                "time": state["time"]
            }
        }

    def step(self, action_input: str):
        """
        1. Parse JSON action.
        2. Update simulator.
        3. Compute Reward.
        4. Return transition.
        """
        action_dict = extract_json(action_input)
        
        # Simulator Update
        next_state = self.simulator.update(action_dict)
        
        # Reward Computation
        reward = compute_reward(self.last_state, action_dict, next_state)
        
        # Check termination (fixed horizon or zero events left)
        # For training, we usually use a fixed step limit
        done = next_state["time"] >= 100
        
        obs = self._get_partial_obs(next_state)
        self.last_state = next_state
        
        return obs, reward, done, False, {"full_state": next_state}

    def render(self):
        state = self.last_state
        print(f"\n--- STEP: {state['time']} | Weather: {state['weather']} ---")
        print(f"SOS Active: {len([e for e in state['events'] if e['status'] in ['active', 'pending']])}")
        print(f"SOS Resolved: {len([e for e in state['events'] if e['status'] == 'resolved'])}")
        print(f"Resources Busy: {len([r for r in state['resources'] if r['status'] == 'busy'])}")
