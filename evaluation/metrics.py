import numpy as np
from typing import List, Dict, Any

class MetricsTracker:
    """
    Evaluation metrics for SirenWorld.
    Tracks rewards, success rates, and response latency.
    """
    
    def __init__(self):
        self.rewards = []
        self.success_counts = []
        self.failure_counts = []
        self.latencies = []

    def log_step(self, reward: float, full_state: Dict[str, Any]):
        self.rewards.append(reward)
        
        events = full_state["events"]
        resolved = [e for e in events if e["status"] == "resolved"]
        failed = [e for e in events if e["status"] == "failed"]
        
        self.success_counts.append(len(resolved))
        self.failure_counts.append(len(failed))
        
        # Latency (Time from creation to resolution)
        for e in resolved:
            # Note: This is an approximation based on current state
            self.latencies.append(full_state["time"] - e["creation_time"])

    def print_report(self):
        print("\n" + "="*30)
        print("FINAL PERFORMANCE REPORT")
        print("="*30)
        print(f"Total Steps:      {len(self.rewards)}")
        print(f"Average Reward:   {np.mean(self.rewards):.2f}")
        print(f"Max Reward:       {np.max(self.rewards):.2f}")
        print(f"Successes:        {self.success_counts[-1] if self.success_counts else 0}")
        print(f"Failures:         {self.failure_counts[-1] if self.success_counts else 0}")
        if self.latencies:
            print(f"Avg Latency:      {np.mean(self.latencies):.1f} steps")
        print("="*30 + "\n")
