import random
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from utils.helpers import calculate_distance

class Resource(BaseModel):
    id: str
    type: str  # 'ambulance', 'fire_truck', 'police_car', 'rescue_team'
    location: List[float]
    status: str  # 'free', 'busy', 'returning'
    speed: float = 5.0  # Units per time step
    target_event_id: Optional[str] = None

class SOSEvent(BaseModel):
    id: str
    description: str
    category: str
    true_severity: int
    location: List[float]
    status: str  # 'pending', 'active', 'resolved', 'failed'
    creation_time: int
    resolution_progress: float = 0.0

class WorldSimulator:
    """
    Sirennet World Simulator: Handles the physics and stochastic transitions
    of the disaster response world.
    """
    
    def __init__(self, grid_size: float = 100.0):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.current_time = 0
        self.weather = random.choice(["clear", "rainy", "stormy", "foggy"])
        self.road_conditions = self._generate_road_conditions()
        
        self.resources = self._init_resources()
        self.events = []
        self._spawn_events(count=2)
        
        return self.get_state()

    def _init_resources(self) -> List[Resource]:
        res = []
        counts = {"ambulance": 3, "fire_truck": 2, "police_car": 3, "rescue_team": 2}
        for r_type, count in counts.items():
            for i in range(count):
                res.append(Resource(
                    id=f"{r_type.upper()}_{i}",
                    type=r_type,
                    location=[random.uniform(0, self.grid_size), random.uniform(0, self.grid_size)],
                    status="free"
                ))
        return res

    def _generate_road_conditions(self):
        # 0.0: blocked, 1.0: clear
        return {"main_road": 1.0, "highways": 1.0 if self.weather == "clear" else 0.7}

    def _spawn_events(self, count=1):
        for _ in range(count):
            eid = f"SOS_{len(self.events) + 1}"
            category = random.choice(["fire", "medical", "disaster", "false_alarm"])
            severity = random.randint(1, 10)
            location = [random.uniform(0, self.grid_size), random.uniform(0, self.grid_size)]
            
            # Simple descriptions
            desc_map = {
                "fire": "Building fire detected.",
                "medical": "Emergency medical request.",
                "disaster": "Large scale disaster event.",
                "false_alarm": "Detected potential false alarm."
            }
            
            self.events.append(SOSEvent(
                id=eid,
                description=desc_map[category],
                category=category,
                true_severity=severity,
                location=location,
                status="pending",
                creation_time=self.current_time
            ))

    def update(self, action: Dict[str, Any]):
        """
        Primary update loop for the world.
        1. Apply agent action (dispatch).
        2. Advance time and move resources.
        3. Resolve events.
        4. Stochastic changes (weather, new SOS).
        """
        # 1. Dispatch Resources
        target_eid = action.get("event_id")
        dispatch_ids = action.get("dispatch", [])
        
        event = next((e for e in self.events if e.id == target_eid), None)
        if event and event.status in ["pending", "active"]:
            for rid in dispatch_ids:
                res = next((r for r in self.resources if r.id == rid), None)
                if res and res.status == "free":
                    res.status = "busy"
                    res.target_event_id = target_eid
                    event.status = "active"

        # 2. Advance Physics (Movement)
        self.current_time += 1
        weather_penalty = 0.5 if self.weather == "stormy" else 1.0
        
        for res in self.resources:
            if res.status == "busy" and res.target_event_id:
                target_ev = next((e for e in self.events if e.id == res.target_event_id), None)
                if target_ev:
                    dist = calculate_distance(res.location, target_ev.location)
                    if dist > 0.5:
                        # Move towards event
                        move_dist = res.speed * weather_penalty
                        direction = [
                            (target_ev.location[0] - res.location[0]) / dist,
                            (target_ev.location[1] - res.location[1]) / dist
                        ]
                        res.location[0] += direction[0] * min(move_dist, dist)
                        res.location[1] += direction[1] * min(move_dist, dist)
                    else:
                        # At event location, contribute to resolution
                        target_ev.resolution_progress += 0.2

        # 3. Resolve Events
        for e in self.events:
            if e.status == "active":
                if e.resolution_progress >= (e.true_severity * 0.1):
                    e.status = "resolved"
                    # Free resources
                    for res in self.resources:
                        if res.target_event_id == e.id:
                            res.status = "free"
                            res.target_event_id = None
            elif e.status == "pending" and (self.current_time - e.creation_time) > 50:
                e.status = "failed" # Too long to respond

        # 4. Stochastic Updates
        if random.random() < 0.15:
            self._spawn_events()
        
        if self.current_time % 20 == 0:
            self.weather = random.choice(["clear", "rainy", "stormy", "foggy"])

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self.current_time,
            "weather": self.weather,
            "resources": [r.dict() for r in self.resources],
            "events": [e.dict() for e in self.events],
            "road_conditions": self.road_conditions
        }
