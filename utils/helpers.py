import json
import math
from typing import List, Dict, Any

def calculate_distance(loc1: List[float], loc2: List[float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def extract_json(response: str) -> Dict[str, Any]:
    """Robustly extract JSON from model response."""
    try:
        # Simplest case: direct JSON
        return json.loads(response)
    except:
        try:
            # Look for code blocks or braces
            if "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                return json.loads(response[start:end])
        except:
            pass
    return {}

def format_location(loc: List[float]) -> str:
    """Format coordinates for display."""
    return f"({loc[0]:.1f}, {loc[1]:.1f})"
