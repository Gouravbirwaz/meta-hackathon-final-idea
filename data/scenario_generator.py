import json
import random
import os

class ScenarioGenerator:
    """
    Synthetic SOS Generator for training scenarios.
    """
    def __init__(self, output_path: str = "data/synthetic_sos.json"):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def generate(self, count: int = 500):
        data = []
        categories = ["fire", "medical", "disaster"]
        descriptions = {
            "fire": ["Structure fire reported.", "Smoke visible from roof.", "Kitchen fire out of control."],
            "medical": ["Patient suffering from heart distress.", "Unconscious person in lobby.", "Accident with trauma."],
            "disaster": ["Flood blocking main artery.", "Chemical spill on highway.", "Structural collapse."]
        }
        
        for i in range(count):
            cat = random.choice(categories)
            data.append({
                "id": f"SOS_GEN_{i}",
                "category": cat,
                "description": random.choice(descriptions[cat]),
                "severity": random.randint(1, 10),
                "loc": [random.uniform(0, 100), random.uniform(0, 100)]
            })
            
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Successfully generated {count} scenarios in {self.output_path}")

if __name__ == "__main__":
    gen = ScenarioGenerator()
    gen.generate()
