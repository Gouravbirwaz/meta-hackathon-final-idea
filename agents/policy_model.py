import json
from typing import Dict, Any
from utils.helpers import extract_json, format_location

class SirenAgentPolicy:
    """
    Enhanced agent policy for Mistral.
    Generates prompts for the partially observable world state.
    """
    
    def __init__(self, model_name: str = "Mistral-7B-Instruct"):
        self.model_name = model_name

    def get_system_prompt(self) -> str:
        return (
            "You are the Emergency Response Dispatcher AI. Your objective is to save lives "
            "and optimize resources in a dynamic disaster environment. You see a "
            "partially observable view of the world.\n\n"
            "Respond in JSON format with your reasoning (thought), the target SOS ID, "
            "the classification, the resources to dispatch, and the routing strategy."
        )

    def format_obs_for_prompt(self, obs: Dict[str, Any]) -> str:
        """Convert observation dict to a clean text-based prompt for the LLM."""
        prompt = f"### WORLD STATE (Time: {obs['env_conditions']['time']} | Weather: {obs['env_conditions']['weather']})\n\n"
        
        prompt += "INCOMING SOS REQUESTS:\n"
        if not obs["sos_requests"]:
            prompt += "- None active.\n"
        for sos in obs["sos_requests"]:
            sev_label = ["LOW", "MEDIUM", "HIGH"][sos["severity_estimate"]]
            prompt += f"- ID: {sos['id']} | Category: {sos['description']} | Severity: {sev_label} | Loc: {format_location(sos['location'])}\n"
            
        prompt += "\nRESOURCE STATUS:\n"
        free_res = [r for r in obs["resource_status"] if r["status"] == "free"]
        if not free_res:
            prompt += "- No units available.\n"
        for r in free_res:
            prompt += f"- ID: {r['id']} | Type: {r['type']} | Loc: {format_location(r['location'])}\n"
            
        return prompt

    def generate_full_prompt(self, obs: Dict[str, Any]) -> str:
        system = self.get_system_prompt()
        state_text = self.format_obs_for_prompt(obs)
        
        return (
            f"<s>[INST] {system}\n\n{state_text}\n"
            "Output your decision in JSON format: [/INST]"
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Extract and clean the JSON response from the LLM."""
        return extract_json(response)
