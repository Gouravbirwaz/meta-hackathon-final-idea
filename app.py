import gradio as gr
import os
import pandas as pd
import torch
import requests
import json
from server.app import KisanEnvironment
from env.models import KisanAction, FarmDecision, ToolName

# --- 1. DATA & CONFIG ---
SUCCESS_DATA = [
    {"Epoch": 0.02, "Format Reward": 0.1500, "Kisan Reward": 0.1812, "Total": 0.3312},
    {"Epoch": 0.12, "Format Reward": 0.1750, "Kisan Reward": 0.2125, "Total": 0.3875},
    {"Epoch": 0.33, "Format Reward": 0.1750, "Kisan Reward": 0.2187, "Total": 0.3937},
    {"Epoch": 0.58, "Format Reward": 0.2000, "Kisan Reward": 0.2562, "Total": 0.4562},
    {"Epoch": 0.88, "Format Reward": 0.1625, "Kisan Reward": 0.1843, "Total": 0.3468},
]
DASHBOARD_IMAGE_URL = "https://huggingface.co/gouravbirwaz/kisanagent-trained-model/resolve/main/training_analysis.png"
DEEP_INSIGHT_URL = "https://huggingface.co/gouravbirwaz/kisanagent-trained-model/resolve/main/training_deep_insight.png"

# Initialize Local Environment for the Playground
env = KisanEnvironment()

# --- 2. PLAYGROUND LOGIC ---
def playground_step(decision, tool, args, reasoning):
    try:
        action = KisanAction(
            farm_decision=FarmDecision(decision) if decision else None,
            tool_name=ToolName(tool) if tool else None,
            tool_args=json.loads(args) if args else {},
            reasoning=reasoning
        )
        obs = env.step(action)
        return json.dumps(obs.model_dump(), indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

def playground_reset(difficulty):
    try:
        obs = env.reset(difficulty=difficulty)
        return json.dumps(obs.model_dump(), indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

def playground_state():
    try:
        return json.dumps(env.state.model_dump(), indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# --- 3. DASHBOARD UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍅 KisanAgent: Meta-PyTorch Hackathon Finale")
    
    with gr.Tab("📈 Training Evidence"):
        gr.Dataframe(pd.DataFrame(SUCCESS_DATA))
        gr.Image(DASHBOARD_IMAGE_URL, label="Reward Convergence")

    with gr.Tab("🧪 Deep Insight"):
        gr.Image(DEEP_INSIGHT_URL, label="Technical Heartbeat")

    with gr.Tab("📜 Audit & Verification"):
        try:
            with open("agent_traing_log.log", "r") as f: log_content = f.read()
        except: log_content = "Log file not found."
        gr.Code(value=log_content, language="json", label="Raw GRPO Logs")
        gr.File("agent_traing_log.log", label="Download Audit Trail")

    with gr.Tab("🚀 Live Playground"):
        gr.Markdown("### Interactive Environment (Official Style)")
        with gr.Row():
            with gr.Column():
                diff = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Difficulty")
                decision = gr.Dropdown(choices=[d.value for d in FarmDecision] + [None], label="Farm Decision")
                tool = gr.Dropdown(choices=[t.value for t in ToolName] + [None], label="Tool Name")
                args = gr.Textbox(label="Tool Args (JSON)", placeholder="{}")
                reason = gr.Textbox(label="Reasoning")
                with gr.Row():
                    step_btn = gr.Button("Step", variant="primary")
                    reset_btn = gr.Button("Reset")
                    state_btn = gr.Button("Get State")
            with gr.Column():
                output = gr.Code(label="Status (Raw JSON response)", language="json", lines=25)

        step_btn.click(playground_step, [decision, tool, args, reason], output)
        reset_btn.click(playground_reset, [diff], output)
        state_btn.click(playground_state, None, output)

    with gr.Tab("🧠 Architecture"):
        gr.Markdown("- **Model**: Qwen-2.5-7B (LoRA) | **RL**: GRPO | **Framework**: OpenEnv")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
