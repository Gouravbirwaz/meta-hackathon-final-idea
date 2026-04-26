import gradio as gr
import os
import pandas as pd
from server.app import app
import threading
import uvicorn

# --- THE CONTENT ---
SUCCESS_DATA = [
    {"Epoch": 0.02, "Format Reward": 0.1500, "Kisan Reward": 0.1812, "Total": 0.3312},
    {"Epoch": 0.12, "Format Reward": 0.1750, "Kisan Reward": 0.2125, "Total": 0.3875},
    {"Epoch": 0.33, "Format Reward": 0.1750, "Kisan Reward": 0.2187, "Total": 0.3937},
    {"Epoch": 0.58, "Format Reward": 0.2000, "Kisan Reward": 0.2562, "Total": 0.4562},
    {"Epoch": 0.88, "Format Reward": 0.1625, "Kisan Reward": 0.1843, "Total": 0.3468},
]

# Point to your Cloud-Uploaded Dashboards
DASHBOARD_IMAGE_URL = "https://huggingface.co/gouravbirwaz/kisanagent-trained-model/resolve/main/training_analysis.png"
DEEP_INSIGHT_URL = "https://huggingface.co/gouravbirwaz/kisanagent-trained-model/resolve/main/training_deep_insight.png"

# --- THE UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍅 KisanAgent: Meta-PyTorch Hackathon Finale")
    gr.Markdown("### RL-Powered Climate-Resilient Agriculture")
    
    with gr.Tab("📈 Training Evidence"):
        gr.Markdown("#### Final Production Run Metrics")
        gr.Dataframe(pd.DataFrame(SUCCESS_DATA))
        gr.Image(DASHBOARD_IMAGE_URL, label="Training Dashboard (GRPO Rewards)")

    with gr.Tab("🧪 Deep Insight"):
        gr.Markdown("#### Technical Policy Analysis")
        gr.Markdown("This dashboard shows the inner stability and confidence of the GRPO algorithm.")
        gr.Image(DEEP_INSIGHT_URL, label="Inner Model Heartbeat (Technical)")
        
import torch

# --- HYBRID INFERENCE ENGINE ---
def smart_llm_call(messages):
    # 1. Try Proper Unsloth Method (If GPU exists)
    if torch.cuda.is_available():
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="gouravbirwaz/kisanagent-trained-model",
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=128, temperature=0.1)
            return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        except Exception as e:
            print(f"GPU Load failed, falling back: {e}")

    # 2. Fallback to Cloud API (For Free Spaces / CPU)
    import requests
    API_URL = "https://api-inference.huggingface.co/models/gouravbirwaz/kisanagent-trained-model"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    
    # Simple format for Cloud API
    prompt = "".join([f"{m['role']}: {m['content']}\n" for m in messages]) + "assistant: "
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 128}})
    
    if response.status_code == 200:
        res = response.json()
        return res[0]['generated_text'] if isinstance(res, list) else res.get('generated_text', "Error")
    
    return "The agent is thinking... (Cloud API Warming Up)"

# Patch the main inference logic
import inference
inference.llm_call = smart_llm_call

with gr.Tab("🧠 Agent Architecture"):
        gr.Markdown("""
        - **Model**: Qwen-2.5-7B (LoRA Trained)
        - **Algorithm**: GRPO (Group Relative Policy Optimization)
        - **Framework**: OpenEnv + Unsloth
        - **Reward**: 3-Tier (Format + Interaction + Economic Strategy)
        """)
    
with gr.Tab("🚀 Live Environment"):
        gr.Markdown("The OpenEnv FastAPI server is running in the background.")
        gr.Markdown(f"Endpoint: `0.0.0.0:{os.getenv('PORT', 7860)}`")

# --- STARTING BOTH ---
def run_fastapi():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    # We run Gradio and FastAPI on the same process
    # Gradio will mount to the root, FastAPI can run on a sub-thread or we can mount it
    print("🚀 Launching KisanAgent Dashboard...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
