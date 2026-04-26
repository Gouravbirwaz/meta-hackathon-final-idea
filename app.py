import gradio as gr
import os
import pandas as pd
import torch
import requests
from fastapi import FastAPI
import uvicorn
from server.app import KisanEnvironment
from openenv.core.env_server import create_web_interface_app
from env.models import KisanAction, FarmerObservation

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

# --- 2. HYBRID INFERENCE ENGINE ---
def smart_llm_call(messages):
    if torch.cuda.is_available():
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(model_name="gouravbirwaz/kisanagent-trained-model", load_in_4bit=True)
            FastLanguageModel.for_inference(model)
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=128, temperature=0.1)
            return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        except Exception: pass

    # Cloud Fallback
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        return "❌ Error: HF_TOKEN is missing. Please add it to your Space Secrets to enable inference."
        
    API_URL = "https://api-inference.huggingface.co/models/gouravbirwaz/kisanagent-trained-model"
    headers = {"Authorization": f"Bearer {hf_token}"}
    prompt = "".join([f"{m['role']}: {m['content']}\n" for m in messages]) + "assistant: "
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 128}}, timeout=10)
        if response.status_code == 200:
            res = response.json()
            return res[0]['generated_text'] if isinstance(res, list) else res.get('generated_text', "Thinking...")
    except: pass
    return "Agent is connecting to cloud..."

# Patch inference
import inference
inference.llm_call = smart_llm_call

# --- 3. CUSTOM DASHBOARD UI ---
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

    with gr.Tab("🚀 Live Environment"):
        gr.Markdown("### Official OpenEnv Interaction")
        gr.Markdown("Interact with the KisanEnv directly using the official buttons and state viewer below.")
        # Pointing to the absolute path from the Space root
        gr.HTML('<iframe src="/env-ui/" width="100%" height="800px" style="border:2px solid #2ecc71; border-radius:10px;"></iframe>')

    with gr.Tab("🧠 Architecture"):
        gr.Markdown("- **Model**: Qwen-2.5-7B (LoRA) | **RL**: GRPO | **Framework**: OpenEnv")

# --- 4. MOUNTING & LAUNCHING ---
main_app = FastAPI()

# A. Get the official UI app
openenv_ui_app = create_web_interface_app(KisanEnvironment, KisanAction, FarmerObservation)

# B. IMPORTANT: Set the root_path so the sub-app knows it lives at /env-ui
openenv_ui_app.root_path = "/env-ui"

# C. Mount the official UI using FastAPI's native mount
main_app.mount("/env-ui", openenv_ui_app)

# D. Mount our custom Gradio dashboard at the root
main_app = gr.mount_gradio_app(main_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(main_app, host="0.0.0.0", port=7860)
