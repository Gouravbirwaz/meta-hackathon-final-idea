import os
import json
import requests
import time
from inference import run_episode, SYSTEM_PROMPT
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
# This points to your online model on Hugging Face
MODEL_NAME = "gouravbirwaz/kisanagent-trained-model"
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ ERROR: HF_TOKEN not found in .env!")
    exit(1)

print(f"🚀 Connecting to Trained KisanAgent on HF Cloud: {MODEL_NAME}...")

def trained_llm_call(messages, retries=5):
    # Use the official HF Inference API
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Format for Qwen
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 128,
            "temperature": 0.1,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            
            # 503 means model is loading
            if response.status_code == 503:
                print(f"⏳ Model is loading on HF Cloud... waiting 20s (Attempt {attempt+1})")
                time.sleep(20)
                continue
            
            if response.status_code != 200:
                print(f"❌ HF API Error ({response.status_code}): {response.text}")
                return json.dumps({"reasoning": "Cloud is busy", "farm_decision": "do_nothing"})
                
            result = response.json()
            # Handle list or dict response
            text = result[0].get("generated_text", "") if isinstance(result, list) else result.get("generated_text", "")

            # Extract JSON
            if "{" in text:
                text = text[text.find("{"):text.rfind("}")+1]
            return text
            
        except Exception as e:
            print(f"⚠️ Connection Error: {e}")
            time.sleep(2)
            
    return json.dumps({"reasoning": "Timeout", "farm_decision": "do_nothing"})

# Patch the inference loop
import inference
inference.client = None 
inference.llm_call = trained_llm_call

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌟 KISANAGENT TRAINED MODEL DEMO (CLOUD) 🌟")
    print("="*50)
    run_episode(difficulty="medium", verbose=True)
