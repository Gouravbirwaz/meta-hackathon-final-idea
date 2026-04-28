import os
import json
import torch
from inference import run_episode
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
MODEL_NAME = "gouravbirwaz/kisanagent-trained-model"

print(f"🚀 Loading Trained KisanAgent using PROPER method (Unsloth)...")

try:
    from unsloth import FastLanguageModel
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True, # Efficient 4-bit loading
    )
    FastLanguageModel.for_inference(model) # Enable 2x faster inference
    
    def trained_llm_call(messages):
        # Format using the proper chat template
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda") # This REQUIRES a GPU

        outputs = model.generate(
            input_ids, 
            max_new_tokens=128,
            temperature=0.1,
            top_p=0.9,
            use_cache=True
        )
        
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return response

    print("✅ Model loaded successfully on GPU!")

except ImportError:
    print("⚠️ Unsloth not installed. Please install it to use the PRO method.")
    exit(1)
except Exception as e:
    print(f"❌ Error loading on local hardware: {e}")
    print("\n💡 NOTE: This 'Proper Method' requires an NVIDIA GPU and 'unsloth' installed.")
    print("For the Hackathon demo, we will use the Cloud Fallback in the Space.")
    exit(1)

# Patch the inference loop
import inference
inference.client = None 
inference.llm_call = trained_llm_call

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌟 KISANAGENT TRAINED MODEL DEMO (PRO) 🌟")
    print("="*50)
    run_episode(difficulty="medium", verbose=True)
