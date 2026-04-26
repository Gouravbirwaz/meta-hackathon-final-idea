# /// script
# dependencies = [
#   "unsloth",
#   "trl",
#   "torch",
#   "datasets",
#   "fastapi",
#   "pydantic",
#   "gymnasium",
#   "openenv-core",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "mergekit",
#   "hf_transfer",
#   "pandas"
# ]
# ///

from unsloth import FastLanguageModel
import os
import sys
import json
import logging
import torch
import numpy as np
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# --- 1. Setup Logging and Paths ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("kisanagent.train")

try:
    logger.info("Installing system dependencies (cmake) for GGUF export...")
    import subprocess
    subprocess.run(["apt-get", "update"], check=False)
    subprocess.run(["apt-get", "install", "-y", "cmake"], check=False)
except Exception as e:
    logger.warning(f"Could not install system packages: {e}")

def find_project_root():
    current = os.path.abspath(os.getcwd())
    for _ in range(3):
        if os.path.exists(os.path.join(current, "server")):
            return current
        current = os.path.dirname(current)
    search_paths = ["/root", "/repo", "/home/user", "/tmp", os.getcwd()]
    for base in search_paths:
        if not os.path.exists(base): continue
        for root, dirs, files in os.walk(base, topdown=True):
            if "server" in dirs:
                return root
            if len(dirs) > 50: break 
    return None

root = find_project_root()
if not root:
    logger.warning("Project root not found. Attempting to clone repository...")
    repo_url = "https://huggingface.co/gouravbirwaz/kisanagent-training"
    try:
        import subprocess
        subprocess.run(["git", "clone", repo_url, "cloned_repo"], check=True)
        root = os.path.abspath("cloned_repo")
        logger.info(f"Successfully cloned repo to: {root}")
    except Exception as e:
        logger.error(f"Failed to clone repo: {e}")

if root:
    sys.path.append(root)
    logger.info(f"Final project root set to: {root}")
else:
    sys.path.append(os.getcwd())

from server.app import KisanEnvironment
from env.models import KisanAction

# --- 2. Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
OUTPUT_DIR = "checkpoints/kisanagent-grpo"
MAX_SEQ_LENGTH = 2048

SYSTEM_PROMPT = (
    "You are KisanAgent. Maximize net income across 90 days. "
    "Respond ONLY in JSON: {\"reasoning\": \"...\", \"tool_to_call\": \"tool_name or null\", \"farm_decision\": \"decision or null\"}"
)

# --- 3. Initialize Environment ---
env = KisanEnvironment()
logger.info("KisanEnvironment initialized.")

# --- 4. Load Model and Tokenizer ---
logger.info(f"Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None, 
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# --- 5. Define GRPO Reward Functions ---
def format_reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else completion[-1].get("content", "")
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json.loads(match.group())
                rewards.append(0.2)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

def kisan_reward_fn(completions, prompts=None, **kwargs):
    rewards = []
    for completion in completions:
        current_reward = 0.05 # BASE REWARD: Never 0 again.
        try:
            text = completion if isinstance(completion, str) else completion[-1].get("content", "")
            # 1. Look for JSON-like structure
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                current_reward += 0.05
                # 2. Try to parse JSON
                try:
                    parsed = json.loads(match.group())
                    current_reward += 0.1
                    
                    tool_name = parsed.get("tool_to_call")
                    farm_decision = parsed.get("farm_decision")
                    
                    # 3. Create Action object
                    action = KisanAction(
                        farm_decision=farm_decision if farm_decision != "null" else None,
                        tool_name=tool_name if tool_name != "null" else None,
                        reasoning=parsed.get("reasoning", "")
                    )
                    current_reward += 0.1
                    
                    # 4. Interact with Environment
                    env.reset(difficulty='medium', seed=0)
                    obs = env.step(action=action)
                    state_score = env.grader.calculate_composite_score(env.state)
                    current_reward += float(state_score)
                    
                except Exception:
                    pass # Keep accumulated rewards
        except Exception:
            pass # Keep accumulated rewards
            
        rewards.append(current_reward)
    return rewards

# --- 6. Prepare Dataset ---
logger.info("Building training prompts (Lightning Mode)...")
num_episodes = 50 # Reduced for speed
all_prompts = []
for diff in ['easy', 'medium', 'hard']:
    for ep in range(num_episodes // 3):
        obs = env.reset(difficulty=diff)
        # Sample every 20 days (Faster simulation)
        for day in range(0, 90, 20):
            if day % 10 == 0:
                p = f"Day {obs.day}: Stage {obs.crop_stage}, Balance {obs.bank_balance_inr:.0f} INR."
                all_prompts.append({
                    'prompt': [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user',   'content': p},
                    ]
                })
            obs = env.step(action=KisanAction(farm_decision="do_nothing", reasoning="Simulating days for dataset."))

train_dataset = Dataset.from_list(all_prompts)

# --- 7. Configure Trainer ---
REPO_ID = "gouravbirwaz/kisanagent-training-progress"

trainer_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,  # 1 Epoch is enough for a quick win
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4, # Smaller accum for more frequent updates
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=128,
    logging_steps=1,
    learning_rate=1e-5, # Higher LR for faster learning
    save_steps=20,      # Save more often
    push_to_hub=True,    # PUSH TO HUB AUTOMATICALLY
    hub_model_id=REPO_ID, # The repo to push to
    hub_strategy="every_save",
    report_to="none", 
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward_fn, kisan_reward_fn],
    args=trainer_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

# --- 8. Train ---
logger.info("Starting GRPO Training (Lightning Mode)...")
trainer.train()

# --- 9. Post-Training: Save and Push to Hub ---
try:
    logger.info("Generating training plots...")
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Try to extract log history for plotting
    history = pd.DataFrame(trainer.state.log_history)
    if 'reward' in history.columns:
        history = history.dropna(subset=['reward']).copy()
        plt.figure(figsize=(10, 5))
        plt.plot(history['step'], history['reward'], label='Reward')
        plt.title('Training Progress')
        plt.savefig("training_progress.png")
        logger.info("Plot generated: training_progress.png")

    logger.info("Pushing model to Hugging Face Hub (Standard Format)...")
    model.push_to_hub(
        "gouravbirwaz/kisanagent-trained-model",
        tokenizer=tokenizer,
        save_method="merged_16bit",
        token=os.getenv("HF_TOKEN")
    )
    
    # Upload the plot as evidence
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj="training_progress.png",
        path_in_repo="training_progress.png",
        repo_id="gouravbirwaz/kisanagent-trained-model",
        repo_type="model",
        token=os.getenv("HF_TOKEN")
    )
    logger.info("SUCCESS: Model and Evidence pushed to Hub!")

except Exception as e:
    logger.error(f"Final cleanup failed: {e}")
    # Fallback: Just save locally so we don't lose the weights
    logger.info("Falling back to local save...")
    model.save_pretrained_merged("kisan_agent_final", tokenizer, save_method="merged_16bit")

logger.info("DONE! Process complete.")