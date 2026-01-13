import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/Qwen3-4B-Instruct-2507")
SFT_ADAPTER_PATH = os.path.join(PROJECT_ROOT, "model/sft/checkpoint-1200")
NEW_SFT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/sft_merged_model")

print(f"Loading base model from {BASE_MODEL_PATH}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
)

print(f"Loading SFT adapter from {SFT_ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)

print("Merging SFT adapter into base model...")
model = model.merge_and_unload()

print(f"Saving new SFT base model to {NEW_SFT_MODEL_PATH}...")
os.makedirs(NEW_SFT_MODEL_PATH, exist_ok=True)
model.save_pretrained(NEW_SFT_MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.save_pretrained(NEW_SFT_MODEL_PATH)

print("SFT Merge Complete! Now use this path for GRPO.")
