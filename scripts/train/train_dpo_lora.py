import torch
import wandb
import os
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback
)
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig

# ================= 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

MODEL_NAME = os.path.join(PROJECT_ROOT, "model/Qwen3-4B-Instruct-2507")
SFT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/sft")
DATA_FILE = os.path.join(PROJECT_ROOT, "processed_data/commentr_dpo_reward_data.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model/dpo")
MAX_LENGTH = 1024

# ================= 自动加载 SFT 权重 =================
def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir): return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[1]), reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])

LATEST_SFT_CHECKPOINT = get_latest_checkpoint(SFT_MODEL_PATH)
MODEL_TO_LOAD = LATEST_SFT_CHECKPOINT if LATEST_SFT_CHECKPOINT else MODEL_NAME
print(f"Loading model from: {MODEL_TO_LOAD}")

# ================= WandB 回调 =================
class WandbLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            safe_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            wandb.log(safe_logs, step=state.global_step)

wandb.init(
    project="qwen3-dpo",
    name="commentr-reply-dpo",
    config={"model": MODEL_TO_LOAD}
)

# ================= Tokenizer & Data =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

def load_and_convert_reward_data(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

def preprocess_function(examples):
    prompts = examples["prompt"]
    chosen = examples["chosen"]
    rejected = examples["rejected"]
    
    new_prompts = []
    new_chosen = []
    new_rejected = []
    
    for prompt, chosen_text, rejected_text in zip(prompts, chosen, rejected):
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        new_prompts.append(prompt_text)
        
    
    return {
        "prompt": new_prompts,
        "chosen": new_chosen,
        "rejected": new_rejected
    }

dataset = load_and_convert_reward_data(DATA_FILE)
dataset = dataset.train_test_split(test_size=0.1)
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# ================= Model & LoRA =================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_TO_LOAD,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.config.pad_token_id = tokenizer.pad_token_id
model.gradient_checkpointing_enable() 

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none"
)

# ================= DPO Config (关键修改区域) =================
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    
    # --- 修改开始 ---
    # 如果你真的是指 epoch，把下面改成 strategy="epoch" (但 DPO 通常只跑 1-3 epoch，没法每隔 200 epoch 保存)
    # 这里我假设你指的是 step (步数)
    eval_strategy="steps",
    eval_steps=200,             # 每 200 步评估一次
    save_strategy="steps",      # 按步数保存
    save_steps=200,             # 每 200 步保存一次
    save_total_limit=None,      # 【重点】设置为 None，表示保留所有 checkpoint，不删除旧的！
    # --- 修改结束 ---
    
    logging_steps=10,
    bf16=True
    report_to="wandb",
    optim="paged_adamw_32bit",  # 记得 pip install bitsandbytes
    max_length=MAX_LENGTH,
    max_prompt_length=512,
    remove_unused_columns=False,
    gradient_checkpointing=True
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    peft_config=peft_config,
    args=dpo_config,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    callbacks=[WandbLoggingCallback()]
)

print("Starting DPO training...")
dpo_trainer.train()

dpo_trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"DPO Model saved to {OUTPUT_DIR}")