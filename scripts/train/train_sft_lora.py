import torch
import wandb
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

MODEL_NAME = os.path.join(PROJECT_ROOT, "model/Qwen3-4B-Instruct-2507")
DATA_FILE = os.path.join(PROJECT_ROOT, "processed_data/commentr_sft_data.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model/sft")
MAX_LENGTH = 1024

class WandbLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs, step=state.global_step)

wandb.init(
    project="qwen3-sft",
    name="commentr-reply-lora",
    config={
        "model": MODEL_NAME,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "epochs": 3,
        "lora_r": 16,
        "lora_alpha": 32,
    }
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None or tokenizer.pad_token == "":
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

def preprocess_function(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    
    texts = []
    for inst, inp, out in zip(instructions, inputs, outputs):
        messages = [
            {"role": "system", "content": inst},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": out}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    
    tokenized = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.train_test_split(test_size=0.1)
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.config.pad_token_id = tokenizer.pad_token_id
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "down_proj", "up_proj", "gate_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
    save_total_limit=3,
    remove_unused_columns=False,
    bf16=True,
    report_to="wandb",
    optim="adamw_torch",
    warmup_steps=100,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    callbacks=[WandbLoggingCallback()]
)

print("Starting SFT training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"SFT Model saved to {OUTPUT_DIR}")
