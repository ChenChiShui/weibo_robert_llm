import torch
import wandb
import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model
from trl import RewardTrainer, RewardConfig

# ================= 1. 补全路径配置 (之前缺失的部分) =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# 如果上面路径不对，请手动指定你的项目根目录，例如：
# PROJECT_ROOT = "/root/autodl-tmp/robort_llm/robert_llm"
MODEL_NAME = os.path.join(PROJECT_ROOT, "model/Qwen3-4B-Instruct-2507")
# 确保这个文件路径是真实存在的
DATA_FILE = os.path.join(PROJECT_ROOT, "processed_data/commentr_dpo_reward_data.jsonl") 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model/reward")
MAX_LENGTH = 1024

print(f"Model Path: {MODEL_NAME}")
print(f"Data File: {DATA_FILE}")

# ================= 2. WandB 初始化 =================
wandb.init(
    project="qwen3-reward",
    name="commentr-reply-reward-pairwise",
    config={"model": MODEL_NAME}
)

# ================= 3. 数据加载与预处理 =================
def load_data_pairwise(data_file):
    """
    读取你的 jsonl 文件，保持 Pairwise 格式
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"找不到数据文件: {data_file}")
        
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # trl 的 RewardTrainer 需要 dataset 有 chosen 和 rejected 列
    # 你的数据里已经是 prompt, chosen, rejected 了，直接转 Dataset 即可
    return Dataset.from_list(data)

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading and processing data...")
dataset = load_data_pairwise(DATA_FILE)
# 划分训练集和测试集
dataset = dataset.train_test_split(test_size=0.1)
# 移除不需要的列，只保留 prompt, chosen, rejected
dataset = dataset.remove_columns(['meta'])
# RewardTrainer 会自动处理 tokenization，不需要预处理

# ================= 4. 模型加载 =================
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=1,           # Reward Model 输出一个标量分数
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    local_files_only=True
)
model.config.pad_token_id = tokenizer.pad_token_id
model.gradient_checkpointing_enable() 

# 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # 注意这里是 Sequence Classification
    inference_mode=False,
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
# 注意：RewardTrainer 会自动处理 peft_model 的封装，这里不需要手动 get_peft_model，
# 只需要把 peft_config 传给 trainer 即可。

# ================= 5. 训练参数配置 =================
training_args = RewardConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # 显存不够改小，够的话改大
    gradient_accumulation_steps=8,     # 累计梯度
    num_train_epochs=1,                # Reward Model 很容易过拟合，1个 epoch 通常足够
    learning_rate=1e-5,                # 学习率要低
    remove_unused_columns=False,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,                     # 频繁验证，防止过拟合
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    report_to="wandb",
    max_length=MAX_LENGTH,
    gradient_checkpointing=True,
)

# ================= 6. 开始训练 =================
trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
)

print("Starting Reward Model training...")
trainer.train()

# 保存
print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)