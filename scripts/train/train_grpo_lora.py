import os
import torch
import warnings
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning)

# ================= 1. 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

SFT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/sft_merged_model")
REWARD_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/reward/checkpoint-1600")

DATA_FILE = os.path.join(PROJECT_ROOT, "processed_data/commentr_dpo_reward_data.jsonl") 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model/grpo")

# ================= 2. 加载 Reward Model =================
print("Loading Reward Model to GPU...")
reward_device = "cuda:0" 

def load_reward_model(model_path, device):
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_lora:
        print("🔍 检测到 LoRA Reward Model，正在合并加载...")
        base_model_path = SFT_MODEL_PATH 
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=1,
            torch_dtype=torch.bfloat16, 
            device_map=device,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        print("🔍 加载全量 Reward Model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
    model.eval()
    return model

torch.cuda.empty_cache()

reward_model = load_reward_model(REWARD_MODEL_PATH, reward_device)
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)

if reward_tokenizer.pad_token is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

# =================================================================
# 3. 奖励函数 (修复重复格式化和标准化问题)
# =================================================================
def reward_func(prompts, completions, **kwargs):
    # 打印预览
    print("\n" + "="*20 + " 模型生成样本预览 " + "="*20)
    print(f"🔹 Prompt: {prompts[0]}")
    print(f"🔸 Completion: {completions[0]}")
    print("="*60 + "\n")

    # 从格式化后的 prompt 中提取原始内容
    # 格式: <|im_start|>user\n{original_content}<|im_end|>\n<|im_start|>assistant\n
    original_prompts = []
    for p in prompts:
        if "<|im_start|>user\n" in p and "<|im_end|>" in p:
            start = p.find("<|im_start|>user\n") + len("<|im_start|>user\n")
            end = p.find("<|im_end|>", start)
            original_content = p[start:end]
            original_prompts.append(original_content)
        else:
            original_prompts.append(p)

    inputs = []
    for p, c in zip(original_prompts, completions):
        messages = [{"role": "user", "content": p}, {"role": "assistant", "content": c}]
        try:
            text = reward_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except:
            text = f"User: {p}\nAssistant: {c}"
        inputs.append(text)

    tokenized_inputs = reward_tokenizer(
        inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024
    ).to(reward_device)

    with torch.no_grad():
        outputs = reward_model(**tokenized_inputs)
        scores = outputs.logits.squeeze(-1)

    # 添加 4.0 的 offset 来调整奖励分数
    scores = scores + 4.0

    return scores.tolist()

# ================= 4. 数据格式化逻辑 (关键修改) =================
def format_data(example, tokenizer):
    # 保留原始 prompt 用于 reward function
    # 将原始文本包装成 Chat 格式用于生成：
    # "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    messages = [{"role": "user", "content": example["prompt"]}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return {"prompt": formatted_prompt, "original_prompt": example["prompt"]}

# ================= 5. 配置 GRPO =================
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    run_name="commentr-reply-grpo-4b-full",
    learning_rate=1e-6,
    num_train_epochs=1,
    per_device_train_batch_size=16, 
    gradient_accumulation_steps=2, 
    beta=0.5,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    bf16=True,
    torch_compile=False,
    use_vllm=False, 
    num_generations=8,
    max_completion_length=512,
    logging_steps=1,
    save_steps=100,
    report_to="wandb",
)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=32, lora_alpha=64, lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "down_proj", "up_proj", "gate_proj"]
)

# ================= 6. 主流程 =================
def main():
    # -------------------------------------------------------------
    # 步骤 1: 先加载 Tokenizer (必须在处理数据前)
    # -------------------------------------------------------------
    print(f"Loading Tokenizer & Checking EOS alignment...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
    
    # 检查 Chat Template
    if tokenizer.chat_template is None:
        print("⚠️ Tokenizer 没有默认 Chat Template，使用 ChatML 格式")
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # EOS 对齐
    correct_eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if correct_eos_id == tokenizer.unk_token_id:
        correct_eos_id = tokenizer.eos_token_id
        print(f"⚠️ 未找到 <|im_end|>，回退使用默认 EOS: {correct_eos_id}")
    else:
        print(f"🔧 检测到 <|im_end|> (ID: {correct_eos_id})，强制将其设为统一 EOS")

    tokenizer.eos_token_id = correct_eos_id
    tokenizer.pad_token_id = correct_eos_id

    # -------------------------------------------------------------
    # 步骤 2: 加载数据并应用格式化
    # -------------------------------------------------------------
    print(f"Loading Dataset from {DATA_FILE}...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    if "input" in dataset.column_names and "prompt" not in dataset.column_names:
        dataset = dataset.rename_column("input", "prompt")
    
    print("⏳ 正在应用 Chat Template 格式化...")
    # 这里是关键：将 dataset 里的 prompt 变成对话格式
    dataset = dataset.map(lambda x: format_data(x, tokenizer), batched=False)
    
    print(f"✅ 数据处理完成，样本预览: {dataset[0]['prompt'][-100:]}")

    # -------------------------------------------------------------
    # 步骤 3: 加载策略模型
    # -------------------------------------------------------------
    print(f"Loading 4B Policy Model in FULL BF16...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH,
        trust_remote_code=True,
        device_map={"": 0}, 
        torch_dtype=torch.bfloat16
    )

    policy_model.config.eos_token_id = correct_eos_id
    policy_model.config.pad_token_id = correct_eos_id

    # 处理 Generation Config
    if policy_model.generation_config is None:
        print("⚠️ 模型缺少 generation_config，正在创建...")
        policy_model.generation_config = GenerationConfig.from_model_config(policy_model.config)
    
    policy_model.generation_config.eos_token_id = correct_eos_id
    policy_model.generation_config.pad_token_id = correct_eos_id
    # GRPO 训练初期建议设为 1.0，避免干扰
    policy_model.generation_config.repetition_penalty = 1.2
    print(f"✅ GenerationConfig 已修正: EOS ID = {policy_model.generation_config.eos_token_id}, Repetition Penalty = 1.0")

    # -------------------------------------------------------------
    # 步骤 4: 开始训练
    # -------------------------------------------------------------
    trainer = GRPOTrainer(
        model=policy_model, 
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    print("Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()