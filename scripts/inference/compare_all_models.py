import torch
import os
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm  # 建议安装: pip install tqdm

# ================= 配置区域 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# 请确保这些路径与你的实际路径一致
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/Qwen3-4B-Instruct-2507")
SFT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/sft")
DPO_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/dpo")
TEST_DATA_FILE = os.path.join(PROJECT_ROOT, "processed_data/commentr_sft_data.jsonl")
OUTPUT_FILE = "model_comparison_results.jsonl"

# ================= 工具函数 =================
def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        # 如果目录下直接就是 adapter_config.json，说明本身就是 checkpoints
        if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
            return checkpoint_dir
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[1]), reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])

def clear_gpu_memory():
    """强制清理显存"""
    gc.collect()
    torch.cuda.empty_cache()

# ================= 核心：模型加载逻辑 =================
def load_specific_model(model_type="base"):
    """
    根据类型加载模型，包含正确的 Merge 逻辑
    model_type: "base", "sft", "dpo"
    """
    print(f"\n[System] Initializing {model_type.upper()} Model...")
    
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. 加载 Base Model
    print(f"Loading Base Model from: {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    if model_type == "base":
        return model, tokenizer

    # 3. 处理 SFT 阶段 (SFT 和 DPO 都需要这一步)
    sft_checkpoint = get_latest_checkpoint(SFT_MODEL_PATH)
    if sft_checkpoint:
        print(f"Loading SFT Adapter from: {sft_checkpoint}")
        model = PeftModel.from_pretrained(model, sft_checkpoint)
        print("Merging SFT Adapter into Base Model...")
        model = model.merge_and_unload() # 【关键】先把 SFT 融进去
    else:
        print("Warning: SFT checkpoint not found! Proceeding with Base Model.")

    if model_type == "sft":
        return model, tokenizer

    # 4. 处理 DPO 阶段
    if model_type == "dpo":
        dpo_checkpoint = get_latest_checkpoint(DPO_MODEL_PATH)
        if dpo_checkpoint:
            print(f"Loading DPO Adapter from: {dpo_checkpoint}")
            model = PeftModel.from_pretrained(model, dpo_checkpoint)
            print("Merging DPO Adapter for Inference...")
            model = model.merge_and_unload() # 【关键】再把 DPO 融进去
        else:
            print("Error: DPO checkpoint not found!")
    
    return model, tokenizer

def generate_batch_responses(model, tokenizer, dataset):
    """批量生成回复"""
    responses = []
    
    for item in tqdm(dataset, desc="Generating"):
        instruction = item.get('instruction', '')
        user_input = item.get('input', '')
        
        # 构建 Prompt
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        responses.append(response_text)
        
    return responses

# ================= 主程序 =================
def main(num_samples=10):
    # 1. 准备数据
    print(f"Loading data from {TEST_DATA_FILE}...")
    with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
        full_data = [json.loads(line) for line in f]
    
    # 取前 N 条做测试
    test_data = full_data[:num_samples]
    print(f"Selected {len(test_data)} samples for comparison.")

    results_storage = {
        "base": [],
        "sft": [],
        "dpo": []
    }

    # 2. 依次跑三个模型 (Load -> Generate -> Unload)
    # 顺序：Base -> SFT -> DPO
    for mode in ["base", "sft", "dpo"]:
        print(f"\n{'='*20} Processing {mode.upper()} {'='*20}")
        
        model, tokenizer = load_specific_model(mode)
        results = generate_batch_responses(model, tokenizer, test_data)
        results_storage[mode] = results
        
        # 清理显存
        del model
        del tokenizer
        clear_gpu_memory()
        print(f"{mode.upper()} finished and unloaded.")

    # 3. 打印对比结果并保存
    print(f"\n{'='*40} FINAL COMPARISON {'='*40}")
    
    final_records = []
    
    for i, item in enumerate(test_data):
        instruction = item.get('instruction', '')
        user_input = item.get('input', '')
        ground_truth = item.get('output', '')
        
        base_res = results_storage["base"][i]
        sft_res = results_storage["sft"][i]
        dpo_res = results_storage["dpo"][i]
        
        # 打印到控制台
        print(f"\n[Sample {i+1}]")
        print(f"User Input: {user_input}")
        print(f"-" * 20)
        print(f"Base Model : {base_res}")
        print(f"SFT Model  : {sft_res}")
        print(f"DPO Model  : {dpo_res}") # 重点观察这个是否比 SFT 更好
        print(f"-" * 80)
        
        # 准备保存
        final_records.append({
            "instruction": instruction,
            "input": user_input,
            "ground_truth": ground_truth,
            "base": base_res,
            "sft": sft_res,
            "dpo": dpo_res
        })

    # 4. 保存文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in final_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\nComparison completed! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    import sys
    # 可以通过命令行传参控制测试样本数： python script.py 20
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n)