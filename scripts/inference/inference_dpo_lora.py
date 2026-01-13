import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/Qwen3-4B-Instruct-2507")
SFT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/sft")
DPO_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/dpo")

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        return checkpoint_dir
    checkpoints.sort(key=lambda x: int(x.split('-')[1]), reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])

def load_dpo_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    if tokenizer.pad_token is None or tokenizer.pad_token == "":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading base model from: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    sft_checkpoint = get_latest_checkpoint(SFT_MODEL_PATH)
    if sft_checkpoint:
        print(f"Loading and Merging SFT LoRA from: {sft_checkpoint}")
        # 加载 SFT Adapter
        model = PeftModel.from_pretrained(base_model, sft_checkpoint)
        # 【关键步骤】立刻合并！把 SFT 权重永久加到 base_model 参数里
        model = model.merge_and_unload()
    else:
        print("No SFT checkpoint found, using base model")
        model = base_model

    # ================= 2. 加载 DPO LoRA =================
    dpo_checkpoint = get_latest_checkpoint(DPO_MODEL_PATH)
    if dpo_checkpoint:
        print(f"Loading DPO LoRA from: {dpo_checkpoint}")
        # 在已经包含 SFT 权重的 model 上加载 DPO Adapter
        model = PeftModel.from_pretrained(model, dpo_checkpoint)
        
        # 可选：如果你追求推理速度，可以把 DPO 也合并了
        # 如果你想保留切换能力，可以不合并
        print("Merging DPO LoRA for faster inference...")
        model = model.merge_and_unload()
    else:
        print("Warning: No DPO checkpoint found!")

    return model, tokenizer

def generate_response(model, tokenizer, user_input, max_new_tokens=512, temperature=0.7, top_p=0.9):
    messages = [
        {"role": "user", "content": user_input}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def interactive_inference():
    print("Loading DPO LoRA model...")
    model, tokenizer = load_dpo_model()
    print("Model loaded successfully!")
    print("\n" + "="*50)
    print("DPO LoRA Interactive Inference")
    print("Enter 'quit' or 'exit' to stop")
    print("="*50 + "\n")
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not user_input:
            continue
        
        response = generate_response(model, tokenizer, user_input)
        print(f"Assistant: {response}\n")

def batch_inference(test_inputs, output_file=None):
    print("Loading DPO LoRA model...")
    model, tokenizer = load_dpo_model()
    print("Model loaded successfully!")
    
    results = []
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"Processing {i}/{len(test_inputs)}...")
        
        response = generate_response(model, tokenizer, user_input)
        
        result = {
            "input": user_input,
            "output": response
        }
        results.append(result)
        
        print(f"Input: {user_input}")
        print(f"Output: {response}")
        print("-" * 50 + "\n")
    
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        test_inputs = [
            "你好，最近怎么样？",
            "今天天气真好！",
            "你能帮我写一首诗吗？",
            "什么是人工智能？",
            "给我讲个笑话吧"
        ]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        batch_inference(test_inputs, output_file)
    else:
        interactive_inference()
