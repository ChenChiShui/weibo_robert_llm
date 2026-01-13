import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/Qwen3-4B-Instruct-2507")
SFT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/sft/checkpoint-400")

def load_sft_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    if tokenizer.pad_token is None or tokenizer.pad_token == "":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, SFT_MODEL_PATH)
    model = model.merge_and_unload()
    
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
    print("Loading SFT LoRA model...")
    model, tokenizer = load_sft_model()
    print("Model loaded successfully!")
    print("\n" + "="*50)
    print("SFT LoRA Interactive Inference")
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
    print("Loading SFT LoRA model...")
    model, tokenizer = load_sft_model()
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
