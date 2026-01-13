import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/Qwen3-4B-Instruct-2507")
REWARD_MODEL_PATH = os.path.join(PROJECT_ROOT, "model/reward")

def get_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[1]), reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])

def load_reward_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    if tokenizer.pad_token is None or tokenizer.pad_token == "":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    reward_checkpoint = get_latest_checkpoint(REWARD_MODEL_PATH)
    if reward_checkpoint:
        print(f"Loading Reward LoRA from: {reward_checkpoint}")
        model = PeftModel.from_pretrained(base_model, reward_checkpoint)
        model = model.merge_and_unload()
    else:
        print("No reward checkpoint found, using base model")
        model = base_model
    
    return model, tokenizer

def compute_reward(model, tokenizer, user_input, assistant_output):
    messages = [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_output}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model(**model_inputs)
        reward = outputs.logits.squeeze().item()
    
    return reward

def interactive_reward_evaluation():
    print("Loading reward model...")
    model, tokenizer = load_reward_model()
    print("Model loaded successfully!")
    print("\n" + "="*50)
    print("Reward Model Interactive Evaluation")
    print("Enter 'quit' or 'exit' to stop")
    print("="*50 + "\n")
    
    while True:
        user_input = input("User Input: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not user_input:
            continue
        
        assistant_output = input("Assistant Output: ").strip()
        
        if not assistant_output:
            continue
        
        reward = compute_reward(model, tokenizer, user_input, assistant_output)
        print(f"Reward Score: {reward:.4f}\n")

def batch_reward_evaluation(data_file, output_file=None):
    print("Loading reward model...")
    model, tokenizer = load_reward_model()
    print("Model loaded successfully!")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    results = []
    
    for i, item in enumerate(data, 1):
        print(f"Processing {i}/{len(data)}...")
        
        user_input = item['input']
        assistant_output = item['output']
        
        reward = compute_reward(model, tokenizer, user_input, assistant_output)
        
        result = {
            "input": user_input,
            "output": assistant_output,
            "predicted_reward": reward,
            "actual_reward": item.get('reward', None)
        }
        results.append(result)
        
        print(f"Input: {user_input}")
        print(f"Output: {assistant_output}")
        print(f"Predicted Reward: {reward:.4f}")
        if 'reward' in item:
            print(f"Actual Reward: {item['reward']:.4f}")
        print("-" * 50 + "\n")
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        data_file = sys.argv[2] if len(sys.argv) > 2 else "processed_data/commentr_reward_data.jsonl"
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        batch_reward_evaluation(data_file, output_file)
    else:
        interactive_reward_evaluation()
