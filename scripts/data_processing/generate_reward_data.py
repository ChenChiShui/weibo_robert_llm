import json
import math
import re
from tqdm import tqdm

SFT_FILE = 'commentr_sft_data.jsonl'
REWARD_FILE = 'commentr_reward_data.jsonl'

MIN_LIKES = 0
MIN_CHARS = 2
MAX_CHARS = 500

AD_KEYWORDS = ['加群', '代购', '兼职', '刷单', '推广', '合作', '商务', '广告', '引流', '私聊']
SPAM_PATTERNS = [r'^[。\.]+$', r'^[！!]+$', r'^[？?]+$', r'^…+$', r'^[^\w\u4e00-\u9fa5]+$']

def remove_mentions(text):
    if not text:
        return ""
    return re.sub(r'@[\w\u4e00-\u9fa5]+', '', text).strip()

def is_spam_content(text):
    if not text:
        return True
    
    text = text.strip()
    
    if len(text) < MIN_CHARS:
        return True
    
    for pattern in SPAM_PATTERNS:
        if re.match(pattern, text):
            return True
    
    unique_chars = len(set(text))
    if len(text) > 10 and unique_chars < 3:
        return True
    
    for keyword in AD_KEYWORDS:
        if keyword in text:
            return True
    
    emoji_count = len(re.findall(r'\[.*?\]', text))
    if emoji_count > 10:
        return True
    
    return False

def calculate_reward(comment_item):
    """
    计算评论的 reward 分数（主要基于点赞数，使用对数缩放）
    """
    likes = comment_item.get('likes_count', 0)
    content = comment_item.get('content', '')
    
    content_len = len(content)
    
    if is_spam_content(content):
        return -1.0
    
    base_reward = math.log(likes + 1) / 2.5 - 0.5
    
    length_bonus = 0.0
    if 10 <= content_len <= 30:
        length_bonus = 0.05
    elif 31 <= content_len <= 80:
        length_bonus = 0.1
    elif content_len > 80:
        length_bonus = 0.05
    
    emoji_count = len(re.findall(r'\[.*?\]', content))
    if emoji_count > 0 and emoji_count <= 3:
        length_bonus += 0.02
    elif emoji_count > 5:
        length_bonus -= 0.05
    
    text_clean = remove_mentions(content)
    if len(text_clean) < MIN_CHARS:
        length_bonus -= 0.1
    
    final_reward = base_reward + length_bonus
    
    final_reward = max(-1.0, min(1.0, final_reward))
    
    return round(final_reward, 4)

def main():
    print("正在读取 SFT 数据集并生成 Reward Model 数据集...")
    
    all_data = []
    
    with open(SFT_FILE, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in, desc="Loading data"):
            data = json.loads(line.strip())
            
            input_text = data.get('input', '')
            output_text = data.get('output', '')
            likes = data.get('meta', {}).get('likes', 0)
            
            if not input_text or not output_text:
                continue
            
            comment_item = {
                'likes_count': likes,
                'content': output_text,
                'input': input_text,
                'output': output_text
            }
            
            reward = calculate_reward(comment_item)
            
            all_data.append({
                'data': data,
                'reward': reward
            })
    
    print(f"共加载 {len(all_data)} 条数据")
    
    reward_stats = {
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'min_reward': float('inf'),
        'max_reward': float('-inf'),
        'min_likes': float('inf'),
        'max_likes': 0
    }
    
    with open(REWARD_FILE, 'w', encoding='utf-8') as f_out:
        for item in all_data:
            reward = item['reward']
            
            data_entry = {
                "input": item['data']['input'],
                "output": item['data']['output'],
                "reward": reward
            }
            
            if reward > 0.2:
                reward_stats['positive'] += 1
            elif reward < -0.2:
                reward_stats['negative'] += 1
            else:
                reward_stats['neutral'] += 1
            
            reward_stats['min_reward'] = min(reward_stats['min_reward'], reward)
            reward_stats['max_reward'] = max(reward_stats['max_reward'], reward)
            
            likes = item['data']['meta']['likes']
            reward_stats['min_likes'] = min(reward_stats['min_likes'], likes)
            reward_stats['max_likes'] = max(reward_stats['max_likes'], likes)
            
            f_out.write(json.dumps(data_entry, ensure_ascii=False) + '\n')

    print("="*50)
    print(f"处理完成！")
    print(f"生成文件: {REWARD_FILE}")
    print(f"有效数据: {len(all_data)} 条")
    print(f"\nReward 统计:")
    print(f"  正向 (reward > 0.2): {reward_stats['positive']} 条 ({reward_stats['positive']/len(all_data)*100:.1f}%)")
    print(f"  负向 (reward < -0.2): {reward_stats['negative']} 条 ({reward_stats['negative']/len(all_data)*100:.1f}%)")
    print(f"  中性 (-0.2 <= reward <= 0.2): {reward_stats['neutral']} 条 ({reward_stats['neutral']/len(all_data)*100:.1f}%)")
    print(f"  最小值: {reward_stats['min_reward']}")
    print(f"  最大值: {reward_stats['max_reward']}")
    print(f"\n点赞数统计:")
    print(f"  最小点赞数: {reward_stats['min_likes']}")
    print(f"  最大点赞数: {reward_stats['max_likes']}")
    print("="*50)

if __name__ == "__main__":
    main()
