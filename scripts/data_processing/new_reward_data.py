import json
import math
import re
import random
from collections import defaultdict
from tqdm import tqdm
import ijson  # 必须安装: pip install ijson

# ================= 配置路径 =================
POSTS_FILE = 'data/Posts_fixed.json'       # 你的帖子数据
COMMENTS_FILE = 'data/Comments.json'       # 你的评论数据
OUTPUT_FILE = 'processed_data/commentr_dpo_reward_data.jsonl' # 输出文件

# ================= 评分与筛选阈值 =================
MIN_LIKES_FOR_CHOSEN = 2      # 只有大于2个赞的才能当 Chosen (正例)
MIN_SCORE_MARGIN = 0.5        # 正负例的分差必须超过这个值 (保证对比足够明显)
SPAM_CHARS_THRESHOLD = 2      # 极短的视为垃圾
MIN_SIMILARITY_THRESHOLD = 0.8  # 文本相似度阈值，超过此值则过滤

# ================= 辅助函数 =================

def remove_mentions(text):
    if not text: return ""
    return re.sub(r'@[\w\u4e00-\u9fa5]+', '', text).strip()

def clean_text(text):
    if not text: return ""
    # 去除换行、多余空格和@
    return remove_mentions(text.replace('\n', ' ').replace('\r', ' ').strip())

def is_spam(text):
    """
    垃圾过滤：我们不希望 spam 进入 Chosen，
    但如果它不是广告而只是无聊，可以考虑作为 Rejected。
    这里我们只过滤掉纯乱码或广告。
    """
    if not text or len(text) < SPAM_CHARS_THRESHOLD: return True
    # 纯标点
    if re.match(r'^[。\.]+$', text): return True
    # 广告关键词
    if any(k in text for k in ['加群', '兼职', '刷单', '推广', 'http']): return True
    return False

def calculate_reward_score(likes, content):
    """
    计算奖励分数 (Heuristic Reward Function)
    这不仅用于排序，也会写入文件供后续 RM 训练参考
    """
    if is_spam(content):
        return -10.0
    
    # 1. 基础分：点赞的对数 (Log Scale)
    # log(1)=0, log(10)=2.3, log(100)=4.6
    score = math.log(likes + 1)
    
    # 2. 长度惩罚/奖励 (中间长度最好)
    length = len(content)
    if length < 5: 
        score -= 1.0  # 太短通常没营养
    elif 10 <= length <= 60: 
        score += 0.5  # 黄金长度
        
    # 3. Emoji 奖励 (微博特色)
    if '[' in content and ']' in content:
        score += 0.2
        
    return round(score, 4)

def format_prompt(post_data):
    """构建输入 Prompt"""
    content = clean_text(post_data.get('content', ''))
    pic_num = post_data.get('pic_num', 0)
    img_tag = f" [包含{pic_num}张图片]" if pic_num > 0 else ""
    return f"{content}{img_tag}"

def calculate_text_similarity(text1, text2):
    """
    计算两个文本的相似度（使用 Jaccard 相似度）
    返回值范围 [0, 1]，1 表示完全相同
    """
    # 去除标点符号和空格，只保留中文字符和字母数字
    def normalize(t):
        t = re.sub(r'[^\w\u4e00-\u9fa5]', '', t)
        return set(t)
    
    set1 = normalize(text1)
    set2 = normalize(text2)
    
    # Jaccard 相似度：交集 / 并集
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0

# ================= 主逻辑 =================

def main():
    print("🚀 开始构建 Reward/DPO 数据集...")

    # 1. 第一遍扫描：收集评论并计算分数
    # 结构: post_id -> list of {'text', 'likes', 'score'}
    post_comments = defaultdict(list)
    # 优质回复池（用于构造随机负例）
    high_quality_pool = []

    print("1. 读取并评分评论数据...")
    with open(COMMENTS_FILE, 'r', encoding='utf-8') as f:
        # 使用 ijson 避免一次性加载导致内存爆炸
        for comment in tqdm(ijson.items(f, 'item'), desc="Processing Comments"):
            pid = comment.get('root_post_mblogid')
            if not pid: continue
            
            raw_text = comment.get('content', '')
            text = clean_text(raw_text)
            
            # 严重垃圾直接丢弃，不作为任何样本
            if is_spam(text): continue
            
            likes = comment.get('likes_count', 0)
            score = calculate_reward_score(likes, text)
            
            item = {
                "text": text,
                "likes": likes,
                "score": score
            }
            post_comments[pid].append(item)
            
            # 如果分数较高，加入池子
            if score > 3.0: 
                high_quality_pool.append(text)

    print(f"   >>> 共整理了 {len(post_comments)} 个帖子的有效评论。")

    # 2. 加载对应的帖子内容
    print("2. 加载 Posts 数据...")
    needed_pids = set(post_comments.keys())
    posts_map = {}
    
    try:
        with open(POSTS_FILE, 'r', encoding='utf-8') as f:
            for post in tqdm(ijson.items(f, 'item'), desc="Loading Posts"):
                mblogid = post.get('mblogid')
                if mblogid in needed_pids:
                    posts_map[mblogid] = {
                        'content': post.get('content', ''),
                        'pic_num': post.get('pic_num', 0)
                    }
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {POSTS_FILE}")
        return

    # 3. 生成配对数据 (Pairwise Generation)
    print("3. 生成 Chosen-Rejected 对...")
    final_dataset = []
    
    stats = {
        "real_negatives": 0,   # 同贴低分
        "random_negatives": 0,  # 异贴随机
        "filtered_similarity": 0  # 因相似度过高被过滤
    }

    for pid, comments in tqdm(post_comments.items(), desc="Pairing"):
        if pid not in posts_map: continue
        
        # 按分数从高到低排序
        comments.sort(key=lambda x: x['score'], reverse=True)
        
        best = comments[0]
        worst = comments[-1]
        
        # 基础门槛：Chosen 必须足够好
        if best['likes'] < MIN_LIKES_FOR_CHOSEN:
            continue
            
        prompt = format_prompt(posts_map[pid])
        
        # === 策略 A: 挖掘“真实负例” (Real Negative) ===
        # 同一个帖子下，既有高分也有低分，且差距明显
        # 这种数据质量最高，能教会模型“语境内的优劣”
        if (best['score'] - worst['score'] > MIN_SCORE_MARGIN) and (best['text'] != worst['text']):
            final_dataset.append({
                "prompt": prompt,
                "chosen": best['text'],
                "rejected": worst['text'],
                "meta": {
                    "type": "real_negative",
                    "chosen_score": best['score'],
                    "rejected_score": worst['score']
                }
            })
            stats["real_negatives"] += 1
            
        # === 策略 B: 构造“随机负例” (Random Negative) ===
        # 如果帖子下没有明显的差评（比如只有一条好评），或者全都是好评
        # 我们随机抽一个不相关的优质回复作为负例
        # 这能教会模型“相关性” (Relevance)
        elif best['score'] > 1.0 and high_quality_pool:
            random_neg = random.choice(high_quality_pool)
            
            # 确保随机抽的不是它自己
            if random_neg != best['text']:
                # 给随机负例一个假想的低分 (因为离题了)
                random_neg_score = -5.0 
                
                final_dataset.append({
                    "prompt": prompt,
                    "chosen": best['text'],
                    "rejected": random_neg,
                    "meta": {
                        "type": "random_negative",
                        "chosen_score": best['score'],
                        "rejected_score": random_neg_score
                    }
                })
                stats["random_negatives"] += 1

    # 4. 写入文件
    print(f"4. 写入结果到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("="*50)
    print(f"✅ 完成！共生成 {len(final_dataset)} 条配对数据。")
    print(f"📊 数据分布:")
    print(f"   - 真实负例 (优化风格/内容): {stats['real_negatives']} 条")
    print(f"   - 随机负例 (优化相关性):   {stats['random_negatives']} 条")
    print("="*50)

if __name__ == "__main__":
    main()