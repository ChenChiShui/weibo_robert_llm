import json
import math
import re
import os
from collections import defaultdict
from tqdm import tqdm
import ijson

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

POSTS_FILE = os.path.join(PROJECT_ROOT, 'data/Posts_fixed.json')
COMMENTS_FILE = os.path.join(PROJECT_ROOT, 'data/Comments.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'processed_data/commentr_sft_data.jsonl')

MIN_LIKES = 2
MIN_CHARS = 4
MAX_CHARS = 500
MIN_OUTPUT_CHARS = 4

AD_KEYWORDS = ['加群', '代购', '兼职', '刷单', '推广', '合作', '商务', '广告', '引流', '私聊']
SPAM_PATTERNS = [r'^[。\.]+$', r'^[！!]+$', r'^[？?]+$', r'^…+$', r'^[^\w\u4e00-\u9fa5]+$']

def remove_mentions(text):
    """去除@提及"""
    if not text:
        return ""
    return re.sub(r'@[\w\u4e00-\u9fa5]+', '', text).strip()

def is_spam_content(text):
    """检测是否为垃圾内容"""
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

def is_low_quality_output(text):
    """检测低质量输出"""
    if not text:
        return True
    
    text = text.strip()
    
    if len(text) < MIN_OUTPUT_CHARS:
        return True
    
    text_clean = remove_mentions(text)
    
    if not text_clean or len(text_clean) < MIN_OUTPUT_CHARS:
        return True
    
    if re.match(r'^http', text):
        return True
    
    if re.match(r'^图片评论', text):
        return True
    
    emoji_only = re.sub(r'\[.*?\]', '', text_clean).strip()
    if not emoji_only:
        return True
    
    return False

def calculate_quality_score(comment_item):
    """计算评论质量分数"""
    likes = comment_item.get('likes_count', 0)
    content = comment_item.get('content', '')
    
    score = math.log(likes + 1)
    
    if len(content) < 6:
        score *= 0.7
    elif len(content) > 20:
        score *= 1.2
    
    if '[' in content and ']' in content:
        score *= 1.05
    
    return round(score, 4)

def clean_text(text):
    """文本清洗"""
    if not text:
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    text = remove_mentions(text)
    return text

def format_prompt(post_data, parent_comment=None):
    """构建输入 Prompt"""
    post_content = clean_text(post_data.get('content', ''))
    
    img_tag = ""
    pic_num = post_data.get('pic_num', 0)
    if pic_num > 0:
        img_tag = f" [包含{pic_num}张图片]"
    
    prompt = f"{post_content}{img_tag}"
    
    if parent_comment:
        parent_text = clean_text(parent_comment.get('content', ''))
        if parent_text:
            prompt += f"\n\n(该用户回复了评论：{parent_text})"
        
    return prompt

def load_posts_minimal(needed_post_ids):
    """流式加载帖子数据，只保留必要字段"""
    print("3. 正在加载 Posts 数据（仅需要的帖子）...")
    posts_map = {}
    try:
        with open(POSTS_FILE, 'r', encoding='utf-8') as f:
            for post in tqdm(ijson.items(f, 'item'), desc="Loading Posts"):
                mblogid = post.get('mblogid')
                if mblogid and mblogid in needed_post_ids:
                    posts_map[mblogid] = {
                        'content': post.get('content', ''),
                        'pic_num': post.get('pic_num', 0)
                    }
        print(f"   已加载 {len(posts_map)} 条帖子。")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {POSTS_FILE}")
        return None
    return posts_map

def main():
    print("1. 正在处理 Comments，收集有效评论...")
    valid_comments = []
    skipped_content = 0
    skipped_likes = 0
    skipped_output = 0
    
    with open(COMMENTS_FILE, 'r', encoding='utf-8') as f_in:
        for comment in tqdm(ijson.items(f_in, 'item'), desc="Processing Comments"):
            post_id = comment.get('root_post_mblogid')
            if not post_id:
                continue
                
            content = clean_text(comment.get('content', ''))
            likes = comment.get('likes_count', 0)
            
            if is_spam_content(content):
                skipped_content += 1
                continue
            
            if likes < MIN_LIKES:
                skipped_likes += 1
                continue
            
            if is_low_quality_output(content):
                skipped_output += 1
                continue
            
            valid_comments.append(comment)
    
    print(f"   有效评论: {len(valid_comments)} 条")
    
    print("2. 正在收集需要的 post_id...")
    needed_post_ids = set()
    for comment in valid_comments:
        needed_post_ids.add(comment.get('root_post_mblogid'))
    print(f"   需要 {len(needed_post_ids)} 个帖子")
    
    posts_map = load_posts_minimal(needed_post_ids)
    
    if not posts_map:
        return
    
    print("4. 正在去重（每个帖子保留最高赞评论）...")
    post_comments_map = defaultdict(list)
    skipped_post = 0
    
    for comment in valid_comments:
        post_id = comment.get('root_post_mblogid')
        if post_id in posts_map:
            post_comments_map[post_id].append(comment)
        else:
            skipped_post += 1
    
    valid_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for post_id, comments in tqdm(post_comments_map.items(), desc="Deduplicating"):
            best_comment = max(comments, key=lambda x: x.get('likes_count', 0))
            
            target_post = posts_map[post_id]
            parent_comment_info = best_comment.get('reply_comment')
            
            input_text = format_prompt(target_post, parent_comment_info)
            
            output_text = clean_text(best_comment.get('content', ''))
            
            quality_score = calculate_quality_score(best_comment)

            data_entry = {
                "instruction": "根据帖子内容进行回复。",
                "input": input_text,
                "output": output_text,
                "meta": {
                    "likes": best_comment.get('likes_count', 0),
                    "quality_score": quality_score,
                    "post_id": post_id,
                    "comment_id": best_comment.get('_id')
                }
            }
            
            f_out.write(json.dumps(data_entry, ensure_ascii=False) + '\n')
            valid_count += 1

    print("="*50)
    print(f"处理完成！")
    print(f"生成文件: {OUTPUT_FILE}")
    print(f"有效数据: {valid_count} 条")
    print(f"过滤统计:")
    print(f"  - 垃圾内容: {skipped_content} 条")
    print(f"  - 无效帖子: {skipped_post} 条")
    print(f"  - 低赞评论: {skipped_likes} 条")
    print(f"  - 低质量输出: {skipped_output} 条")
    print("="*50)

if __name__ == "__main__":
    main()
