## 梗概

微博鲁伯特交互数据集处理后的数据

处理后数据在 https://huggingface.co/datasets/Chishui-Chen/robert-llm-data

源数据集 https://github.com/FDUDataNET 

后训练模型在 https://huggingface.co/Chishui-Chen/robert-llm


## 数据来源与语言类型

### 数据来源

本项目使用的数据来自**微博**，具体包括：

| 数据文件           | 内容         | 说明                                                         |
| ------------------ | ------------ | ------------------------------------------------------------ |
| `Posts_fixed.json` | 微博帖子数据 | 包含帖子ID、创建时间、地理位置、IP位置、转发次数、评论次数、点赞数、来源、内容、图片URL/编号等 |
| `Comments.json`    | 微博评论数据 | 包含评论ID、根帖ID、根评论ID、创建时间、点赞数、IP位置、内容、评论用户信息、回复评论信息等 |



**重要说明**：数据处理脚本评论是来自人类还是罗伯特。它们只是根据**点赞数**和**质量分数**来筛选评论。

| 类型                       | 来源             | 语言类型                          | 处理方式                             |
| -------------------------- | ---------------- | --------------------------------- | ------------------------------------ |
| **input/prompt**           | Posts_fixed.json | 人类语言（人类发布的帖子）        | 直接使用                             |
| **output/chosen/rejected** | Comments.json    | **混合**（人类评论 + 罗伯特回复） | 根据点赞数和质量分数筛选，不区分来源 |

**训练目标**：训练一个**新的罗伯特模型**，让它学习如何根据人类帖子生成符合人类偏好的回复。

---

## SFT 数据处理


### 核心思想
生成单样本的监督学习数据，每个帖子只保留最高赞的评论作为标准答案，教会模型"如何回复"。

### 筛选标准

#### 1. 基础阈值
```python
MIN_LIKES = 2           # 评论至少获得2个赞
MIN_CHARS = 4           # 评论至少4个字符
MAX_CHARS = 500         # 评论最多500个字符
MIN_OUTPUT_CHARS = 4    # 输出至少4个字符
```

#### 2. 垃圾内容过滤
过滤以下类型的评论：
- **广告关键词**：`['加群', '代购', '兼职', '刷单', '推广', '合作', '商务', '广告', '引流', '私聊']`
- **纯标点符号**：`^[。\.]+$`, `^[！!]+$`, `^[？?]+$`, `^…+$`
- **纯特殊字符**：`^[^\w\u4e00-\u9fa5]+$`
- **字符多样性低**：长度>10但唯一字符<3
- **Emoji 过多**：超过10个 emoji

#### 3. 低质量输出过滤
- 纯 emoji 内容
- 以 `http` 开头的链接
- 以 `图片评论` 开头的内容
- 去除 @ 提及后内容为空

#### 4. 去重策略
每个帖子只保留**最高赞**的评论，确保数据多样性和质量。

### 质量评分算法
```python
score = log(likes + 1)  # 基础分：点赞对数

# 长度调整
if len(content) < 6:
    score *= 0.7  # 惩罚过短
elif len(content) > 20:
    score *= 1.2  # 奖励适中长度

# Emoji 奖励
if '[' in content and ']' in content:
    score *= 1.05
```

---

## DPO 数据处理


### 核心思想
生成配对数据（Chosen vs Rejected），通过对比学习教会模型"哪个回复更好"，用于 DPO（Direct Preference Optimization）或 Reward Model 训练。

### 筛选标准

#### 1. 基础阈值
```python
MIN_LIKES_FOR_CHOSEN = 2      # Chosen 至少2个赞
MIN_SCORE_MARGIN = 0.5        # 正负例分数差至少0.5
SPAM_CHARS_THRESHOLD = 2      # 过滤极短内容（<2字符）
```

#### 2. 奖励评分算法
```python
def calculate_reward_score(likes, content):
    # 垃圾内容直接给低分
    if is_spam(content):
        return -10.0
    
    # 1. 基础分：点赞对数
    score = log(likes + 1)
    
    # 2. 长度调整
    if len(content) < 5:
        score -= 1.0  # 太短惩罚
    elif 10 <= len(content) <= 60:
        score += 0.5  # 黄金长度奖励
    
    # 3. Emoji 奖励
    if '[' in content and ']' in content:
        score += 0.2
    
    return round(score, 4)
```

#### 3. 配对生成策略

##### 策略 A：真实负例（Real Negative）
- **条件**：同一帖子下，最高分和最低分评论的分数差 > `MIN_SCORE_MARGIN`
- **目的**：教会模型在相同语境下识别优劣
- **质量**：最高，能学习风格和内容质量

##### 策略 B：随机负例（Random Negative）
- **条件**：帖子下没有明显的低分评论，且 Chosen 分数 > 1.0
- **方法**：从优质回复池（score > 3.0）中随机抽取不相关的评论
- **目的**：教会模型"相关性"（回复应该与帖子相关）
- **质量**：中等，主要学习相关性

#### 4. 优质回复池
- 收集所有 `score > 3.0` 的评论
- 用于构造随机负例
- 确保负例本身质量不错，只是不相关

---


## 数据格式说明

### SFT 数据格式

```json
{
  "instruction": "根据帖子内容进行回复。",
  "input": "咱俩的关系有点亲密了[害羞] [包含1张图片]",
  "output": "当然！如果你希望继续和我对话 来评论吧",
  "meta": {
    "likes": 2,
    "quality_score": 1.0986,
    "post_id": "a3c591ab27e739017762b45c3e23a88c",
    "comment_id": "7bd3073aece9f606330c8dd579ce50f4"
  }
}
```

**字段说明：**
- `instruction`: 任务描述，固定为"根据帖子内容进行回复。"
- `input`: 帖子内容 + 图片标签（如果有）
- `output`: 标准回复（最高赞评论）
- `meta.likes`: 评论获得的点赞数
- `meta.quality_score`: 质量评分（基于点赞、长度、emoji）
- `meta.post_id`: 原帖子ID
- `meta.comment_id`: 评论ID

### DPO 数据格式

```json
{
  "prompt": "咱俩的关系有点亲密了[害羞] [包含1张图片]",
  "chosen": "哈哈哈哈哈哈[doge]",
  "rejected": "哈哈哈哈",
  "meta": {
    "type": "real_negative",
    "chosen_score": 1.7986,
    "rejected_score": -0.3069
  }
}
```

**字段说明：**
- `prompt`: 帖子内容 + 图片标签（如果有）
- `chosen`: 高质量回复（优选）
- `rejected`: 低质量或不相关回复（劣选）
- `meta.type`: 负例类型
  - `real_negative`: 真实负例（同帖低分评论）
  - `random_negative`: 随机负例（不相关优质评论）
- `meta.chosen_score`: Chosen 的奖励分数
- `meta.rejected_score`: Rejected 的奖励分数

---
