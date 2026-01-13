# Robert LLM - 基于 CommentR 数据集的微博社区评论大模型后训练

基于 Qwen3-4B(2025.07) 和 CommentR Interaction Dataset(2025.11) 的微博评论机器人训练项目，通过多阶段训练（SFT → Reward Model → RL）学习生成符合人类偏好的高质量评论回复。

后训练模型：[![Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-yellow)](https://huggingface.co/Chishui-Chen/robert-llm)

处理后数据：[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-green)](https://huggingface.co/datasets/Chishui-Chen/robert-llm-data)

base_model：[![Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-yellow)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)



## 项目概述

大模型后训练练手项目，大量使用 vibe coding，因此可能有细节问题，请谅解！

本项目旨在训练一个能够根据微博帖子内容自动生成评论回复的智能机器人。基于 **CommentR Interaction Dataset**，包含 557,645 个提及 @CommentR 的帖子、304,400 个独立用户以及 1,028,364 条回复。

通过分析真实的人类-AI 互动数据，学习人类用户的回复风格和偏好，生成自然、有趣且符合社区氛围的评论。

### 核心特性

- **基于真实人类-AI 交互数据**：使用 CommentR Interaction Dataset，包含 55万+ 真实交互场景
- **多阶段训练流程**：SFT（监督微调）→ Reward Model（奖励模型）→ DPO/GRPO（偏好对齐）
- **训练框架**：PyTorch, VLLM, TRL

## 快速开始

### 环境配置

```bash
git clone https://github.com/ChenChiShui/robert-llm.git
cd robert-llm
pip install -r requirements.txt
```

### 下载模型和数据

#### 下载模型

模型托管在 Hugging Face，使用以下命令下载：

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct --local-dir ./model/Qwen3-4B-Instruct
huggingface-cli download Chishui-Chen/robert-llm-sft --local-dir ./model/sft
huggingface-cli download Chishui-Chen/robert-llm-dpo --local-dir ./model/dpo
huggingface-cli download Chishui-Chen/robert-llm-grpo --local-dir ./model/grpo
```

#### 下载数据集

### 快速推理

```bash
cd scripts/inference
python inference_dpo_lora.py
```

## 项目结构

```
weibo_robert_llm/
├── data/                          # 数据文件夹
│   ├── sample_posts.json          # 示例帖子数据
│   ├── sample_comments.json       # 示例评论数据
│   ├── sample_users.json          # 示例用户数据
│   └── README.md                  # 数据说明文档
├── processed_data/                # 处理后的数据
│   ├── commentr_sft_data.jsonl    # SFT训练数据
│   └── commentr_dpo_reward_data.jsonl  # DPO/Reward训练数据
├── scripts/
│   ├── data_processing/           # 数据处理脚本
│   │   ├── convert_data.py        # SFT数据生成
│   │   ├── generate_reward_data.py # Reward数据生成
│   │   └── new_reward_data.py     # 新版Reward数据生成
│   ├── train/                     # 训练脚本
│   │   ├── train_sft_lora.py      # SFT训练
│   │   ├── merge_sft.py           # 合并SFT模型
│   │   ├── train_reward_model.py # Reward模型训练
│   │   ├── train_dpo_lora.py      # DPO训练
│   │   └── train_grpo_lora.py     # GRPO训练
│   └── inference/                 # 推理脚本
│       ├── inference.py           # 基础推理
│       ├── inference_sft_lora.py   # SFT模型推理
│       ├── inference_sft_merged.py # SFT合并模型推理
│       ├── inference_dpo_lora.py   # DPO模型推理
│       ├── inference_reward_model.py # Reward模型推理
│       └── compare_all_models.py  # 模型对比
├── results/                       # 推理结果
│   ├── inference_comparison_results.jsonl  # 模型对比结果
│   └── reward_model.json          # Reward 模型评估结果
├── docs/                          # 文档
│   └── DATA_PROCESSING.md         # 数据处理说明文档
├── .gitignore                     # Git 忽略文件
├── LICENSE                        # MIT 许可证
├── README.md                      # 本文件
└── requirements.txt               # 依赖列表
```

## 训练策略说明

### 为什么需要多阶段训练？

现代大语言模型的训练通常分为多个阶段：

1. **预训练**：在海量文本数据上学习语言知识（本项目使用预训练好的 Qwen3）
2. **SFT（监督微调）**：学习特定任务的能力，即"如何回复"
3. **偏好对齐（DPO/GRPO）**：对齐人类价值观和偏好，即"哪个回复更好"

### SFT vs DPO

| 阶段 | 目标 | 数据类型 | 训练方式 | 结果 |
|------|------|----------|----------|------|
| SFT | 学会回复 | 单样本（input→output） | 监督学习 | 模型能生成回复 |
| DPO | 学会偏好 | 配对样本（prompt→chosen/rejected） | 偏好优化 | 模型生成高质量回复 |

**类比**：
- **SFT**：像教学生"如何写作文"，学习基本语法和结构
- **DPO**：像教学生"如何写好作文"，学习优秀作文的特点
