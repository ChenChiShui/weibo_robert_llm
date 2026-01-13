# Data

本文件夹包含示例数据，用于展示数据格式。

## 数据来源

本项目的数据来源于 **CommentR Interaction Dataset**，由复旦大学 FDUDataNET 团队发布。

- **GitHub 仓库**: https://github.com/FDUDataNET/Comment-Robert
- **数据集描述**: 大规模人类-LLM 交互数据集，来自微博平台，专注于用户主动提及 LLM 账号 @CommentR 的帖子

## 数据文件说明

- `sample_posts.json` - 微博帖子示例数据
- `sample_comments.json` - 微博评论示例数据
- `sample_users.json` - 微博用户示例数据

## 数据格式

### Posts.json

```json
{
    "_id": "帖子ID",
    "mblogid": "微博ID",
    "created_at": "创建时间",
    "geo": "地理位置",
    "ip_location": "IP属地",
    "reposts_count": 转发数,
    "comments_count": "评论数",
    "source": "发布来源",
    "content": "帖子内容",
    "pic_urls": ["图片URL列表"],
    "pic_num": 图片数量,
    "isLongText": 是否长文本,
    "is_retweet": 是否转发,
    "user": {
        "_id": "用户ID",
        "avatar_hd": "头像URL",
        "nick_name": "昵称",
        "verified": 是否认证,
        "description": "简介",
        "followers_count": 粉丝数,
        "follow_count": 关注数,
        "gender": "性别"
    }
}
```

### Comments.json

```json
{
    "created_at": "创建时间",
    "_id": "评论ID",
    "ip_location": "IP属地",
    "content": "评论内容",
    "comment_user": {
        "_id": "用户ID",
        "avatar_hd": "头像URL",
        "nick_name": "昵称",
        "verified": 是否认证,
        "description": "简介",
        "followers_count": 粉丝数,
        "friends_count": 关注数,
        "statuses_count": "微博数",
        "gender": "性别",
        "location": "位置"
    }
}
```

### Users.json

```json
{
    "_id": "用户ID",
    "avatar_hd": "头像URL",
    "nick_name": "昵称",
    "verified": 是否认证,
    "description": "简介",
    "followers_count": 粉丝数,
    "friends_count": 关注数,
    "statuses_count": "微博数",
    "gender": "性别",
    "location": "位置"
}
```

## 获取完整数据集

完整的 CommentR Interaction Dataset 可通过以下方式获取：

### 方式一：从 GitHub 下载

访问 [FDUDataNET/Comment-Robert](https://github.com/FDUDataNET/Comment-Robert) 仓库，在仓库中可以找到示例数据（1,000 个帖子）。

### 方式二：从 Zenodo 下载完整数据集

完整数据集可通过 Zenodo 获取（请查看 GitHub 仓库中的下载链接）。

### 数据集统计

- 提及 @CommentR 的帖子：**557,645** 个
- 独立用户：**304,400** 个
- 总回复数：**1,028,364** 条
- 时间跨度：2023年12月1日 至 2025年4月30日
- CommentR 发布时间：2023年12月8日
- 数据收集开始：2024年3月（回填早期数据，每月更新）

### 数据收集方法

- 通过微博搜索查询包含 @CommentR 的帖子
- 请求间隔 1 秒，避免服务器过载

## 数据使用说明

下载完整数据集后，请将文件重命名为：
- `Posts.json`
- `Comments.json`
- `Users.json`

并放置在 `data/` 目录下，然后运行数据处理脚本。

## 数据引用

如果使用此数据集，请引用相关论文。
