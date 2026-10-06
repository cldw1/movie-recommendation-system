# 🎬 基于召回与排序的个性化电影推荐系统

一个基于 MovieLens Latest Small 数据集构建的两阶段个性化推荐系统。

项目采用：

**ItemCF + UserCF 多路召回 → 特征工程 → XGBRanker 排序 → Top-N 推荐**

完整实现了从数据处理、召回模型、排序模型、离线评估到 Streamlit 推荐系统 Demo 的推荐算法流程。

---

## 1. 项目背景

推荐系统通常需要解决两个核心问题：

1. 如何从大量物品中快速找到用户可能感兴趣的候选物品；
2. 如何对候选物品进行更加精细的个性化排序。

因此，本项目采用推荐系统中常见的两阶段架构：

```text
用户历史行为
      │
      ▼
┌─────────────────────┐
│      多路召回        │
│                     │
│   ItemCF Top300      │
│          +           │
│   UserCF Top300      │
└──────────┬──────────┘
           │
           ▼
       候选电影池
           │
           ▼
┌─────────────────────┐
│      特征工程        │
│                     │
│ 用户特征             │
│ 电影特征             │
│ 类型偏好特征         │
│ ItemCF / UserCF 得分 │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      XGBRanker       │
│   Learning to Rank   │
└──────────┬──────────┘
           │
           ▼
      Top-10 推荐
```

---

## 2. 数据集

项目使用：

**MovieLens Latest Small**

主要包含：

```text
ratings.csv
movies.csv
tags.csv
links.csv
```

其中核心使用的数据为：

```text
ratings.csv
movies.csv
```

数据规模：

```text
用户数量：610
电影数量：9742
评分数量：100836
```

评分字段：

```text
userId
movieId
rating
timestamp
```

在本项目中定义：

```python
rating >= 4
```

表示用户对该电影存在正向偏好。

---

## 3. 项目结构

```text
movie-recommendation-system/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── ratings.csv
│   │   ├── movies.csv
│   │   ├── tags.csv
│   │   └── links.csv
│   │
│   └── processed/
│       ├── train_ratings.csv
│       ├── test_ratings.csv
│       └── ranker_data.csv
│
├── models/
│   └── rank_model_ltr.pkl
│
├── results/
│
└── src/
    ├── popular_recommend.py
    ├── split_data.py
    ├── build_ranker_data.py
    ├── train_ranker.py
    ├── evaluate.py
    ├── evaluate_ranker.py
    └── recommend_service.py
```

---

## 4. 数据划分

为了避免推荐系统中常见的时间泄漏问题，本项目按照用户内部时间顺序进行训练集与测试集划分。

对每个用户的评分记录按照：

```text
timestamp
```

升序排列。

前 80%：

```text
训练集
```

后 20%：

```text
测试集
```

最终得到：

```text
训练集：80419 条
测试集：20417 条
```

这种划分方式模拟真实推荐场景：

```text
使用用户过去的行为
        ↓
预测用户未来喜欢什么
```

而不是随机打乱用户的历史行为。

---

## 5. Popular Baseline

首先实现 Popular Recommendation 作为基础模型。

综合考虑：

```text
电影平均评分
+
电影评分数量
```

构造 Popular Score：

```python
popular_score = avg_rating * log(1 + rating_count)
```

该模型不考虑用户个性化信息，主要作为推荐算法效果的 Baseline。

---

## 6. ItemCF 召回

ItemCF 基于：

> 喜欢相似电影的用户行为具有相似性。

首先构建：

```text
电影 × 用户
```

的二值交互矩阵。

当：

```python
rating >= 4
```

时设置为 1，否则为 0。

使用：

```python
NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)
```

计算电影之间的余弦相似度。

根据用户历史喜欢电影召回相似电影，并累计：

```text
itemcf_score
```

作为 ItemCF 召回得分。

---

## 7. UserCF 召回

UserCF 基于：

> 相似用户可能喜欢相似的电影。

构建：

```text
用户 × 电影
```

交互矩阵。

通过余弦相似度寻找与目标用户最相似的用户。

再根据相似用户喜欢的电影生成推荐候选，并累计：

```text
usercf_score
```

作为 UserCF 召回得分。

---

## 8. 多路召回

最终候选集由：

```text
ItemCF Top300
+
UserCF Top300
```

进行 Outer Merge 得到。

候选召回实验结果：

```text
Top100 Recall ≈ 0.3599
Top200 Recall ≈ 0.4809
Top300 Recall ≈ 0.5529
```

Top300 + Top300 最终召回：

```text
4100 / 7415
```

个未来正样本。

因此最终采用：

```text
ItemCF Top300
+
UserCF Top300
```

作为排序模型候选池。

---

## 9. 排序模型数据构造

为了避免排序模型看到未来信息，在原训练集内部再次按照时间进行划分：

```text
rank_history
+
rank_target
```

其中：

```text
rank_history：64081
rank_target：16338
```

排序模型只能使用：

```text
rank_history
```

构建召回模型和特征。

若候选电影出现在用户未来的：

```text
rank_target
```

中，并且：

```python
rating >= 4
```

则：

```python
label = 1
```

否则：

```python
label = 0
```

最终得到 Learning to Rank 训练数据。

---

## 10. 排序特征

最终 XGBRanker 使用 8 个特征：

| 特征 | 含义 |
|---|---|
| user_rating_count | 用户历史评分数量 |
| user_avg_rating | 用户历史平均评分 |
| movie_rating_count | 电影历史评分数量 |
| movie_avg_rating | 电影历史平均评分 |
| genre_match_score | 用户类型偏好与电影类型匹配程度 |
| itemcf_score | ItemCF 召回得分 |
| usercf_score | UserCF 召回得分 |
| is_popular_movie | 是否属于热门电影 |

---

## 11. Genre Match 特征

为了进一步增强个性化能力，本项目构建：

```text
genre_match_score
```

首先统计用户历史喜欢电影中的类型分布：

```text
Action
Drama
Comedy
Sci-Fi
...
```

计算：

```text
用户对每种电影类型的偏好程度
```

再根据候选电影包含的 genres 计算平均偏好值。

最终得到：

```text
genre_match_score ∈ [0, 1]
```

该特征用于衡量：

> 一部候选电影的类型与用户历史兴趣是否匹配。

实验表明加入该特征后：

```text
Recall@10
HitRate@10
NDCG@10
```

均获得提升。

---

## 12. XGBRanker

最终排序模型使用：

```python
XGBRanker
```

目标函数：

```text
rank:ndcg
```

主要参数：

```python
n_estimators=300
max_depth=5
learning_rate=0.05
subsample=0.8
colsample_bytree=0.8
eval_metric="ndcg@10"
```

训练集和验证集按照：

```text
用户 ID
```

划分。

保证同一个用户的候选电影不会同时出现在训练集和验证集中。

---

## 13. 特征消融实验

项目进一步比较了 CF Score 和 CF Rank 两类特征。

实验结果：

| 特征组合 | Validation NDCG@10 |
|---|---:|
| CF Score | **0.11923** |
| CF Score + CF Rank | 0.11843 |
| CF Rank | 0.11754 |

最终发现：

```text
Score-only
```

效果最好。

说明虽然 Recall Rank 本身包含较强信息，但：

```text
rank
```

会损失不同电影之间具体的相似度差异。

因此最终模型使用：

```text
itemcf_score
usercf_score
```

而没有使用 Rank 特征。

---

## 14. 模型效果对比

严格按照外层测试集进行离线评估。

有效测试用户：

```text
592
```

结果如下：

| Model | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Popular | 0.0606 | 0.0492 | 0.3277 | 0.0793 |
| ItemCF | 0.0784 | 0.0738 | 0.4307 | 0.0985 |
| UserCF | **0.0916** | **0.0910** | 0.4696 | **0.1158** |
| XGBClassifier | 0.0856 | 0.0891 | 0.4764 | 0.1093 |
| XGBRanker | 0.0863 | 0.0857 | **0.4949** | 0.1071 |

实验可以看到：

```text
UserCF
```

在：

```text
Precision
Recall
NDCG
```

上表现最好。

而：

```text
XGBRanker
```

在：

```text
HitRate@10
```

上达到最高：

```text
0.4949
```

意味着约：

```text
49.5%
```

的测试用户在 Top-10 推荐中至少命中一部未来喜欢的电影。

该实验也说明：

> 在 MovieLens Small 这样规模较小的数据集上，更复杂的排序模型并不一定在所有指标上全面超过协同过滤模型。

---

## 15. 推荐系统 Demo

项目使用：

```text
Streamlit
```

构建交互式推荐系统。

用户可以输入：

```text
User ID
```

系统自动完成：

```text
用户画像
    ↓
ItemCF + UserCF 多路召回
    ↓
8 维特征构建
    ↓
XGBRanker 排序
    ↓
Top-10 个性化推荐
```

Demo 页面包含：

```text
用户历史评分数量
用户平均评分
用户喜欢电影数量
用户电影类型偏好
最近喜欢的电影
Top-10 推荐电影
推荐得分
Genre 匹配度
ItemCF / UserCF 得分
简单推荐解释
```

---

## 16. 推荐解释

为了增强系统可解释性，根据：

```text
genre_match_score
itemcf_score
usercf_score
```

生成简单的规则式推荐解释。

例如：

```text
类型兴趣匹配较高
相似用户偏好明显
与历史喜欢电影相似
```

需要说明的是：

> 该推荐解释属于基于模型特征的规则式解释，并非 XGBRanker 自身直接生成的因果解释。

---

## 17. 环境安装

推荐使用：

```text
Python 3.10+
```

创建虚拟环境后安装：

```bash
pip install -r requirements.txt
```

---

## 18. 项目运行

首先完成数据划分：

```bash
python src/split_data.py
```

生成排序训练数据：

```bash
python src/build_ranker_data.py
```

训练 XGBRanker：

```bash
python src/train_ranker.py
```

进行离线评估：

```bash
python src/evaluate_ranker.py
```

启动推荐系统：

```bash
streamlit run app.py
```

如果使用 Windows 项目虚拟环境，也可以运行：

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器访问 Streamlit 提供的本地地址即可使用推荐系统。

---

## 19. 项目亮点

本项目完整实现了推荐算法项目中的：

```text
时间序列数据划分
Popular Baseline
ItemCF
UserCF
多路召回
候选集构建
用户特征
物品特征
Genre 偏好特征
Learning to Rank
XGBRanker
Precision / Recall / HitRate / NDCG
特征消融实验
严格离线评估
推荐解释
Streamlit Demo
```

相比只实现单个协同过滤算法，本项目更接近真实推荐系统中的：

```text
Recall → Ranking
```

两阶段推荐架构。

---

## 20. 后续优化方向

后续可以进一步尝试：

```text
矩阵分解 / SVD
LightFM
双塔召回模型
Embedding 特征
时间衰减特征
用户近期兴趣特征
电影年份特征
Tag 特征
LightGBM Ranker
更大规模 MovieLens 数据集
深度学习推荐模型
```

同时可以加入：

```text
新用户冷启动
新电影冷启动
在线推荐接口
模型部署
A/B Test
```

进一步提升系统的完整性。

---

## 21. 技术栈

```text
Python
Pandas
NumPy
SciPy
Scikit-learn
XGBoost
Joblib
Matplotlib
Streamlit
```

---

## 22. 总结

本项目从 MovieLens 用户评分数据出发，逐步实现了：

```text
Popular
    ↓
ItemCF
    ↓
UserCF
    ↓
多路召回
    ↓
特征工程
    ↓
XGBRanker
    ↓
离线评估
    ↓
Streamlit 推荐系统
```

不仅关注模型训练结果，同时重点考虑：

```text
时间泄漏
召回覆盖率
候选集构造
训练数据构造
排序特征设计
模型消融实验
严格测试集评估
推荐可解释性
工程化展示
```

最终形成一个较完整的个性化电影推荐系统项目。