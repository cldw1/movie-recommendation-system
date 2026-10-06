from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

import pandas as pd
import math
import numpy as np

# ==============================
# 数据路径
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_ratings.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_ratings.csv"


# ==============================
# 读取训练集和测试集
# ==============================

train_ratings = pd.read_csv(TRAIN_PATH)
test_ratings = pd.read_csv(TEST_PATH)

print("===== 评估数据 =====")
print("训练集数量：", len(train_ratings))
print("测试集数量：", len(test_ratings))

# ==============================
# 1. Precision@K
# ==============================

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    计算 Precision@K

    参数：
        recommended_items: 推荐出来的电影 ID 列表
        relevant_items: 用户测试集中真正喜欢的电影 ID 集合
        k: 只看推荐列表前 K 个

    返回：
        Precision@K
    """

    recommended_top_k = recommended_items[:k]

    hit_count = len(
        set(recommended_top_k) & set(relevant_items)
    )

    precision = hit_count / k

    return precision

# ==============================
# # 2. 简单测试
# # ==============================
#
# recommended_items = [10, 20, 30, 40, 50]
# relevant_items = [20, 40, 60]
#
# precision = precision_at_k(
#     recommended_items,
#     relevant_items,
#     k=5
# )
#
# print("Precision@5 =", precision)

# ==============================
# 3. Recall@K
# ==============================

def recall_at_k(recommended_items, relevant_items, k=10):
    """
    计算 Recall@K

    参数：
        recommended_items: 推荐出来的电影 ID 列表
        relevant_items: 用户测试集中真正喜欢的电影 ID 集合
        k: 只看推荐列表前 K 个

    返回：
        Recall@K
    """

    recommended_top_k = recommended_items[:k]

    hit_count = len(
        set(recommended_top_k) & set(relevant_items)
    )

    if len(relevant_items) == 0:
        return 0.0

    recall = hit_count / len(relevant_items)

    return recall

# recommended_items = [10, 20, 30, 40, 50]
# relevant_items = [20, 40, 60]
#
# recall = recall_at_k(
#     recommended_items,
#     relevant_items,
#     k=5
# )
#
# print("Recall@5 =", recall)

# ==============================
# 4. HitRate@K
# ==============================

def hitrate_at_k(recommended_items, relevant_items, k=10):
    """
    计算 HitRate@K

    参数：
        recommended_items: 推荐出来的电影 ID 列表
        relevant_items: 用户测试集中真正喜欢的电影 ID 集合
        k: 只看推荐列表前 K 个

    返回：
        HitRate@K
    """

    recommended_top_k = recommended_items[:k]

    hit_count = len(
        set(recommended_top_k) & set(relevant_items)
    )

    if hit_count > 0:
        return 1.0

    return 0.0

# recommended_items = [10, 20, 30, 40, 50]
# relevant_items = [20, 40, 60]
#
# hitrate = hitrate_at_k(
#     recommended_items,
#     relevant_items,
#     k=5
# )
#
# print("HitRate@5 =", hitrate)

# ==============================
# 5. NDCG@K
# ==============================

def ndcg_at_k(recommended_items, relevant_items, k=10):
    """
    计算 NDCG@K

    参数：
        recommended_items: 推荐出来的电影 ID 列表
        relevant_items: 用户测试集中真正喜欢的电影 ID 集合
        k: 只看推荐列表前 K 个

    返回：
        NDCG@K
    """

    recommended_top_k = recommended_items[:k]
    relevant_set = set(relevant_items)

    # 计算 DCG
    dcg = 0.0

    for i, movie_id in enumerate(recommended_top_k):

        if movie_id in relevant_set:
            dcg += 1 / math.log2(i + 2)

    # 计算理想情况下的 DCG
    ideal_hit_count = min(
        len(relevant_set),
        k
    )

    if ideal_hit_count == 0:
        return 0.0

    idcg = 0.0

    for i in range(ideal_hit_count):
        idcg += 1 / math.log2(i + 2)

    ndcg = dcg / idcg

    return ndcg

# recommended_items = [10, 20, 30, 40, 50]
# relevant_items = [20, 40, 60]
#
# ndcg = ndcg_at_k(
#     recommended_items,
#     relevant_items,
#     k=5
# )
#
# print("NDCG@5 =", ndcg)

# ==============================
# 提取测试集中的真实喜欢电影
# ==============================

test_positive = test_ratings[
    test_ratings["rating"] >= 4
]

user_relevant_items = (
    test_positive
    .groupby("userId")["movieId"]
    .apply(list)
    .to_dict()
)

print("\n===== 测试集真实喜欢行为 =====")
print("有正向测试样本的用户数量：", len(user_relevant_items))

print("\n用户 1 在测试集中真正喜欢的电影：")
print(user_relevant_items.get(1, []))

print(
    "用户 1 真正喜欢的电影数量：",
    len(user_relevant_items.get(1, []))
)

# ==============================
# 使用训练集构建 Popular Baseline
# ==============================

popular_stats = (
    train_ratings
    .groupby("movieId")
    .agg(
        rating_count=("rating", "count"),
        avg_rating=("rating", "mean")
    )
    .reset_index()
)

popular_stats["popular_score"] = (
    popular_stats["avg_rating"]
    * np.log1p(popular_stats["rating_count"])
)

popular_ranking = (
    popular_stats
    .sort_values(
        by="popular_score",
        ascending=False
    )
)

print("\n===== 基于训练集的热门电影 Top 10 =====")

print(
    popular_ranking[
        ["movieId", "rating_count", "avg_rating", "popular_score"]
    ]
    .head(10)
    .to_string(index=False)
)

# ==============================
# Popular Baseline：给指定用户推荐
# ==============================

def recommend_popular_for_user(user_id, top_n=10):

    # 用户在训练集中已经看过的电影
    watched_movies = set(
        train_ratings[
            train_ratings["userId"] == user_id
        ]["movieId"]
    )

    recommendations = []

    for movie_id in popular_ranking["movieId"]:

        # 已经看过的不再推荐
        if movie_id in watched_movies:
            continue

        recommendations.append(movie_id)

        if len(recommendations) >= top_n:
            break

    return recommendations

popular_recommendations = recommend_popular_for_user(
    user_id=1,
    top_n=10
)

print("\n===== 用户 1 的 Popular Top-10 =====")
print(popular_recommendations)

# ==============================
# 评估用户 1 的 Popular Baseline
# ==============================

user_id = 1

recommended_items = recommend_popular_for_user(
    user_id=user_id,
    top_n=10
)

relevant_items = user_relevant_items.get(
    user_id,
    []
)

precision = precision_at_k(
    recommended_items,
    relevant_items,
    k=10
)

recall = recall_at_k(
    recommended_items,
    relevant_items,
    k=10
)

hitrate = hitrate_at_k(
    recommended_items,
    relevant_items,
    k=10
)

ndcg = ndcg_at_k(
    recommended_items,
    relevant_items,
    k=10
)

print("\n===== 用户 1 的 Popular Baseline 评估 =====")
print("Precision@10 =", precision)
print("Recall@10 =", recall)
print("HitRate@10 =", hitrate)
print("NDCG@10 =", ndcg)

# ==============================
# 评估全部有效用户
# ==============================

precision_list = []
recall_list = []
hitrate_list = []
ndcg_list = []

for user_id, relevant_items in user_relevant_items.items():

    recommended_items = recommend_popular_for_user(
        user_id=user_id,
        top_n=10
    )

    precision_list.append(
        precision_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )

    recall_list.append(
        recall_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )

    hitrate_list.append(
        hitrate_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )

    ndcg_list.append(
        ndcg_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )


print("\n===== Popular Baseline 全体用户评估 =====")

print("评估用户数量：", len(user_relevant_items))
print("Precision@10 =", sum(precision_list) / len(precision_list))
print("Recall@10 =", sum(recall_list) / len(recall_list))
print("HitRate@10 =", sum(hitrate_list) / len(hitrate_list))
print("NDCG@10 =", sum(ndcg_list) / len(ndcg_list))

# ==============================
# 使用训练集重新构建 ItemCF
# ==============================

train_interactions = train_ratings.copy()

train_interactions["liked"] = (
    train_interactions["rating"] >= 4
).astype(int)


# 构建 用户-电影 喜欢矩阵
train_user_item_matrix = (
    train_interactions
    .pivot(
        index="userId",
        columns="movieId",
        values="liked"
    )
    .fillna(0)
)


# 转置成 电影-用户 矩阵
train_item_user_matrix = train_user_item_matrix.T


# 转成稀疏矩阵
train_item_user_sparse = csr_matrix(
    train_item_user_matrix.values
)


# 构建 ItemCF 最近邻模型
itemcf_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

itemcf_model.fit(
    train_item_user_sparse
)


print("\n===== 基于训练集的 ItemCF 模型 =====")
print(
    "用户-电影矩阵：",
    train_user_item_matrix.shape
)

print(
    "电影-用户矩阵：",
    train_item_user_matrix.shape
)

print("ItemCF 模型构建完成")

# ==============================
# 训练集版 ItemCF 推荐函数
# ==============================

def recommend_itemcf_for_user(
    user_id,
    top_n=10,
    neighbors_per_item=20
):
    """
    使用仅基于训练集构建的 ItemCF
    为指定用户生成 Top-N 推荐。

    参数：
        user_id: 用户 ID
        top_n: 最终推荐数量
        neighbors_per_item: 每部历史喜欢电影寻找多少个相似电影

    返回：
        推荐电影 ID 列表
    """

    # 用户训练集中的历史记录
    user_history = train_interactions[
        train_interactions["userId"] == user_id
    ]

    if user_history.empty:
        return []


    # 用户训练集中喜欢过的电影
    liked_movies = user_history[
        user_history["liked"] == 1
    ]["movieId"].tolist()


    # 用户训练集中已经看过的所有电影
    watched_movies = set(
        user_history["movieId"].tolist()
    )


    # 保存候选电影得分
    candidate_scores = defaultdict(float)


    # 遍历用户喜欢过的电影
    for movie_id in liked_movies:

        # 如果电影不在训练 ItemCF 矩阵里，跳过
        if movie_id not in train_item_user_matrix.index:
            continue

        movie_index = train_item_user_matrix.index.get_loc(
            movie_id
        )

        distances, indices = itemcf_model.kneighbors(
            train_item_user_sparse[movie_index],
            n_neighbors=neighbors_per_item + 1
        )

        for distance, index in zip(
            distances[0][1:],
            indices[0][1:]
        ):

            candidate_movie_id = (
                train_item_user_matrix.index[index]
            )

            # 已经看过的不推荐
            if candidate_movie_id in watched_movies:
                continue

            similarity = 1 - distance

            candidate_scores[candidate_movie_id] += similarity


    # 按 ItemCF 得分排序
    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # 只返回电影 ID
    recommendations = [
        int(movie_id)
        for movie_id, score in sorted_candidates[:top_n]
    ]

    return recommendations

itemcf_recommendations = recommend_itemcf_for_user(
    user_id=1,
    top_n=10
)

print("\n===== 用户 1 的 ItemCF Top-10 =====")
print(itemcf_recommendations)

# ==============================
# 评估用户 1 的 ItemCF
# ==============================

user_id = 1

itemcf_recommended_items = recommend_itemcf_for_user(
    user_id=user_id,
    top_n=10
)

relevant_items = user_relevant_items.get(
    user_id,
    []
)

print("\n===== 用户 1 的 ItemCF 评估 =====")

print(
    "Precision@10 =",
    precision_at_k(
        itemcf_recommended_items,
        relevant_items,
        k=10
    )
)

print(
    "Recall@10 =",
    recall_at_k(
        itemcf_recommended_items,
        relevant_items,
        k=10
    )
)

print(
    "HitRate@10 =",
    hitrate_at_k(
        itemcf_recommended_items,
        relevant_items,
        k=10
    )
)

print(
    "NDCG@10 =",
    ndcg_at_k(
        itemcf_recommended_items,
        relevant_items,
        k=10
    )
)

# ==============================
# ItemCF 全体用户评估
# ==============================

itemcf_precision_list = []
itemcf_recall_list = []
itemcf_hitrate_list = []
itemcf_ndcg_list = []

for user_id, relevant_items in user_relevant_items.items():

    recommended_items = recommend_itemcf_for_user(
        user_id=user_id,
        top_n=10
    )

    itemcf_precision_list.append(
        precision_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )

    itemcf_recall_list.append(
        recall_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )

    itemcf_hitrate_list.append(
        hitrate_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )

    itemcf_ndcg_list.append(
        ndcg_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )


print("\n===== ItemCF 全体用户评估 =====")

print("评估用户数量：", len(user_relevant_items))

print(
    "Precision@10 =",
    sum(itemcf_precision_list) / len(itemcf_precision_list)
)

print(
    "Recall@10 =",
    sum(itemcf_recall_list) / len(itemcf_recall_list)
)

print(
    "HitRate@10 =",
    sum(itemcf_hitrate_list) / len(itemcf_hitrate_list)
)

print(
    "NDCG@10 =",
    sum(itemcf_ndcg_list) / len(itemcf_ndcg_list)
)

# ==============================
# 使用训练集重新构建 UserCF
# ==============================

# UserCF 直接使用 用户-电影 矩阵
train_user_item_sparse = csr_matrix(
    train_user_item_matrix.values
)

usercf_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

usercf_model.fit(
    train_user_item_sparse
)

print("\n===== 基于训练集的 UserCF 模型 =====")

print(
    "用户-电影矩阵：",
    train_user_item_matrix.shape
)

print("UserCF 模型构建完成")

# ==============================
# 训练集版 UserCF 推荐函数
# ==============================

def recommend_usercf_for_user(
    user_id,
    top_n=10,
    similar_user_count=20
):
    """
    使用仅基于训练集构建的 UserCF
    为指定用户生成 Top-N 推荐。

    参数：
        user_id: 目标用户 ID
        top_n: 最终推荐电影数量
        similar_user_count: 使用多少个相似用户

    返回：
        推荐电影 ID 列表
    """

    # ------------------------------
    # 1. 检查用户是否存在
    # ------------------------------

    if user_id not in train_user_item_matrix.index:
        return []


    # ------------------------------
    # 2. 找目标用户在矩阵中的位置
    # ------------------------------

    user_index = train_user_item_matrix.index.get_loc(
        user_id
    )


    # ------------------------------
    # 3. 找最相似的用户
    # ------------------------------

    distances, indices = usercf_model.kneighbors(
        train_user_item_sparse[user_index],
        n_neighbors=similar_user_count + 1
    )


    # ------------------------------
    # 4. 用户训练集中已经看过的电影
    # ------------------------------

    watched_movies = set(
        train_interactions[
            train_interactions["userId"] == user_id
        ]["movieId"].tolist()
    )


    # ------------------------------
    # 5. 收集候选电影及得分
    # ------------------------------

    candidate_scores = defaultdict(float)

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        similar_user_id = (
            train_user_item_matrix.index[index]
        )

        similarity = 1 - distance

        # 找这个相似用户喜欢的电影
        liked_movies = train_interactions[
            (train_interactions["userId"] == similar_user_id)
            &
            (train_interactions["liked"] == 1)
        ]["movieId"].tolist()

        for movie_id in liked_movies:

            # 目标用户已经看过的电影不推荐
            if movie_id in watched_movies:
                continue

            candidate_scores[movie_id] += similarity


    # ------------------------------
    # 6. 按 UserCF 得分排序
    # ------------------------------

    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    recommendations = [
        int(movie_id)
        for movie_id, score in sorted_candidates[:top_n]
    ]

    return recommendations

# ==============================
# 测试用户 1 的 UserCF 推荐
# ==============================

usercf_recommendations = recommend_usercf_for_user(
    user_id=1,
    top_n=10,
    similar_user_count=20
)

print("\n===== 用户 1 的 UserCF Top-10 =====")
print(usercf_recommendations)

# ==============================
# 评估用户 1 的 UserCF
# ==============================

user_id = 1

usercf_recommended_items = recommend_usercf_for_user(
    user_id=user_id,
    top_n=10,
    similar_user_count=20
)

relevant_items = user_relevant_items.get(
    user_id,
    []
)

print("\n===== 用户 1 的 UserCF 评估 =====")

print(
    "Precision@10 =",
    precision_at_k(
        usercf_recommended_items,
        relevant_items,
        k=10
    )
)

print(
    "Recall@10 =",
    recall_at_k(
        usercf_recommended_items,
        relevant_items,
        k=10
    )
)

print(
    "HitRate@10 =",
    hitrate_at_k(
        usercf_recommended_items,
        relevant_items,
        k=10
    )
)

print(
    "NDCG@10 =",
    ndcg_at_k(
        usercf_recommended_items,
        relevant_items,
        k=10
    )
)

# ==============================
# UserCF 全体用户评估
# ==============================

usercf_precision_list = []
usercf_recall_list = []
usercf_hitrate_list = []
usercf_ndcg_list = []

for user_id, relevant_items in user_relevant_items.items():

    recommended_items = recommend_usercf_for_user(
        user_id=user_id,
        top_n=10,
        similar_user_count=20
    )

    usercf_precision_list.append(
        precision_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )

    usercf_recall_list.append(
        recall_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )

    usercf_hitrate_list.append(
        hitrate_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )

    usercf_ndcg_list.append(
        ndcg_at_k(
            recommended_items,
            relevant_items,
            k=10
        )
    )


print("\n===== UserCF 全体用户评估 =====")

print("评估用户数量：", len(user_relevant_items))

print(
    "Precision@10 =",
    sum(usercf_precision_list) / len(usercf_precision_list)
)

print(
    "Recall@10 =",
    sum(usercf_recall_list) / len(usercf_recall_list)
)

print(
    "HitRate@10 =",
    sum(usercf_hitrate_list) / len(usercf_hitrate_list)
)

print(
    "NDCG@10 =",
    sum(usercf_ndcg_list) / len(usercf_ndcg_list)
)

# ==============================
# 保存模型对比结果
# ==============================

RESULTS_DIR = PROJECT_ROOT / "results"

model_comparison = pd.DataFrame({
    "model": [
        "Popular",
        "ItemCF",
        "UserCF"
    ],

    "Precision@10": [
        sum(precision_list) / len(precision_list),
        sum(itemcf_precision_list) / len(itemcf_precision_list),
        sum(usercf_precision_list) / len(usercf_precision_list)
    ],

    "Recall@10": [
        sum(recall_list) / len(recall_list),
        sum(itemcf_recall_list) / len(itemcf_recall_list),
        sum(usercf_recall_list) / len(usercf_recall_list)
    ],

    "HitRate@10": [
        sum(hitrate_list) / len(hitrate_list),
        sum(itemcf_hitrate_list) / len(itemcf_hitrate_list),
        sum(usercf_hitrate_list) / len(usercf_hitrate_list)
    ],

    "NDCG@10": [
        sum(ndcg_list) / len(ndcg_list),
        sum(itemcf_ndcg_list) / len(itemcf_ndcg_list),
        sum(usercf_ndcg_list) / len(usercf_ndcg_list)
    ]
})


print("\n===== 模型效果对比 =====")
print(
    model_comparison.to_string(
        index=False
    )
)


model_comparison.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

print("\n模型对比结果已保存到：")
print(
    RESULTS_DIR / "model_comparison.csv"
)