from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

import pandas as pd
import joblib


# ==============================
# 1. 设置项目路径
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_PATH = PROJECT_ROOT / "data" / "raw" / "ratings.csv"
MOVIES_PATH = PROJECT_ROOT / "data" / "raw" / "movies.csv"


# ==============================
# 2. 读取数据
# ==============================

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)


# ==============================
# 3. 构建用户-电影评分矩阵
# ==============================

user_item_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating"
)

print("===== 用户-电影评分矩阵 =====")
print("矩阵形状：", user_item_matrix.shape)

print("\n前 5 个用户、前 10 部电影：")
print(user_item_matrix.iloc[:5, :10])

# ==============================
# 4. 构建用户-电影喜欢矩阵
# ==============================

interaction_data = ratings.copy()

# rating >= 4 认为用户喜欢该电影
interaction_data["liked"] = (
    interaction_data["rating"] >= 4
).astype(int)

user_item_like_matrix = (
    interaction_data.pivot(
        index="userId",
        columns="movieId",
        values="liked"
    )
    .fillna(0)
)

print("\n===== 用户-电影喜欢矩阵 =====")
print("矩阵形状：", user_item_like_matrix.shape)

print("\n前 5 个用户、前 10 部电影：")
print(user_item_like_matrix.iloc[:5, :10])

# ==============================
# 5. 构建 ItemCF 相似度模型
# ==============================

# 转置：
# 原来是 用户 × 电影
# 现在变成 电影 × 用户
item_user_matrix = user_item_like_matrix.T

# 转换成稀疏矩阵
item_user_sparse = csr_matrix(item_user_matrix.values)

# 创建 ItemCF 模型
itemcf_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

# 训练模型
itemcf_model.fit(item_user_sparse)

print("\n===== ItemCF 模型 =====")
print("电影-用户矩阵形状：", item_user_matrix.shape)
print("ItemCF 模型构建完成")

# ==============================
# 6. 查找相似电影函数
# ==============================

def get_similar_movies(movie_id, top_n=5):
    """
    根据 ItemCF 查找与指定电影最相似的电影。

    参数：
        movie_id: 目标电影 ID
        top_n: 返回相似电影数量

    返回：
        相似电影列表
    """

    if movie_id not in item_user_matrix.index:
        print(f"movieId={movie_id} 不存在于评分数据中")
        return None

    target_index = item_user_matrix.index.get_loc(movie_id)

    distances, indices = itemcf_model.kneighbors(
        item_user_sparse[target_index],
        n_neighbors=top_n + 1
    )

    results = []

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):
        similar_movie_id = item_user_matrix.index[index]

        movie_info = movies.loc[
            movies["movieId"] == similar_movie_id
        ].iloc[0]

        results.append({
            "movieId": similar_movie_id,
            "title": movie_info["title"],
            "genres": movie_info["genres"],
            "similarity": 1 - distance
        })

    return pd.DataFrame(results)

similar_movies = get_similar_movies(
    movie_id=1,
    top_n=5
)

print("\n===== ItemCF 相似电影 =====")
print(similar_movies.to_string(index=False))

# ==============================
# 7. ItemCF 用户推荐函数
# ==============================

def recommend_for_user(user_id, top_n=10, neighbors_per_item=20):
    """
    使用 ItemCF 为指定用户生成推荐结果。

    参数：
        user_id: 用户 ID
        top_n: 最终推荐电影数量
        neighbors_per_item: 每部历史喜欢电影寻找多少部相似电影

    返回：
        用户的 Top-N 推荐结果
    """

    # ------------------------------
    # 1. 找到用户所有评分记录
    # ------------------------------

    user_history = interaction_data[
        interaction_data["userId"] == user_id
    ]

    if user_history.empty:
        print(f"userId={user_id} 不存在")
        return None


    # ------------------------------
    # 2. 找到用户喜欢过的电影
    # ------------------------------

    liked_movies = user_history[
        user_history["liked"] == 1
    ]["movieId"].tolist()


    # 用户已经看过的所有电影
    watched_movies = set(
        user_history["movieId"].tolist()
    )


    print(f"\n用户 {user_id}：")
    print("评分电影数量：", len(watched_movies))
    print("喜欢电影数量：", len(liked_movies))


    # ------------------------------
    # 3. 收集候选电影及其分数
    # ------------------------------

    candidate_scores = defaultdict(float)

    for movie_id in liked_movies:

        if movie_id not in item_user_matrix.index:
            continue

        movie_index = item_user_matrix.index.get_loc(movie_id)

        distances, indices = itemcf_model.kneighbors(
            item_user_sparse[movie_index],
            n_neighbors=neighbors_per_item + 1
        )

        for distance, index in zip(
            distances[0][1:],
            indices[0][1:]
        ):

            candidate_movie_id = item_user_matrix.index[index]

            # 已经看过的电影不再推荐
            if candidate_movie_id in watched_movies:
                continue

            similarity = 1 - distance

            candidate_scores[candidate_movie_id] += similarity


    # ------------------------------
    # 4. 按推荐得分排序
    # ------------------------------

    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]


    # ------------------------------
    # 5. 整理推荐结果
    # ------------------------------

    results = []

    for movie_id, score in sorted_candidates:

        movie_info = movies[
            movies["movieId"] == movie_id
        ].iloc[0]

        results.append({
            "movieId": movie_id,
            "title": movie_info["title"],
            "genres": movie_info["genres"],
            "itemcf_score": score
        })

    return pd.DataFrame(results)

# ==============================
# 8. 测试用户推荐
# ==============================

recommendations = recommend_for_user(
    user_id=1,
    top_n=10
)

print("\n===== 用户 1 的 ItemCF 推荐结果 =====")
print(recommendations.to_string(index=False))

# ==============================
# 9. 保存 ItemCF 推荐结果
# ==============================

RESULTS_DIR = PROJECT_ROOT / "results"

recommendations.to_csv(
    RESULTS_DIR / "itemcf_recommendations.csv",
    index=False
)

print("\nItemCF 推荐结果已保存到：")
print(RESULTS_DIR / "itemcf_recommendations.csv")

# ==============================
# 10. 保存 ItemCF 模型
# ==============================

MODELS_DIR = PROJECT_ROOT / "models"

itemcf_model_data = {
    "model": itemcf_model,
    "movie_ids": item_user_matrix.index.to_list(),
    "item_user_sparse": item_user_sparse
}

joblib.dump(
    itemcf_model_data,
    MODELS_DIR / "itemcf_model.pkl"
)

print("\nItemCF 模型已保存到：")
print(MODELS_DIR / "itemcf_model.pkl")

# ==============================
# 11. 测试重新加载 ItemCF 模型
# ==============================

loaded_itemcf_data = joblib.load(
    MODELS_DIR / "itemcf_model.pkl"
)

loaded_model = loaded_itemcf_data["model"]
loaded_movie_ids = loaded_itemcf_data["movie_ids"]
loaded_sparse_matrix = loaded_itemcf_data["item_user_sparse"]

print("\n===== ItemCF 模型加载测试 =====")
print("模型加载成功")
print("电影数量：", len(loaded_movie_ids))
print("矩阵形状：", loaded_sparse_matrix.shape)