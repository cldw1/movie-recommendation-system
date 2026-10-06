from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

import pandas as pd
from scipy.sparse import csr_matrix
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
# 3. 构建用户喜欢行为
# ==============================

interaction_data = ratings.copy()

interaction_data["liked"] = (
    interaction_data["rating"] >= 4
).astype(int)


# ==============================
# 4. 构建用户-电影喜欢矩阵
# ==============================

user_item_matrix = (
    interaction_data.pivot(
        index="userId",
        columns="movieId",
        values="liked"
    )
    .fillna(0)
)

print("===== UserCF 用户-电影喜欢矩阵 =====")
print("矩阵形状：", user_item_matrix.shape)

print("\n前 5 个用户、前 10 部电影：")
print(user_item_matrix.iloc[:5, :10])


# ==============================
# 5. 转换为稀疏矩阵
# ==============================

user_item_sparse = csr_matrix(
    user_item_matrix.values
)

print("\n稀疏矩阵形状：", user_item_sparse.shape)

# ==============================
# 6. 构建 UserCF 相似度模型
# ==============================

usercf_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

usercf_model.fit(user_item_sparse)

print("\nUserCF 模型构建完成")

# ==============================
# 7. 查找相似用户函数
# ==============================

def get_similar_users(user_id, top_n=5):
    """
    查找与指定用户最相似的用户。

    参数：
        user_id: 目标用户 ID
        top_n: 返回相似用户数量

    返回：
        相似用户及相似度
    """

    if user_id not in user_item_matrix.index:
        print(f"userId={user_id} 不存在")
        return None

    user_index = user_item_matrix.index.get_loc(user_id)

    distances, indices = usercf_model.kneighbors(
        user_item_sparse[user_index],
        n_neighbors=top_n + 1
    )

    results = []

    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):
        similar_user_id = user_item_matrix.index[index]

        results.append({
            "userId": similar_user_id,
            "similarity": 1 - distance
        })

    return pd.DataFrame(results)

similar_users = get_similar_users(
    user_id=1,
    top_n=5
)

print("\n===== 与用户 1 最相似的用户 =====")
print(similar_users.to_string(index=False))

# ==============================
# 8. UserCF 用户推荐函数
# ==============================

def recommend_for_user(user_id, top_n=10, similar_user_count=20):
    """
    使用 UserCF 为指定用户生成推荐结果。

    参数：
        user_id: 目标用户 ID
        top_n: 最终推荐电影数量
        similar_user_count: 使用多少个相似用户参与推荐

    返回：
        Top-N 推荐结果
    """

    # ------------------------------
    # 1. 获取目标用户历史行为
    # ------------------------------

    user_history = interaction_data[
        interaction_data["userId"] == user_id
    ]

    if user_history.empty:
        print(f"userId={user_id} 不存在")
        return None


    # 用户已经看过的电影
    watched_movies = set(
        user_history["movieId"].tolist()
    )


    # ------------------------------
    # 2. 找到相似用户
    # ------------------------------

    similar_users = get_similar_users(
        user_id=user_id,
        top_n=similar_user_count
    )


    # ------------------------------
    # 3. 计算候选电影得分
    # ------------------------------

    candidate_scores = defaultdict(float)

    for _, row in similar_users.iterrows():

        similar_user_id = int(row["userId"])
        similarity = row["similarity"]

        # 找到这个相似用户喜欢的电影
        liked_movies = interaction_data[
            (interaction_data["userId"] == similar_user_id)
            & (interaction_data["liked"] == 1)
        ]["movieId"].tolist()

        for movie_id in liked_movies:

            # 已经看过的电影不推荐
            if movie_id in watched_movies:
                continue

            candidate_scores[movie_id] += similarity


    # ------------------------------
    # 4. 推荐分数排序
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
            "usercf_score": score
        })

    return pd.DataFrame(results)

recommendations = recommend_for_user(
    user_id=1,
    top_n=10,
    similar_user_count=20
)

print("\n===== 用户 1 的 UserCF 推荐结果 =====")
print(recommendations.to_string(index=False))

# ==============================
# 9. 保存 UserCF 推荐结果
# ==============================

RESULTS_DIR = PROJECT_ROOT / "results"

recommendations.to_csv(
    RESULTS_DIR / "usercf_recommendations.csv",
    index=False
)

print("\nUserCF 推荐结果已保存到：")
print(RESULTS_DIR / "usercf_recommendations.csv")

# ==============================
# 10. 保存 UserCF 模型
# ==============================

MODELS_DIR = PROJECT_ROOT / "models"

usercf_model_data = {
    "model": usercf_model,
    "user_ids": user_item_matrix.index.to_list(),
    "user_item_sparse": user_item_sparse
}

joblib.dump(
    usercf_model_data,
    MODELS_DIR / "usercf_model.pkl"
)

print("\nUserCF 模型已保存到：")
print(MODELS_DIR / "usercf_model.pkl")

# ==============================
# 11. 测试重新加载 UserCF 模型
# ==============================

loaded_usercf_data = joblib.load(
    MODELS_DIR / "usercf_model.pkl"
)

loaded_model = loaded_usercf_data["model"]
loaded_user_ids = loaded_usercf_data["user_ids"]
loaded_sparse_matrix = loaded_usercf_data["user_item_sparse"]

print("\n===== UserCF 模型加载测试 =====")
print("模型加载成功")
print("用户数量：", len(loaded_user_ids))
print("矩阵形状：", loaded_sparse_matrix.shape)