from pathlib import Path
from collections import defaultdict

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# =========================================================
# 1. 设置项目路径
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_ratings.csv"
)

MOVIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "movies.csv"
)

RANKER_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ranker_data.csv"
)


# =========================================================
# 2. 读取数据
# =========================================================

train_ratings = pd.read_csv(
    TRAIN_PATH
)

movies = pd.read_csv(
    MOVIES_PATH
)


print("===== 原始训练数据 =====")

print(
    "train_ratings 数量：",
    len(train_ratings)
)

print(
    "movies 数量：",
    len(movies)
)


# =========================================================
# 3. 按用户和时间排序
# =========================================================

train_ratings = (
    train_ratings
    .sort_values(
        by=[
            "userId",
            "timestamp"
        ]
    )
    .reset_index(drop=True)
)


# =========================================================
# 4. 再次按用户进行时间划分
# =========================================================

history_list = []
target_list = []


for user_id, user_data in train_ratings.groupby(
    "userId"
):

    split_index = int(
        len(user_data) * 0.8
    )

    history_part = (
        user_data.iloc[:split_index]
    )

    target_part = (
        user_data.iloc[split_index:]
    )

    history_list.append(
        history_part
    )

    target_list.append(
        target_part
    )


rank_history = pd.concat(
    history_list,
    ignore_index=True
)

rank_target = pd.concat(
    target_list,
    ignore_index=True
)


print(
    "\n===== Ranker 数据划分 ====="
)

print(
    "rank_history 数量：",
    len(rank_history)
)

print(
    "rank_target 数量：",
    len(rank_target)
)

print(
    "rank_history 占比：",
    round(
        len(rank_history)
        / len(train_ratings)
        * 100,
        2
    ),
    "%"
)

print(
    "rank_target 占比：",
    round(
        len(rank_target)
        / len(train_ratings)
        * 100,
        2
    ),
    "%"
)


# =========================================================
# 5. 检查时间划分
# =========================================================

time_check = (
    rank_history
    .groupby("userId")["timestamp"]
    .max()
    <=
    rank_target
    .groupby("userId")["timestamp"]
    .min()
)


error_user_count = (
    ~time_check
).sum()


print(
    "\n===== Ranker 时间划分检查 ====="
)

print(
    "检查用户数量：",
    len(time_check)
)

print(
    "时间划分异常用户数量：",
    error_user_count
)


# =========================================================
# 6. 构造 Ranker 标签
# =========================================================

rank_target = (
    rank_target.copy()
)

rank_target["label"] = (
    rank_target["rating"] >= 4
).astype(int)


print(
    "\n===== Ranker 标签统计 ====="
)

print(
    rank_target["label"]
    .value_counts()
)


print(
    "\n标签比例："
)

print(
    rank_target["label"]
    .value_counts(
        normalize=True
    )
)


# =========================================================
# 7. 提取 Ranker 正样本
# =========================================================

rank_positive = (
    rank_target[
        rank_target["label"] == 1
    ]
    .copy()
)


user_positive_items = (
    rank_positive
    .groupby(
        "userId"
    )[
        "movieId"
    ]
    .apply(set)
    .to_dict()
)


print(
    "\n===== Ranker 正样本 ====="
)

print(
    "正样本数量：",
    len(rank_positive)
)

print(
    "有正样本的用户数量：",
    len(user_positive_items)
)

print(
    "用户 1 的正样本电影数量：",
    len(
        user_positive_items.get(
            1,
            set()
        )
    )
)


# =========================================================
# 8. 构造用户电影类型偏好
# =========================================================

# 只使用 rank_history 中 rating >= 4 的电影
liked_history = (
    rank_history[
        rank_history[
            "rating"
        ] >= 4
    ]
    .copy()
)


# 加入电影 genres
liked_history = (
    liked_history
    .merge(
        movies[
            [
                "movieId",
                "genres"
            ]
        ],
        on="movieId",
        how="left"
    )
)


liked_history[
    "genres"
] = (
    liked_history[
        "genres"
    ]
    .fillna(
        "(no genres listed)"
    )
)


print(
    "\n===== 用户喜欢电影的类型数据 ====="
)

print(
    liked_history[
        [
            "userId",
            "movieId",
            "rating",
            "genres"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# =========================================================
# 9. 拆分电影类型
# =========================================================

user_genre_data = (
    liked_history[
        [
            "userId",
            "genres"
        ]
    ]
    .assign(
        genres=lambda x:
        x["genres"]
        .str.split("|")
    )
    .explode(
        "genres"
    )
)


print(
    "\n===== 拆分后的用户类型偏好 ====="
)

print(
    user_genre_data
    .head(20)
    .to_string(
        index=False
    )
)


# =========================================================
# 10. 统计用户喜欢各种类型次数
# =========================================================

user_genre_preference = (
    user_genre_data
    .groupby(
        [
            "userId",
            "genres"
        ]
    )
    .size()
    .reset_index(
        name="genre_like_count"
    )
)


# =========================================================
# 11. 构造用户类型偏好强度
# =========================================================

# 每个用户最常喜欢的类型 = 1
# 其他类型按照喜欢次数进行相对缩放
user_genre_preference[
    "genre_preference"
] = (
    user_genre_preference[
        "genre_like_count"
    ]
    /
    user_genre_preference
    .groupby(
        "userId"
    )[
        "genre_like_count"
    ]
    .transform(
        "max"
    )
)


print(
    "\n===== 用户 1 类型偏好强度 ====="
)

print(
    user_genre_preference[
        user_genre_preference[
            "userId"
        ] == 1
    ]
    .sort_values(
        by="genre_preference",
        ascending=False
    )
    .to_string(
        index=False
    )
)


# =========================================================
# 12. 类型匹配特征函数
# =========================================================

def add_genre_match_score(
    candidate_pool
):
    """
    计算候选电影与用户历史类型偏好的匹配程度。

    返回：
        genre_match_score

    范围：
        0 ~ 1

    数值越高：
        候选电影越符合用户历史类型兴趣。
    """

    if candidate_pool.empty:

        result = (
            candidate_pool.copy()
        )

        result[
            "genre_match_score"
        ] = pd.Series(
            dtype="float64"
        )

        return result


    # 只保留计算需要的字段
    candidate_genres = (
        candidate_pool[
            [
                "userId",
                "movieId"
            ]
        ]
        .copy()
    )


    # 加入候选电影 genres
    candidate_genres = (
        candidate_genres
        .merge(
            movies[
                [
                    "movieId",
                    "genres"
                ]
            ],
            on="movieId",
            how="left"
        )
    )


    candidate_genres[
        "genres"
    ] = (
        candidate_genres[
            "genres"
        ]
        .fillna(
            "(no genres listed)"
        )
    )


    # 拆分多类型
    candidate_genres[
        "genres"
    ] = (
        candidate_genres[
            "genres"
        ]
        .str.split("|")
    )


    candidate_genres = (
        candidate_genres
        .explode(
            "genres"
        )
    )


    # 加入用户对每个类型的偏好
    candidate_genres = (
        candidate_genres
        .merge(
            user_genre_preference[
                [
                    "userId",
                    "genres",
                    "genre_preference"
                ]
            ],
            on=[
                "userId",
                "genres"
            ],
            how="left"
        )
    )


    # 用户以前没有喜欢过的类型记为 0
    candidate_genres[
        "genre_preference"
    ] = (
        candidate_genres[
            "genre_preference"
        ]
        .fillna(0.0)
    )


    # 一部电影可能有多个类型
    # 取这些类型偏好的平均值
    genre_match = (
        candidate_genres
        .groupby(
            [
                "userId",
                "movieId"
            ],
            as_index=False
        )[
            "genre_preference"
        ]
        .mean()
        .rename(
            columns={
                "genre_preference":
                "genre_match_score"
            }
        )
    )


    result = (
        candidate_pool
        .merge(
            genre_match,
            on=[
                "userId",
                "movieId"
            ],
            how="left"
        )
    )


    result[
        "genre_match_score"
    ] = (
        result[
            "genre_match_score"
        ]
        .fillna(0.0)
    )


    return result


# =========================================================
# 13. 基于 rank_history 构建 ItemCF
# =========================================================

rank_interactions = (
    rank_history.copy()
)


rank_interactions[
    "liked"
] = (
    rank_interactions[
        "rating"
    ] >= 4
).astype(int)


# 用户 × 电影喜欢矩阵
rank_user_item_matrix = (
    rank_interactions
    .pivot(
        index="userId",
        columns="movieId",
        values="liked"
    )
    .fillna(0)
)


# ItemCF 使用：
# 电影 × 用户
rank_item_user_matrix = (
    rank_user_item_matrix.T
)


rank_item_user_sparse = (
    csr_matrix(
        rank_item_user_matrix.values
    )
)


rank_itemcf_model = (
    NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )
)


rank_itemcf_model.fit(
    rank_item_user_sparse
)


print(
    "\n===== Ranker ItemCF 模型 ====="
)

print(
    "用户-电影矩阵：",
    rank_user_item_matrix.shape
)

print(
    "电影-用户矩阵：",
    rank_item_user_matrix.shape
)

print(
    "Ranker ItemCF 模型构建完成"
)


# =========================================================
# 14. ItemCF 候选召回函数
# =========================================================

def recall_itemcf_candidates(
    user_id,
    top_n=100,
    neighbors_per_item=30
):

    user_history = (
        rank_interactions[
            rank_interactions[
                "userId"
            ] == user_id
        ]
    )


    if user_history.empty:

        return pd.DataFrame({
            "movieId":
            pd.Series(
                dtype="int64"
            ),

            "itemcf_score":
            pd.Series(
                dtype="float64"
            )
        })


    liked_movies = (
        user_history[
            user_history[
                "liked"
            ] == 1
        ][
            "movieId"
        ]
        .tolist()
    )


    watched_movies = set(
        user_history[
            "movieId"
        ]
        .tolist()
    )


    candidate_scores = (
        defaultdict(float)
    )


    for movie_id in liked_movies:

        if (
            movie_id
            not in
            rank_item_user_matrix.index
        ):
            continue


        movie_index = (
            rank_item_user_matrix
            .index
            .get_loc(
                movie_id
            )
        )


        # 防止邻居数量超过电影总数
        actual_neighbors = min(
            neighbors_per_item + 1,
            len(
                rank_item_user_matrix
            )
        )


        distances, indices = (
            rank_itemcf_model
            .kneighbors(
                rank_item_user_sparse[
                    movie_index
                ],
                n_neighbors=(
                    actual_neighbors
                )
            )
        )


        for distance, index in zip(
            distances[0][1:],
            indices[0][1:]
        ):

            candidate_movie_id = (
                rank_item_user_matrix
                .index[index]
            )


            if (
                candidate_movie_id
                in watched_movies
            ):
                continue


            similarity = (
                1 - distance
            )


            candidate_scores[
                candidate_movie_id
            ] += similarity


    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]


    result = pd.DataFrame(
        sorted_candidates,
        columns=[
            "movieId",
            "itemcf_score"
        ]
    )


    if not result.empty:

        result[
            "movieId"
        ] = (
            result[
                "movieId"
            ]
            .astype(int)
        )

        result[
            "itemcf_score"
        ] = (
            result[
                "itemcf_score"
            ]
            .astype(float)
        )


    return result


# =========================================================
# 15. 基于 rank_history 构建 UserCF
# =========================================================

rank_user_item_sparse = (
    csr_matrix(
        rank_user_item_matrix.values
    )
)


rank_usercf_model = (
    NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )
)


rank_usercf_model.fit(
    rank_user_item_sparse
)


print(
    "\n===== Ranker UserCF 模型 ====="
)

print(
    "用户-电影矩阵：",
    rank_user_item_matrix.shape
)

print(
    "Ranker UserCF 模型构建完成"
)


# =========================================================
# 16. UserCF 候选召回函数
# =========================================================

def recall_usercf_candidates(
    user_id,
    top_n=100,
    similar_user_count=30
):

    if (
        user_id
        not in
        rank_user_item_matrix.index
    ):

        return pd.DataFrame({
            "movieId":
            pd.Series(
                dtype="int64"
            ),

            "usercf_score":
            pd.Series(
                dtype="float64"
            )
        })


    user_index = (
        rank_user_item_matrix
        .index
        .get_loc(
            user_id
        )
    )


    # 防止邻居数量超过用户总数
    actual_neighbors = min(
        similar_user_count + 1,
        len(
            rank_user_item_matrix
        )
    )


    distances, indices = (
        rank_usercf_model
        .kneighbors(
            rank_user_item_sparse[
                user_index
            ],
            n_neighbors=(
                actual_neighbors
            )
        )
    )


    watched_movies = set(
        rank_interactions[
            rank_interactions[
                "userId"
            ] == user_id
        ][
            "movieId"
        ]
        .tolist()
    )


    candidate_scores = (
        defaultdict(float)
    )


    for distance, index in zip(
        distances[0][1:],
        indices[0][1:]
    ):

        similar_user_id = (
            rank_user_item_matrix
            .index[index]
        )


        similarity = (
            1 - distance
        )


        liked_movies = (
            rank_interactions[
                (
                    rank_interactions[
                        "userId"
                    ]
                    ==
                    similar_user_id
                )
                &
                (
                    rank_interactions[
                        "liked"
                    ]
                    == 1
                )
            ][
                "movieId"
            ]
            .tolist()
        )


        for movie_id in liked_movies:

            if (
                movie_id
                in watched_movies
            ):
                continue


            candidate_scores[
                movie_id
            ] += similarity


    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]


    result = pd.DataFrame(
        sorted_candidates,
        columns=[
            "movieId",
            "usercf_score"
        ]
    )


    if not result.empty:

        result[
            "movieId"
        ] = (
            result[
                "movieId"
            ]
            .astype(int)
        )

        result[
            "usercf_score"
        ] = (
            result[
                "usercf_score"
            ]
            .astype(float)
        )


    return result


# =========================================================
# 17. 合并 ItemCF / UserCF 候选
#
# 新增：
# itemcf_rank
# usercf_rank
# =========================================================

def build_candidate_pool(
    user_id,
    itemcf_top_n=100,
    usercf_top_n=100
):

    # -----------------------------------------------------
    # 1. ItemCF 召回
    # -----------------------------------------------------

    itemcf_candidates = (
        recall_itemcf_candidates(
            user_id=user_id,
            top_n=itemcf_top_n,
            neighbors_per_item=30
        )
    )


    # 根据 ItemCF score 再次明确排序
    if not itemcf_candidates.empty:

        itemcf_candidates = (
            itemcf_candidates
            .sort_values(
                by="itemcf_score",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


        # 第一名 rank = 1
        # 第二名 rank = 2
        # ...
        itemcf_candidates[
            "itemcf_rank"
        ] = (
            itemcf_candidates.index
            + 1
        )

    else:

        itemcf_candidates[
            "itemcf_rank"
        ] = pd.Series(
            dtype="float64"
        )


    # -----------------------------------------------------
    # 2. UserCF 召回
    # -----------------------------------------------------

    usercf_candidates = (
        recall_usercf_candidates(
            user_id=user_id,
            top_n=usercf_top_n,
            similar_user_count=30
        )
    )


    if not usercf_candidates.empty:

        usercf_candidates = (
            usercf_candidates
            .sort_values(
                by="usercf_score",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


        usercf_candidates[
            "usercf_rank"
        ] = (
            usercf_candidates.index
            + 1
        )

    else:

        usercf_candidates[
            "usercf_rank"
        ] = pd.Series(
            dtype="float64"
        )


    # -----------------------------------------------------
    # 3. 合并两路召回
    # -----------------------------------------------------

    candidate_pool = (
        itemcf_candidates
        .merge(
            usercf_candidates,
            on="movieId",
            how="outer"
        )
    )


    # -----------------------------------------------------
    # 4. 某一路未召回到时，
    #    对应 score 填 0
    # -----------------------------------------------------

    candidate_pool[
        [
            "itemcf_score",
            "usercf_score"
        ]
    ] = (
        candidate_pool[
            [
                "itemcf_score",
                "usercf_score"
            ]
        ]
        .fillna(0.0)
    )


    # -----------------------------------------------------
    # 5. 某一路没有召回到的电影，
    #    rank 设置为 top_n + 1
    #
    # 例如 Top300：
    #
    # ItemCF 没召回
    # → itemcf_rank = 301
    #
    # UserCF 没召回
    # → usercf_rank = 301
    # -----------------------------------------------------

    candidate_pool[
        "itemcf_rank"
    ] = (
        candidate_pool[
            "itemcf_rank"
        ]
        .fillna(
            itemcf_top_n + 1
        )
        .astype(int)
    )


    candidate_pool[
        "usercf_rank"
    ] = (
        candidate_pool[
            "usercf_rank"
        ]
        .fillna(
            usercf_top_n + 1
        )
        .astype(int)
    )


    # -----------------------------------------------------
    # 6. 加入 userId
    # -----------------------------------------------------

    candidate_pool.insert(
        0,
        "userId",
        user_id
    )


    return candidate_pool


# =========================================================
# 18. 给候选电影添加标签
# =========================================================

def add_candidate_labels(
    candidate_pool,
    user_id
):

    positive_items = (
        user_positive_items.get(
            user_id,
            set()
        )
    )


    result = (
        candidate_pool.copy()
    )


    result[
        "label"
    ] = (
        result[
            "movieId"
        ]
        .isin(
            positive_items
        )
        .astype(int)
    )


    return result


# =========================================================
# 19. 构造用户特征
# =========================================================

user_features = (
    rank_history
    .groupby(
        "userId"
    )
    .agg(
        user_rating_count=(
            "rating",
            "count"
        ),

        user_avg_rating=(
            "rating",
            "mean"
        )
    )
    .reset_index()
)


print(
    "\n===== 用户特征 ====="
)

print(
    user_features
    .head(10)
    .to_string(
        index=False
    )
)


# =========================================================
# 20. 构造电影特征
# =========================================================

movie_features = (
    rank_history
    .groupby(
        "movieId"
    )
    .agg(
        movie_rating_count=(
            "rating",
            "count"
        ),

        movie_avg_rating=(
            "rating",
            "mean"
        )
    )
    .reset_index()
)


print(
    "\n===== 电影特征 ====="
)

print(
    movie_features
    .head(10)
    .to_string(
        index=False
    )
)


# =========================================================
# 21. 构造热门电影特征阈值
# =========================================================

popular_threshold = (
    movie_features[
        "movie_rating_count"
    ]
    .quantile(0.8)
)


print(
    "\n===== 热门电影阈值 ====="
)

print(
    "评分人数达到",
    popular_threshold,
    "及以上认为是热门电影"
)


# =========================================================
# 22. 测试用户 1 的完整候选特征
# =========================================================

candidate_pool_test = (
    build_candidate_pool(
        user_id=1,
        itemcf_top_n=20,
        usercf_top_n=20
    )
)


candidate_pool_test = (
    add_candidate_labels(
        candidate_pool_test,
        user_id=1
    )
)


candidate_pool_test = (
    candidate_pool_test
    .merge(
        user_features,
        on="userId",
        how="left"
    )
)


candidate_pool_test = (
    candidate_pool_test
    .merge(
        movie_features,
        on="movieId",
        how="left"
    )
)


candidate_pool_test = (
    add_genre_match_score(
        candidate_pool_test
    )
)


candidate_pool_test[
    "is_popular_movie"
] = (
    candidate_pool_test[
        "movie_rating_count"
    ]
    >=
    popular_threshold
).astype(int)


print(
    "\n===== 用户 1 完整候选特征 ====="
)


print(
    candidate_pool_test[
        [
            "userId",
            "movieId",

            "user_rating_count",
            "user_avg_rating",

            "movie_rating_count",
            "movie_avg_rating",

            "genre_match_score",

            "itemcf_score",
            "itemcf_rank",

            "usercf_score",
            "usercf_rank",

            "is_popular_movie",

            "label"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# =========================================================
# 23. 构造单个用户 Ranker 训练样本
# =========================================================

def build_ranker_samples_for_user(
    user_id
):

    # -----------------------------------------------------
    # 1. 多路召回
    # -----------------------------------------------------

    candidate_pool = (
        build_candidate_pool(
            user_id=user_id,
            itemcf_top_n=300,
            usercf_top_n=300
        )
    )


    if candidate_pool.empty:

        return candidate_pool


    # -----------------------------------------------------
    # 2. 添加 label
    # -----------------------------------------------------

    candidate_pool = (
        add_candidate_labels(
            candidate_pool,
            user_id
        )
    )


    # -----------------------------------------------------
    # 3. 添加用户特征
    # -----------------------------------------------------

    candidate_pool = (
        candidate_pool
        .merge(
            user_features,
            on="userId",
            how="left"
        )
    )


    # -----------------------------------------------------
    # 4. 添加电影特征
    # -----------------------------------------------------

    candidate_pool = (
        candidate_pool
        .merge(
            movie_features,
            on="movieId",
            how="left"
        )
    )


    # -----------------------------------------------------
    # 5. 添加电影类型匹配特征
    # -----------------------------------------------------

    candidate_pool = (
        add_genre_match_score(
            candidate_pool
        )
    )


    # -----------------------------------------------------
    # 6. 添加热门电影特征
    # -----------------------------------------------------

    candidate_pool[
        "is_popular_movie"
    ] = (
        candidate_pool[
            "movie_rating_count"
        ]
        >=
        popular_threshold
    ).astype(int)


    return candidate_pool


# =========================================================
# 24. 构造全部用户 Ranker 训练集
# =========================================================

ranker_sample_list = []


ranker_user_ids = list(
    user_positive_items.keys()
)


print(
    "\n===== 开始构造 Ranker 训练集 ====="
)

print(
    "用户数量：",
    len(
        ranker_user_ids
    )
)


for i, user_id in enumerate(
    ranker_user_ids,
    start=1
):

    user_samples = (
        build_ranker_samples_for_user(
            user_id
        )
    )


    if not user_samples.empty:

        ranker_sample_list.append(
            user_samples
        )


    if (
        i % 50 == 0
    ):

        print(
            f"已处理："
            f"{i}/"
            f"{len(ranker_user_ids)}"
        )


ranker_data = pd.concat(
    ranker_sample_list,
    ignore_index=True
)


# =========================================================
# 25. 查看 Ranker 训练集
# =========================================================

print(
    "\n===== Ranker 训练集 ====="
)

print(
    "训练样本数量：",
    len(
        ranker_data
    )
)

print(
    "用户数量：",
    ranker_data[
        "userId"
    ].nunique()
)

print(
    "电影数量：",
    ranker_data[
        "movieId"
    ].nunique()
)


print(
    "\n标签统计："
)

print(
    ranker_data[
        "label"
    ]
    .value_counts()
)


print(
    "\n标签比例："
)

print(
    ranker_data[
        "label"
    ]
    .value_counts(
        normalize=True
    )
)


print(
    "\n前 10 条训练样本："
)

print(
    ranker_data[
        [
            "userId",
            "movieId",

            "user_rating_count",
            "user_avg_rating",

            "movie_rating_count",
            "movie_avg_rating",

            "genre_match_score",

            "itemcf_score",
            "itemcf_rank",

            "usercf_score",
            "usercf_rank",

            "is_popular_movie",

            "label"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# =========================================================
# 26. 检查 Ranker 特征缺失值
# =========================================================

feature_check_columns = [

    "user_rating_count",
    "user_avg_rating",

    "movie_rating_count",
    "movie_avg_rating",

    "genre_match_score",

    "itemcf_score",
    "itemcf_rank",

    "usercf_score",
    "usercf_rank",

    "is_popular_movie"
]


print(
    "\n===== Ranker 特征缺失值 ====="
)

print(
    ranker_data[
        feature_check_columns
    ]
    .isnull()
    .sum()
)


# =========================================================
# 27. 检查新增召回排名特征
# =========================================================

print(
    "\n===== 新增召回排名特征检查 ====="
)

print(
    ranker_data[
        [
            "itemcf_rank",
            "usercf_rank"
        ]
    ]
    .describe()
)


print(
    "\n用户 1 的召回排名特征前 20 条："
)

print(
    ranker_data[
        ranker_data[
            "userId"
        ] == 1
    ][
        [
            "movieId",

            "itemcf_score",
            "itemcf_rank",

            "usercf_score",
            "usercf_rank",

            "label"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# =========================================================
# 28. 保存 Ranker 原始训练数据
# =========================================================

ranker_data.to_csv(
    RANKER_DATA_PATH,
    index=False
)


print(
    "\nRanker 数据已保存到："
)

print(
    RANKER_DATA_PATH
)


# =========================================================
# 29. 检查候选召回覆盖率
# =========================================================

total_positive_count = (
    len(
        rank_positive
    )
)


recalled_positive_count = (
    ranker_data[
        "label"
    ] == 1
).sum()


candidate_recall = (
    recalled_positive_count
    /
    total_positive_count
)


user_hit_stats = (
    ranker_data
    .groupby(
        "userId"
    )[
        "label"
    ]
    .sum()
)


hit_user_count = (
    user_hit_stats > 0
).sum()


print(
    "\n===== 候选召回覆盖情况 ====="
)

print(
    "rank_target 正样本数量：",
    total_positive_count
)

print(
    "候选池召回到的正样本数量：",
    recalled_positive_count
)

print(
    "候选正样本召回率：",
    round(
        candidate_recall,
        4
    )
)

print(
    "参与 Ranker 的用户数量：",
    ranker_data[
        "userId"
    ].nunique()
)

print(
    "候选池至少命中 1 个正样本的用户数量：",
    hit_user_count
)