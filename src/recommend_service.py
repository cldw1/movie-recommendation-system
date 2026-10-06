from pathlib import Path
from collections import defaultdict

import joblib
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# =========================================================
# 1. 项目路径
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

RANK_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "rank_model_ltr.pkl"
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


# =========================================================
# 3. 加载最终 XGBRanker
# =========================================================

rank_model_data = joblib.load(
    RANK_MODEL_PATH
)

rank_model = (
    rank_model_data[
        "model"
    ]
)

feature_columns = (
    rank_model_data[
        "feature_columns"
    ]
)


# =========================================================
# 4. 构建用户喜欢行为
# =========================================================

train_interactions = (
    train_ratings
    .copy()
)

train_interactions[
    "liked"
] = (
    train_interactions[
        "rating"
    ] >= 4
).astype(int)


# =========================================================
# 5. 构建用户-电影矩阵
# =========================================================

train_user_item_matrix = (
    train_interactions
    .pivot(
        index="userId",
        columns="movieId",
        values="liked"
    )
    .fillna(0)
)


# =========================================================
# 6. 初始化 ItemCF
# =========================================================

# ItemCF：
# 电影 × 用户
train_item_user_matrix = (
    train_user_item_matrix.T
)

train_item_user_sparse = (
    csr_matrix(
        train_item_user_matrix.values
    )
)

itemcf_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

itemcf_model.fit(
    train_item_user_sparse
)


# =========================================================
# 7. 初始化 UserCF
# =========================================================

train_user_item_sparse = (
    csr_matrix(
        train_user_item_matrix.values
    )
)

usercf_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

usercf_model.fit(
    train_user_item_sparse
)


# =========================================================
# 8. ItemCF 候选召回
# =========================================================

def recall_itemcf_candidates(
    user_id,
    top_n=300,
    neighbors_per_item=30
):
    """
    使用 ItemCF 为指定用户召回候选电影。

    参数：
        user_id:
            用户 ID

        top_n:
            最终保留多少部候选电影

        neighbors_per_item:
            对用户历史中每部喜欢的电影，
            查找多少部相似电影

    返回：
        DataFrame

        包含：
            movieId
            itemcf_score
    """

    user_history = (
        train_interactions[
            train_interactions[
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
            train_item_user_matrix.index
        ):
            continue


        movie_index = (
            train_item_user_matrix
            .index
            .get_loc(
                movie_id
            )
        )


        actual_neighbors = min(
            neighbors_per_item + 1,
            len(
                train_item_user_matrix
            )
        )


        distances, indices = (
            itemcf_model
            .kneighbors(
                train_item_user_sparse[
                    movie_index
                ],
                n_neighbors=actual_neighbors
            )
        )


        for distance, index in zip(
            distances[0][1:],
            indices[0][1:]
        ):

            candidate_movie_id = (
                train_item_user_matrix
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


    return result


# =========================================================
# 9. UserCF 候选召回
# =========================================================

def recall_usercf_candidates(
    user_id,
    top_n=300,
    similar_user_count=30
):
    """
    使用 UserCF 为指定用户召回候选电影。

    参数：
        user_id:
            用户 ID

        top_n:
            最终保留多少部候选电影

        similar_user_count:
            使用多少个相似用户

    返回：
        DataFrame

        包含：
            movieId
            usercf_score
    """

    if (
        user_id
        not in
        train_user_item_matrix.index
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
        train_user_item_matrix
        .index
        .get_loc(
            user_id
        )
    )


    actual_neighbors = min(
        similar_user_count + 1,
        len(
            train_user_item_matrix
        )
    )


    distances, indices = (
        usercf_model
        .kneighbors(
            train_user_item_sparse[
                user_index
            ],
            n_neighbors=actual_neighbors
        )
    )


    watched_movies = set(
        train_interactions[
            train_interactions[
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
            train_user_item_matrix
            .index[index]
        )


        similarity = (
            1 - distance
        )


        liked_movies = (
            train_interactions[
                (
                    train_interactions[
                        "userId"
                    ]
                    ==
                    similar_user_id
                )
                &
                (
                    train_interactions[
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


    return result


# =========================================================
# 10. 合并 ItemCF / UserCF 候选池
# =========================================================

def build_candidate_pool(
    user_id,
    itemcf_top_n=300,
    usercf_top_n=300
):
    """
    合并 ItemCF 和 UserCF 两路召回结果。

    返回字段：
        userId
        movieId
        itemcf_score
        usercf_score
    """

    itemcf_candidates = (
        recall_itemcf_candidates(
            user_id=user_id,
            top_n=itemcf_top_n,
            neighbors_per_item=30
        )
    )


    usercf_candidates = (
        recall_usercf_candidates(
            user_id=user_id,
            top_n=usercf_top_n,
            similar_user_count=30
        )
    )


    candidate_pool = (
        itemcf_candidates
        .merge(
            usercf_candidates,
            on="movieId",
            how="outer"
        )
    )


    if candidate_pool.empty:

        return pd.DataFrame({
            "userId":
                pd.Series(
                    dtype="int64"
                ),

            "movieId":
                pd.Series(
                    dtype="int64"
                ),

            "itemcf_score":
                pd.Series(
                    dtype="float64"
                ),

            "usercf_score":
                pd.Series(
                    dtype="float64"
                )
        })


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


    candidate_pool.insert(
        0,
        "userId",
        user_id
    )


    return candidate_pool


# =========================================================
# 11. 构造用户特征
# =========================================================

user_features = (
    train_ratings
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


# =========================================================
# 12. 构造电影特征
# =========================================================

movie_features = (
    train_ratings
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


# =========================================================
# 13. 热门电影阈值
# =========================================================

popular_threshold = (
    movie_features[
        "movie_rating_count"
    ]
    .quantile(
        0.8
    )
)


# =========================================================
# 14. 添加基础 Ranker 特征
# =========================================================

def add_basic_ranker_features(
    candidate_pool
):
    """
    添加 Ranker 基础特征：

        user_rating_count
        user_avg_rating
        movie_rating_count
        movie_avg_rating
        is_popular_movie
    """

    result = (
        candidate_pool
        .copy()
    )


    result = (
        result
        .merge(
            user_features,
            on="userId",
            how="left"
        )
    )


    result = (
        result
        .merge(
            movie_features,
            on="movieId",
            how="left"
        )
    )


    result[
        "is_popular_movie"
    ] = (
        result[
            "movie_rating_count"
        ]
        >=
        popular_threshold
    ).astype(int)


    return result


# =========================================================
# 15. 构造用户电影类型偏好
# =========================================================

liked_train = (
    train_ratings[
        train_ratings[
            "rating"
        ] >= 4
    ]
    .copy()
)


liked_train = (
    liked_train
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


liked_train[
    "genres"
] = (
    liked_train[
        "genres"
    ]
    .fillna(
        "(no genres listed)"
    )
)


user_genre_data = (
    liked_train[
        [
            "userId",
            "genres"
        ]
    ]
    .assign(
        genres=lambda x:
        x[
            "genres"
        ]
        .str.split(
            "|"
        )
    )
    .explode(
        "genres"
    )
)


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


# =========================================================
# 16. 添加 genre_match_score
# =========================================================

def add_genre_match_score(
    candidate_pool
):
    """
    计算候选电影与用户历史电影类型偏好的匹配程度。

    返回新增字段：

        genre_match_score

    范围：
        0 ~ 1
    """

    if candidate_pool.empty:

        result = (
            candidate_pool
            .copy()
        )

        result[
            "genre_match_score"
        ] = pd.Series(
            dtype="float64"
        )

        return result


    candidate_genres = (
        candidate_pool[
            [
                "userId",
                "movieId"
            ]
        ]
        .copy()
    )


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
        .str.split(
            "|"
        )
    )


    candidate_genres = (
        candidate_genres
        .explode(
            "genres"
        )
    )


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


    candidate_genres[
        "genre_preference"
    ] = (
        candidate_genres[
            "genre_preference"
        ]
        .fillna(
            0.0
        )
    )


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
        .fillna(
            0.0
        )
    )


    return result


# =========================================================
# 17. XGBRanker 排序
# =========================================================

def rank_candidates(
    candidate_pool,
    top_n=10
):
    """
    使用最终 XGBRanker 对候选电影重新排序。
    """

    if candidate_pool.empty:

        return candidate_pool


    candidate_features = (
        candidate_pool[
            feature_columns
        ]
        .apply(
            pd.to_numeric,
            errors="raise"
        )
        .astype(
            float
        )
    )


    result = (
        candidate_pool
        .copy()
    )


    result[
        "rank_score"
    ] = (
        rank_model
        .predict(
            candidate_features
        )
    )


    result = (
        result
        .sort_values(
            by="rank_score",
            ascending=False
        )
        .head(
            top_n
        )
        .copy()
    )


    return result


# =========================================================
# 18. 添加电影信息
# =========================================================

def add_movie_info(
    ranked_candidates
):
    """
    添加电影名称和电影类型。
    """

    result = (
        ranked_candidates
        .merge(
            movies[
                [
                    "movieId",
                    "title",
                    "genres"
                ]
            ],
            on="movieId",
            how="left"
        )
    )


    return result


# =========================================================
# 19. 最终推荐函数
# =========================================================

def recommend(
    user_id,
    top_n=10
):
    """
    为指定用户生成最终个性化电影推荐。

    流程：

        ItemCF Top300
              +
        UserCF Top300
              ↓
        多路召回候选池
              ↓
        构造 Ranker 特征
              ↓
        XGBRanker 排序
              ↓
        Top-N 推荐
    """

    if (
        user_id
        not in
        train_user_item_matrix.index
    ):

        raise ValueError(
            f"用户 {user_id} 不存在。"
        )


    candidate_pool = (
        build_candidate_pool(
            user_id=user_id,
            itemcf_top_n=300,
            usercf_top_n=300
        )
    )


    if candidate_pool.empty:

        return pd.DataFrame()


    candidate_pool = (
        add_basic_ranker_features(
            candidate_pool
        )
    )


    candidate_pool = (
        add_genre_match_score(
            candidate_pool
        )
    )


    missing_features = [
        feature
        for feature
        in feature_columns
        if feature
        not in candidate_pool.columns
    ]


    if missing_features:

        raise ValueError(
            "候选池缺少 Ranker 特征："
            f"{missing_features}"
        )


    ranked_candidates = (
        rank_candidates(
            candidate_pool,
            top_n=top_n
        )
    )


    recommendations = (
        add_movie_info(
            ranked_candidates
        )
    )


    recommendations.insert(
        0,
        "recommend_rank",
        range(
            1,
            len(
                recommendations
            ) + 1
        )
    )


    return recommendations


# =========================================================
# 20. 获取用户画像
# =========================================================

def get_user_profile(
    user_id,
    top_genres=5
):
    """
    获取指定用户的基础画像。

    返回：
        userId
        rating_count
        avg_rating
        liked_count
        top_genres
    """

    user_history = (
        train_ratings[
            train_ratings[
                "userId"
            ] == user_id
        ]
        .copy()
    )


    if user_history.empty:

        raise ValueError(
            f"用户 {user_id} 不存在。"
        )


    rating_count = (
        len(
            user_history
        )
    )


    avg_rating = (
        user_history[
            "rating"
        ]
        .mean()
    )


    liked_count = (
        user_history[
            "rating"
        ] >= 4
    ).sum()


    user_genres = (
        user_genre_preference[
            user_genre_preference[
                "userId"
            ] == user_id
        ]
        .sort_values(
            by=[
                "genre_preference",
                "genre_like_count"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(
            top_genres
        )
        .copy()
    )


    top_genre_list = (
        user_genres[
            "genres"
        ]
        .tolist()
    )


    profile = {

        "userId":
            int(
                user_id
            ),

        "rating_count":
            int(
                rating_count
            ),

        "avg_rating":
            round(
                float(
                    avg_rating
                ),
                2
            ),

        "liked_count":
            int(
                liked_count
            ),

        "top_genres":
            top_genre_list
    }


    return profile


# =========================================================
# 21. 获取用户最近喜欢的电影
# =========================================================

def get_recent_liked_movies(
    user_id,
    top_n=5
):
    """
    获取用户最近喜欢过的电影。

    喜欢定义：
        rating >= 4

    按 timestamp 从新到旧排序。
    """

    if (
        user_id
        not in
        train_user_item_matrix.index
    ):

        raise ValueError(
            f"用户 {user_id} 不存在。"
        )


    user_history = (
        train_ratings[
            (
                train_ratings[
                    "userId"
                ] == user_id
            )
            &
            (
                train_ratings[
                    "rating"
                ] >= 4
            )
        ]
        .copy()
    )


    if user_history.empty:

        return pd.DataFrame(
            columns=[
                "movieId",
                "rating",
                "timestamp",
                "title",
                "genres"
            ]
        )


    user_history = (
        user_history
        .sort_values(
            by="timestamp",
            ascending=False
        )
        .head(
            top_n
        )
    )


    result = (
        user_history[
            [
                "movieId",
                "rating",
                "timestamp"
            ]
        ]
        .merge(
            movies[
                [
                    "movieId",
                    "title",
                    "genres"
                ]
            ],
            on="movieId",
            how="left"
        )
    )


    return result


# =========================================================
# 22. 本地测试
# =========================================================

if __name__ == "__main__":

    test_user_id = 1


    print(
        f"\n===== 用户 {test_user_id} 推荐结果 ====="
    )

    recommendations = (
        recommend(
            user_id=test_user_id,
            top_n=10
        )
    )

    print(
        recommendations[
            [
                "recommend_rank",
                "movieId",
                "title",
                "genres",
                "rank_score"
            ]
        ]
        .to_string(
            index=False
        )
    )


    print(
        f"\n===== 用户 {test_user_id} 用户画像 ====="
    )

    profile = (
        get_user_profile(
            user_id=test_user_id
        )
    )

    print(
        profile
    )


    print(
        f"\n===== 用户 {test_user_id} 最近喜欢的电影 ====="
    )

    recent_movies = (
        get_recent_liked_movies(
            user_id=test_user_id,
            top_n=5
        )
    )

    print(
        recent_movies[
            [
                "title",
                "genres",
                "rating"
            ]
        ]
        .to_string(
            index=False
        )
    )