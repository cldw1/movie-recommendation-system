from pathlib import Path

import pandas as pd
import numpy as np

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
# 3. 统计每部电影的评分情况
# ==============================

movie_stats = (
    ratings.groupby("movieId")
    .agg(
        rating_count=("rating", "count"),
        avg_rating=("rating", "mean")
    )
    .reset_index()
)

print("===== 电影评分统计 =====")
print(movie_stats.head(10).to_string(index=False))

# ==============================
# 4. 合并电影信息
# ==============================

movie_stats = movie_stats.merge(
    movies,
    on="movieId",
    how="left"
)

print("\n===== 合并电影信息后 =====")
print(
    movie_stats[
        ["movieId", "title", "rating_count", "avg_rating"]
    ].head(10).to_string(index=False)
)

# ==============================
# 5. 热门推荐函数
# ==============================

def get_popular_recommendations(movie_stats, top_n=10):
    """
    根据平均评分和评分人数计算热门推荐结果。

    参数：
        movie_stats: 电影评分统计数据
        top_n: 返回前多少部电影

    返回：
        热门电影推荐结果
    """

    result = movie_stats.copy()

    result["popular_score"] = (
        result["avg_rating"]
        * np.log1p(result["rating_count"])
    )

    result = (
        result
        .sort_values(
            by="popular_score",
            ascending=False
        )
        .head(top_n)
    )

    return result

# ==============================
# 6. 生成 Top 20 热门推荐
# ==============================

popular_movies = get_popular_recommendations(
    movie_stats,
    top_n=20
)

print("\n===== 热门推荐 Top 20 =====")

print(
    popular_movies[
        [
            "movieId",
            "title",
            "rating_count",
            "avg_rating",
            "popular_score"
        ]
    ].to_string(index=False)
)

# ==============================
# 7. 保存结果
# ==============================

RESULTS_DIR = PROJECT_ROOT / "results"

popular_movies.to_csv(
    RESULTS_DIR / "popular_movies.csv",
    index=False
)