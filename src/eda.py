from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# 1. 设置项目路径
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_PATH = PROJECT_ROOT / "data" / "raw" / "ratings.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MOVIES_PATH = PROJECT_ROOT / "data" / "raw" / "movies.csv"


# ==============================
# 2. 读取评分数据
# ==============================

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

# ==============================
# 3. 统计评分分布
# ==============================

rating_distribution = (
    ratings["rating"]
    .value_counts()
    .sort_index()
)

print("===== 评分分布 =====")
print(rating_distribution)


# ==============================
# 4. 绘制评分分布图
# ==============================

rating_distribution.plot(
    kind="bar"
)

plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "rating_distribution.png",
    dpi=300
)

plt.show()

# ==============================
# 5. 用户活跃度分析
# ==============================

user_rating_count = (
    ratings.groupby("userId")
    .size()
)

print("\n===== 每个用户的评分数量统计 =====")
print(user_rating_count.describe())


# ==============================
# 6. 绘制用户评分数量分布图
# ==============================

plt.figure()

plt.hist(
    user_rating_count,
    bins=50
)

plt.title("User Rating Count Distribution")
plt.xlabel("Number of Ratings per User")
plt.ylabel("Number of Users")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "user_rating_count_distribution.png",
    dpi=300
)

plt.show()

# ==============================
# 7. 电影热度分析
# ==============================

movie_rating_count = (
    ratings.groupby("movieId")
    .size()
)

print("\n===== 每部电影的评分数量统计 =====")
print(movie_rating_count.describe())


# ==============================
# 8. 绘制电影评分数量分布图
# ==============================

plt.figure()

plt.hist(
    movie_rating_count,
    bins=50
)

plt.title("Movie Rating Count Distribution")
plt.xlabel("Number of Ratings per Movie")
plt.ylabel("Number of Movies")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "movie_rating_count_distribution.png",
    dpi=300
)

plt.show()

# ==============================
# 9. 热门电影 Top 20
# ==============================

movie_rating_stats = (
    ratings.groupby("movieId")
    .agg(
        rating_count=("rating", "count"),
        avg_rating=("rating", "mean")
    )
    .reset_index()
)

movie_rating_stats = movie_rating_stats.merge(
    movies,
    on="movieId",
    how="left"
)

top_popular_movies = (
    movie_rating_stats
    .sort_values(
        by="rating_count",
        ascending=False
    )
    .head(20)
)

print("\n===== 热门电影 Top 20 =====")
print(
    top_popular_movies[
        ["movieId", "title", "rating_count", "avg_rating"]
    ].to_string(index=False)
)

# ==============================
# 10. 高分电影 Top 20
# ==============================

high_rating_movies = (
    movie_rating_stats[
        movie_rating_stats["rating_count"] >= 50
    ]
    .sort_values(
        by="avg_rating",
        ascending=False
    )
    .head(20)
)

print("\n===== 高分电影 Top 20 =====")

print(
    high_rating_movies[
        ["movieId", "title", "rating_count", "avg_rating"]
    ].to_string(index=False)
)

# ==============================
# 11. 电影类型分布
# ==============================

genre_count = (
    movies["genres"]
    .str.split("|")
    .explode()
    .value_counts()
)

print("\n===== 电影类型分布 =====")
print(genre_count)


# ==============================
# 12. 绘制电影类型分布图
# ==============================

plt.figure(figsize=(10, 6))

genre_count.sort_values().plot(
    kind="barh"
)

plt.title("Movie Genre Distribution")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "genre_distribution.png",
    dpi=300
)

plt.show()

# ==============================
# 13. 保存 EDA 汇总结果
# ==============================

eda_summary = pd.DataFrame({
    "metric": [
        "user_count",
        "movie_count",
        "rating_count",
        "avg_ratings_per_user",
        "median_ratings_per_user",
        "avg_ratings_per_movie",
        "median_ratings_per_movie"
    ],
    "value": [
        ratings["userId"].nunique(),
        movies["movieId"].nunique(),
        len(ratings),
        user_rating_count.mean(),
        user_rating_count.median(),
        movie_rating_count.mean(),
        movie_rating_count.median()
    ]
})

eda_summary.to_csv(
    RESULTS_DIR / "eda_summary.csv",
    index=False
)

print("\n===== EDA 汇总 =====")
print(eda_summary.to_string(index=False))