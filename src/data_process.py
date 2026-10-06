from pathlib import Path

import pandas as pd


# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据文件路径
RATINGS_PATH = PROJECT_ROOT / "data" / "raw" / "ratings.csv"
MOVIES_PATH = PROJECT_ROOT / "data" / "raw" / "movies.csv"


# 读取数据
ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)


# 查看基本信息
print("ratings 数据规模：", ratings.shape)
print(ratings.head())

print("\nmovies 数据规模：", movies.shape)
print(movies.head())

print("\nratings 字段：")
print(ratings.columns)

print("\nratings 数据类型：")
print(ratings.dtypes)

print("\nmovies 字段：")
print(movies.columns)

print("\nmovies 数据类型：")
print(movies.dtypes)

print("\nratings 缺失值：")
print(ratings.isnull().sum())

print("\nmovies 缺失值：")
print(movies.isnull().sum())

print("\nratings 重复行数量：")
print(ratings.duplicated().sum())

print("\nmovies 重复行数量：")
print(movies.duplicated().sum())


print("\n===== 数据基本统计 =====")

user_count = ratings["userId"].nunique()
movie_count = movies["movieId"].nunique()
rating_count = len(ratings)

print("用户数量：", user_count)
print("电影数量：", movie_count)
print("评分记录数量：", rating_count)

# ==============================
# 用户-电影交互稀疏度
# ==============================

total_possible_ratings = user_count * movie_count
actual_ratings = rating_count

sparsity = 1 - actual_ratings / total_possible_ratings

print("\n===== 用户-电影交互稀疏度 =====")
print("理论最大评分数量：", total_possible_ratings)
print("实际评分数量：", actual_ratings)
print("数据稀疏度：", round(sparsity * 100, 2), "%")

# ==============================
# 评分数据进一步检查
# ==============================

print("\n===== 评分取值检查 =====")

print("最低评分：", ratings["rating"].min())
print("最高评分：", ratings["rating"].max())
print("所有评分取值：", sorted(ratings["rating"].unique()))


# 检查 userId + movieId 是否重复
duplicate_user_movie = ratings.duplicated(
    subset=["userId", "movieId"]
).sum()

print("\n===== 用户-电影重复评分检查 =====")
print("重复的用户-电影组合数量：", duplicate_user_movie)