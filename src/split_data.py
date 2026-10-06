from pathlib import Path

import pandas as pd


# ==============================
# 1. 设置项目路径
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_PATH = PROJECT_ROOT / "data" / "raw" / "ratings.csv"


# ==============================
# 2. 读取评分数据
# ==============================

ratings = pd.read_csv(RATINGS_PATH)


# ==============================
# 3. 按用户和时间排序
# ==============================

ratings_sorted = ratings.sort_values(
    by=["userId", "timestamp"]
).reset_index(drop=True)


print("===== 排序后的评分数据 =====")
print(ratings_sorted.head(10).to_string(index=False))

# ==============================
# 4. 按用户进行 80% / 20% 划分
# ==============================

train_list = []
test_list = []

for user_id, user_data in ratings_sorted.groupby("userId"):

    split_index = int(len(user_data) * 0.8)

    train_part = user_data.iloc[:split_index]
    test_part = user_data.iloc[split_index:]

    train_list.append(train_part)
    test_list.append(test_part)


train_ratings = pd.concat(
    train_list,
    ignore_index=True
)

test_ratings = pd.concat(
    test_list,
    ignore_index=True
)


print("\n===== 训练集 / 测试集划分结果 =====")

print("训练集数量：", len(train_ratings))
print("测试集数量：", len(test_ratings))

print(
    "训练集占比：",
    round(len(train_ratings) / len(ratings) * 100, 2),
    "%"
)

print(
    "测试集占比：",
    round(len(test_ratings) / len(ratings) * 100, 2),
    "%"
)

# ==============================
# 5. 检查时间划分是否正确
# ==============================

time_check = (
    train_ratings.groupby("userId")["timestamp"].max()
    <=
    test_ratings.groupby("userId")["timestamp"].min()
)

error_user_count = (~time_check).sum()

print("\n===== 时间划分检查 =====")
print("检查用户数量：", len(time_check))
print("时间划分异常用户数量：", error_user_count)

# ==============================
# 6. 保存训练集和测试集
# ==============================

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

train_ratings.to_csv(
    PROCESSED_DIR / "train_ratings.csv",
    index=False
)

test_ratings.to_csv(
    PROCESSED_DIR / "test_ratings.csv",
    index=False
)

print("\n训练集和测试集已保存：")
print(PROCESSED_DIR / "train_ratings.csv")
print(PROCESSED_DIR / "test_ratings.csv")