from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBRanker


# =========================================================
# 1. 项目路径
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANKER_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ranker_data.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

RANK_MODEL_PATH = (
    MODELS_DIR
    / "rank_model_ltr.pkl"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. 最终 Ranker 特征
# =========================================================

feature_columns = [

    # 用户特征
    "user_rating_count",
    "user_avg_rating",

    # 电影特征
    "movie_rating_count",
    "movie_avg_rating",

    # 用户 × 电影类型匹配特征
    "genre_match_score",

    # 协同过滤召回特征
    "itemcf_score",
    "usercf_score",

    # 热门电影特征
    "is_popular_movie"
]


# =========================================================
# 3. 读取 Ranker 数据
# =========================================================

if not RANKER_DATA_PATH.exists():

    raise FileNotFoundError(
        "没有找到 ranker_data.csv。\n"
        "请先运行：\n"
        "python src/build_ranker_data.py"
    )


ranker_data = pd.read_csv(
    RANKER_DATA_PATH
)


print(
    "===== XGBRanker 训练 ====="
)

print(
    "Ranker 数据路径：",
    RANKER_DATA_PATH
)

print(
    "原始样本数量：",
    len(ranker_data)
)

print(
    "原始用户数量：",
    ranker_data[
        "userId"
    ].nunique()
)


# =========================================================
# 4. 检查训练数据
# =========================================================

required_columns = (
    [
        "userId",
        "movieId",
        "label"
    ]
    +
    feature_columns
)


missing_columns = [
    column
    for column
    in required_columns
    if column
    not in ranker_data.columns
]


if missing_columns:

    raise ValueError(
        "ranker_data.csv 缺少以下字段：\n"
        f"{missing_columns}\n\n"
        "请重新运行 src/build_ranker_data.py。"
    )


feature_missing = (
    ranker_data[
        feature_columns
    ]
    .isnull()
    .sum()
)


if (
    feature_missing
    .sum()
    > 0
):

    raise ValueError(
        "Ranker 特征存在缺失值：\n"
        f"{feature_missing}"
    )


# 确保模型特征全部为数值类型
ranker_data[
    feature_columns
] = (
    ranker_data[
        feature_columns
    ]
    .apply(
        pd.to_numeric,
        errors="raise"
    )
)


ranker_data[
    "label"
] = (
    pd.to_numeric(
        ranker_data[
            "label"
        ],
        errors="raise"
    )
    .astype(int)
)


# =========================================================
# 5. 只保留具有正样本的用户
# =========================================================
#
# Learning to Rank 需要一个用户的候选列表中
# 至少存在一个真正相关的电影，
# 否则没有有效的排序学习目标。
# =========================================================

positive_count_by_user = (
    ranker_data
    .groupby(
        "userId"
    )[
        "label"
    ]
    .sum()
)


valid_user_ids = (
    positive_count_by_user[
        positive_count_by_user
        > 0
    ]
    .index
)


xgb_ranker_data = (
    ranker_data[
        ranker_data[
            "userId"
        ]
        .isin(
            valid_user_ids
        )
    ]
    .copy()
)


# XGBRanker 要求同一个 query / 用户的数据连续
xgb_ranker_data = (
    xgb_ranker_data
    .sort_values(
        by="userId"
    )
    .reset_index(
        drop=True
    )
)


print(
    "\n===== 有效 Ranker 数据 ====="
)

print(
    "样本数量：",
    len(xgb_ranker_data)
)

print(
    "用户数量：",
    xgb_ranker_data[
        "userId"
    ].nunique()
)

print(
    "正样本数量：",
    int(
        (
            xgb_ranker_data[
                "label"
            ] == 1
        )
        .sum()
    )
)

print(
    "负样本数量：",
    int(
        (
            xgb_ranker_data[
                "label"
            ] == 0
        )
        .sum()
    )
)


if xgb_ranker_data.empty:

    raise ValueError(
        "没有可以用于 XGBRanker "
        "训练的有效用户数据。"
    )


# =========================================================
# 6. 按用户划分训练集 / 验证集
# =========================================================
#
# 注意：
# 这里不是随机划分候选行，
# 而是直接划分 userId。
#
# 这样可以保证：
# 同一个用户不会同时出现在
# 训练集和验证集中。
# =========================================================

ranker_user_ids = (
    xgb_ranker_data[
        "userId"
    ]
    .unique()
)


train_user_ids, valid_user_ids = (
    train_test_split(
        ranker_user_ids,
        test_size=0.2,
        random_state=42
    )
)


rank_train_data = (
    xgb_ranker_data[
        xgb_ranker_data[
            "userId"
        ]
        .isin(
            train_user_ids
        )
    ]
    .sort_values(
        by="userId"
    )
    .reset_index(
        drop=True
    )
)


rank_valid_data = (
    xgb_ranker_data[
        xgb_ranker_data[
            "userId"
        ]
        .isin(
            valid_user_ids
        )
    ]
    .sort_values(
        by="userId"
    )
    .reset_index(
        drop=True
    )
)


print(
    "\n===== 用户划分 ====="
)

print(
    "训练用户数量：",
    len(
        train_user_ids
    )
)

print(
    "验证用户数量：",
    len(
        valid_user_ids
    )
)

print(
    "训练候选数量：",
    len(
        rank_train_data
    )
)

print(
    "验证候选数量：",
    len(
        rank_valid_data
    )
)


# =========================================================
# 7. 构造训练特征和标签
# =========================================================

X_rank_train = (
    rank_train_data[
        feature_columns
    ]
    .astype(float)
    .copy()
)

y_rank_train = (
    rank_train_data[
        "label"
    ]
    .astype(int)
    .copy()
)


X_rank_valid = (
    rank_valid_data[
        feature_columns
    ]
    .astype(float)
    .copy()
)

y_rank_valid = (
    rank_valid_data[
        "label"
    ]
    .astype(int)
    .copy()
)


# =========================================================
# 8. 构造 XGBRanker Group
# =========================================================
#
# 每一个 group 数字表示：
#
# 某个用户共有多少部候选电影。
#
# 例如：
#
# [425, 386, 417, ...]
#
# 表示：
#
# 用户 A：425 个候选
# 用户 B：386 个候选
# 用户 C：417 个候选
# =========================================================

train_group = (
    rank_train_data
    .groupby(
        "userId",
        sort=False
    )
    .size()
    .to_list()
)


valid_group = (
    rank_valid_data
    .groupby(
        "userId",
        sort=False
    )
    .size()
    .to_list()
)


# =========================================================
# 9. Group 安全检查
# =========================================================

if (
    sum(
        train_group
    )
    !=
    len(
        X_rank_train
    )
):

    raise ValueError(
        "train_group 与训练样本数量不一致。"
    )


if (
    sum(
        valid_group
    )
    !=
    len(
        X_rank_valid
    )
):

    raise ValueError(
        "valid_group 与验证样本数量不一致。"
    )


print(
    "\n===== Group 检查 ====="
)

print(
    "训练 Group 数量：",
    len(
        train_group
    )
)

print(
    "训练 Group 样本总数：",
    sum(
        train_group
    )
)

print(
    "验证 Group 数量：",
    len(
        valid_group
    )
)

print(
    "验证 Group 样本总数：",
    sum(
        valid_group
    )
)


# =========================================================
# 10. 创建最终 XGBRanker
# =========================================================

xgb_ltr_ranker = XGBRanker(

    objective="rank:ndcg",

    n_estimators=300,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    eval_metric="ndcg@10",

    random_state=42,

    n_jobs=-1
)


# =========================================================
# 11. 训练 XGBRanker
# =========================================================

print(
    "\n开始训练 XGBRanker..."
)


xgb_ltr_ranker.fit(

    X_rank_train,

    y_rank_train,

    group=train_group,

    eval_set=[
        (
            X_rank_valid,
            y_rank_valid
        )
    ],

    eval_group=[
        valid_group
    ],

    verbose=False
)


print(
    "XGBRanker 训练完成"
)


# =========================================================
# 12. 验证集 NDCG@10
# =========================================================

eval_results = (
    xgb_ltr_ranker
    .evals_result()
)


validation_ndcg = (
    eval_results[
        "validation_0"
    ][
        "ndcg@10"
    ][-1]
)


print(
    "\n===== XGBRanker 验证结果 ====="
)

print(
    "Validation NDCG@10：",
    validation_ndcg
)


# =========================================================
# 13. 特征重要性
# =========================================================

feature_importance = (
    pd.DataFrame(
        {
            "feature":
                feature_columns,

            "importance":
                xgb_ltr_ranker
                .feature_importances_
        }
    )
    .sort_values(
        by="importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


print(
    "\n===== XGBRanker 特征重要性 ====="
)

print(
    feature_importance
    .to_string(
        index=False
    )
)


# =========================================================
# 14. 保存训练结果
# =========================================================

feature_importance.to_csv(
    RESULTS_DIR
    / "xgbranker_feature_importance.csv",
    index=False
)


ranker_metrics = pd.DataFrame(
    {
        "model": [
            "XGBRanker"
        ],

        "Validation_NDCG@10": [
            validation_ndcg
        ],

        "train_users": [
            len(
                train_user_ids
            )
        ],

        "valid_users": [
            len(
                valid_user_ids
            )
        ]
    }
)


ranker_metrics.to_csv(
    RESULTS_DIR
    / "xgbranker_metrics.csv",
    index=False
)


# =========================================================
# 15. 保存最终模型
# =========================================================

rank_model_data = {

    "model":
        xgb_ltr_ranker,

    "feature_columns":
        feature_columns,

    "model_type":
        "XGBRanker",

    "validation_ndcg@10":
        validation_ndcg
}


joblib.dump(
    rank_model_data,
    RANK_MODEL_PATH
)


print(
    "\n===== 模型保存完成 ====="
)

print(
    "模型路径：",
    RANK_MODEL_PATH
)

print(
    "特征重要性：",
    RESULTS_DIR
    / "xgbranker_feature_importance.csv"
)

print(
    "验证结果：",
    RESULTS_DIR
    / "xgbranker_metrics.csv"
)

print(
    "\n======================================"
)

print(
    "最终 XGBRanker 训练流程完成"
)