from pathlib import Path
import math
import sys

import pandas as pd


# =========================================================
# 1. 项目路径
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# 将项目根目录加入 Python 搜索路径
# 这样可以正确导入 src.recommend_service
if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.recommend_service import recommend


TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_ratings.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 2. 评估参数
# =========================================================

TOP_K = 10

POSITIVE_RATING_THRESHOLD = 4.0


# =========================================================
# 3. 读取测试集
# =========================================================

if not TEST_PATH.exists():

    raise FileNotFoundError(
        "没有找到 test_ratings.csv。\n"
        "请先运行：\n"
        "python src/split_data.py"
    )


test_ratings = pd.read_csv(
    TEST_PATH
)


print(
    "===== 最终 XGBRanker 离线评估 ====="
)

print(
    "测试集路径：",
    TEST_PATH
)

print(
    "测试集评分数量：",
    len(
        test_ratings
    )
)

print(
    "测试集用户数量：",
    test_ratings[
        "userId"
    ].nunique()
)


# =========================================================
# 4. Precision@K
# =========================================================

def precision_at_k(
    recommended_items,
    relevant_items,
    k=10
):
    """
    Precision@K：

    Top-K 推荐结果中，
    真正相关电影所占的比例。
    """

    recommended_top_k = (
        recommended_items[
            :k
        ]
    )


    relevant_set = set(
        relevant_items
    )


    hit_count = len(
        set(
            recommended_top_k
        )
        &
        relevant_set
    )


    return (
        hit_count
        /
        k
    )


# =========================================================
# 5. Recall@K
# =========================================================

def recall_at_k(
    recommended_items,
    relevant_items,
    k=10
):
    """
    Recall@K：

    用户未来真正喜欢的电影中，
    有多少被 Top-K 推荐结果召回。
    """

    relevant_set = set(
        relevant_items
    )


    if not relevant_set:

        return 0.0


    recommended_top_k = (
        recommended_items[
            :k
        ]
    )


    hit_count = len(
        set(
            recommended_top_k
        )
        &
        relevant_set
    )


    return (
        hit_count
        /
        len(
            relevant_set
        )
    )


# =========================================================
# 6. HitRate@K
# =========================================================

def hitrate_at_k(
    recommended_items,
    relevant_items,
    k=10
):
    """
    HitRate@K：

    Top-K 中只要至少命中一部
    用户未来喜欢的电影，
    当前用户记为 1，否则为 0。
    """

    recommended_top_k = (
        recommended_items[
            :k
        ]
    )


    hit_count = len(
        set(
            recommended_top_k
        )
        &
        set(
            relevant_items
        )
    )


    if hit_count > 0:

        return 1.0


    return 0.0


# =========================================================
# 7. NDCG@K
# =========================================================

def ndcg_at_k(
    recommended_items,
    relevant_items,
    k=10
):
    """
    NDCG@K：

    不仅考虑是否命中，
    同时考虑命中的电影是否排在更靠前的位置。
    """

    recommended_top_k = (
        recommended_items[
            :k
        ]
    )


    relevant_set = set(
        relevant_items
    )


    if not relevant_set:

        return 0.0


    # ------------------------------
    # DCG
    # ------------------------------

    dcg = 0.0


    for rank, movie_id in enumerate(
        recommended_top_k
    ):

        if movie_id in relevant_set:

            dcg += (
                1.0
                /
                math.log2(
                    rank + 2
                )
            )


    # ------------------------------
    # IDCG
    # ------------------------------

    ideal_hit_count = min(
        len(
            relevant_set
        ),
        k
    )


    idcg = sum(
        (
            1.0
            /
            math.log2(
                rank + 2
            )
        )

        for rank in range(
            ideal_hit_count
        )
    )


    if idcg == 0:

        return 0.0


    return (
        dcg
        /
        idcg
    )


# =========================================================
# 8. 提取测试集正样本
# =========================================================
#
# rating >= 4
#
# 表示用户未来真正喜欢的电影。
# =========================================================

test_positive = (
    test_ratings[
        test_ratings[
            "rating"
        ]
        >=
        POSITIVE_RATING_THRESHOLD
    ]
    .copy()
)


user_relevant_items = (
    test_positive
    .groupby(
        "userId"
    )[
        "movieId"
    ]
    .apply(
        lambda x:
        x
        .astype(int)
        .tolist()
    )
    .to_dict()
)


print(
    "\n===== 测试集正样本 ====="
)

print(
    "正样本数量：",
    len(
        test_positive
    )
)

print(
    "有效测试用户数量：",
    len(
        user_relevant_items
    )
)


# =========================================================
# 9. 全体用户离线评估
# =========================================================

user_metrics = []


total_users = len(
    user_relevant_items
)


print(
    "\n开始评估..."
)


for index, (
    user_id,
    relevant_items
) in enumerate(
    user_relevant_items.items(),
    start=1
):

    # ------------------------------
    # 调用正式推荐服务
    # ------------------------------

    recommendations = (
        recommend(
            user_id=int(
                user_id
            ),
            top_n=TOP_K
        )
    )


    if recommendations.empty:

        recommended_items = []

    else:

        recommended_items = (
            recommendations[
                "movieId"
            ]
            .astype(int)
            .tolist()
        )


    # ------------------------------
    # 计算四个指标
    # ------------------------------

    precision = (
        precision_at_k(
            recommended_items,
            relevant_items,
            k=TOP_K
        )
    )


    recall = (
        recall_at_k(
            recommended_items,
            relevant_items,
            k=TOP_K
        )
    )


    hitrate = (
        hitrate_at_k(
            recommended_items,
            relevant_items,
            k=TOP_K
        )
    )


    ndcg = (
        ndcg_at_k(
            recommended_items,
            relevant_items,
            k=TOP_K
        )
    )


    # ------------------------------
    # 当前用户命中数量
    # ------------------------------

    hit_count = len(
        set(
            recommended_items[
                :TOP_K
            ]
        )
        &
        set(
            relevant_items
        )
    )


    user_metrics.append(
        {
            "userId":
                int(
                    user_id
                ),

            "relevant_count":
                len(
                    set(
                        relevant_items
                    )
                ),

            "recommended_count":
                len(
                    recommended_items
                ),

            "hit_count":
                hit_count,

            f"Precision@{TOP_K}":
                precision,

            f"Recall@{TOP_K}":
                recall,

            f"HitRate@{TOP_K}":
                hitrate,

            f"NDCG@{TOP_K}":
                ndcg
        }
    )


    # 每 50 个用户显示一次进度
    if (
        index % 50 == 0
        or
        index == total_users
    ):

        print(
            f"已评估："
            f"{index}/"
            f"{total_users}"
        )


# =========================================================
# 10. 转换为 DataFrame
# =========================================================

user_metrics_df = pd.DataFrame(
    user_metrics
)


if user_metrics_df.empty:

    raise ValueError(
        "没有得到任何有效评估结果。"
    )


# =========================================================
# 11. 计算全体用户平均指标
# =========================================================

precision_value = (
    user_metrics_df[
        f"Precision@{TOP_K}"
    ]
    .mean()
)


recall_value = (
    user_metrics_df[
        f"Recall@{TOP_K}"
    ]
    .mean()
)


hitrate_value = (
    user_metrics_df[
        f"HitRate@{TOP_K}"
    ]
    .mean()
)


ndcg_value = (
    user_metrics_df[
        f"NDCG@{TOP_K}"
    ]
    .mean()
)


# =========================================================
# 12. 输出最终结果
# =========================================================

print(
    "\n======================================"
)

print(
    "===== XGBRanker 最终测试集结果 ====="
)

print(
    "评估用户数量：",
    len(
        user_metrics_df
    )
)

print(
    f"Precision@{TOP_K} =",
    precision_value
)

print(
    f"Recall@{TOP_K} =",
    recall_value
)

print(
    f"HitRate@{TOP_K} =",
    hitrate_value
)

print(
    f"NDCG@{TOP_K} =",
    ndcg_value
)

print(
    "======================================"
)


# =========================================================
# 13. 保存全体用户指标
# =========================================================

USER_METRICS_PATH = (
    RESULTS_DIR
    / "xgbranker_user_metrics.csv"
)


user_metrics_df.to_csv(
    USER_METRICS_PATH,
    index=False
)


# =========================================================
# 14. 保存最终平均指标
# =========================================================

FINAL_METRICS_PATH = (
    RESULTS_DIR
    / "xgbranker_test_metrics.csv"
)


final_metrics = pd.DataFrame(
    {
        "model": [
            "XGBRanker"
        ],

        "test_users": [
            len(
                user_metrics_df
            )
        ],

        f"Precision@{TOP_K}": [
            precision_value
        ],

        f"Recall@{TOP_K}": [
            recall_value
        ],

        f"HitRate@{TOP_K}": [
            hitrate_value
        ],

        f"NDCG@{TOP_K}": [
            ndcg_value
        ]
    }
)


final_metrics.to_csv(
    FINAL_METRICS_PATH,
    index=False
)


# =========================================================
# 15. 保存完成
# =========================================================

print(
    "\n评估结果已保存："
)

print(
    USER_METRICS_PATH
)

print(
    FINAL_METRICS_PATH
)

print(
    "\n最终 XGBRanker 离线评估完成。"
)