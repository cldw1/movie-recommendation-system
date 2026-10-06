import streamlit as st

from src.recommend_service import (
    recommend,
    get_user_profile,
    get_recent_liked_movies
)


# =========================================================
# 1. 页面配置
# =========================================================

st.set_page_config(
    page_title="个性化电影推荐系统",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. 侧边栏：系统信息
# =========================================================

with st.sidebar:

    st.title(
        "🎬 推荐系统"
    )

    st.caption(
        "MovieLens Latest Small"
    )

    st.divider()


    # -----------------------------------------------------
    # 系统架构
    # -----------------------------------------------------

    st.subheader(
        "🧠 系统架构"
    )

    st.markdown(
        """
        **第一阶段：多路召回**

        `ItemCF Top300`

        ＋

        `UserCF Top300`

        ↓

        **第二阶段：特征工程**

        - 用户评分数量
        - 用户平均评分
        - 电影评分数量
        - 电影平均评分
        - 类型匹配度
        - ItemCF 得分
        - UserCF 得分
        - 热门电影标记

        ↓

        **第三阶段：Learning to Rank**

        `XGBRanker`

        ↓

        **最终 Top-10 推荐**
        """
    )

    st.divider()


    # -----------------------------------------------------
    # 最终模型
    # -----------------------------------------------------

    st.subheader(
        "⚙️ 最终模型"
    )

    st.write(
        "召回模型"
    )

    st.code(
        "ItemCF + UserCF",
        language=None
    )

    st.write(
        "排序模型"
    )

    st.code(
        "XGBRanker",
        language=None
    )

    st.metric(
        label="排序特征数量",
        value=8
    )

    st.divider()


    # -----------------------------------------------------
    # 离线评估
    # -----------------------------------------------------

    st.subheader(
        "📊 离线评估"
    )

    st.caption(
        "基于 592 个有效测试用户"
    )


    metric_col1, metric_col2 = (
        st.columns(2)
    )


    with metric_col1:

        st.metric(
            "Precision@10",
            "0.0863"
        )

        st.metric(
            "HitRate@10",
            "0.4949"
        )


    with metric_col2:

        st.metric(
            "Recall@10",
            "0.0857"
        )

        st.metric(
            "NDCG@10",
            "0.1071"
        )


    st.caption(
        "HitRate@10 ≈ 49.5%，"
        "即约一半测试用户的 Top-10 "
        "至少命中一部未来喜欢的电影。"
    )

    st.divider()


    # -----------------------------------------------------
    # 项目说明
    # -----------------------------------------------------

    st.subheader(
        "ℹ️ 项目说明"
    )

    st.caption(
        "该系统采用“召回 → 特征工程 → 排序”的"
        "两阶段推荐架构。"
    )


# =========================================================
# 3. 页面标题
# =========================================================

st.title(
    "🎬 个性化电影推荐系统"
)

st.caption(
    "基于 ItemCF + UserCF 多路召回与 XGBRanker 排序"
)


# =========================================================
# 4. 用户输入
# =========================================================

user_id = st.number_input(
    "请输入用户 ID",
    min_value=1,
    max_value=610,
    value=1,
    step=1
)


# =========================================================
# 5. 生成推荐
# =========================================================

if st.button(
    "生成推荐",
    type="primary"
):

    try:

        # =================================================
        # 加载用户数据并生成推荐
        # =================================================

        with st.spinner(
            "正在分析用户兴趣并生成推荐..."
        ):

            profile = (
                get_user_profile(
                    user_id=int(user_id)
                )
            )


            recent_liked_movies = (
                get_recent_liked_movies(
                    user_id=int(user_id),
                    top_n=5
                )
            )


            recommendations = (
                recommend(
                    user_id=int(user_id),
                    top_n=10
                )
            )


        # =================================================
        # 6. 推荐完成提示
        # =================================================

        st.success(
            f"用户 {int(user_id)} 推荐生成完成"
        )


        # =================================================
        # 7. 用户画像
        # =================================================

        st.subheader(
            "👤 用户画像"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                label="历史评分电影",
                value=profile[
                    "rating_count"
                ]
            )


        with col2:

            st.metric(
                label="平均评分",
                value=profile[
                    "avg_rating"
                ]
            )


        with col3:

            st.metric(
                label="喜欢的电影",
                value=profile[
                    "liked_count"
                ]
            )


        # =================================================
        # 8. 用户类型偏好
        # =================================================

        st.markdown(
            "#### ❤️ 偏好电影类型"
        )


        genre_text = "　".join(
            [
                f"`{genre}`"

                for genre

                in profile[
                    "top_genres"
                ]
            ]
        )


        st.markdown(
            genre_text
        )


        # =================================================
        # 9. 最近喜欢的电影
        # =================================================

        st.markdown(
            "#### 🕘 最近喜欢的电影"
        )


        if recent_liked_movies.empty:

            st.info(
                "当前用户暂无历史喜欢记录。"
            )

        else:

            recent_display = (
                recent_liked_movies[
                    [
                        "title",
                        "genres",
                        "rating"
                    ]
                ]
                .copy()
            )


            recent_display.columns = [
                "电影",
                "类型",
                "评分"
            ]


            recent_display[
                "评分"
            ] = (
                recent_display[
                    "评分"
                ]
                .round(1)
            )


            st.dataframe(
                recent_display,
                use_container_width=True,
                hide_index=True
            )


        st.divider()


        # =================================================
        # 10. Top-10 推荐
        # =================================================

        st.subheader(
            "🍿 为你推荐的 Top-10 电影"
        )


        if recommendations.empty:

            st.warning(
                "当前用户暂时没有可推荐的电影。"
            )

        else:

            # -------------------------------------------------
            # 两列电影卡片
            # -------------------------------------------------

            left_col, right_col = (
                st.columns(2)
            )


            for _, row in recommendations.iterrows():

                rank = int(
                    row[
                        "recommend_rank"
                    ]
                )


                title = (
                    row[
                        "title"
                    ]
                )


                genres = (
                    str(
                        row[
                            "genres"
                        ]
                    )
                    .replace(
                        "|",
                        " · "
                    )
                )


                rank_score = float(
                    row[
                        "rank_score"
                    ]
                )


                genre_match_score = float(
                    row[
                        "genre_match_score"
                    ]
                )


                itemcf_score = float(
                    row[
                        "itemcf_score"
                    ]
                )


                usercf_score = float(
                    row[
                        "usercf_score"
                    ]
                )


                # 奇数放左边
                # 偶数放右边
                target_column = (
                    left_col

                    if rank % 2 == 1

                    else right_col
                )


                with target_column:

                    with st.container(
                        border=True
                    ):

                        # ------------------------------
                        # 排名图标
                        # ------------------------------

                        if rank == 1:

                            rank_icon = "🥇"

                        elif rank == 2:

                            rank_icon = "🥈"

                        elif rank == 3:

                            rank_icon = "🥉"

                        else:

                            rank_icon = "🎬"


                        # ------------------------------
                        # 电影标题
                        # ------------------------------

                        st.markdown(
                            f"### {rank_icon} #{rank} {title}"
                        )


                        st.caption(
                            f"🎭 {genres}"
                        )


                        # ------------------------------
                        # 主要指标
                        # ------------------------------

                        score_col1, score_col2 = (
                            st.columns(2)
                        )


                        with score_col1:

                            st.metric(
                                label="推荐得分",
                                value=f"{rank_score:.4f}"
                            )


                        with score_col2:

                            st.metric(
                                label="类型匹配度",
                                value=(
                                    f"{genre_match_score:.2%}"
                                )
                            )


                        # ------------------------------
                        # 简单推荐依据
                        # ------------------------------

                        reasons = []


                        if (
                            genre_match_score
                            >= 0.65
                        ):

                            reasons.append(
                                "类型兴趣匹配较高"
                            )


                        if (
                            usercf_score
                            >= 4
                        ):

                            reasons.append(
                                "相似用户偏好明显"
                            )


                        if (
                            itemcf_score
                            >= 7
                        ):

                            reasons.append(
                                "与历史喜欢电影相似"
                            )


                        if reasons:

                            reason_text = (
                                " · ".join(
                                    reasons
                                )
                            )

                            st.caption(
                                f"💡 推荐理由：{reason_text}"
                            )


            # =================================================
            # 11. 推荐算法详细信息
            # =================================================

            st.markdown("")


            with st.expander(
                "🔍 查看推荐算法详细信息"
            ):

                st.caption(
                    "这里展示召回与排序阶段的核心特征，"
                    "用于观察最终推荐结果的生成依据。"
                )


                detail_columns = (
                    recommendations[
                        [
                            "recommend_rank",
                            "movieId",
                            "title",
                            "genre_match_score",
                            "itemcf_score",
                            "usercf_score",
                            "rank_score"
                        ]
                    ]
                    .copy()
                )


                detail_columns.columns = [
                    "排名",
                    "电影ID",
                    "电影",
                    "类型匹配度",
                    "ItemCF得分",
                    "UserCF得分",
                    "Ranker得分"
                ]


                detail_columns[
                    "类型匹配度"
                ] = (
                    detail_columns[
                        "类型匹配度"
                    ]
                    .round(4)
                )


                detail_columns[
                    "ItemCF得分"
                ] = (
                    detail_columns[
                        "ItemCF得分"
                    ]
                    .round(4)
                )


                detail_columns[
                    "UserCF得分"
                ] = (
                    detail_columns[
                        "UserCF得分"
                    ]
                    .round(4)
                )


                detail_columns[
                    "Ranker得分"
                ] = (
                    detail_columns[
                        "Ranker得分"
                    ]
                    .round(4)
                )


                st.dataframe(
                    detail_columns,
                    use_container_width=True,
                    hide_index=True
                )


    except ValueError as e:

        st.error(
            str(e)
        )