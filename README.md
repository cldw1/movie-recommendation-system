# 🎬 Two-stage Movie Recommendation System

A portfolio-ready, two-stage personalized movie recommender built with the MovieLens Latest Small dataset.

**ItemCF + UserCF multi-channel recall → feature engineering → XGBRanker (Learning to Rank) → Top-N recommendations**

## Highlights

- Uses a per-user chronological 80/20 split to avoid training on future interactions.
- Combines ItemCF and UserCF Top-300 recall results, then ranks the union with an 8-feature `XGBRanker` model.
- Includes a runnable Streamlit interface, offline evaluation, and simple feature-based recommendation explanations.
- The repository includes only the two small generated artifacts needed to run the demo immediately: `train_ratings.csv` (about 2 MB) and `rank_model_ltr.pkl` (about 0.8 MB). Larger intermediate data and other generated models remain ignored.

## Final offline results

Evaluation uses the chronological holdout test set (592 users with relevant items) and `rating >= 4` as positive feedback. All figures are @10.

| Model | Precision | Recall | HitRate | NDCG |
| --- | ---: | ---: | ---: | ---: |
| Popular | 0.0606 | 0.0492 | 0.3277 | 0.0793 |
| ItemCF | 0.0784 | 0.0738 | 0.4307 | 0.0985 |
| UserCF | **0.0916** | **0.0910** | 0.4696 | **0.1158** |
| XGBClassifier | 0.0856 | 0.0891 | 0.4764 | 0.1093 |
| Final XGBRanker | 0.0863 | 0.0857 | **0.4949** | 0.1071 |

The final XGBRanker has the highest HitRate@10, while UserCF remains stronger on Precision@10, Recall@10, and NDCG@10. The results therefore do **not** claim that the ranker is best on every metric.

## System design

```text
User history
    │
    ├── ItemCF Top-300 recall
    └── UserCF Top-300 recall
             │
             ▼
      Deduplicated candidate pool
             │
             ▼
  User / item / genre / recall-score features
             │
             ▼
      XGBRanker (Learning to Rank)
             │
             ▼
      Personalized Top-10 movies
```

The ranker uses user activity and average-rating features, item popularity and average-rating features, an item-popularity flag, `itemcf_score`, `usercf_score`, and `genre_match_score`.

## Dataset

The project uses [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) data included under `data/raw/`.

- Users: 610
- Movies: 9,742
- Ratings: 100,836
- Positive feedback: `rating >= 4`

For each user, interactions are sorted by timestamp: the earliest 80% form the training data and the final 20% the test data. This produces 80,419 training ratings and 20,417 test ratings.

## Repository structure

```text
movie-recommendation-system/
├── app.py                              # Streamlit UI
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                            # MovieLens source CSV files
│   │   ├── ratings.csv
│   │   ├── movies.csv
│   │   ├── tags.csv
│   │   └── links.csv
│   └── processed/
│       └── train_ratings.csv           # committed: required by the demo
├── models/
│   └── rank_model_ltr.pkl              # committed: final XGBRanker for the demo
├── results/                            # committed EDA figures; generated CSVs ignored
│   ├── rating_distribution.png
│   ├── user_rating_count_distribution.png
│   ├── movie_rating_count_distribution.png
│   └── genre_distribution.png
└── src/
    ├── data_process.py                 # data checks and descriptive statistics
    ├── eda.py                          # exploratory analysis and plots
    ├── split_data.py                   # chronological train/test split
    ├── popular_recommend.py            # popularity baseline
    ├── itemcf.py                       # ItemCF recall
    ├── usercf.py                       # UserCF recall
    ├── build_ranker_data.py            # ranking-data construction
    ├── train_ranker.py                 # XGBRanker training and model export
    ├── evaluate.py                     # baseline model evaluation
    ├── evaluate_ranker.py              # final ranker evaluation
    └── recommend_service.py            # inference service used by the app
```

## Quick start: run the demo

The clone already contains the small `data/processed/train_ratings.csv` and `models/rank_model_ltr.pkl` artifacts needed by `app.py`; no training step is needed for the demo.

```bash
git clone https://github.com/cldw1/movie-recommendation-system.git
cd movie-recommendation-system
python -m venv .venv
```

Activate the environment, install dependencies, then start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows, the last command can also be run as:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit and enter a valid MovieLens user ID such as `1`.

## Reproduce the experiment from scratch

This route regenerates the ignored intermediate files and overwrites the committed demo artifacts locally.

```bash
python src/data_process.py       # optional data checks
python src/eda.py                # optional EDA plots
python src/split_data.py
python src/build_ranker_data.py
python src/train_ranker.py
python src/evaluate_ranker.py
```

`build_ranker_data.py` creates the large `data/processed/ranker_data.csv`, and the evaluation scripts create result CSV files. These are intentionally ignored by Git. The full pipeline can take a while because it builds candidate pools and evaluates all eligible users.

## Environment

The included model was trained and smoke-tested with Python 3.13.9 and the exact package versions in `requirements.txt`, especially `xgboost==3.4.1` and `scikit-learn==1.9.1`. Using these versions avoids serialized-model compatibility surprises.

## Streamlit demo

The application shows a user profile, genre preferences, recently liked movies, ranked Top-10 recommendations, recall scores, genre-match scores, and a rule-based explanation. These explanations describe contributing features; they are not causal explanations emitted by XGBRanker itself.

## Scope and next steps

This is an offline MovieLens portfolio project, not an online serving system. Useful extensions include cold-start handling, temporal and tag features, matrix-factorization or embedding recall, a larger dataset, and online A/B testing.

## Tech stack

Python · Pandas · NumPy · SciPy · scikit-learn · XGBoost · Joblib · Matplotlib · Streamlit
