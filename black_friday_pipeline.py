from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor, early_stopping

    HAS_LIGHTGBM = True
except Exception:
    LGBMRegressor = None
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except Exception:
    CatBoostRegressor = None
    HAS_CATBOOST = False


RANDOM_STATE = 42
N_SPLITS = 5
RANK_SAMPLE = 30_000
TOP_FEATURES = 30

AGE_MAP = {
    "0-17": 0,
    "18-25": 1,
    "26-35": 2,
    "36-45": 3,
    "46-50": 4,
    "51-55": 5,
    "55+": 6,
}
STAY_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4+": 4}
CITY_MAP = {"A": 0, "B": 1, "C": 2}


@dataclass
class ModelResult:
    name: str
    rmse: float
    r2: float
    mape: float
    oof_path: str
    pred_path: str
    importance_path: str | None = None


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100)


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    figures_dir = base_dir / "figures"
    notebook_dir = base_dir / "notebook"
    report_dir = base_dir / "report"
    for path in (figures_dir, notebook_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "base": base_dir,
        "figures": figures_dir,
        "notebook": notebook_dir,
        "report": report_dir,
    }


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def factorize_from_full(
    full_df: pd.DataFrame, train_len: int, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded = full_df.copy()
    for column in columns:
        encoded[column], _ = pd.factorize(encoded[column].astype(str), sort=True)
        encoded[column] = encoded[column].astype(np.int32)
    return encoded.iloc[:train_len].copy(), encoded.iloc[train_len:].copy()


def build_feature_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = train.copy()
    test_raw = test.copy()

    train_target = train_raw["Purchase"].copy()
    test_comb = test_raw["Comb"].copy() if "Comb" in test_raw.columns else None
    test_user_original = test_raw["User_ID"].copy()
    test_product_original = test_raw["Product_ID"].copy()

    test_no_target = test_raw.drop(columns=["Comb"], errors="ignore")
    full = pd.concat([train_raw.drop(columns=["Purchase"]), test_no_target], ignore_index=True)

    full["Product_Category_2_filled"] = full["Product_Category_2"].fillna(0).astype(int)
    full["Product_Category_3_filled"] = full["Product_Category_3"].fillna(0).astype(int)
    full["Product_Category_2_missing"] = full["Product_Category_2"].isna().astype(int)
    full["Product_Category_3_missing"] = full["Product_Category_3"].isna().astype(int)
    full["Category_Count"] = (
        1
        + full["Product_Category_2"].notna().astype(int)
        + full["Product_Category_3"].notna().astype(int)
    )
    full["Age_Ordinal"] = full["Age"].map(AGE_MAP).fillna(2).astype(int)
    full["Stay_Ordinal"] = (
        full["Stay_In_Current_City_Years"].astype(str).map(STAY_MAP).fillna(2).astype(int)
    )
    full["Gender_Binary"] = (full["Gender"] == "M").astype(int)
    full["City_Ordinal"] = full["City_Category"].map(CITY_MAP).fillna(1).astype(int)

    product_popularity = full["Product_ID"].value_counts()
    user_total_orders = full["User_ID"].value_counts()
    user_unique_products = full.groupby("User_ID")["Product_ID"].nunique()
    user_unique_cat1 = full.groupby("User_ID")["Product_Category_1"].nunique()
    user_cat_freq = full.groupby(["User_ID", "Product_Category_1"]).size()
    product_city_freq = full.groupby(["Product_ID", "City_Category"]).size()
    category_combo_freq = full.groupby(
        ["Product_Category_1", "Product_Category_2_filled", "Product_Category_3_filled"]
    ).size()

    full["Product_Popularity"] = full["Product_ID"].map(product_popularity).astype(np.int32)
    full["User_Total_Orders"] = full["User_ID"].map(user_total_orders).astype(np.int32)
    full["User_Unique_Products"] = full["User_ID"].map(user_unique_products).astype(np.int32)
    full["User_Unique_Cat1"] = full["User_ID"].map(user_unique_cat1).astype(np.int16)
    full["User_Cat_Freq"] = (
        full.set_index(["User_ID", "Product_Category_1"]).index.map(user_cat_freq).astype(np.int32)
    )
    full["Product_City_Freq"] = (
        full.set_index(["Product_ID", "City_Category"]).index.map(product_city_freq).astype(np.int32)
    )
    full["Category_Combo_Freq"] = (
        full.set_index(
            ["Product_Category_1", "Product_Category_2_filled", "Product_Category_3_filled"]
        )
        .index.map(category_combo_freq)
        .astype(np.int32)
    )

    full["Category_12"] = (
        full["Product_Category_1"].astype(str) + "_" + full["Product_Category_2_filled"].astype(str)
    )
    full["Category_13"] = (
        full["Product_Category_1"].astype(str) + "_" + full["Product_Category_3_filled"].astype(str)
    )
    full["Category_23"] = (
        full["Product_Category_2_filled"].astype(str) + "_" + full["Product_Category_3_filled"].astype(str)
    )
    full["Category_Combo"] = (
        full["Product_Category_1"].astype(str)
        + "_"
        + full["Product_Category_2_filled"].astype(str)
        + "_"
        + full["Product_Category_3_filled"].astype(str)
    )
    full["Product12_Age"] = (
        full["Product_Category_1"].astype(str)
        + "_"
        + full["Product_Category_2_filled"].astype(str)
        + "_"
        + full["Age"].astype(str)
    )
    full["Product12_City"] = (
        full["Product_Category_1"].astype(str)
        + "_"
        + full["Product_Category_2_filled"].astype(str)
        + "_"
        + full["City_Category"].astype(str)
    )
    full["Product1_City_Gender"] = (
        full["Product_Category_1"].astype(str)
        + "_"
        + full["City_Category"].astype(str)
        + "_"
        + full["Gender"].astype(str)
    )
    full["Product13_Gender"] = (
        full["Product_Category_1"].astype(str)
        + "_"
        + full["Product_Category_3_filled"].astype(str)
        + "_"
        + full["Gender"].astype(str)
    )
    full["Product23_Gender"] = (
        full["Product_Category_2_filled"].astype(str)
        + "_"
        + full["Product_Category_3_filled"].astype(str)
        + "_"
        + full["Gender"].astype(str)
    )
    full["Product1_Marital"] = (
        full["Product_Category_1"].astype(str) + "_" + full["Marital_Status"].astype(str)
    )
    full["Age_Occupation"] = full["Age"].astype(str) + "_" + full["Occupation"].astype(str)
    full["Age_City_Marital"] = (
        full["Age"].astype(str)
        + "_"
        + full["City_Category"].astype(str)
        + "_"
        + full["Marital_Status"].astype(str)
    )
    full["Occupation_City_Gender"] = (
        full["Occupation"].astype(str)
        + "_"
        + full["City_Category"].astype(str)
        + "_"
        + full["Gender"].astype(str)
    )
    full["Occupation_Marital"] = (
        full["Occupation"].astype(str) + "_" + full["Marital_Status"].astype(str)
    )
    full["Occupation_City"] = full["Occupation"].astype(str) + "_" + full["City_Category"].astype(str)
    full["City_Stay"] = (
        full["City_Category"].astype(str) + "_" + full["Stay_In_Current_City_Years"].astype(str)
    )
    full["Stay_Marital"] = (
        full["Stay_In_Current_City_Years"].astype(str) + "_" + full["Marital_Status"].astype(str)
    )
    full["Stay_Gender"] = (
        full["Stay_In_Current_City_Years"].astype(str) + "_" + full["Gender"].astype(str)
    )
    full["Age_Gender"] = full["Age"].astype(str) + "_" + full["Gender"].astype(str)
    full["Occupation_Gender"] = full["Occupation"].astype(str) + "_" + full["Gender"].astype(str)

    encoded_columns = [
        "Product_ID",
        "Gender",
        "Age",
        "City_Category",
        "Stay_In_Current_City_Years",
        "Category_12",
        "Category_13",
        "Category_23",
        "Category_Combo",
        "Product12_Age",
        "Product12_City",
        "Product1_City_Gender",
        "Product13_Gender",
        "Product23_Gender",
        "Product1_Marital",
        "Age_Occupation",
        "Age_City_Marital",
        "Occupation_City_Gender",
        "Occupation_Marital",
        "Occupation_City",
        "City_Stay",
        "Stay_Marital",
        "Stay_Gender",
        "Age_Gender",
        "Occupation_Gender",
    ]
    train_fe, test_fe = factorize_from_full(full, len(train_raw), encoded_columns)
    train_fe["Purchase"] = train_target.values
    if test_comb is not None:
        test_fe["Comb"] = test_comb.values
    test_fe["User_ID_Original"] = test_user_original.values
    test_fe["Product_ID_Original"] = test_product_original.values
    return train_fe, test_fe


def add_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    column: str,
    target: str = "Purchase",
    smoothing: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    global_mean = train_df[target].mean()
    train_encoded = np.full(len(train_df), global_mean, dtype=float)

    for train_idx, valid_idx in kf.split(train_df):
        fold_train = train_df.iloc[train_idx]
        stats = fold_train.groupby(column)[target].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        train_encoded[valid_idx] = (
            train_df.iloc[valid_idx][column].map(smooth).fillna(global_mean).values
        )

    full_stats = train_df.groupby(column)[target].agg(["mean", "count"])
    full_smooth = (full_stats["count"] * full_stats["mean"] + smoothing * global_mean) / (
        full_stats["count"] + smoothing
    )
    test_encoded = test_df[column].map(full_smooth).fillna(global_mean).values
    return train_encoded, test_encoded


def apply_target_encodings(train_fe: pd.DataFrame, test_fe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoding_plan = {
        "User_ID": "User_ID_TE",
        "Product_ID": "Product_ID_TE",
        "Category_Combo": "Category_Combo_TE",
        "Product1_Marital": "Product1_Marital_TE",
        "Age_Occupation": "Age_Occupation_TE",
        "Product_Category_1": "Product_Category_1_TE",
        "Category_12": "Category_12_TE",
        "Occupation_City_Gender": "Occupation_City_Gender_TE",
    }
    for source, target_name in encoding_plan.items():
        train_fe[target_name], test_fe[target_name] = add_target_encoding(train_fe, test_fe, source)
    return train_fe, test_fe


def feature_sets(train_fe: pd.DataFrame) -> tuple[list[str], list[str]]:
    baseline = [
        "Gender_Binary",
        "Age_Ordinal",
        "Occupation",
        "City_Ordinal",
        "Stay_Ordinal",
        "Marital_Status",
        "Product_Category_1",
        "Product_Category_2_filled",
        "Product_Category_3_filled",
        "Category_Count",
    ]
    advanced = [
        "Gender_Binary",
        "Age_Ordinal",
        "Occupation",
        "City_Ordinal",
        "Stay_Ordinal",
        "Marital_Status",
        "Product_Category_1",
        "Product_Category_2_filled",
        "Product_Category_3_filled",
        "Product_Category_2_missing",
        "Product_Category_3_missing",
        "Category_Count",
        "Product_Popularity",
        "User_Total_Orders",
        "User_Unique_Products",
        "User_Unique_Cat1",
        "User_Cat_Freq",
        "Product_City_Freq",
        "Category_Combo_Freq",
        "User_ID_TE",
        "Product_ID_TE",
        "Category_Combo_TE",
        "Product1_Marital_TE",
        "Age_Occupation_TE",
        "Product_Category_1_TE",
        "Category_12_TE",
        "Occupation_City_Gender_TE",
    ]
    return baseline, advanced


def compute_statistical_ranking(
    train_fe: pd.DataFrame, features: list[str], figures_dir: Path
) -> pd.DataFrame:
    sample = train_fe.sample(n=min(RANK_SAMPLE, len(train_fe)), random_state=RANDOM_STATE)
    X_sample = sample[features].fillna(0)
    y_sample = sample["Purchase"]

    mi = mutual_info_regression(X_sample, y_sample, random_state=RANDOM_STATE)
    f_scores, _ = f_regression(X_sample, y_sample)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_sample, y_sample)
    rf_importance = rf.feature_importances_

    ranking = pd.DataFrame(
        {
            "feature": features,
            "mutual_info": mi,
            "f_score": f_scores,
            "rf_importance": rf_importance,
        }
    )
    for metric in ("mutual_info", "f_score", "rf_importance"):
        ranking[f"{metric}_rank"] = ranking[metric].rank(ascending=False, method="average")
    ranking["mean_rank"] = ranking[
        ["mutual_info_rank", "f_score_rank", "rf_importance_rank"]
    ].mean(axis=1)
    ranking = ranking.sort_values("mean_rank").reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    top_plot = ranking.head(15).sort_values("mutual_info", ascending=True)
    sns.barplot(data=top_plot, x="mutual_info", y="feature", color=sns.color_palette("magma", 15)[8])
    plt.title("Mutual Information: сила нелинейной связи с Purchase")
    plt.xlabel("Mutual information")
    plt.ylabel("Признак")
    save_figure(figures_dir / "eda_mutual_info.png")
    return ranking


def select_features(ranking: pd.DataFrame, train_fe: pd.DataFrame) -> list[str]:
    must_have = [
        "User_ID_TE",
        "Product_ID_TE",
        "Category_Combo_TE",
        "Product1_Marital_TE",
        "Age_Occupation_TE",
        "Product_Category_1_TE",
        "Category_12_TE",
        "Occupation_City_Gender_TE",
        "Product_Popularity",
        "User_Cat_Freq",
        "User_Total_Orders",
        "User_Unique_Products",
        "Product_City_Freq",
        "Category_Count",
        "Age_Ordinal",
        "Stay_Ordinal",
        "City_Ordinal",
        "Product_Category_1",
        "Product_Category_2_filled",
        "Product_Category_3_filled",
        "Product_Category_2_missing",
        "Product_Category_3_missing",
        "Occupation",
        "Marital_Status",
        "Gender_Binary",
    ]
    selected = ranking.head(TOP_FEATURES)["feature"].tolist()
    final_features = []
    for feature in must_have + selected:
        if feature in train_fe.columns and feature not in final_features:
            final_features.append(feature)
    return final_features


def baseline_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=300,
        max_depth=8,
        min_samples_leaf=40,
        random_state=RANDOM_STATE,
    )


def lightgbm_model() -> Any:
    if not HAS_LIGHTGBM:
        return None
    return LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        n_estimators=1000,
        num_leaves=127,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def xgboost_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.05,
        n_estimators=500,
        max_depth=8,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.5,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def catboost_model() -> Any:
    if not HAS_CATBOOST:
        return None
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        learning_rate=0.05,
        depth=8,
        iterations=900,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        verbose=False,
    )


def train_cv(
    name: str,
    model_factory: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    output_dir: Path,
) -> tuple[ModelResult, np.ndarray, np.ndarray, pd.DataFrame | None]:
    if model_factory is None:
        raise ValueError(f"Model factory for {name} is not available.")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_train), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    importances: list[np.ndarray] = []
    target_floor = float(y_train.min())

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        model = model_factory()

        if name == "LightGBM":
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[early_stopping(50, verbose=False)],
            )
            val_pred = np.clip(model.predict(X_val), target_floor, None)
            test_pred += np.clip(model.predict(X_test), target_floor, None) / N_SPLITS
            importances.append(model.feature_importances_)
        elif name == "XGBoost":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            val_pred = np.clip(model.predict(X_val), target_floor, None)
            test_pred += np.clip(model.predict(X_test), target_floor, None) / N_SPLITS
            importances.append(model.feature_importances_)
        elif name == "CatBoost":
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
            val_pred = np.clip(model.predict(X_val), target_floor, None)
            test_pred += np.clip(model.predict(X_test), target_floor, None) / N_SPLITS
            importances.append(model.get_feature_importance())
        else:
            model.fit(X_tr, y_tr)
            val_pred = np.clip(model.predict(X_val), target_floor, None)
            test_pred += np.clip(model.predict(X_test), target_floor, None) / N_SPLITS

        oof[valid_idx] = val_pred
        fold_rmse = rmse(y_val.values, val_pred)
        print(f"[{name}] fold {fold}/{N_SPLITS}: RMSE={fold_rmse:.2f}")

    importance_df = None
    importance_path = None
    if importances:
        importance_df = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": np.mean(importances, axis=0),
            }
        ).sort_values("importance", ascending=False)
        importance_path = str(output_dir / f"{name.lower()}_importance.csv")
        importance_df.to_csv(importance_path, index=False)

    oof_path = output_dir / f"{name.lower()}_oof.npy"
    pred_path = output_dir / f"{name.lower()}_test_pred.npy"
    np.save(oof_path, oof)
    np.save(pred_path, test_pred)

    result = ModelResult(
        name=name,
        rmse=rmse(y_train.values, oof),
        r2=float(r2_score(y_train.values, oof)),
        mape=mape(y_train.values, oof),
        oof_path=str(oof_path),
        pred_path=str(pred_path),
        importance_path=importance_path,
    )
    return result, oof, test_pred, importance_df


def optimize_blend(
    candidates: list[tuple[str, np.ndarray, np.ndarray]],
    y_true: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, float]:
    target_floor = float(np.min(y_true))
    if len(candidates) == 1:
        name, oof, pred = candidates[0]
        oof = np.clip(oof, target_floor, None)
        pred = np.clip(pred, target_floor, None)
        return {name: 1.0}, oof, pred, rmse(y_true, oof)

    best_score = math.inf
    best_weights: dict[str, float] = {}
    best_oof = None
    best_pred = None

    if len(candidates) == 2:
        for weight in np.linspace(0.0, 1.0, 21):
            w1 = float(weight)
            w2 = 1.0 - w1
            blend_oof = np.clip(w1 * candidates[0][1] + w2 * candidates[1][1], target_floor, None)
            score = rmse(y_true, blend_oof)
            if score < best_score:
                best_score = score
                best_weights = {candidates[0][0]: w1, candidates[1][0]: w2}
                best_oof = blend_oof
                best_pred = np.clip(w1 * candidates[0][2] + w2 * candidates[1][2], target_floor, None)
    else:
        grid = np.linspace(0.0, 1.0, 11)
        for w1 in grid:
            for w2 in grid:
                w3 = 1.0 - w1 - w2
                if w3 < 0:
                    continue
                blend_oof = np.clip(
                    w1 * candidates[0][1] + w2 * candidates[1][1] + w3 * candidates[2][1],
                    target_floor,
                    None,
                )
                score = rmse(y_true, blend_oof)
                if score < best_score:
                    best_score = score
                    best_weights = {
                        candidates[0][0]: float(w1),
                        candidates[1][0]: float(w2),
                        candidates[2][0]: float(w3),
                    }
                    best_oof = blend_oof
                    best_pred = np.clip(
                        w1 * candidates[0][2] + w2 * candidates[1][2] + w3 * candidates[2][2],
                        target_floor,
                        None,
                    )

    if best_oof is None or best_pred is None:
        raise RuntimeError("Не удалось подобрать веса ансамбля.")
    return best_weights, best_oof, best_pred, best_score


def generate_eda_plots(
    train: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Any]:
    insights: dict[str, Any] = {}
    global_mean = float(train["Purchase"].mean())
    insights["global_mean_purchase"] = global_mean

    missing_share = (train[["Product_Category_2", "Product_Category_3"]].isna().mean() * 100).round(2)
    insights["missing_share"] = missing_share.to_dict()

    cat_stats = (
        train.groupby("Product_Category_1")["Purchase"]
        .agg(["mean", "std", "count"])
        .sort_index()
        .reset_index()
    )
    insights["top_category_by_mean"] = int(cat_stats.sort_values("mean", ascending=False).iloc[0]["Product_Category_1"])
    insights["top_category_mean_purchase"] = float(cat_stats["mean"].max())
    insights["most_volatile_category"] = int(cat_stats.sort_values("std", ascending=False).iloc[0]["Product_Category_1"])

    plt.figure(figsize=(12, 6))
    sns.barplot(data=cat_stats, x="Product_Category_1", y="std", palette="rocket")
    plt.axhline(cat_stats["std"].mean(), color="black", linestyle="--", label="Среднее стандартное отклонение")
    plt.title("Разброс цен (STD) по категориям")
    plt.xlabel("Product_Category_1")
    plt.ylabel("Стандартное отклонение цены")
    plt.legend()
    save_figure(figures_dir / "volatility_by_cat.png")

    age_cat = train.pivot_table(
        index="Age", columns="Product_Category_1", values="Purchase", aggfunc="mean"
    )
    plt.figure(figsize=(16, 7))
    sns.heatmap(age_cat, annot=True, fmt=".0f", cmap="YlOrRd")
    plt.title("Тепловая карта взаимодействия: Возраст vs Категория 1")
    plt.xlabel("Номер основной категории продукта")
    plt.ylabel("Возрастная группа")
    save_figure(figures_dir / "age_cat1_heatmap.png")
    insights["age_category_peak"] = {
        "age": str(age_cat.stack().idxmax()[0]),
        "category": int(age_cat.stack().idxmax()[1]),
        "mean_purchase": float(age_cat.stack().max()),
    }

    plt.figure(figsize=(14, 6))
    sns.boxenplot(
        data=train,
        x="City_Category",
        y="Purchase",
        hue="Gender",
        palette="pastel",
    )
    plt.title("Распределение покупательской способности по городам и полу")
    save_figure(figures_dir / "city_gender_purchase.png")

    user_stats = (
        train.groupby("User_ID")
        .agg(
            avg_spend=("Purchase", "mean"),
            order_count=("Purchase", "size"),
            spend_std=("Purchase", "std"),
            occupation=("Occupation", "first"),
        )
        .reset_index()
    )
    user_stats["spend_std"] = user_stats["spend_std"].fillna(0)

    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(
        user_stats["order_count"],
        user_stats["avg_spend"],
        c=user_stats["occupation"],
        s=np.clip(user_stats["spend_std"], 20, 200),
        cmap="viridis",
        alpha=0.5,
    )
    plt.colorbar(scatter, label="Occupation")
    plt.title("Сегментация пользователей: Активность (Count) vs Лояльность (Avg Spend)")
    plt.xlabel("Количество покупок (Активность)")
    plt.ylabel("Средний чек (Лояльность)")
    save_figure(figures_dir / "user_segmentation_rfm.png")

    top_occupations = (
        train["Occupation"].value_counts().head(3).index.tolist()
    )
    profile = train[train["Occupation"].isin(top_occupations)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.countplot(data=profile, x="Occupation", hue="Age", ax=axes[0], palette="viridis")
    axes[0].set_title("Возрастной профиль топ-профессий")
    axes[0].set_ylabel("Количество записей")
    gender_mix = (
        profile.groupby(["Occupation", "Gender"]).size().reset_index(name="count")
    )
    gender_mix["share"] = gender_mix.groupby("Occupation")["count"].transform(lambda s: 100 * s / s.sum())
    sns.barplot(data=gender_mix, x="Occupation", y="share", hue="Gender", ax=axes[1], palette="magma")
    axes[1].set_title("Гендерный состав (%)")
    axes[1].set_ylabel("Процент")
    save_figure(figures_dir / "top_occ_profiles.png")
    insights["top_occupations"] = [int(x) for x in top_occupations]

    occ_mean = train.groupby("Occupation")["Purchase"].agg(["mean", "count", "std"]).reset_index()
    occ_mean = occ_mean.sort_values("mean", ascending=False)
    plt.figure(figsize=(16, 8))
    sns.barplot(data=occ_mean, x="Occupation", y="mean", palette="viridis")
    plt.axhline(global_mean, color="red", linestyle="--", label="Общее среднее")
    plt.title("Средняя сумма покупки по кодам профессий (отсортировано)")
    plt.ylabel("Средний чек (Purchase)")
    plt.legend()
    save_figure(figures_dir / "occupation_purchase_anova.png")

    top_cat = int(train["Product_Category_1"].value_counts().idxmax())
    subset = train[train["Product_Category_1"] == top_cat]
    plt.figure(figsize=(14, 7))
    sns.kdeplot(
        data=subset,
        x="Purchase",
        hue="Marital_Status",
        fill=True,
        common_norm=False,
        palette="crest",
    )
    plt.title(
        f"Распределение трат (Purchase) внутри Категории {top_cat} в разрезе Marital_Status"
    )
    plt.xlabel("Сумма покупки")
    save_figure(figures_dir / "cat_purchase_dist_marital.png")

    plt.figure(figsize=(14, 7))
    sns.pointplot(
        data=train,
        x="Age",
        y="Purchase",
        hue="Marital_Status",
        errorbar=("ci", 95),
    )
    plt.title("Влияние семейного положения на траты в разных возрастных группах")
    save_figure(figures_dir / "marital_age_purchase.png")

    temp = train.copy()
    temp["Cat_Count"] = temp[["Product_Category_1", "Product_Category_2", "Product_Category_3"]].notna().sum(axis=1)
    temp["Gender_Binary"] = (temp["Gender"] == "M").astype(int)
    plt.figure(figsize=(14, 7))
    sns.boxenplot(data=temp, x="Cat_Count", y="Purchase", hue="Gender_Binary")
    plt.title("Зависимость суммы от количества заполненных категорий товара")
    save_figure(figures_dir / "purchase_vs_cat_count.png")

    product_stats = (
        train.groupby("Product_ID")["Purchase"]
        .agg(popularity="size", mean_purchase="mean")
        .reset_index()
    )
    plt.figure(figsize=(14, 7))
    sns.scatterplot(
        data=product_stats,
        x="popularity",
        y="mean_purchase",
        alpha=0.3,
        s=50,
    )
    z = np.polyfit(product_stats["popularity"], product_stats["mean_purchase"], 1)
    p = np.poly1d(z)
    xs = np.linspace(product_stats["popularity"].min(), product_stats["popularity"].max(), 100)
    plt.plot(xs, p(xs), color="red", linewidth=2)
    plt.title("Связь популярности товара и его цены")
    plt.xlabel("Product_Popularity")
    plt.ylabel("Purchase")
    save_figure(figures_dir / "popularity_vs_price.png")

    occ_city = (
        train.groupby(["Occupation", "City_Category"])["Purchase"].mean().unstack()
    )
    deviation = (occ_city - global_mean).round(0)
    plt.figure(figsize=(12, 10))
    sns.heatmap(deviation, annot=True, fmt=".0f", cmap="RdYlGn")
    plt.title("Отклонение среднего чека от общего среднего (Профессия vs Город)")
    save_figure(figures_dir / "occ_city_deviation.png")

    user_stats["User_Spend_Level"] = pd.qcut(
        user_stats["avg_spend"], q=5, labels=["Low", "Below_Avg", "Average", "Above_Avg", "High"]
    )
    user_profiles = user_stats.merge(train[["User_ID", "Gender"]].drop_duplicates(), on="User_ID", how="left")
    user_profiles["Gender_Binary"] = (user_profiles["Gender"] == "M").astype(int)
    plt.figure(figsize=(14, 7))
    sns.violinplot(
        data=user_profiles,
        x="User_Spend_Level",
        y="avg_spend",
        hue="Gender_Binary",
        split=True,
        palette="tab10",
    )
    plt.title("Ценовые профили пользователей")
    plt.xlabel("User_Spend_Level")
    plt.ylabel("Purchase")
    save_figure(figures_dir / "user_spend_profiles.png")

    return insights


def plot_rmse_comparison(baseline_rmse: float, final_rmse: float, figures_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(["Baseline", "With New Features"], [baseline_rmse, final_rmse], color=["gray", "green"])
    plt.title("Сравнение RMSE до и после добавления признаков")
    plt.ylabel("RMSE (ниже - лучше)")
    save_figure(figures_dir / "rmse_improvement_comparison.png")


def plot_feature_importance(
    best_model_name: str,
    importance_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    model_top = importance_df.head(10).sort_values("importance", ascending=True)
    stat_top = ranking_df.head(10).sort_values("mean_rank", ascending=False)
    axes[0].barh(model_top["feature"], model_top["importance"], color=sns.color_palette("viridis", len(model_top)))
    axes[0].set_title(f"ТОП-10: Выбор модели ({best_model_name})")
    axes[1].barh(stat_top["feature"], stat_top["mean_rank"], color=sns.color_palette("magma", len(stat_top)))
    axes[1].set_title("ТОП-10: Статистическая значимость")
    axes[1].invert_xaxis()
    save_figure(figures_dir / "feature_importance_comparison.png")

    plt.figure(figsize=(11, 8))
    ordered = importance_df.head(30).sort_values("importance", ascending=True)
    plt.barh(ordered["feature"], ordered["importance"], color="steelblue")
    plt.title(f"Важность признаков ({best_model_name} — среднее по фолдам)")
    plt.xlabel("Importance")
    if best_model_name == "LightGBM":
        save_figure(figures_dir / "lgbm_avg_importance_horizontal.png")
    else:
        save_figure(figures_dir / "model_avg_importance_horizontal.png")


def plot_final_diagnostics(
    y_true: pd.Series,
    oof_pred: np.ndarray,
    product_popularity: pd.Series,
    importance_df: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sample_size = min(5000, len(y_true))
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(y_true), size=sample_size, replace=False)
    axes[0].scatter(y_true.iloc[sample_idx], oof_pred[sample_idx], alpha=0.2, s=8, color="steelblue")
    lims = [0, max(float(y_true.max()), float(np.max(oof_pred)))]
    axes[0].plot(lims, lims, "r--", linewidth=1.2)
    axes[0].set_xlabel("Факт (Purchase)")
    axes[0].set_ylabel("Прогноз (Purchase)")
    axes[0].set_title("Факт vs Прогноз (5k точек)")
    top_imp = importance_df.head(15).sort_values("importance", ascending=True)
    axes[1].barh(top_imp["feature"], top_imp["importance"], color=sns.color_palette("magma", len(top_imp)))
    axes[1].set_title("Топ-15 признаков по важности")
    save_figure(figures_dir / "final_ensemble_performance.png")

    error_df = pd.DataFrame(
        {
            "abs_pct_error": np.abs(y_true.values - oof_pred) / np.maximum(y_true.values, 1),
            "Product_Popularity": product_popularity.values,
        }
    )
    error_curve = (
        error_df.groupby("Product_Popularity")["abs_pct_error"]
        .mean()
        .reset_index()
        .sort_values("Product_Popularity")
    )
    plt.figure(figsize=(14, 7))
    plt.plot(error_curve["Product_Popularity"], error_curve["abs_pct_error"], color="red")
    plt.axhline(error_curve["abs_pct_error"].mean(), color="black", linestyle="--", label="Средняя ошибка")
    plt.fill_between(
        error_curve["Product_Popularity"],
        error_curve["abs_pct_error"] * 0.9,
        error_curve["abs_pct_error"] * 1.1,
        color="red",
        alpha=0.15,
    )
    plt.title("Как популярность товара снижает ошибку модели")
    plt.xlabel("Популярность товара (кол-во покупок в базе)")
    plt.ylabel("Абсолютная ошибка предсказания")
    plt.legend()
    save_figure(figures_dir / "error_vs_popularity.png")

    corr = np.corrcoef(error_df["Product_Popularity"], error_df["abs_pct_error"])[0, 1]
    return {"error_popularity_corr": float(corr)}


def make_submission(
    test_fe: pd.DataFrame,
    prediction: np.ndarray,
    output_dir: Path,
    stem: str = "black_friday_submission",
) -> dict[str, str]:
    prediction = np.clip(prediction, 0, None)
    submission = pd.DataFrame(
        {
            "User_ID": test_fe.get("User_ID_Original", test_fe["User_ID"]),
            "Product_ID": test_fe.get("Product_ID_Original", test_fe["Product_ID"]),
            "Purchase": np.round(prediction, 0).astype(int),
        }
    )
    full_submission = submission.copy()
    if "Comb" in test_fe.columns:
        full_submission.insert(0, "Comb", test_fe["Comb"])

    submission_path = output_dir / f"{stem}.csv"
    full_submission_path = output_dir / f"{stem}_full.csv"
    submission.to_csv(submission_path, index=False)
    full_submission.to_csv(full_submission_path, index=False)
    return {
        "submission": str(submission_path),
        "submission_full": str(full_submission_path),
    }


def run_pipeline(train_path: Path, test_path: Path, output_root: Path) -> dict[str, Any]:
    np.random.seed(RANDOM_STATE)
    sns.set_theme(style="whitegrid")
    dirs = ensure_dirs(output_root)

    train, test = load_data(train_path, test_path)
    train_fe, test_fe = build_feature_frames(train, test)
    train_fe, test_fe = apply_target_encodings(train_fe, test_fe)

    eda_insights = generate_eda_plots(train, dirs["figures"])
    baseline_features, advanced_candidates = feature_sets(train_fe)
    ranking = compute_statistical_ranking(train_fe, advanced_candidates, dirs["figures"])
    ranking_path = dirs["base"] / "statistical_feature_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    selected_features = select_features(ranking, train_fe)

    y_train = train_fe["Purchase"]
    X_baseline = train_fe[baseline_features].fillna(0)
    X_train = train_fe[selected_features].fillna(0)
    X_test = test_fe[selected_features].fillna(0)

    baseline_result, baseline_oof, baseline_test_pred, _ = train_cv(
        name="BaselineHGB",
        model_factory=baseline_model,
        X_train=X_baseline,
        y_train=y_train,
        X_test=test_fe[baseline_features].fillna(0),
        output_dir=dirs["base"],
    )

    model_results: list[ModelResult] = [baseline_result]
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
    importance_for_best = None
    best_model_name = None

    if HAS_LIGHTGBM:
        lgb_result, lgb_oof, lgb_test, lgb_importance = train_cv(
            name="LightGBM",
            model_factory=lightgbm_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            output_dir=dirs["base"],
        )
        model_results.append(lgb_result)
        candidates.append(("LightGBM", lgb_oof, lgb_test))
        if lgb_importance is not None:
            importance_for_best = lgb_importance
            best_model_name = "LightGBM"

    xgb_result, xgb_oof, xgb_test, xgb_importance = train_cv(
        name="XGBoost",
        model_factory=xgboost_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        output_dir=dirs["base"],
    )
    model_results.append(xgb_result)
    candidates.append(("XGBoost", xgb_oof, xgb_test))

    if not HAS_LIGHTGBM:
        importance_for_best = xgb_importance
        best_model_name = "XGBoost"

    if HAS_CATBOOST:
        cbt_result, cbt_oof, cbt_test, _ = train_cv(
            name="CatBoost",
            model_factory=catboost_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            output_dir=dirs["base"],
        )
        model_results.append(cbt_result)
        candidates.append(("CatBoost", cbt_oof, cbt_test))

    weights, blend_oof, blend_pred, blend_rmse = optimize_blend(candidates, y_train.values)
    blend_r2 = float(r2_score(y_train.values, blend_oof))
    blend_mape = mape(y_train.values, blend_oof)
    plot_rmse_comparison(baseline_result.rmse, blend_rmse, dirs["figures"])

    if importance_for_best is None:
        importance_for_best = xgb_importance
        best_model_name = "XGBoost"
    plot_feature_importance(best_model_name, importance_for_best, ranking, dirs["figures"])
    diagnostics = plot_final_diagnostics(
        y_true=y_train,
        oof_pred=blend_oof,
        product_popularity=train_fe["Product_Popularity"],
        importance_df=importance_for_best,
        figures_dir=dirs["figures"],
    )

    train_mean = float(y_train.mean())
    blend_mean = float(np.mean(blend_pred))
    baseline_mean = float(np.mean(baseline_test_pred))
    ensemble_is_stable = 0.5 * train_mean <= blend_mean <= 1.5 * train_mean

    recommended_name = "ensemble" if ensemble_is_stable else "baseline"
    recommended_pred = blend_pred if ensemble_is_stable else baseline_test_pred
    recommended_paths = make_submission(
        test_fe,
        recommended_pred,
        dirs["base"],
        stem="black_friday_submission",
    )
    experimental_paths = make_submission(
        test_fe,
        blend_pred,
        dirs["base"],
        stem="black_friday_submission_experimental_ensemble",
    )
    baseline_paths = make_submission(
        test_fe,
        baseline_test_pred,
        dirs["base"],
        stem="black_friday_submission_baseline",
    )
    final_head = pd.read_csv(recommended_paths["submission"]).head(10).to_dict(orient="records")

    top_features = importance_for_best.head(15)["feature"].tolist()
    top_features_path = dirs["base"] / "top_features.json"
    top_features_path.write_text(json.dumps(top_features, ensure_ascii=False, indent=2))

    results = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "baseline_features": baseline_features,
        "selected_features": selected_features,
        "selected_feature_count": len(selected_features),
        "statistical_ranking_path": str(ranking_path),
        "top_features_path": str(top_features_path),
        "model_results": [asdict(result) for result in model_results],
        "ensemble": {
            "weights": weights,
            "rmse": blend_rmse,
            "r2": blend_r2,
            "mape": blend_mape,
            "best_importance_model": best_model_name,
            "test_mean_prediction": blend_mean,
        },
        "baseline_submission_mean": baseline_mean,
        "recommended_submission": recommended_name,
        "eda_insights": eda_insights,
        "diagnostics": diagnostics,
        "submission_paths": recommended_paths,
        "baseline_submission_paths": baseline_paths,
        "experimental_submission_paths": experimental_paths,
        "submission_head": final_head,
        "figures_dir": str(dirs["figures"]),
    }

    results_path = dirs["base"] / "results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train and compare Black Friday purchase models.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/black_friday"))
    args = parser.parse_args()
    for path in (args.train, args.test):
        if not path.is_file():
            parser.error(f"Dataset not found: {path}. See data/README.md.")
    results = run_pipeline(args.train, args.test, args.output)
    print(json.dumps(results["ensemble"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
