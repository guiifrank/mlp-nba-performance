"""Carregamento, limpeza e divisão treino/teste do dataset NBA."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Age",
    "G",
    "GS",
    "MP",
    "FG",
    "FGA",
    "FG%",
    "3P",
    "3PA",
    "3P%",
    "2P",
    "2PA",
    "2P%",
    "eFG%",
    "FT",
    "FTA",
    "FT%",
    "ORB",
    "DRB",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Performance"])
    df = df.drop(columns=["FG%.1"], errors="ignore")
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS)
    df["Performance"] = df["Performance"].map({"Good": 1, "Bad": 0})
    return df


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in (0, 1):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_ratio))
        test_idx.extend(indices[:n_test].tolist())
        train_idx.extend(indices[n_test:].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """Padronização z-score (média 0, desvio 1) — implementação manual."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> StandardScaler:
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Chame fit antes de transform")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def prepare_data(
    csv_path: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    df = load_and_clean(csv_path)
    x = df[FEATURE_COLUMNS].values.astype(float)
    y = df["Performance"].values.astype(float)
    x_train, x_test, y_train, y_test = stratified_split(
        x, y, test_ratio=test_ratio, seed=seed
    )
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    n_good = int(np.sum(y_train == 1))
    n_bad = int(np.sum(y_train == 0))
    w_bad = 1.0
    w_good = n_bad / max(n_good, 1)
    return {
        "df": df,
        "feature_names": FEATURE_COLUMNS,
        "x_train": x_train_s,
        "x_test": x_test_s,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "class_weight": (w_bad, w_good),
        "n_features": len(FEATURE_COLUMNS),
    }
