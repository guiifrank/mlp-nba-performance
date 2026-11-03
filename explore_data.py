#!/usr/bin/env python3
"""Visualização exploratória do dataset (Pandas + Matplotlib)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_utils import FEATURE_COLUMNS, load_and_clean

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_and_clean(str(ROOT / "dataset.csv"))

    print("=== Amostra do dataset (5 primeiras linhas, atributos selecionados) ===")
    cols = ["Player", "Pos", "PTS", "MP", "AST", "Performance"]
    print(df[cols].head().to_string(index=False))

    print("\n=== Estatísticas descritivas ===")
    print(df[FEATURE_COLUMNS].describe().round(3).to_string())

    print("\n=== Distribuição da variável alvo ===")
    print(df["Performance"].value_counts().rename({0: "Bad", 1: "Good"}))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df["Performance"].map({0: "Bad", 1: "Good"}).value_counts()
    axes[0].bar(counts.index.astype(str), counts.values, color=["#dc2626", "#16a34a"])
    axes[0].set_title("Distribuição de Performance")
    axes[0].set_ylabel("Quantidade")

    good = df[df["Performance"] == 1]["PTS"]
    bad = df[df["Performance"] == 0]["PTS"]
    axes[1].hist([bad, good], bins=20, label=["Bad", "Good"], color=["#f87171", "#4ade80"])
    axes[1].set_xlabel("PTS (pontos por jogo)")
    axes[1].set_ylabel("Frequência")
    axes[1].set_title("PTS por classe")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "eda_distribuicao.png", dpi=150)
    plt.close()

    corr = df[FEATURE_COLUMNS + ["Performance"]].corr()["Performance"].drop("Performance")
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    plt.figure(figsize=(8, 6))
    plt.barh(corr.index[::-1][:15], corr.values[::-1][:15], color="#6366f1")
    plt.xlabel("Correlação de Pearson com Performance")
    plt.title("Top 15 atributos correlacionados com o alvo")
    plt.tight_layout()
    plt.savefig(OUT / "eda_correlacao.png", dpi=150)
    plt.close()
    print(f"\nFiguras salvas em {OUT}")


if __name__ == "__main__":
    main()
