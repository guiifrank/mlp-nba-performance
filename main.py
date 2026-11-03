#!/usr/bin/env python3
"""
Treino e avaliação do MLP — previsão de Performance (Good/Bad) de jogadores NBA.
Execute: python main.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_utils import prepare_data
from metrics import accuracy, confusion_matrix, format_metrics_report, precision_recall_f1
from mlp import MLP

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
DATA_PATH = ROOT / "dataset.csv"


def feature_importance_permutation(
    model: MLP,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    seed: int = 42,
) -> list[tuple[str, float]]:
    """Queda de recall da classe Good ao permutar cada atributo (baseline: F1 Good)."""
    baseline_pred = model.predict(x_test)
    cm0 = confusion_matrix(y_test, baseline_pred)
    baseline_f1 = precision_recall_f1(cm0)["f1"]
    rng = np.random.default_rng(seed)
    importances: list[tuple[str, float]] = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            x_perm = x_test.copy()
            rng.shuffle(x_perm[:, j])
            pred = model.predict(x_perm)
            cm = confusion_matrix(y_test, pred)
            f1 = precision_recall_f1(cm)["f1"]
            drops.append(baseline_f1 - f1)
        importances.append((name, float(np.mean(drops))))
    importances.sort(key=lambda t: t[1], reverse=True)
    return importances


def first_layer_weight_importance(model: MLP, feature_names: list[str]) -> list[tuple[str, float]]:
    w = model.weights[0]
    scores = np.mean(np.abs(w), axis=1)
    pairs = list(zip(feature_names, scores.tolist()))
    pairs.sort(key=lambda t: t[1], reverse=True)
    return pairs


def plot_training_curve(history: list[float], path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history, color="#2563eb")
    plt.xlabel("Época")
    plt.ylabel("Loss (entropia cruzada binária)")
    plt.title("Curva de treinamento — MLP")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred. Bad", "Pred. Good"])
    ax.set_yticklabels(["Real Bad", "Real Good"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Matriz de confusão (conjunto de teste)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_bars(
    importances: list[tuple[str, float]], path: Path, title: str, top_k: int = 12
) -> None:
    items = importances[:top_k]
    names = [t[0] for t in items][::-1]
    vals = [t[1] for t in items][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(names, vals, color="#059669")
    plt.xlabel("Importância (score)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def hyperparameter_search(data: dict) -> dict:
    configs = [
        {"hidden": [16], "lr": 0.05, "epochs": 800},
        {"hidden": [24, 12], "lr": 0.05, "epochs": 1000},
        {"hidden": [32, 16], "lr": 0.03, "epochs": 1200},
        {"hidden": [16, 8], "lr": 0.1, "epochs": 600},
        {"hidden": [20], "lr": 0.08, "epochs": 900},
    ]
    best: dict | None = None
    best_score = -1.0
    results = []
    n_in = data["n_features"]
    for cfg in configs:
        sizes = [n_in] + cfg["hidden"] + [1]
        model = MLP(sizes, learning_rate=cfg["lr"], seed=42)
        history = model.fit(
            data["x_train"],
            data["y_train"],
            epochs=cfg["epochs"],
            class_weight=data["class_weight"],
        )
        y_pred = model.predict(data["x_test"])
        cm = confusion_matrix(data["y_test"], y_pred)
        pr = precision_recall_f1(cm)
        acc = accuracy(data["y_test"], y_pred)
        score = pr["f1"]
        entry = {
            "architecture": sizes,
            "learning_rate": cfg["lr"],
            "epochs": cfg["epochs"],
            "accuracy": acc,
            "precision": pr["precision"],
            "recall": pr["recall"],
            "f1": pr["f1"],
            "confusion_matrix": cm.tolist(),
            "final_loss": history[-1],
        }
        results.append(entry)
        if score > best_score:
            best_score = score
            best = {
                "config": cfg,
                "model": model,
                "history": history,
                "metrics": entry,
            }
    assert best is not None
    return {"best": best, "all_results": results}


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("=== 1. Preparação dos dados ===")
    data = prepare_data(str(DATA_PATH))
    df = data["df"]
    print(f"Instâncias após limpeza: {len(df)}")
    print(
        f"Classes — Bad: {(df['Performance'] == 0).sum()}, "
        f"Good: {(df['Performance'] == 1).sum()}"
    )
    print(f"Treino: {len(data['y_train'])} | Teste: {len(data['y_test'])}")
    print(f"Atributos numéricos: {data['n_features']}")

    print("\n=== 2–3. Treinamento e busca de hiperparâmetros ===")
    search = hyperparameter_search(data)
    best = search["best"]
    model: MLP = best["model"]
    cfg = best["config"]
    print(
        f"Melhor configuração: ocultas={cfg['hidden']}, "
        f"lr={cfg['lr']}, épocas={cfg['epochs']}"
    )
    print(format_metrics_report(
        np.array(best["metrics"]["confusion_matrix"]),
        best["metrics"]["accuracy"],
    ))

    print("\n=== 4. Gráficos e importância de variáveis ===")
    plot_training_curve(best["history"], OUTPUT_DIR / "curva_treinamento.png")
    cm = np.array(best["metrics"]["confusion_matrix"])
    plot_confusion(cm, OUTPUT_DIR / "matriz_confusao.png")

    perm_imp = feature_importance_permutation(
        model,
        data["x_test"],
        data["y_test"],
        data["feature_names"],
    )
    weight_imp = first_layer_weight_importance(model, data["feature_names"])
    plot_feature_bars(
        perm_imp,
        OUTPUT_DIR / "importancia_permutacao.png",
        "Top atributos (permutação — impacto no F1 da classe Good)",
    )
    plot_feature_bars(
        weight_imp,
        OUTPUT_DIR / "importancia_pesos.png",
        "Top atributos (média |peso| na 1ª camada)",
    )

    report_data = {
        "n_samples": len(df),
        "class_counts": {"Bad": int((df["Performance"] == 0).sum()), "Good": int((df["Performance"] == 1).sum())},
        "train_size": len(data["y_train"]),
        "test_size": len(data["y_test"]),
        "features": data["feature_names"],
        "best_hyperparameters": {
            "hidden_layers": cfg["hidden"],
            "learning_rate": cfg["lr"],
            "epochs": cfg["epochs"],
            "class_weight_bad_good": list(data["class_weight"]),
        },
        "best_metrics": best["metrics"],
        "hyperparameter_search": search["all_results"],
        "top_features_permutation": perm_imp[:10],
        "top_features_weights": weight_imp[:10],
    }
    with open(OUTPUT_DIR / "resultados.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print("\nResultados salvos em:", OUTPUT_DIR)
    print("Top 5 atributos (permutação):", [t[0] for t in perm_imp[:5]])


if __name__ == "__main__":
    main()
