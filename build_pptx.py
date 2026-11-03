#!/usr/bin/env python3
"""
Gera relatorio-apresentacao.pptx a partir de modelo-relatorio.pptx
com textos, tabela comparativa e figuras do projeto.
Execute: python build_pptx.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "modelo-relatorio.pptx"
OUTPUT = ROOT / "relatorio-apresentacao.pptx"
RESULTS = ROOT / "outputs" / "resultados.json"
OUT_IMG = ROOT / "outputs"

# Preencha com seus dados antes de entregar
ALUNOS = "Nome Aluno 1  |  Nome Aluno 2"
SEMESTRE = "2025/2"


def set_shape_text(shape, text: str, font_size: int | None = None) -> None:
    if not shape.has_text_frame:
        return
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    if font_size:
        for run in p.runs:
            run.font.size = Pt(font_size)


def replace_largest_text(slide, text: str, font_size: int | None = 11) -> None:
    candidates = [s for s in slide.shapes if s.has_text_frame]
    if not candidates:
        return
    shape = max(candidates, key=lambda s: len(s.text_frame.text))
    set_shape_text(shape, text, font_size)


def add_table_slide(prs, results: dict) -> None:
    """Adiciona slide extra com tabela de hiperparâmetros (opcional backup)."""
    pass  # usamos slide 6 existente


def main() -> None:
    with open(RESULTS, encoding="utf-8") as f:
        results = json.load(f)

    prs = Presentation(str(TEMPLATE))

    # --- Slide 1: capa ---
    s1 = prs.slides[0]
    for sh in s1.shapes:
        if sh.has_text_frame:
            if "Análise" in sh.text_frame.text:
                set_shape_text(
                    sh,
                    "MLP — Previsão de Desempenho\nJogadores NBA",
                    32,
                )
            elif "Disciplina" in sh.text_frame.text:
                set_shape_text(
                    sh,
                    f"Disciplina: FBX4054 — Inteligência Computacional\n"
                    f"Alunos: {ALUNOS}\n"
                    f"Semestre: {SEMESTRE}\n"
                    f"Dataset: {results['n_samples']} instâncias (após limpeza)",
                    14,
                )

    # --- Slide 2: arquitetura e execuções ---
    arch = results["best_hyperparameters"]
    s2_text = (
        "Projeto: MLP implementado do zero (Python + NumPy)\n"
        "Entrada: 26 estatísticas por jogo (PTS, AST, MP, …)\n"
        "Saída: 1 neurônio sigmoide — classe Good (1) vs Bad (0)\n\n"
        f"Arquitetura vencedora: 26 → {arch['hidden_layers'][0]} (sigmoide) → 1\n"
        f"Treino: η={arch['learning_rate']}, {arch['epochs']} épocas, "
        f"peso classe Good≈{arch['class_weight_bad_good'][1]:.1f}×\n"
        f"Split estratificado 80/20 — treino {results['train_size']}, "
        f"teste {results['test_size']}\n\n"
        "Número de execuções: 5 combinações de hiperparâmetros\n"
        "Métricas no teste: acurácia, erro, recall, especificidade, precisão, F1\n"
        "(Curva ROC opcional — não incluída nesta entrega)"
    )
    replace_largest_text(prs.slides[1], s2_text, 13)

    # Slides 3–4: definições do modelo — manter texto original do template

    # --- Slide 5: nota sobre ROC ---
    s5 = prs.slides[4]
    replace_largest_text(
        s5,
        "ROC = Receiver Operating Characteristic\n"
        "Eixo Y: TPR (Recall) = TP/(TP+FN)\n"
        "Eixo X: FPR = FP/(FP+TN)\n\n"
        "Neste trabalho priorizamos matriz de confusão, F1 e curva de loss.\n"
        "A curva ROC pode ser gerada variando o limiar de decisão (0,5 padrão).\n\n"
        "Figuras geradas: curva_treinamento.png, matriz_confusao.png",
        12,
    )

    # --- Slide 6: resultados + tabela ---
    s6 = prs.slides[5]
    best = results["best_metrics"]
    intro = (
        "Melhor configuração: 1 camada oculta (20 neurônios), η=0,08, 900 épocas\n\n"
        f"Acurácia: {best['accuracy']*100:.2f}%  |  Erro: {(1-best['accuracy'])*100:.2f}%\n"
        f"Precisão (Good): {best['precision']*100:.2f}%  |  "
        f"Recall (Good): {best['recall']*100:.2f}%\n"
        f"Especificidade (Bad): "
        f"{(best['confusion_matrix'][0][0]/(best['confusion_matrix'][0][0]+best['confusion_matrix'][0][1]))*100:.2f}%\n"
        f"F1 (Good): {best['f1']:.3f}\n"
        f"Matriz [TN,FP; FN,TP]: {best['confusion_matrix']}"
    )
    replace_largest_text(s6, intro, 11)

    # Tabela comparativa
    rows = len(results["hyperparameter_search"]) + 1
    cols = 6
    left, top, width, height = Inches(0.4), Inches(3.8), Inches(9.2), Inches(2.8)
    table_shape = s6.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    headers = ["Ocultas", "η", "Épocas", "Acurácia", "Recall", "F1 Good"]
    for c, h in enumerate(headers):
        table.cell(0, c).text = h
    for r, run in enumerate(results["hyperparameter_search"], start=1):
        hidden = run["architecture"][1:-1]
        hidden_s = "→".join(str(x) for x in hidden) if len(hidden) > 1 else str(hidden[0])
        table.cell(r, 0).text = hidden_s
        table.cell(r, 1).text = str(run["learning_rate"])
        table.cell(r, 2).text = str(run["epochs"])
        table.cell(r, 3).text = f"{run['accuracy']*100:.1f}%"
        table.cell(r, 4).text = f"{run['recall']*100:.0f}%"
        table.cell(r, 5).text = f"{run['f1']:.3f}"

    # Imagens
    if (OUT_IMG / "matriz_confusao.png").exists():
        s6.shapes.add_picture(
            str(OUT_IMG / "matriz_confusao.png"),
            Inches(6.8),
            Inches(0.3),
            width=Inches(2.8),
        )

    # --- Slide 7: conclusões ---
    top_feats = ", ".join(t[0] for t in results["top_features_permutation"][:5])
    concl = (
        "Conclusões\n"
        "• O MLP treinado com backpropagation classificou bem a classe Good (F1≈0,91), "
        "mesmo com desbalanceamento 501 Bad vs 52 Good.\n"
        "• Padronização z-score e pesos de classe foram essenciais para convergência.\n"
        "• Atributos mais relevantes (permutação): "
        f"{top_feats}.\n"
        "• Rede rasa (20 neurônios) superou arquiteturas mais profundas testadas.\n\n"
        "Aprendizagens: implementação manual fixa entendimento de forward/backward pass.\n\n"
        "Trabalhos futuros: imputação de missing, validação cruzada, mini-batch SGD, "
        "regularização L2, curva ROC e one-hot de posição."
    )
    replace_largest_text(prs.slides[6], concl, 12)

    # Slide extra: preparação + figuras EDA (duplicar layout em branco)
    blank = prs.slide_layouts[6] if len(prs.slide_layouts) > 6 else prs.slide_layouts[0]
    slide_eda = prs.slides.add_slide(blank)
    title_box = slide_eda.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6))
    title_box.text_frame.text = "Preparação dos dados e visualizações"
    body = slide_eda.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(4.5), Inches(2.5))
    body.text_frame.text = (
        "• CSV NBA: limpeza de missing, remoção FG%.1\n"
        "• 26 features numéricas; alvo Bad=0 / Good=1\n"
        f"• {results['class_counts']['Bad']} Bad, {results['class_counts']['Good']} Good\n"
        "• explore_data.py → histogramas e correlação\n"
        "• Código: data_utils.py, mlp.py, metrics.py, main.py"
    )
    for img, x in [("eda_distribuicao.png", 5.0), ("curva_treinamento.png", 7.0)]:
        p = OUT_IMG / img
        if p.exists():
            slide_eda.shapes.add_picture(str(p), Inches(x), Inches(1.2), width=Inches(2.3))

    slide_imp = prs.slides.add_slide(blank)
    slide_imp.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.6)).text_frame.text = (
        "Importância das variáveis"
    )
    p = OUT_IMG / "importancia_permutacao.png"
    if p.exists():
        slide_imp.shapes.add_picture(str(p), Inches(0.8), Inches(1.0), width=Inches(8.5))

    prs.save(str(OUTPUT))
    print(f"Apresentação salva em: {OUTPUT}")
    print("Edite ALUNOS e SEMESTRE no topo de build_pptx.py e rode de novo se necessário.")


if __name__ == "__main__":
    main()
