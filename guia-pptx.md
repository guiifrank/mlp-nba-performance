# Guia para o PowerPoint (`modelo-relatorio.pptx`)

Este guia complementa o arquivo **`relatorio-apresentacao.pptx`**, gerado automaticamente por `python build_pptx.py` (rode `python main.py` antes, para ter figuras e `resultados.json`).

---

## Antes de entregar

1. Abra `build_pptx.py` e altere **`ALUNOS`** e **`SEMESTRE`**.
2. Execute: `python build_pptx.py`
3. Abra `relatorio-apresentacao.pptx` no PowerPoint/LibreOffice e revise layout (mover imagens se sobrepor texto).

---

## Slide a slide (template original + extras)

### Slide 1 — Capa

| Campo | Sugestão |
|-------|----------|
| Título | MLP — Previsão de Desempenho de Jogadores NBA |
| Disciplina | FBX4054 — Inteligência Computacional |
| Alunos | Seus nomes |
| Semestre | Ex.: 2025/2 |

---

### Slide 2 — Comparação / arquitetura

**Texto sugerido (bullets):**

- Objetivo: classificar **Performance** (Good/Bad) com MLP **sem** Scikit-learn/TensorFlow.
- **26 entradas:** estatísticas (MP, PTS, AST, FG%, rebotes, etc.).
- **Arquitetura escolhida:** 26 → **20** (sigmoide) → **1** (sigmoide = P(Good)).
- **5 execuções** com hiperparâmetros diferentes (ver tabela no slide 6).
- Pré-processamento: limpeza, **z-score** no treino, split **estratificado** 443/110.

**Figura opcional:** `outputs/eda_distribuicao.png`

---

### Slides 3 e 4 — Fórmulas (deixe o template)

O modelo já traz definições de **acurácia**, **erro**, **recall**, **precisão** e **especificidade**. Na apresentação oral, diga:

- Classe positiva = **Good** (desempenho bom).
- Classe negativa = **Bad**.

---

### Slide 5 — ROC

O template fala de curva ROC. Você pode:

- **Opção A:** Manter a explicação teórica e dizer que o foco foi matriz de confusão + F1 (como no PPTX gerado).
- **Opção B (extra):** Variar limiar de 0,1 a 0,9 e plotar TPR vs FPR (trabalho futuro).

---

### Slide 6 — Resultados

**Tabela comparativa (já no PPTX gerado):**

| Ocultas | η | Épocas | Acurácia | Recall Good | F1 |
|---------|---|--------|----------|-------------|-----|
| 16 | 0,05 | 800 | 96,4% | 100% | 0,833 |
| 24→12 | 0,05 | 1000 | 95,5% | 100% | 0,800 |
| 32→16 | 0,03 | 1200 | 95,5% | 100% | 0,800 |
| 16→8 | 0,10 | 600 | 95,5% | 100% | 0,800 |
| **20** | **0,08** | **900** | **98,2%** | **100%** | **0,909** |

**Melhor modelo — métricas teste:**

- Acurácia: **98,18%** → Erro: **1,82%**
- Precisão (Good): **83,33%**
- Recall (Good): **100%**
- Especificidade (Bad): **98%**
- Matriz: TN=98, FP=2, FN=0, TP=10

**Figuras:** `matriz_confusao.png`, `curva_treinamento.png`

---

### Slide 7 — Conclusões

**Bullets para falar:**

1. MLP manual atingiu bom **F1 na classe minoritária** (Good).
2. **Desbalanceamento** exige peso na loss; acurácia sozinha engana.
3. Variáveis importantes: **STL, PF, GS, G, BLK** (participação + defesa).
4. Aprendizado: backprop + normalização na prática.
5. Futuro: imputação, k-fold, mini-batches, ROC, regularização.

---

### Slides extras (no `relatorio-apresentacao.pptx`)

- **Preparação dos dados** + figuras EDA.
- **Importância das variáveis** (`importancia_permutacao.png`).

Se o professor limitar número de slides, mescle os extras no slide 2 e 6.

---

## Roteiro oral (~5 min)

1. Problema e dataset (30 s)
2. Pipeline de dados (30 s)
3. Arquitetura MLP + backprop (1 min)
4. Tabela de 5 configs + melhor resultado (1,5 min)
5. Matriz de confusão e interpretação (1 min)
6. Conclusão e limitações (30 s)

---

## O que falar sobre o código (resumo)

Ver também a explicação na resposta do assistente ou `relatorio.md` seção “Estrutura do código”.
