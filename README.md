# MLP — Previsão de Performance NBA (TDE)

Implementação de rede neural multicamada **do zero** (backpropagation + gradient descent) para classificar desempenho **Good/Bad** de jogadores.

## Requisitos

```bash
pip install -r requirements.txt
```

## Execução

```bash
# Exploração e gráficos EDA
python explore_data.py

# Treino, busca de hiperparâmetros e avaliação
python main.py
```

Saídas em `outputs/` (PNG + `resultados.json`).

## Relatório

Texto completo para o trabalho: **`relatorio.md`** (adaptar ao `modelo-relatorio.pptx`).

## Arquivos principais

- `mlp.py` — MLP sem bibliotecas de ML
- `metrics.py` — acurácia, precisão, recall, matriz de confusão
- `data_utils.py` — pré-processamento
# mlp-nba-performance
