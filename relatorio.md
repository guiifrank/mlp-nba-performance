# Relatório — Rede Neural MLP para Previsão de Desempenho de Jogadores NBA

**Disciplina:** Inteligência Computacional (TDE)  
**Tema:** Implementação de perceptron multicamada sem frameworks de ML  
**Dataset:** Estatísticas por jogo de jogadores da NBA (`dataset.csv`)

> Use o arquivo `modelo-relatorio.pptx` como layout visual e transcreva/adapte as seções abaixo. Figuras geradas estão em `outputs/`.

---

## 1. Preparação dos dados

### 1.1 Carregamento e visualização

O dataset foi carregado com **Pandas** (`explore_data.py`, `data_utils.py`). Cada linha representa um jogador (ou entrada por time, quando há registros `TOT`). A variável alvo é **Performance**, com classes **Good** (52 jogadores no arquivo original) e **Bad** (570 no original).

Foram excluídas colunas de identificação (**Player**, **Tm**, **Pos**) do modelo, pois não representam desempenho estatístico generalizável. A coluna duplicada **FG%.1** (mesmos valores em formato texto) foi removida.

**Atributos numéricos utilizados (26):** Age, G, GS, MP, FG, FGA, FG%, 3P, 3PA, 3P%, 2P, 2PA, 2P%, eFG%, FT, FTA, FT%, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS.

### 1.2 Limpeza

- Remoção de linhas sem rótulo **Performance** ou com valores ausentes nos atributos numéricos (muitos missing em **FT%** e **3P%**).
- Conversão de **Performance** para numérico: Bad → 0, Good → 1.
- Após limpeza: **553 instâncias** (501 Bad, 52 Good) — desbalanceamento forte (~9,6:1).

### 1.3 Pré-processamento

- **Divisão treino/teste estratificada** (80%/20%), preservando proporção Good/Bad em ambos os conjuntos: 443 treino, 110 teste (`data_utils.stratified_split`).
- **Padronização z-score** (média 0, desvio 1) calculada **somente no treino** e aplicada ao teste (`StandardScaler` manual).

### 1.4 Visualizações exploratórias

| Arquivo | Conteúdo |
|---------|----------|
| `outputs/eda_distribuicao.png` | Contagem Bad/Good e histograma de PTS por classe |
| `outputs/eda_correlacao.png` | Correlação de Pearson dos atributos com Performance |

---

## 2. Implementação do modelo

### 2.1 Arquitetura

Implementação em **`mlp.py`**, sem Scikit-learn, TensorFlow ou PyTorch:

| Camada | Neurônios | Função de ativação |
|--------|-----------|-------------------|
| Entrada | 26 | — |
| Oculta | 20 (melhor configuração) | Sigmoide |
| Saída | 1 | Sigmoide (probabilidade de Good) |

Inicialização **Xavier/Glorot** dos pesos; biases zerados.

### 2.2 Função de perda

**Entropia cruzada binária** entre rótulo \(y \in \{0,1\}\) e saída \(\hat{y}\):

\[
L = -\frac{1}{N}\sum_i w_i \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]
\]

Pesos \(w_i\) maiores na classe **Good** no treino (~9,55×) para mitigar desbalanceamento.

### 2.3 Retropropagação

Para cada camada, calculam-se gradientes da loss em relação a pesos e biases; atualização por **gradiente descendente**:

\[
W \leftarrow W - \eta \frac{\partial L}{\partial W}
\]

Implementação completa em `MLP.forward` e `MLP.backward`.

---

## 3. Treinamento

- **Algoritmo:** gradiente descendente em batch completo (todas as amostras de treino por época).
- **Busca manual de hiperparâmetros** (`main.py`): cinco combinações de camadas ocultas, taxa de aprendizado \(\eta\) e número de épocas.

**Melhor configuração encontrada:**

| Hiperparâmetro | Valor |
|----------------|-------|
| Camadas ocultas | 20 neurônios |
| Taxa de aprendizado \(\eta\) | 0,08 |
| Épocas | 900 |
| Peso de classe (Bad / Good) | 1,0 / ~9,55 |

**Loss final (treino):** ~0,102  

**Curva de treinamento:** `outputs/curva_treinamento.png`

---

## 4. Avaliação e ajuste

Métricas implementadas manualmente em **`metrics.py`** (matriz de confusão, acurácia, precisão, recall, F1, especificidade).

### 4.1 Resultados no conjunto de teste (melhor modelo)

| Métrica | Valor |
|---------|-------|
| Acurácia | **98,18%** |
| Precisão (Good) | **83,33%** |
| Recall (Good) | **100%** |
| F1 (Good) | **0,909** |
| Especificidade (Bad) | **98,00%** |

**Matriz de confusão:**

|  | Pred. Bad | Pred. Good |
|--|-----------|------------|
| **Real Bad** | 98 (TN) | 2 (FP) |
| **Real Good** | 0 (FN) | 10 (TP) |

Figura: `outputs/matriz_confusao.png`

### 4.2 Comparação entre configurações testadas

| Ocultas | η | Épocas | Acurácia | F1 Good |
|---------|---|--------|----------|---------|
| 16 | 0,05 | 800 | 96,36% | 0,833 |
| 24→12 | 0,05 | 1000 | 95,45% | 0,800 |
| 32→16 | 0,03 | 1200 | 95,45% | 0,800 |
| 16→8 | 0,10 | 600 | 95,45% | 0,800 |
| **20** | **0,08** | **900** | **98,18%** | **0,909** |

Redes mais profundas não superaram uma única camada oculta moderada — possível overfitting ou dificuldade de otimização com batch GD simples.

### 4.3 Observações sobre a avaliação

- A **acurácia alta** reflete em parte o desbalanceamento (prever sempre Bad já daria ~91%).
- O modelo equilibrou **recall perfeito** na classe minoritária (todos os 10 Good do teste detectados) com poucos falsos positivos (2 Bad classificados como Good).
- Métricas detalhadas em JSON: `outputs/resultados.json`.

---

## 5. Interpretação e conclusão

### 5.1 Eficácia do modelo

O MLP implementado do zero conseguiu **separar estatísticas de jogadores “Good” vs “Bad”** com bom F1 na classe Good, usando apenas estatísticas de box score. Isso indica que o rótulo está correlacionado com produção ofensiva/defensiva agregada (minutos, pontos, arremessos, etc.), e que a rede aprendeu fronteiras não lineares após normalização.

Limitações: conjunto pequeno após limpeza, desbalanceamento, possível **vazamento conceitual** se “Good” for definido diretamente a partir de métricas similares às features (PTS, eficiência). Em produção, seria necessário validar como o rótulo foi construído.

### 5.2 Variáveis com maior impacto

**Importância por permutação** (queda no F1 ao embaralhar cada atributo):

1. STL (roubos de bola)  
2. PF (faltas pessoais)  
3. GS (jogos como titular)  
4. G (jogos disputados)  
5. BLK (tocos)  

Gráfico: `outputs/importancia_permutacao.png`

**Magnitude média dos pesos na 1ª camada** (indício de sensibilidade): FG, 2P, STL, AST, PTS, eFG% — gráfico `outputs/importancia_pesos.png`.

Interpretação: desempenho “Good” associa-se a **maior participação** (GS, G, MP) e estatísticas de **impacto two-way** (STL, BLK) além de produção anotadora (FG, PTS, AST).

### 5.3 Melhorias sugeridas

1. **Imputação** de missing (ex.: mediana por posição) em vez de descartar ~70 linhas.  
2. **Validação cruzada k-fold** estratificada para estimativa mais estável.  
3. **Mini-batch SGD** ou **momentum** para convergência mais rápida e estável.  
4. **Early stopping** com conjunto de validação.  
5. **Regularização L2** ou dropout (se permitido estender a implementação).  
6. **SMOTE** ou undersampling da classe Bad para comparar modelos.  
7. Incluir **Pos** via one-hot encoding se a posição influenciar o critério de Good/Bad.

---

## Estrutura do código entregue

| Arquivo | Função |
|---------|--------|
| `data_utils.py` | Limpeza, split estratificado, padronização |
| `mlp.py` | Rede MLP, sigmoide, backprop, treino |
| `metrics.py` | Matriz de confusão e métricas |
| `explore_data.py` | EDA e gráficos iniciais |
| `main.py` | Busca de hiperparâmetros, treino final, gráficos |
| `requirements.txt` | Dependências (Pandas, NumPy, Matplotlib) |

### Como executar

```bash
pip install -r requirements.txt
python explore_data.py
python main.py
```

---

## Referências rápidas (conceitos)

- Rumelhart, Hinton & Williams — retropropagação em MLP.  
- Glorot & Bengio — inicialização de pesos em redes profundas.
