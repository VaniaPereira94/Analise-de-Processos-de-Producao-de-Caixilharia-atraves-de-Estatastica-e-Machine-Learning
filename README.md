# Previsão de Atrasos em Ordens de Produção

Modelos de Machine Learning para identificar, numa empresa industrial, as ordens de produção (OP) com risco de terminar depois da data prevista.

O projeto foi desenvolvido no âmbito do Mestrado em Estatística para Ciência de Dados (Universidade do Minho), a partir de dados reais de produção. A análise foi feita originalmente em **R** e está a ser reimplementada em **Python**.

---

## Objetivo

Prever se uma ordem de produção vai terminar com atraso, a partir das suas características (série, tratamento, número de referências, operações e planos de corte, carga mensal de produção, duração planeada), e perceber que fatores estão mais associados aos atrasos, como apoio à melhoria do planeamento.

## Dados

> Os dados pertencem à empresa e são confidenciais, por isso **não estão incluídos neste repositório**.

O dataset original tem uma linha por picagem (registo de uma operação numa OP), com informação sobre:

- a ordem de produção, a série do produto e o tratamento (anodizado, lacado);
- as datas de início e fim previstas e as datas das picagens;
- as operações realizadas (triagem, corte, corte e mecanização, assemblagem, embalagem, logística);
- os planos de corte e as ocorrências registadas.

Na análise dos dados foi identificada uma falha no processo de registo: as ocorrências que não exigiam substituição de produto eram registadas sem a referência, o que impedia a análise completa por produto.

## Metodologia

### 1. Preparação dos dados

- Normalização dos nomes das colunas e dos valores (acentos, espaços, maiúsculas).
- Conversão de datas e tratamento de valores em falta.
- Agrupamento das séries e dos tratamentos nas categorias principais.
- Remoção das OP sem data de fim prevista.

### 2. Definição da variável alvo

A data final real de cada OP é a data da última picagem de **Logística**; se não existir, a de **Embalagem**; e, se também não existir, a última picagem registada. A OP é considerada **atrasada** quando a data final real é posterior à data de fim prevista.

### 3. Construção de variáveis (uma linha por OP)

| Variável | Descrição |
|---|---|
| `serie`, `tratamento` | Série do produto e tipo de tratamento |
| `ref_distinta` | Número de referências distintas |
| `operacao_total` | Número total de operações registadas |
| `operacoes_distintas` | Número de tipos de operação distintos |
| `total_planos_corte` | Número de planos de corte distintos |
| `ops_mes` | Número de OP previstas para o mesmo mês (carga de produção) |
| `mes` | Mês previsto de conclusão |
| `duracao_planeada` | Dias entre o início e o fim previstos |
| `ratio_*` | Proporção de cada tipo de operação (triagem, corte, corte e mecanização, assemblagem) |

### 4. Modelação e validação

- Validação cruzada estratificada com 10 folds.
- Métricas: AUC (ROC), sensibilidade, especificidade e acurácia.
- Análise de multicolinearidade com o GVIF.

Modelos comparados:

- Regressão logística
- Regressão logística com regularização LASSO
- Árvore de decisão
- Random forest
- Gradient boosting
- Regressão logística com efeitos mistos (mês como efeito aleatório)

## Resultados

Resultados da versão em Python (média da validação cruzada com 10 folds):

| Modelo | AUC | Acurácia |
|---|---|---|
| Regressão logística | 0,736 | 0,69 |
| Random forest | 0,763 | 0,70 |

Na versão em R, o melhor modelo atingiu uma AUC de 0,78.

Principais conclusões:

- Os modelos baseados em árvores tiveram um desempenho ligeiramente superior à regressão logística.
- A carga mensal de produção e a sazonalidade estão associadas ao risco de atraso.
- Não foram encontrados indícios de multicolinearidade grave entre as variáveis.

## Tecnologias

- **R:** tidyverse, caret, lme4, car, pROC, randomForest, gbm, rpart
- **Python:** pandas, NumPy, scikit-learn, statsmodels, matplotlib
- **Power BI:** modelação de dados e dashboards

## Estrutura do repositório

```
├── R/
│   └── analise_atrasos.R      # análise completa em R
├── python/
│   ├── preparacao_dados.py    # limpeza e construção de variáveis
│   └── modelos.py             # treino e avaliação dos modelos
└── README.md
```

## Como executar

1. Colocar o ficheiro de dados (não incluído) na pasta `data/`.
2. Instalar as dependências:
   ```
   pip install pandas numpy scikit-learn statsmodels matplotlib
   ```
3. Correr primeiro `preparacao_dados.py` e depois `modelos.py`.

## Estado do projeto

🚧 Em desenvolvimento: a reimplementação em Python está concluída para a regressão logística e a random forest; a árvore de decisão, o LASSO, o gradient boosting e o modelo de efeitos mistos estão em curso.

## Autora

**Vânia Pereira**
Mestrado em Estatística para Ciência de Dados, Universidade do Minho
