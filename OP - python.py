import unicodedata

import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def sem_acentos(texto):
    """Remove acentos e cedilhas: 'Logística' -> 'Logistica'."""
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()


#---------------------------------------------------------
# Carregar dados do csv
#---------------------------------------------------------

# Se os acentos aparecerem estranhos (ex.: 'Ã§'), trocar para encoding="utf-8"
dados = pd.read_csv(
    r"C:/Users/vania/Downloads/OP e Plano Corte.csv",
    sep=";",
    encoding="latin-1",
)

dados.info()
dados.isna().sum()

#---------------------------------------------------------
# Limpar nomes das colunas (sem acentos, minúsculas, _ em vez de espaços)
#---------------------------------------------------------

dados.columns = [sem_acentos(c).strip().lower().replace(" ", "_") for c in dados.columns]

dados.info()

#---------------------------------------------------------
# Alterar nomes
#---------------------------------------------------------

dados = dados.rename(columns={
    "referencias_distintas": "ref_distinta",
    "operacoes_totais": "operacao_total",
    "operacao_correspondente_ocorrencias": "operacao_ocorrencias",
    "qtdplanosdistintos": "total_planos_corte",
    "data_final_real_op": "data_final",
    "ocorrencias": "numero_ocorrencias",
    "operacao_correspondente": "operacao",
    "tramento_s/tamanho": "tratamento",
    "data_picagem_(s/horas)": "data_picagens",
    "x": "referencia",
    "classe_desvio_final": "classe_data",
})

#---------------------------------------------------------
# Selecionar colunas
#---------------------------------------------------------

dados = dados[[
    "ordem_producao", "serie", "op_submetida", "plano_de_corte", "posto",
    "operacao", "previsao_fim", "quantidade", "tratamento", "data_final",
    "referencia", "previsao_inicio", "data_picagens", "classe_data",
    "operacao_ocorrencias", "operacoes_distintas", "operacao_total",
    "ref_distinta", "numero_ocorrencias", "atraso", "status_picagem",
]].copy()

dados.info()

#---------------------------------------------------------
# Converter variáveis data
#---------------------------------------------------------

colunas_data = ["previsao_fim", "data_final", "previsao_inicio", "data_picagens"]

for col in colunas_data:
    if pd.api.types.is_datetime64_any_dtype(dados[col]):
        # já é data: só retiro a hora, se existir
        dados[col] = dados[col].dt.normalize()
    else:
        # é texto: corto a hora e converto
        dados[col] = pd.to_datetime(
            dados[col].astype(str).str.split(" ").str[0],
            format="%d/%m/%Y",
            errors="coerce",
        )

#---------------------------------------------------------
# Tratamento NA
#---------------------------------------------------------

dados.isna().sum()

dados = dados.fillna({
    "numero_ocorrencias": 0,
    "posto": "Não identificado",
    "operacao": "Não identificado",
    "operacao_ocorrencias": "Não identificado",
})

#---------------------------------------------------------
# Tirar acentos dos valores usados em comparações e nomes de colunas
# (evita que "Logística" falhe na comparação com "Logistica")
#---------------------------------------------------------

dados["operacao"] = dados["operacao"].map(sem_acentos).str.strip()
dados["operacao_ocorrencias"] = dados["operacao_ocorrencias"].map(sem_acentos).str.strip()

#---------------------------------------------------------
# Converter variáveis categóricas
#---------------------------------------------------------

colunas_cat = [
    "operacao", "posto", "operacao_ocorrencias", "classe_data",
    "tratamento", "serie", "op_submetida", "status_picagem",
]
dados[colunas_cat] = dados[colunas_cat].astype("category")

dados.info()

# confirmar os nomes das categorias (devem aparecer sem acentos)
print(dados["posto"].cat.categories)
print(dados["operacao"].cat.categories)
print(dados["operacao_ocorrencias"].cat.categories)

#---------------------------------------------------------
# Tratamento - considerar apenas anodizado e lacado
#---------------------------------------------------------

dados["tratamento"] = np.select(
    [
        dados["tratamento"].str.contains("Ano", case=False, na=False),
        dados["tratamento"].str.contains("Lac", case=False, na=False),
    ],
    ["Anodizado", "Lacado"],
    default="outro",
)
dados["tratamento"] = dados["tratamento"].astype("category")

#---------------------------------------------------------
# Série - agrupar nas 4 séries principais
# (as versões IF vêm primeiro para não serem apanhadas pelas genéricas)
#---------------------------------------------------------

dados["serie"] = np.select(
    [
        dados["serie"].str.contains("AH!38IF", case=False, na=False, regex=False),
        dados["serie"].str.contains("AH!38", case=False, na=False, regex=False),
        dados["serie"].str.contains("AH!60IF", case=False, na=False, regex=False),
        dados["serie"].str.contains("AH!60", case=False, na=False, regex=False),
    ],
    ["AH!38IF", "AH!38", "AH!60IF", "AH!60"],
    default="outro",
)
dados["serie"] = dados["serie"].astype("category")

#---------------------------------------------------------
# Remoção de OPs que não têm data de previsão final
#---------------------------------------------------------

dados = dados[dados["previsao_fim"].notna()].copy()

#---------------------------------------------------------
# Variável mês previsto
#---------------------------------------------------------

dados["mes_previsto"] = dados["previsao_fim"].dt.to_period("M").dt.to_timestamp()  # ano + mês + dia 01
dados["mes"] = pd.Categorical(dados["mes_previsto"].dt.month.astype("Int64"))

#---------------------------------------------------------
# Total de OPs por mês
#---------------------------------------------------------

dados["ops_mes"] = dados.groupby("mes_previsto")["ordem_producao"].transform("nunique")

#---------------------------------------------------------
# Duração planeada (dias)
#---------------------------------------------------------

dados["duracao_planeada"] = (dados["previsao_fim"] - dados["previsao_inicio"]).dt.days

#---------------------------------------------------------
# Data final real da OP: Logistica -> Embalagem -> última picagem
#---------------------------------------------------------

data_final_op = (
    dados.assign(
        data_logistica=dados["data_picagens"].where(dados["operacao_ocorrencias"] == "Logistica"),
        data_embalagem=dados["data_picagens"].where(dados["operacao_ocorrencias"] == "Embalagem"),
    )
    .groupby("ordem_producao", dropna=False)
    .agg(
        data_logistica=("data_logistica", "max"),
        data_embalagem=("data_embalagem", "max"),
        data_ultima_picagem=("data_picagens", "max"),
    )
)

# equivalente ao coalesce
data_final_op["data_final_real_op"] = (
    data_final_op["data_logistica"]
    .fillna(data_final_op["data_embalagem"])
    .fillna(data_final_op["data_ultima_picagem"])
)
data_final_op = data_final_op[["data_final_real_op"]].reset_index()

# juntar ao dataset original
dados = dados.merge(data_final_op, on="ordem_producao", how="left")

#---------------------------------------------------------
# Atraso ao nível da linha (1 = atrasada, 0 = no prazo, NaN = sem informação)
#---------------------------------------------------------

dados["atraso"] = np.select(
    [
        dados["data_final_real_op"] > dados["previsao_fim"],
        dados["data_final_real_op"] <= dados["previsao_fim"],
    ],
    [1.0, 0.0],
    default=np.nan,
)

dados_concluidos = dados[dados["data_final_real_op"].notna() & dados["atraso"].notna()].copy()

#---------------------------------------------------------
# Dataset com uma linha por OP
#---------------------------------------------------------

op_unica = (
    dados_concluidos
    .groupby("ordem_producao", dropna=False)
    .agg(
        atraso=("atraso", "max"),
        serie=("serie", "first"),
        tratamento=("tratamento", "first"),
        ref_distinta=("referencia", "nunique"),
        ops_mes=("ops_mes", "first"),
        operacoes_distintas=("operacao", "nunique"),
        operacao_total=("ordem_producao", "size"),
        total_planos_corte=("plano_de_corte", "nunique"),
        mes=("mes", "first"),
        duracao_planeada=("duracao_planeada", "first"),
        mes_previsto=("mes_previsto", "first"),
    )
    .reset_index()
)

op_unica["atraso"] = pd.Categorical(
    op_unica["atraso"].map({1: "Sim", 0: "Não"}),
    categories=["Não", "Sim"],
)

#---------------------------------------------------------
# Nº de linhas de cada operação por OP (uma coluna por operação)
#---------------------------------------------------------

contagem_op = (
    dados_concluidos
    .groupby(["ordem_producao", "operacao"], observed=True)
    .size()
    .unstack(fill_value=0)
)
contagem_op.columns = [sem_acentos(c).strip().lower().replace(" ", "_") for c in contagem_op.columns]

print(contagem_op.columns.tolist())  # confirmar os nomes usados nos rácios abaixo

# guardado com outro nome para a célula poder ser corrida várias vezes
op_com_contagens = op_unica.merge(contagem_op.reset_index(), on="ordem_producao", how="left")

#---------------------------------------------------------
# Dataset final para o modelo
#---------------------------------------------------------

op_modelo = (
    op_com_contagens
    .assign(
        ratio_triagem=lambda d: d["triagem"] / d["operacao_total"],
        ratio_corte=lambda d: d["corte"] / d["operacao_total"],
        ratio_corte_mecanizacao=lambda d: d["corte_e_mecanizacao"] / d["operacao_total"],
        ratio_assemblagem=lambda d: d["assemblagem"] / d["operacao_total"],
    )
    [[
        "atraso", "serie", "tratamento", "ref_distinta", "operacao_total",
        "operacoes_distintas", "ops_mes", "mes",
        "duracao_planeada", "total_planos_corte",
        "ratio_triagem", "ratio_corte", "ratio_corte_mecanizacao", "ratio_assemblagem",
    ]]
)

op_modelo.info()


dados_mod = op_modelo.dropna().copy()



for col in dados_mod.select_dtypes("category").columns:
    dados_mod[col] = dados_mod[col].cat.remove_unused_categories()


#Criar variável Y dummy para prever atraso (1) ou não atraso (0)

y = (dados_mod["atraso"]== "Sim").astype(int)

#criar variável X com as variáveis independentes

#o get_dummies transforma as variáveis categóricas em variáveis dummy (0 ou 1) e
#drop_first=True evita a multicolinearidade, removendo a primeira categoria de cada variável categórica.

X = pd.get_dummies(dados_mod.drop(columns="atraso"), drop_first=True, dtype=float)

print(y.value_counts())
print(X.columns.tolist())


#---------------------------------------------------------
# Modelos previsão de atraso
#---------------------------------------------------------

# Treino do modelo - k fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

metricas = {"ROC": "roc_auc",
            "Sens": make_scorer(recall_score, pos_label=0),
            "Spec": make_scorer(recall_score, pos_label=1),
            "Acurácion": "accuracy"}
            
#####################

# Regressão logística
modelo_log = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=5000))
resultados = cross_validate(modelo_log, X, y, cv=cv, scoring=metricas)
print({m: round(resultados[f"test_{m}"].mean(), 3) for m in metricas})

# summary(modelo_log$finalModel)
print(sm.Logit(y, sm.add_constant(X)).fit().summary())


#Multicolinearidade - Variance Inflation Factor (VIF)

import numpy as np

originais = dados_mod.drop(columns="atraso").columns
categoricas = dados_mod.drop(columns="atraso").select_dtypes("category").columns

grupo = {
    c: c if c in originais else next(v for v in categoricas if c.startswith(v + "_"))
    for c in X.columns
}

R = X.corr().values
det_R = np.linalg.det(R)

linhas = []
for var in dict.fromkeys(grupo.values()):
    idx = [i for i, c in enumerate(X.columns) if grupo[c] == var]
    resto = [i for i in range(len(X.columns)) if i not in idx]
    gvif = np.linalg.det(R[np.ix_(idx, idx)]) * np.linalg.det(R[np.ix_(resto, resto)]) / det_R
    linhas.append({"variavel": var, "GVIF": gvif, "Df": len(idx),
                   "GVIF^(1/(2*Df))": gvif ** (1 / (2 * len(idx)))})

gvif_tab = pd.DataFrame(linhas).sort_values("GVIF", ascending=False).round(2)
gvif_tab["moderada"] = gvif_tab["GVIF"] > 3
print(gvif_tab)


#---------------------------------------------------------
# Random Forest
#---------------------------------------------------------

p = X.shape[1]
valores_mtry = sorted(set(np.floor(np.linspace(2, p, 3)).astype(int)))
print("Nº de variáveis:", p)
print("mtry a testar:", valores_mtry)

#encontrar o melhor valor de mtry (max_features) com GridSearchCV
modelo_rf = GridSearchCV(
    RandomForestClassifier(n_estimators=500, random_state=123),
    param_grid={"max_features": valores_mtry},
    cv=cv,
    scoring=metricas,
    refit="ROC",
)

modelo_rf.fit(X, y)

#Ver os resultados

res_rf = pd.DataFrame(modelo_rf.cv_results_).filter(regex="param_|mean_test_")
print(res_rf.round(3))
print("Melhor:", modelo_rf.best_params_)


prev_rf = cross_val_predict(modelo_rf.best_estimator_, X, y, cv=cv)

matriz = pd.DataFrame(
    confusion_matrix(y, prev_rf),
    index=["Real: Não", "Real: Sim"],
    columns=["Previsto: Não", "Previsto: Sim"],
)
print(matriz)
print(classification_report(y, prev_rf, target_names=["Não", "Sim"], digits=3))



#---------------------------------------------------------
# árvore de decisão
#--------------------------------------------------------

#primeiro procurar o melhor valor do cp
modelo_arvore = GridSearchCV(
    DecisionTreeClassifier(random_state=123),
    param_grid={"ccp_alpha": np.arange(0.001, 0.011, 0.001)},
    cv = cv,
    scoring=metricas,
    refit="ROC",
)

modelo_arvore.fit(X, y)

res_arvore = pd.DataFrame(modelo_arvore.cv_results_).filter(regex="param_|mean_test_")
print(res_arvore.round(3))
print("Melhor:", modelo_arvore.best_params_)
  
