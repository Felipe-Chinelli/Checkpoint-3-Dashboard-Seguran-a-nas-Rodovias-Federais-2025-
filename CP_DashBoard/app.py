"""
Checkpoint 3 - Dashboard: Segurança nas Rodovias Federais Brasileiras (2025)
Parte 1: análise descritiva (KPIs e gráficos)
Parte 2: modelagem preditiva (classificação binária de acidente fatal - XGBoost)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Segurança nas Rodovias Federais 2025",
    page_icon="🚧",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "data" / "datatran2025.csv"

COLUMN_NAMES = [
    "id", "data_inversa", "dia_semana", "horario", "uf", "br", "km",
    "municipio", "causa_acidente", "tipo_acidente", "classificacao_acidente",
    "fase_dia", "sentido_via", "condicao_metereologica", "tipo_pista",
    "tracado_via", "uso_solo", "pessoas", "mortos", "feridos_leves",
    "feridos_graves", "ilesos", "ignorados", "feridos", "veiculos",
    "latitude", "longitude", "regional", "delegacia", "uop",
]
CLASSES_VALIDAS = ["Com Vítimas Fatais", "Com Vítimas Feridas", "Sem Vítimas"]

FEATURES = ["uf", "fase_dia", "condicao_metereologica", "tipo_pista",
            "tracado_via", "tipo_acidente"]
FATAL = "Com Vítimas Fatais"

MESES = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
         7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
ORDEM_DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
              "sexta-feira", "sábado", "domingo"]


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando dados da PRF...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH, sep=";", encoding="latin1", header=None,
        names=COLUMN_NAMES, skiprows=1, dtype=str, on_bad_lines="skip",
    )

    # Remove linhas corrompidas (campos deslocados): mantém só classificações válidas
    df = df[df["classificacao_acidente"].str.strip().isin(CLASSES_VALIDAS)].copy()

    # Texto: remove espaços extras
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})

    # Data
    data = pd.to_datetime(df["data_inversa"], format="%Y-%m-%d", errors="coerce")
    if data.isna().mean() > 0.5:
        data = pd.to_datetime(df["data_inversa"], dayfirst=True, errors="coerce")
    df["data"] = data
    df["mes"] = df["data"].dt.month

    # Hora
    df["hora"] = pd.to_numeric(df["horario"].astype(str).str[:2], errors="coerce")

    # Numéricos (a PRF usa vírgula decimal)
    for col in ["km", "latitude", "longitude"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    for col in ["pessoas", "mortos", "feridos_leves", "feridos_graves",
                "ilesos", "ignorados", "feridos", "veiculos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["fatal"] = (df["classificacao_acidente"] == FATAL).astype(int)
    df["dia_semana"] = df["dia_semana"].str.lower()
    return df


# --------------------------------------------------------------------------
# Modelo (mesmo pipeline da CP2: 70/30 estratificado, get_dummies, XGBoost)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Treinando o modelo XGBoost...")
def train_model(df: pd.DataFrame):
    dfm = df[FEATURES + ["fatal"]].dropna().copy()
    X = pd.get_dummies(dfm[FEATURES], drop_first=True)
    y = dfm["fatal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(scale_pos_weight=spw, random_state=42, n_jobs=-1,
                          eval_metric="logloss")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    importances = pd.Series(model.feature_importances_, index=X.columns)
    return {
        "model": model,
        "columns": list(X.columns),
        "y_test": y_test.to_numpy(),
        "proba": proba,
        "importances": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "spw": spw,
    }


def prever(art: dict, valores: dict) -> float:
    linha = pd.get_dummies(pd.DataFrame([valores]), drop_first=False)
    linha = linha.reindex(columns=art["columns"], fill_value=0)
    return float(art["model"].predict_proba(linha)[0, 1])


# --------------------------------------------------------------------------
# Carregamento
# --------------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(
        "Arquivo `data/datatran2025.csv` não encontrado. "
        "Baixe a base de acidentes da PRF (dados abertos) e coloque nessa pasta. "
        "Veja o README."
    )
    st.stop()

df_all = load_data()

# --------------------------------------------------------------------------
# Filtros (sidebar)
# --------------------------------------------------------------------------
st.sidebar.header("Filtros")

ufs = sorted(df_all["uf"].dropna().unique())
sel_uf = st.sidebar.multiselect("UF", ufs, default=ufs)

meses_disp = sorted(df_all["mes"].dropna().astype(int).unique())
mes_ini, mes_fim = st.sidebar.select_slider(
    "Período (mês)", options=meses_disp,
    value=(meses_disp[0], meses_disp[-1]), format_func=lambda m: MESES[m],
)

fases = sorted(df_all["fase_dia"].dropna().unique())
sel_fase = st.sidebar.multiselect("Fase do dia", fases, default=fases)

clima = sorted(df_all["condicao_metereologica"].dropna().unique())
sel_clima = st.sidebar.multiselect("Condição meteorológica", clima, default=clima)

class_ac = sorted(df_all["classificacao_acidente"].dropna().unique())
sel_class = st.sidebar.multiselect("Classificação do acidente", class_ac, default=class_ac)

df = df_all[
    df_all["uf"].isin(sel_uf)
    & df_all["mes"].between(mes_ini, mes_fim)
    & df_all["fase_dia"].isin(sel_fase)
    & df_all["condicao_metereologica"].isin(sel_clima)
    & df_all["classificacao_acidente"].isin(sel_class)
]

st.sidebar.caption(f"{len(df):,} de {len(df_all):,} registros".replace(",", "."))

# --------------------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------------------
st.title("🚧 Segurança nas Rodovias Federais Brasileiras — 2025")
st.caption("Checkpoint 3 · Ciência e Análise de Dados · "
           "Fonte: Polícia Rodoviária Federal (PRF) — Dados Abertos")

if df.empty:
    st.warning("Nenhum registro com os filtros atuais. Ajuste os filtros na barra lateral.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Visão geral", "🔎 Análises", "🗺️ Mapa", "🤖 Modelo preditivo"]
)

# --------------------------------------------------------------------------
# Tab 1 - KPIs
# --------------------------------------------------------------------------
with tab1:
    total = len(df)
    fatais = int(df["fatal"].sum())
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Acidentes", f"{total:,}".replace(",", "."))
    k2.metric("Acidentes com vítimas fatais", f"{fatais:,}".replace(",", "."))
    k3.metric("% fatais", f"{fatais / total:.2%}")
    k4.metric("Mortos", f"{int(df['mortos'].sum()):,}".replace(",", "."))
    k5.metric("Feridos", f"{int(df['feridos'].sum()):,}".replace(",", "."))

    c1, c2 = st.columns(2)

    # Pergunta 1 - estados com mais acidentes
    por_uf = df["uf"].value_counts().head(10).reset_index()
    por_uf.columns = ["UF", "Acidentes"]
    fig = px.bar(por_uf, x="UF", y="Acidentes", text="Acidentes",
                 title="Top 10 estados com mais acidentes")
    c1.plotly_chart(fig, use_container_width=True)

    # Pergunta 2 - principais causas
    causas = df["causa_acidente"].value_counts().head(10).reset_index()
    causas.columns = ["Causa", "Acidentes"]
    fig = px.bar(causas.sort_values("Acidentes"), x="Acidentes", y="Causa",
                 orientation="h", title="Top 10 causas de acidentes")
    c2.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    # Pergunta 3 - vítimas fatais
    cls = df["classificacao_acidente"].value_counts().reset_index()
    cls.columns = ["Classificação", "Acidentes"]
    fig = px.pie(cls, names="Classificação", values="Acidentes", hole=0.45,
                 title="Classificação dos acidentes")
    c3.plotly_chart(fig, use_container_width=True)

    fat_uf = (df[df["fatal"] == 1]["uf"].value_counts().head(10).reset_index())
    fat_uf.columns = ["UF", "Acidentes fatais"]
    fig = px.bar(fat_uf, x="UF", y="Acidentes fatais", text="Acidentes fatais",
                 title="Top 10 estados em acidentes com vítimas fatais",
                 color_discrete_sequence=["#c0392b"])
    c4.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 2 - Análises
# --------------------------------------------------------------------------
with tab2:
    c1, c2 = st.columns(2)

    mensal = (df.groupby("mes").agg(acidentes=("id", "count"), fatais=("fatal", "sum"))
              .reset_index())
    mensal["Mês"] = mensal["mes"].map(MESES)
    fig = px.line(mensal, x="Mês", y=["acidentes", "fatais"], markers=True,
                  title="Evolução mensal: acidentes e acidentes fatais")
    fig.update_layout(legend_title_text="", yaxis_title="Quantidade")
    c1.plotly_chart(fig, use_container_width=True)

    # Taxa de letalidade por fase do dia
    fase = df.groupby("fase_dia")["fatal"].mean().mul(100).reset_index()
    fase.columns = ["Fase do dia", "% fatais"]
    fig = px.bar(fase, x="Fase do dia", y="% fatais", text=fase["% fatais"].round(2),
                 title="Letalidade (%) por fase do dia")
    c2.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    heat = (df.pivot_table(index="dia_semana", columns="hora", values="id",
                           aggfunc="count", fill_value=0)
            .reindex([d for d in ORDEM_DIAS if d in df["dia_semana"].unique()]))
    fig = px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(x="Hora", y="Dia da semana", color="Acidentes"),
                    title="Acidentes por dia da semana × hora (mapa de calor)")
    c3.plotly_chart(fig, use_container_width=True)

    tipo = (df.groupby("tipo_acidente")
            .agg(acidentes=("id", "count"), letalidade=("fatal", "mean"))
            .reset_index())
    tipo["letalidade"] *= 100
    tipo = tipo[tipo["acidentes"] >= 50]
    fig = px.scatter(tipo, x="acidentes", y="letalidade", size="acidentes",
                     hover_name="tipo_acidente", log_x=True,
                     title="Tipo de acidente: volume × letalidade (%)",
                     labels={"acidentes": "Nº de acidentes (log)",
                             "letalidade": "% fatais"})
    c4.plotly_chart(fig, use_container_width=True)

    # Gráfico estático (matplotlib/seaborn)
    st.subheader("Gráfico estático")
    fig_s, ax = plt.subplots(figsize=(9, 3.5))
    clima_fatal = (df.groupby("condicao_metereologica")["fatal"].mean().mul(100)
                   .sort_values(ascending=False))
    sns.barplot(x=clima_fatal.values, y=clima_fatal.index, ax=ax, color="#2c7fb8")
    ax.set_xlabel("% de acidentes fatais")
    ax.set_ylabel("")
    ax.set_title("Letalidade por condição meteorológica")
    st.pyplot(fig_s)

    with st.expander("Ver dados filtrados"):
        st.dataframe(df.drop(columns=["data_inversa"]).head(1000), use_container_width=True)
        st.download_button("Baixar dados filtrados (CSV)",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name="acidentes_filtrados.csv", mime="text/csv")

# --------------------------------------------------------------------------
# Tab 3 - Mapa
# --------------------------------------------------------------------------
with tab3:
    geo = df.dropna(subset=["latitude", "longitude"])
    geo = geo[geo["latitude"].between(-34, 6) & geo["longitude"].between(-74, -34)]
    n_max = st.slider("Máximo de pontos no mapa", 500, 10000, 4000, step=500)
    if len(geo) > n_max:
        geo = geo.sample(n_max, random_state=1)
    if geo.empty:
        st.info("Sem coordenadas válidas para os filtros atuais.")
    else:
        fig = px.scatter_map(
            geo, lat="latitude", lon="longitude", color="classificacao_acidente",
            hover_data=["uf", "br", "km", "causa_acidente", "mortos"],
            zoom=3, height=600, map_style="open-street-map",
            title="Localização dos acidentes (amostra)",
        )
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 4 - Modelo preditivo
# --------------------------------------------------------------------------
with tab4:
    st.markdown(
        "**Problema:** classificação binária — prever se um acidente terá "
        "**vítimas fatais** (1) ou não (0), usando apenas variáveis conhecidas "
        "no momento do registro: `uf`, `fase_dia`, `condicao_metereologica`, "
        "`tipo_pista`, `tracado_via` e `tipo_acidente`. "
        "O modelo usa a base completa (independente dos filtros ao lado), com "
        "divisão 70% treino / 30% teste."
    )

    art = train_model(df_all)
    y_test, proba = art["y_test"], art["proba"]

    st.subheader("Desempenho do XGBoost")
    thr = st.slider("Limiar de decisão (probabilidade mínima para prever 'fatal')",
                    0.05, 0.95, 0.50, step=0.05,
                    help="Limiar menor = mais acidentes fatais detectados (recall maior), "
                         "porém mais falsos alarmes.")
    pred = (proba >= thr).astype(int)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Recall (fatais)", f"{recall_score(y_test, pred):.2f}")
    m2.metric("Precisão (fatais)", f"{precision_score(y_test, pred, zero_division=0):.2f}")
    m3.metric("F1 (fatais)", f"{f1_score(y_test, pred, zero_division=0):.2f}")
    m4.metric("ROC AUC", f"{roc_auc_score(y_test, proba):.2f}")
    m5.metric("Acurácia", f"{(pred == y_test).mean():.2f}")

    c1, c2 = st.columns(2)
    cm = confusion_matrix(y_test, pred)
    fig_cm, ax = plt.subplots(figsize=(4.5, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Não fatal", "Fatal"], yticklabels=["Não fatal", "Fatal"])
    ax.set_xlabel("Previsão do modelo")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusão (teste)")
    c1.pyplot(fig_cm)

    imp = art["importances"].sort_values(ascending=False).head(12)
    fig = px.bar(imp[::-1], orientation="h",
                 labels={"value": "Importância", "index": "Variável"},
                 title="Variáveis mais importantes")
    fig.update_layout(showlegend=False)
    c2.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparação dos modelos (resultados da CP2)")
    comp = pd.DataFrame({
        "Modelo": ["Random Forest", "XGBoost", "Rede Neural (MLP)"],
        "Acurácia": [0.81, 0.78, 0.92],
        "Recall (fatais)": [0.43, 0.59, 0.10],
        "Precisão (fatais)": [0.17, 0.18, 0.28],
        "F1 (fatais)": [0.25, 0.28, 0.15],
    })
    st.dataframe(comp, hide_index=True, use_container_width=True)
    st.caption("A MLP tem a maior acurácia, mas quase não detecta acidentes fatais "
               "(a classe fatal é uma pequena parte da base). Por isso o XGBoost, com maior "
               "recall, foi o escolhido. Valores da tabela são os da CP2; as métricas acima "
               "são calculadas ao vivo com as colunas já corrigidas e podem diferir um pouco.")

    st.subheader("Simulador: probabilidade de acidente fatal")
    s1, s2, s3 = st.columns(3)
    v_uf = s1.selectbox("UF", ufs, key="p_uf")
    v_fase = s2.selectbox("Fase do dia", fases, key="p_fase")
    v_clima = s3.selectbox("Condição meteorológica", clima, key="p_clima")
    s4, s5, s6 = st.columns(3)
    v_pista = s4.selectbox("Tipo de pista", sorted(df_all["tipo_pista"].dropna().unique()),
                           key="p_pista")
    v_traco = s5.selectbox("Traçado da via", sorted(df_all["tracado_via"].dropna().unique()),
                           key="p_traco")
    v_tipo = s6.selectbox("Tipo de acidente", sorted(df_all["tipo_acidente"].dropna().unique()),
                          key="p_tipo")

    p = prever(art, {"uf": v_uf, "fase_dia": v_fase, "condicao_metereologica": v_clima,
                     "tipo_pista": v_pista, "tracado_via": v_traco, "tipo_acidente": v_tipo})
    base = float(df_all["fatal"].mean())
    r1, r2 = st.columns(2)
    r1.metric("Probabilidade estimada de vítimas fatais", f"{p:.1%}",
              delta=f"{(p - base) * 100:+.1f} p.p. vs. média geral ({base:.1%})",
              delta_color="inverse")
    r2.markdown(
        "**Uso prático:** com essa probabilidade, o centro de comando pode priorizar o "
        "acionamento de UTI móvel ou apoio aéreo assim que o acidente é registrado."
    )

st.divider()
st.caption("Fonte dos dados: PRF — Dados Abertos (datatran2025.csv). "
           "Projeto acadêmico FIAP — Ciência e Análise de Dados.")