# Checkpoint 3 — Dashboard: Segurança nas Rodovias Federais (2025)

Dashboard interativo em **Streamlit** que reúne:

- **Parte 1** — análise descritiva: KPIs, estados com mais acidentes, principais causas, acidentes com vítimas fatais, evolução mensal, mapa de calor e mapa geográfico.
- **Parte 2 / CP2** — modelagem preditiva: classificação binária (acidente com vítimas fatais ou não) com XGBoost, métricas, matriz de confusão, importância das variáveis e simulador de probabilidade.

**Turma:** 2CCPW · 

**Integrantes:**
 
- Guilherme Ferraz (RM: 564743)
- Roberto Marques (RM: 564935)
- Manoela Oliveira Bello (RM: 563952)
- Felipe Rodrigues Ribeiro dos Santos (RM: 565274)

**App online:** https://segurancanasrodovias.streamlit.app/

## Fonte dos dados

Polícia Rodoviária Federal (PRF) — Dados Abertos, base de acidentes `datatran2025.csv`
(https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-acidentes).
Separador `;`, codificação `latin1`. O arquivo deve ficar em `data/datatran2025.csv`.

## Estrutura

```
cp3_dashboard/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── datatran2025.csv
```

## Como executar localmente

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Abra http://localhost:8501.

## Como publicar (Streamlit Community Cloud)

1. Suba esta pasta (incluindo `data/datatran2025.csv`) para um repositório público no GitHub.
2. Acesse https://share.streamlit.io, entre com o GitHub e clique em **Create app**.
3. Escolha o repositório, branch `main` e arquivo principal `app.py`, depois **Deploy**.
4. Cole a URL pública gerada neste README e na entrega.

## Tecnologias

Python, Streamlit, Pandas, Plotly, Matplotlib/Seaborn, scikit-learn, XGBoost.

## Modelo preditivo (resumo)

- Alvo: `classificacao_acidente` == "Com Vítimas Fatais" → 1; demais → 0 (~7% da base).
- Features: `uf`, `fase_dia`, `condicao_metereologica`, `tipo_pista`, `tracado_via`, `tipo_acidente` (one-hot com `pd.get_dummies`).
- Divisão 70% treino / 30% teste, estratificada, `random_state=42`.
- XGBoost com `scale_pos_weight` (razão negativos/positivos) para lidar com o desbalanceamento; escolhido por ter o maior recall na classe fatal.
