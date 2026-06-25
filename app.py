import streamlit as st
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="Recomendação de Adubação",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Estado da aplicação
# ---------------------------------------------------------------------------
if "gerar_recomendacao" not in st.session_state:
    st.session_state["gerar_recomendacao"] = False
if "dados_entrada" not in st.session_state:
    st.session_state["dados_entrada"] = {}
if "recomendacao" not in st.session_state:
    st.session_state["recomendacao"] = None

# ---------------------------------------------------------------------------
# CSS customizado para design profissional
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main {
            background-color: #f8fafc;
        }

        h1 {
            color: #1e293b;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        h2, h3, h4 {
            color: #334155;
            font-weight: 600;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .stSlider > div > div > div > div {
            background: #10b981;
        }

        .stButton>button {
            background-color: #10b981;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            border: none;
            transition: all 0.2s ease;
        }

        .stButton>button:hover {
            background-color: #059669;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }

        .result-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 5px solid #10b981;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-top: 1rem;
        }

        .metric-label {
            color: #64748b;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .metric-value {
            color: #1e293b;
            font-size: 1.5rem;
            font-weight: 700;
        }

        .unit-badge {
            display: inline-block;
            background-color: #ecfdf5;
            color: #065f46;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }

        .info-box {
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 1.5rem;
        }

        .disclaimer {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.875rem;
            color: #92400e;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Dicionários de tradução
# ---------------------------------------------------------------------------
SOIL_EN_TO_PT = {
    "Clay": "Argiloso",
    "Silty": "Siltoso",
    "Sandy": "Arenoso",
    "Loamy": "Franco",
}

CROP_EN_TO_PT = {
    "Soybean": "Soja",
    "Corn": "Milho",
    "Wheat": "Trigo",
    "Bean": "Feijão",
    "Coffee": "Café",
}

# Mapeamentos invertidos para conversão PT -> EN
SOIL_PT_TO_EN = {v: k for k, v in SOIL_EN_TO_PT.items()}
CROP_PT_TO_EN = {v: k for k, v in CROP_EN_TO_PT.items()}


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------
def converter_entrada_para_modelo(n_g_kg, p_mg_dm3, k_mg_dm3):
    """
    Converte os valores de entrada do laudo de análise de solo
    para as unidades esperadas pelo modelo de recomendação.
    """
    n_modelo = n_g_kg
    p_modelo = p_mg_dm3 * 2
    k_modelo = k_mg_dm3 * 2
    return n_modelo, p_modelo, k_modelo


def modelo_recomendacao(n, p, k, cultura, expectativa_rendimento, tipo_solo):
    """
    Modelo simplificado de recomendação de adubação.
    """
    fatores = {
        "Soybean": {"n": 0.08, "p": 0.12, "k": 0.18},
        "Corn": {"n": 0.15, "p": 0.10, "k": 0.14},
        "Wheat": {"n": 0.12, "p": 0.09, "k": 0.13},
        "Bean": {"n": 0.06, "p": 0.11, "k": 0.16},
        "Coffee": {"n": 0.10, "p": 0.08, "k": 0.15},
    }

    f = fatores.get(cultura, {"n": 0.10, "p": 0.10, "k": 0.10})

    ajuste_solo = {
        "Clay": 1.05,
        "Silty": 1.00,
        "Sandy": 0.95,
        "Loamy": 1.00,
    }
    solo_fator = ajuste_solo.get(tipo_solo, 1.00)

    necessidade_n = max(0, (expectativa_rendimento * f["n"] * 10) - (n * 10))
    necessidade_p = max(0, (expectativa_rendimento * f["p"] * 10) - p)
    necessidade_k = max(0, (expectativa_rendimento * f["k"] * 10) - k)

    return {
        "N": round(necessidade_n * solo_fator, 1),
        "P2O5": round(necessidade_p * solo_fator, 1),
        "K2O": round(necessidade_k * solo_fator, 1),
    }


def exibir_recomendacao(dados, recomendacao):
    """Renderiza o card de resultado e os detalhes da recomendação."""
    cultura_pt = CROP_EN_TO_PT.get(dados["cultura_en"], dados["cultura_en"])
    tipo_solo_pt = SOIL_EN_TO_PT.get(dados["tipo_solo_en"], dados["tipo_solo_en"])

    st.markdown(
        f"""
        <div class="result-card">
            <h3>🧪 Recomendação Final para {cultura_pt}</h3>
            <p style="margin:0.2rem 0;"><strong>Tipo de solo:</strong> {tipo_solo_pt}</p>
            <p class="metric-label">Unidade de medida:</p>
            <p class="metric-value">kg/ha <span class="unit-badge">quilogramas por hectare</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Nitrogênio (N)",
            value=f"{recomendacao['N']} kg/ha",
        )

    with col2:
        st.metric(
            label="Fósforo (P₂O₅)",
            value=f"{recomendacao['P2O5']} kg/ha",
        )

    with col3:
        st.metric(
            label="Potássio (K₂O)",
            value=f"{recomendacao['K2O']} kg/ha",
        )

    st.markdown("---")

    st.subheader("Resumo dos valores de entrada")
    dados_resumo = pd.DataFrame(
        {
            "Nutriente": ["Nitrogênio (N)", "Fósforo (P)", "Potássio (K)"],
            "Valor no laudo": [
                f"{dados['n_g_kg']} g/kg",
                f"{dados['p_mg_dm3']} mg/dm³",
                f"{dados['k_mg_dm3']} mg/dm³",
            ],
            "Valor após conversão": [
                f"{dados['n_modelo']} g/kg",
                f"{dados['p_modelo']} kg/ha",
                f"{dados['k_modelo']} kg/ha",
            ],
        }
    )
    st.dataframe(dados_resumo, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Atenção:</strong> a recomendação apresentada tem caráter orientativo.
            Consulte sempre um agrônomo para validação técnica considerando as características
            específicas da área, da cultura e do manejo adotado.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Interface principal
# ---------------------------------------------------------------------------
st.title("🌱 Recomendação de Adubação e Calagem")

st.markdown(
    """
    <div class="info-box">
        <strong>Como utilizar:</strong> insira os valores obtidos no laudo de análise de solo,
        selecione o tipo de solo, a cultura e a expectativa de rendimento. O sistema calculará
        a recomendação final em <strong>kg/ha</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)

# Container reservado para garantir a ordem de renderização do resultado
resultado_container = st.container()

st.header("Dados do Laudo de Análise de Solo")

with st.form(key="form_recomendacao", clear_on_submit=False):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Macronutrientes do Solo")

        n_g_kg = st.slider(
            "Nitrogênio (N)",
            min_value=0.0,
            max_value=5.0,
            value=1.2,
            step=0.1,
            format="%.1f g/kg",
            help="Valor de nitrogênio (N) conforme informado no laudo de análise de solo, em g/kg.",
        )

        p_mg_dm3 = st.slider(
            "Fósforo (P)",
            min_value=0.0,
            max_value=60.0,
            value=10.0,
            step=0.5,
            format="%.1f mg/dm³",
            help="Valor de fósforo (P) conforme informado no laudo de análise de solo, em mg/dm³.",
        )

        k_mg_dm3 = st.slider(
            "Potássio (K)",
            min_value=0.0,
            max_value=400.0,
            value=80.0,
            step=1.0,
            format="%.1f mg/dm³",
            help="Valor de potássio (K) conforme informado no laudo de análise de solo, em mg/dm³.",
        )

    with col2:
        st.subheader("Cultura, Solo e Rendimento")

        tipo_solo_pt = st.selectbox(
            "Tipo de Solo",
            options=list(SOIL_EN_TO_PT.values()),
            index=3,  # "Franco"
            help="Selecione o tipo de solo da área avaliada.",
        )

        cultura_pt = st.selectbox(
            "Cultura",
            options=list(CROP_EN_TO_PT.values()),
            help="Selecione a cultura para a qual deseja obter a recomendação.",
        )

        expectativa_rendimento = st.slider(
            "Expectativa de rendimento",
            min_value=1.0,
            max_value=12.0,
            value=4.0,
            step=0.5,
            format="%.1f t/ha",
            help="Rendimento esperado da cultura, em toneladas por hectare (t/ha).",
        )

    st.markdown("---")
    submitted = st.form_submit_button("Gerar Recomendação", type="primary")

    if submitted:
        # Conversão PT -> EN para uso interno no modelo
        tipo_solo_en = SOIL_PT_TO_EN[tipo_solo_pt]
        cultura_en = CROP_PT_TO_EN[cultura_pt]

        # Conversão interna dos valores do laudo para o modelo
        n_modelo, p_modelo, k_modelo = converter_entrada_para_modelo(
            n_g_kg, p_mg_dm3, k_mg_dm3
        )

        # Chamada do modelo com valores convertidos
        recomendacao = modelo_recomendacao(
            n_modelo, p_modelo, k_modelo, cultura_en, expectativa_rendimento, tipo_solo_en
        )

        # Persiste os dados no session_state para manter o resultado
        st.session_state["gerar_recomendacao"] = True
        st.session_state["dados_entrada"] = {
            "n_g_kg": n_g_kg,
            "p_mg_dm3": p_mg_dm3,
            "k_mg_dm3": k_mg_dm3,
            "tipo_solo_pt": tipo_solo_pt,
            "tipo_solo_en": tipo_solo_en,
            "cultura_pt": cultura_pt,
            "cultura_en": cultura_en,
            "expectativa_rendimento": expectativa_rendimento,
            "n_modelo": n_modelo,
            "p_modelo": p_modelo,
            "k_modelo": k_modelo,
        }
        st.session_state["recomendacao"] = recomendacao

        # Exibe a recomendação imediatamente após o clique, dentro do container reservado
        with resultado_container:
            exibir_recomendacao(st.session_state["dados_entrada"], recomendacao)

# Mantém a recomendação visível em reexecuções subsequentes (quando não houve novo submit)
if not submitted and st.session_state.get("gerar_recomendacao"):
    with resultado_container:
        exibir_recomendacao(
            st.session_state["dados_entrada"], st.session_state["recomendacao"]
        )