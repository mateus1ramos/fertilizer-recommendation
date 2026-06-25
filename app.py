import streamlit as st
import numpy as np
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="Recomendação de Adubação",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inicialização robusta do estado da aplicação
if "gerar_recomendacao" not in st.session_state:
    st.session_state["gerar_recomendacao"] = False
if "dados_entrada" not in st.session_state:
    st.session_state["dados_entrada"] = {}

# CSS customizado para design profissional
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


def converter_entrada_para_modelo(n_g_kg, p_mg_dm3, k_mg_dm3):
    """
    Converte os valores de entrada do laudo de análise de solo
    para as unidades esperadas pelo modelo de recomendação.

    Parâmetros:
    - n_g_kg: Nitrogênio em g/kg (conforme laudo)
    - p_mg_dm3: Fósforo em mg/dm³ (conforme laudo)
    - k_mg_dm3: Potássio em mg/dm³ (conforme laudo)

    Retorna:
    - n_modelo: Nitrogênio mantido em g/kg
    - p_modelo: Fósforo convertido multiplicando por 2 (kg/ha na camada 0-20cm)
    - k_modelo: Potássio convertido multiplicando por 2 (kg/ha na camada 0-20cm)
    """
    n_modelo = n_g_kg
    p_modelo = p_mg_dm3 * 2
    k_modelo = k_mg_dm3 * 2
    return n_modelo, p_modelo, k_modelo


def modelo_recomendacao(n, p, k, cultura, expectativa_rendimento):
    """
    Modelo simplificado de recomendação de adubação.
    Em aplicações reais, substituir por modelo treinado ou equações agronômicas.
    """
    # Fatores de resposta por cultura (exemplo didático)
    fatores = {
        "Soja": {"n": 0.08, "p": 0.12, "k": 0.18},
        "Milho": {"n": 0.15, "p": 0.10, "k": 0.14},
        "Trigo": {"n": 0.12, "p": 0.09, "k": 0.13},
        "Feijão": {"n": 0.06, "p": 0.11, "k": 0.16},
        "Café": {"n": 0.10, "p": 0.08, "k": 0.15},
    }

    f = fatores.get(cultura, {"n": 0.10, "p": 0.10, "k": 0.10})

    # Cálculo de necessidade considerando teor do solo e expectativa de rendimento
    necessidade_n = max(0, (expectativa_rendimento * f["n"] * 10) - (n * 10))
    necessidade_p = max(0, (expectativa_rendimento * f["p"] * 10) - p)
    necessidade_k = max(0, (expectativa_rendimento * f["k"] * 10) - k)

    return {
        "N": round(necessidade_n, 1),
        "P2O5": round(necessidade_p, 1),
        "K2O": round(necessidade_k, 1),
    }


# Título principal
st.title("🌱 Recomendação de Adubação e Calagem")

st.markdown(
    """
    <div class="info-box">
        <strong>Como utilizar:</strong> insira os valores obtidos no laudo de análise de solo e 
        selecione a cultura e a expectativa de rendimento. O sistema calculará a recomendação 
        final em <strong>kg/ha</strong>.
    </div>
    """,
    unsafe_allow_html=True,
)

# Abas da aplicação
aba_entrada, aba_recomendacao, aba_sobre = st.tabs(
    ["📊 Entrada de Dados", "✅ Recomendação", "ℹ️ Sobre"]
)

with aba_entrada:
    st.header("Dados do Laudo de Análise de Solo")

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
            key="n_g_kg",
        )

        p_mg_dm3 = st.slider(
            "Fósforo (P)",
            min_value=0.0,
            max_value=60.0,
            value=10.0,
            step=0.5,
            format="%.1f mg/dm³",
            help="Valor de fósforo (P) conforme informado no laudo de análise de solo, em mg/dm³.",
            key="p_mg_dm3",
        )

        k_mg_dm3 = st.slider(
            "Potássio (K)",
            min_value=0.0,
            max_value=400.0,
            value=80.0,
            step=1.0,
            format="%.1f mg/dm³",
            help="Valor de potássio (K) conforme informado no laudo de análise de solo, em mg/dm³.",
            key="k_mg_dm3",
        )

    with col2:
        st.subheader("Cultura e Rendimento")

        cultura = st.selectbox(
            "Cultura",
            options=["Soja", "Milho", "Trigo", "Feijão", "Café"],
            help="Selecione a cultura para a qual deseja obter a recomendação.",
            key="cultura",
        )

        expectativa_rendimento = st.slider(
            "Expectativa de rendimento",
            min_value=1.0,
            max_value=12.0,
            value=4.0,
            step=0.5,
            format="%.1f t/ha",
            help="Rendimento esperado da cultura, em toneladas por hectare (t/ha).",
            key="expectativa_rendimento",
        )

    st.markdown("---")

    # Botão que ativa a geração da recomendação e persiste os dados de entrada
    if st.button("Gerar Recomendação", type="primary"):
        st.session_state["gerar_recomendacao"] = True
        st.session_state["dados_entrada"] = {
            "n_g_kg": n_g_kg,
            "p_mg_dm3": p_mg_dm3,
            "k_mg_dm3": k_mg_dm3,
            "cultura": cultura,
            "expectativa_rendimento": expectativa_rendimento,
        }

with aba_recomendacao:
    st.header("Recomendação de Nutrientes")

    if st.session_state.get("gerar_recomendacao", False):
        dados = st.session_state.get("dados_entrada", {})

        if not dados:
            st.warning(
                "Os dados de entrada não foram encontrados. "
                "Volte à aba **📊 Entrada de Dados** e clique em **Gerar Recomendação**.",
                icon="⚠️",
            )
        else:
            # Recupera as variáveis salvas no session_state para garantir acesso na aba atual
            n_g_kg = dados.get("n_g_kg", 0.0)
            p_mg_dm3 = dados.get("p_mg_dm3", 0.0)
            k_mg_dm3 = dados.get("k_mg_dm3", 0.0)
            cultura = dados.get("cultura", "Soja")
            expectativa_rendimento = dados.get("expectativa_rendimento", 0.0)

            # Conversão interna dos valores do laudo para o modelo
            n_modelo, p_modelo, k_modelo = converter_entrada_para_modelo(
                n_g_kg, p_mg_dm3, k_mg_dm3
            )

            recomendacao = modelo_recomendacao(
                n_modelo, p_modelo, k_modelo, cultura, expectativa_rendimento
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <h3>🧪 Recomendação Final para {cultura}</h3>
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
                    "Valor no laudo": [f"{n_g_kg} g/kg", f"{p_mg_dm3} mg/dm³", f"{k_mg_dm3} mg/dm³"],
                    "Valor após conversão": [f"{n_modelo} g/kg", f"{p_modelo} kg/ha", f"{k_modelo} kg/ha"],
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
    else:
        st.info(
            "Vá até a aba **📊 Entrada de Dados** e clique em **Gerar Recomendação** para visualizar os resultados em kg/ha."
        )

with aba_sobre:
    st.header("Sobre a Aplicação")

    st.markdown(
        """
        Esta aplicação foi desenvolvida para auxiliar na recomendação de adubação baseada em 
        laudos de análise de solo.

        ### Unidades de medida utilizadas
        - **Nitrogênio (N):** g/kg (gramas por quilograma) — unidade comum em laudos de solo.
        - **Fósforo (P):** mg/dm³ (miligramas por decímetro cúbico) — unidade comum em laudos de solo.
        - **Potássio (K):** mg/dm³ (miligramas por decímetro cúbico) — unidade comum em laudos de solo.

        ### Conversão interna
        Os valores de fósforo e potássio informados em mg/dm³ são multiplicados por 2 antes de 
        serem processados pelo modelo. Essa é a conversão padrão para kg/ha na camada de solo 
        de 0–20 cm.

        ### Recomendação final
        Todos os valores de adubação sugeridos são exibidos em **kg/ha** (quilogramas por hectare), 
        unidade padrão para aplicação de fertilizantes em lavouras.
        """
    )