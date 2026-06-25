import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# Configuração da página
st.set_page_config(
    page_title="Recomendação de Fertilizantes",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos customizados
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 6px solid #2E7D32;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2E7D32;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATASET_PATH = "Fertilizer-Prediction.csv"


@st.cache_data(show_spinner=False)
def load_and_train_model(path: str):
    """Carrega o dataset e treina um modelo Random Forest."""
    df = pd.read_csv(path)

    # Normaliza nomes de colunas (remove espaços extras)
    df.columns = [c.strip() for c in df.columns]

    # Colunas esperadas
    expected_cols = [
        "Temparature",
        "Humidity",
        "Moisture",
        "Soil Type",
        "Crop Type",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
        "Fertilizer Name",
    ]

    # Verifica se as colunas esperadas existem (com tolerância a case/acentos)
    available_cols = [c for c in expected_cols if c in df.columns]
    if len(available_cols) < len(expected_cols):
        missing = [c for c in expected_cols if c not in df.columns]
        raise ValueError(f"Colunas ausentes no dataset: {missing}. Colunas encontradas: {list(df.columns)}")

    # Separa features e target
    target_col = "Fertilizer Name"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Codifica variáveis categóricas
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Codifica target
    target_le = LabelEncoder()
    y = target_le.fit_transform(y.astype(str))

    # Treina o modelo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return df, model, label_encoders, target_le, accuracy


# Cabeçalho
st.markdown('<div class="main-header">🌱 Recomendação de Fertilizantes</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Sistema inteligente baseado em Machine Learning para indicar o fertilizante ideal</div>',
    unsafe_allow_html=True,
)

# Verifica existência do dataset
if not os.path.exists(DATASET_PATH):
    st.error(f"⚠️ Arquivo não encontrado: `{DATASET_PATH}`. Por favor, coloque o dataset no mesmo diretório deste app.")
    st.stop()

# Carrega modelo e dados (com spinner para indicar processamento em segundo plano)
with st.spinner("Carregando dataset e treinando modelo Random Forest..."):
    try:
        df, model, label_encoders, target_le, accuracy = load_and_train_model(DATASET_PATH)
    except Exception as e:
        st.error(f"Erro ao processar o dataset: {e}")
        st.stop()

# Sidebar para entrada de dados
with st.sidebar:
    st.header("🧪 Dados de Entrada")
    st.markdown("Preencha as informações abaixo para obter a recomendação.")

    # Valores padrão baseados nas médias do dataset
    temperature = st.number_input(
        "🌡️ Temperatura (°C)",
        min_value=0.0,
        max_value=60.0,
        value=float(round(df["Temparature"].mean(), 1)),
        step=0.1,
    )
    humidity = st.number_input(
        "💧 Umidade (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(round(df["Humidity"].mean(), 1)),
        step=0.1,
    )
    moisture = st.number_input(
        "🌾 Umidade do Solo (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(round(df["Moisture"].mean(), 1)),
        step=0.1,
    )

    soil_options = sorted(df["Soil Type"].unique().tolist())
    crop_options = sorted(df["Crop Type"].unique().tolist())

    soil_type = st.selectbox("🪨 Tipo de Solo", soil_options)
    crop_type = st.selectbox("🌽 Tipo de Cultura", crop_options)

    nitrogen = st.slider("🧪 Nitrogênio (N)", 0, 50, int(df["Nitrogen"].median()))
    phosphorus = st.slider("🧪 Fósforo (P)", 0, 50, int(df["Phosphorous"].median()))
    potassium = st.slider("🧪 Potássio (K)", 0, 50, int(df["Potassium"].median()))

    predict_button = st.button("🔍 Recomendar Fertilizante", use_container_width=True)

# Layout principal em abas
tab1, tab2 = st.tabs(["🎯 Recomendação", "📊 Análise Exploratória"])

with tab1:
    st.subheader("Resultado da Predição")

    if predict_button:
        # Constrói DataFrame de entrada com as colunas na mesma ordem do treino
        input_data = pd.DataFrame({
            "Temparature": [temperature],
            "Humidity": [humidity],
            "Moisture": [moisture],
            "Soil Type": [soil_type],
            "Crop Type": [crop_type],
            "Nitrogen": [nitrogen],
            "Potassium": [potassium],
            "Phosphorous": [phosphorus],
        })

        # Aplica os mesmos encoders do treino
        for col, le in label_encoders.items():
            input_data[col] = le.transform(input_data[col].astype(str))

        # Predição e probabilidades
        prediction_encoded = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        prediction = target_le.inverse_transform([prediction_encoded])[0]
        confidence = float(probabilities[prediction_encoded]) * 100

        # Exibe resultado em destaque
        st.markdown(
            f"""
            <div class="prediction-box">
                <h3 style="margin-top: 0; color: #1B5E20;">✅ Fertilizante Recomendado</h3>
                <p style="font-size: 2rem; font-weight: 700; color: #2E7D32; margin: 0;">{prediction}</p>
                <p style="font-size: 1rem; color: #555; margin-top: 0.5rem;">
                    Confiança do modelo: <b>{confidence:.1f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probabilidades por classe
        prob_df = pd.DataFrame({
            "Fertilizante": target_le.inverse_transform(np.arange(len(probabilities))),
            "Probabilidade": probabilities * 100,
        }).sort_values("Probabilidade", ascending=False)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 📋 Resumo dos Dados Informados")
            summary = pd.DataFrame({
                "Parâmetro": [
                    "Temperatura", "Umidade", "Umidade do Solo", "Tipo de Solo",
                    "Tipo de Cultura", "Nitrogênio", "Fósforo", "Potássio"
                ],
                "Valor": [
                    f"{temperature} °C", f"{humidity}%", f"{moisture}%", soil_type,
                    crop_type, nitrogen, phosphorus, potassium
                ],
            })
            st.dataframe(summary, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("#### 📈 Probabilidades por Fertilizante")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=prob_df,
                y="Fertilizante",
                x="Probabilidade",
                palette="Greens_d",
                ax=ax,
            )
            ax.set_xlabel("Probabilidade (%)")
            ax.set_xlim(0, 100)
            ax.set_title("Confiança do Modelo por Classe")
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f%%", padding=3)
            st.pyplot(fig)
    else:
        st.info("👈 Preencha os dados na barra lateral e clique em **Recomendar Fertilizante** para ver o resultado.")

    # Métrica do modelo
    st.divider()
    st.markdown(
        f"<p class='metric-label'>Acurácia do modelo Random Forest no conjunto de teste:</p>"
        f"<p class='metric-value'>{accuracy:.2%}</p>",
        unsafe_allow_html=True,
    )

with tab2:
    st.subheader("Análise Exploratória")

    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total de registros", f"{len(df):,}")
    kpi2.metric("Tipos de solo", df["Soil Type"].nunique())
    kpi3.metric("Tipos de cultura", df["Crop Type"].nunique())
    kpi4.metric("Fertilizantes distintos", df["Fertilizer Name"].nunique())

    st.divider()

    # Gráficos de distribuição de nutrientes por tipo de solo
    st.markdown("#### Distribuição de Nutrientes por Tipo de Solo")

    nutrient_cols = ["Nitrogen", "Phosphorous", "Potassium"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, nutrient in zip(axes, nutrient_cols):
        sns.boxplot(
            data=df,
            x="Soil Type",
            y=nutrient,
            palette="Set2",
            ax=ax,
        )
        ax.set_title(f"{nutrient} por Tipo de Solo")
        ax.set_xlabel("Tipo de Solo")
        ax.set_ylabel(f"Nível de {nutrient}")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # Heatmap de correlação entre variáveis numéricas
    st.markdown("#### Correlação entre Variáveis Numéricas")
    numeric_df = df.select_dtypes(include=[np.number])
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.divider()

    # Distribuição de fertilizantes recomendados
    st.markdown("#### Distribuição de Fertilizantes no Dataset")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fertilizer_counts = df["Fertilizer Name"].value_counts().reset_index()
    fertilizer_counts.columns = ["Fertilizante", "Quantidade"]
    sns.barplot(
        data=fertilizer_counts,
        x="Quantidade",
        y="Fertilizante",
        palette="viridis",
        ax=ax3,
    )
    ax3.set_title("Frequência de cada Fertilizante")
    ax3.set_xlabel("Quantidade")
    for container in ax3.containers:
        ax3.bar_label(container, padding=3)
    st.pyplot(fig3)

    st.divider()

    # Preview do dataset
    st.markdown("#### Visualização dos Dados")
    st.dataframe(df.head(10), use_container_width=True)