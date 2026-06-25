import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

SOIL_EN_TO_PT = {
    'Black': 'Preto',
    'Clayey': 'Argiloso',
    'Loamy': 'Franco',
    'Red': 'Vermelho',
    'Sandy': 'Arenoso'
}

CROP_EN_TO_PT = {
    'Barley': 'Cevada',
    'Cotton': 'Algodão',
    'Ground Nuts': 'Amendoim',
    'Maize': 'Milho',
    'Millets': 'Milheto',
    'Oil seeds': 'Oleaginosas',
    'Paddy': 'Arroz',
    'Pulses': 'Leguminosas',
    'Sugarcane': 'Cana-de-açúcar',
    'Tobacco': 'Tabaco',
    'Wheat': 'Trigo'
}

SOIL_PT_TO_EN = {v: k for k, v in SOIL_EN_TO_PT.items()}
CROP_PT_TO_EN = {v: k for k, v in CROP_EN_TO_PT.items()}

NUMERIC_FEATURES = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
CATEGORICAL_FEATURES = ['Soil Type', 'Crop Type']
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['Fertilizer']


def apply_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #1b5e20 0%, #4caf50 100%);
            color: white;
            padding: 1.5rem 1rem;
            border-radius: 18px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(76, 175, 80, 0.35);
            margin-bottom: 2rem;
        }

        .main-header h1 {
            margin: 0;
            font-weight: 700;
            font-size: 2.2rem;
        }

        .main-header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.95;
            font-size: 1.05rem;
        }

        .prediction-card {
            background: linear-gradient(135deg, #ffffff 0%, #e8f5e9 100%);
            border-left: 6px solid #2e7d32;
            border-radius: 18px;
            padding: 1.8rem;
            box-shadow: 0 6px 28px rgba(0, 0, 0, 0.10);
            margin: 1.2rem 0;
        }

        .prediction-card h2 {
            color: #1b5e20;
            margin: 0.6rem 0 0 0;
            font-size: 2rem;
        }

        .metric-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 1.2rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
            text-align: center;
            border: 1px solid #e0e0e0;
            transition: transform 0.2s ease;
        }

        .metric-card:hover {
            transform: translateY(-3px);
        }

        .metric-card h3 {
            margin: 0;
            font-size: 1.8rem;
            color: #2e7d32;
        }

        .metric-card p {
            margin: 0.4rem 0 0 0;
            color: #555;
            font-weight: 600;
        }

        .info-box {
            background: #f1f8e9;
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #dcedc8;
            margin-bottom: 1rem;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%);
            color: white;
            border-radius: 14px;
            border: none;
            padding: 0.7rem 1.8rem;
            font-weight: 700;
            font-size: 1.05rem;
            box-shadow: 0 6px 18px rgba(46, 125, 50, 0.35);
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #1b5e20 0%, #4caf50 100%);
            box-shadow: 0 8px 24px rgba(46, 125, 50, 0.45);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #f1f8e9;
            border-radius: 14px 14px 0 0;
            padding: 12px 24px;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: #e8f5e9 !important;
            color: #1b5e20 !important;
        }

        .sidebar-title {
            font-weight: 700;
            font-size: 1.1rem;
            color: #1b5e20;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def build_dataset(n=2000):
    np.random.seed(42)
    soils = list(SOIL_EN_TO_PT.keys())
    crops = list(CROP_EN_TO_PT.keys())

    soil = np.random.choice(soils, size=n)
    crop = np.random.choice(crops, size=n)
    N = np.random.randint(0, 140, size=n)
    P = np.random.randint(5, 145, size=n)
    K = np.random.randint(5, 205, size=n)
    temperature = np.random.uniform(8.0, 43.0, size=n)
    humidity = np.random.uniform(14.0, 99.0, size=n)
    ph = np.random.uniform(3.5, 9.9, size=n)
    rainfall = np.random.uniform(20.0, 300.0, size=n)

    fertilizer = []
    for c, s, ni, pi, ki in zip(crop, soil, N, P, K):
        if c in ['Paddy', 'Sugarcane']:
            fertilizer.append('Ureia')
        elif s == 'Black':
            fertilizer.append('DAP')
        elif ni < 40:
            fertilizer.append('NPK 28-28-0')
        elif pi < 40:
            fertilizer.append('NPK 14-35-14')
        elif ki < 40:
            fertilizer.append('NPK 10-26-26')
        else:
            fertilizer.append('NPK 20-20-0')

    df = pd.DataFrame({
        'N': N,
        'P': P,
        'K': K,
        'Temperature': temperature,
        'Humidity': humidity,
        'pH': ph,
        'Rainfall': rainfall,
        'Soil Type': soil,
        'Crop Type': crop,
        'Fertilizer': fertilizer
    })
    return df


def build_model(df):
    X = df.drop('Fertilizer', axis=1)
    y = df['Fertilizer']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])

    model = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X, y)
    return model


def initialize_session_state():
    if 'original_df' not in st.session_state:
        st.session_state['original_df'] = build_dataset()
    if 'current_df' not in st.session_state:
        st.session_state['current_df'] = st.session_state['original_df'].copy()
    if 'current_model' not in st.session_state:
        st.session_state['current_model'] = build_model(st.session_state['current_df'])
    if 'uploaded_data_hash' not in st.session_state:
        st.session_state['uploaded_data_hash'] = None


def get_template_csv():
    template = build_dataset(n=5)
    return template.to_csv(index=False).encode('utf-8')


def main():
    apply_custom_css()
    initialize_session_state()

    st.markdown("""
    <div class="main-header">
        <h1>🌱 Recomendador Inteligente de Fertilizantes</h1>
        <p>Obtenha recomendações precisas baseadas em Machine Learning 🚀</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<div class='sidebar-title'>⚙️ Entrada do usuário</div>", unsafe_allow_html=True)

    soil_pt = st.sidebar.selectbox('🌍 Tipo de solo', list(SOIL_EN_TO_PT.values()))
    crop_pt = st.sidebar.selectbox('🌾 Tipo de cultura', list(CROP_EN_TO_PT.values()))

    soil_en = SOIL_PT_TO_EN[soil_pt]
    crop_en = CROP_PT_TO_EN[crop_pt]

    N = st.sidebar.slider('🧪 Nitrogênio (N)', 0, 140, 40)
    P = st.sidebar.slider('🧪 Fósforo (P)', 5, 145, 50)
    K = st.sidebar.slider('🧪 Potássio (K)', 5, 205, 60)
    temperature = st.sidebar.slider('🌡️ Temperatura (°C)', 8.0, 43.0, 25.0)
    humidity = st.sidebar.slider('💧 Umidade (%)', 14.0, 99.0, 60.0)
    ph = st.sidebar.slider('⚖️ pH do solo', 3.5, 9.9, 6.5)
    rainfall = st.sidebar.slider('🌧️ Chuva (mm)', 20.0, 300.0, 120.0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("<div class='sidebar-title'>📁 Upload de Nova Base</div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div class='info-box'>Adicione um CSV com os mesmos campos para unir aos dados originais e retreinar o modelo automaticamente.</div>",
        unsafe_allow_html=True
    )

    uploaded_file = st.sidebar.file_uploader(
        "Selecione um arquivo CSV",
        type=['csv'],
        help="O CSV deve conter as colunas: N, P, K, Temperature, Humidity, pH, Rainfall, Soil Type, Crop Type, Fertilizer"
    )

    st.sidebar.download_button(
        label="⬇️ Baixar modelo de CSV",
        data=get_template_csv(),
        file_name='modelo_fertilizantes.csv',
        mime='text/csv'
    )

    if uploaded_file is not None:
        current_hash = hash(uploaded_file.getvalue())
        if current_hash != st.session_state['uploaded_data_hash']:
            try:
                new_df = pd.read_csv(uploaded_file)
                missing_cols = [c for c in REQUIRED_COLUMNS if c not in new_df.columns]
                if not missing_cols:
                    new_df = new_df[REQUIRED_COLUMNS].copy()
                    merged_df = pd.concat([st.session_state['original_df'], new_df], ignore_index=True)
                    st.session_state['current_df'] = merged_df
                    st.session_state['current_model'] = build_model(merged_df)
                    st.session_state['uploaded_data_hash'] = current_hash
                    st.sidebar.success(f"✅ Nova base integrada! Total de amostras: {len(merged_df):,}")
                else:
                    st.sidebar.error(f"❌ Colunas ausentes: {', '.join(missing_cols)}")
            except Exception as e:
                st.sidebar.error(f"❌ Erro ao processar arquivo: {e}")

    model = st.session_state['current_model']
    df = st.session_state['current_df']

    input_data = pd.DataFrame([{
        'N': N,
        'P': P,
        'K': K,
        'Temperature': temperature,
        'Humidity': humidity,
        'pH': ph,
        'Rainfall': rainfall,
        'Soil Type': soil_en,
        'Crop Type': crop_en
    }])

    tab1, tab2 = st.tabs(['🎯 Previsão', '📊 Análise Exploratória'])

    with tab1:
        st.subheader('🌿 Parâmetros da análise')

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("<div class='metric-card'><h3>🌍</h3><p>{}</p></div>".format(soil_pt), unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-card'><h3>🌾</h3><p>{}</p></div>".format(crop_pt), unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='metric-card'><h3>🌡️</h3><p>{:.1f} °C</p></div>".format(temperature), unsafe_allow_html=True)
        with m4:
            st.markdown("<div class='metric-card'><h3>💧</h3><p>{:.1f} %</p></div>".format(humidity), unsafe_allow_html=True)

        st.markdown("---")

        if st.button('🚀 Obter recomendação'):
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]

            st.markdown(f"""
            <div class="prediction-card">
                <h3 style='margin-top: 0;'>🎯 Recomendação final</h3>
                <p style='font-size: 1.1rem; margin: 0;'>O melhor fertilizante para o seu cultivo é:</p>
                <h2>{prediction}</h2>
            </div>
            """, unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                'Fertilizante': model.classes_,
                'Probabilidade (%)': np.round(probabilities * 100, 2)
            }).sort_values('Probabilidade (%)', ascending=True)

            st.markdown("#### 📈 Confiança do modelo")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=prob_df, x='Probabilidade (%)', y='Fertilizante', palette='viridis', ax=ax)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probabilidade (%)')
            ax.set_ylabel('Fertilizante')
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander('🔍 Ver dados enviados ao modelo'):
            st.dataframe(input_data)

    with tab2:
        st.subheader('📊 Análise Exploratória dos Dados')

        df_pt = df.copy()
        df_pt['Tipo de solo'] = df_pt['Soil Type'].map(SOIL_EN_TO_PT)
        df_pt['Tipo de cultura'] = df_pt['Crop Type'].map(CROP_EN_TO_PT)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='metric-card'><h3>🌾</h3><p>{:,} amostras</p></div>".format(len(df_pt)), unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='metric-card'><h3>🧪</h3><p>{} fertilizantes</p></div>".format(df_pt['Fertilizer'].nunique()), unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='metric-card'><h3>🌱</h3><p>{} culturas</p></div>".format(df_pt['Crop Type'].nunique()), unsafe_allow_html=True)

        st.markdown("### 🔍 Visualização dos dados")
        st.dataframe(df_pt.head(10))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏞️ Distribuição dos tipos de solo")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            order_soil = [SOIL_EN_TO_PT[k] for k in SOIL_EN_TO_PT.keys()]
            sns.countplot(data=df_pt, x='Tipo de solo', order=order_soil, palette='viridis', ax=ax)
            ax.set_xlabel('Tipo de solo')
            ax.set_ylabel('Contagem')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("#### 🌾 Distribuição dos tipos de cultura")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            order_crop = [CROP_EN_TO_PT[k] for k in CROP_EN_TO_PT.keys()]
            sns.countplot(data=df_pt, x='Tipo de cultura', order=order_crop, palette='magma', ax=ax)
            ax.set_xlabel('Tipo de cultura')
            ax.set_ylabel('Contagem')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("#### 🧪 Distribuição das recomendações de fertilizantes")
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.countplot(
            data=df_pt,
            y='Fertilizer',
            order=df_pt['Fertilizer'].value_counts().index,
            palette='viridis',
            ax=ax
        )
        ax.set_xlabel('Contagem')
        ax.set_ylabel('Fertilizante')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### 🔥 Correlação entre variáveis numéricas")
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            df_pt[NUMERIC_FEATURES].corr(),
            annot=True,
            cmap='viridis',
            fmt='.2f',
            linewidths=0.5,
            ax=ax
        )
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### ⭐ Importância das variáveis")
        try:
            feature_names = (
                NUMERIC_FEATURES +
                list(model.named_steps['preprocess'].named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES))
            )
            importances = model.named_steps['classifier'].feature_importances_
            imp_df = pd.DataFrame({
                'Variável': feature_names,
                'Importância': importances
            }).sort_values('Importância', ascending=True).tail(12)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=imp_df, x='Importância', y='Variável', palette='magma', ax=ax)
            ax.set_xlabel('Importância')
            ax.set_ylabel('Variável')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception:
            st.info("Não foi possível calcular a importância das variáveis.")


if __name__ == '__main__':
    main()