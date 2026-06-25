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


@st.cache_data
def build_dataset(n=2000):
    np.random.seed(42)
    soils = list(SOIL_EN_TO_PT.keys())
    crops = list(CROP_EN_TO_PT.keys())
    fertilizers = [
        'Ureia', 'DAP', 'NPK 28-28-0', 'NPK 14-35-14',
        'NPK 20-20-0', 'NPK 17-17-17', 'NPK 10-26-26'
    ]

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
    for s, c, ni, pi, ki in zip(soil, crop, N, P, K):
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

    numeric_features = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    categorical_features = ['Soil Type', 'Crop Type']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    model = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X, y)
    return model


@st.cache_resource
def get_model_and_data():
    df = build_dataset()
    model = build_model(df)
    return model, df


def main():
    st.title('Recomendador de Fertilizantes')
    st.sidebar.header('Entrada do usuário')

    soil_pt = st.sidebar.selectbox('Tipo de solo', list(SOIL_EN_TO_PT.values()))
    crop_pt = st.sidebar.selectbox('Tipo de cultura', list(CROP_EN_TO_PT.values()))

    soil_en = SOIL_PT_TO_EN[soil_pt]
    crop_en = CROP_PT_TO_EN[crop_pt]

    N = st.sidebar.slider('Nitrogênio (N)', 0, 140, 40)
    P = st.sidebar.slider('Fósforo (P)', 5, 145, 50)
    K = st.sidebar.slider('Potássio (K)', 5, 205, 60)
    temperature = st.sidebar.slider('Temperatura (°C)', 8.0, 43.0, 25.0)
    humidity = st.sidebar.slider('Umidade (%)', 14.0, 99.0, 60.0)
    ph = st.sidebar.slider('pH do solo', 3.5, 9.9, 6.5)
    rainfall = st.sidebar.slider('Chuva (mm)', 20.0, 300.0, 120.0)

    model, df = get_model_and_data()

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

    tab1, tab2 = st.tabs(['Previsão', 'Análise Exploratória'])

    with tab1:
        st.subheader('Recomendação')
        st.markdown(f'**Tipo de solo selecionado:** {soil_pt}')
        st.markdown(f'**Tipo de cultura selecionada:** {crop_pt}')

        if st.button('Obter recomendação'):
            prediction = model.predict(input_data)[0]
            st.success(f'Fertilizante recomendado: {prediction}')

        with st.expander('Ver dados enviados ao modelo (em inglês)'):
            st.dataframe(input_data)

    with tab2:
        st.subheader('Análise exploratória dos dados')

        df_pt = df.copy()
        df_pt['Tipo de solo'] = df_pt['Soil Type'].map(SOIL_EN_TO_PT)
        df_pt['Tipo de cultura'] = df_pt['Crop Type'].map(CROP_EN_TO_PT)

        st.dataframe(df_pt.head())

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            order_soil = [SOIL_EN_TO_PT[k] for k in SOIL_EN_TO_PT.keys()]
            sns.countplot(data=df_pt, x='Tipo de solo', order=order_soil, ax=ax)
            ax.set_xlabel('Tipo de solo')
            ax.set_ylabel('Contagem')
            ax.set_title('Distribuição dos tipos de solo')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            order_crop = [CROP_EN_TO_PT[k] for k in CROP_EN_TO_PT.keys()]
            sns.countplot(data=df_pt, x='Tipo de cultura', order=order_crop, ax=ax)
            ax.set_xlabel('Tipo de cultura')
            ax.set_ylabel('Contagem')
            ax.set_title('Distribuição dos tipos de cultura')
            plt.xticks(rotation=45)
            st.pyplot(fig)


if __name__ == '__main__':
    main()