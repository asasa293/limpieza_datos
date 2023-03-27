import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, roc_auc_score, mean_absolute_error
import statsmodels.api as sm
from fbprophet.plot import plot_plotly as go
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.express as px

# Load data
# Load monedas.csv


st.set_page_config(page_title="Tasas de Cambio",
                   page_icon=":bar_chart:",
                   layout="wide",
                   initial_sidebar_state="collapsed")

page = st.selectbox("Selecciona un modelo:", ["Multiples Monedas", "Series de Tiempo"])

if page == "Multiples Monedas":
    monedas = pd.read_csv('TP4/datos/monedas.csv', index_col=0)
    year = st.sidebar.multiselect('Selecciona un año:',
                                  options=monedas['year'].unique(),
                                  default=[])

    if not year:
        year = monedas['year'].unique()

    seleccion = monedas.query("year in @year")
    seleccion = seleccion.drop(['year'], axis=1)
    returns = np.log(seleccion / seleccion.shift(1))
    returns = returns.iloc[1:]

    X = returns.drop(['EUR_USD'], axis=1)
    y = returns['EUR_USD']

    # Load pickles
    with open('TP4/datos/regressor.pkl', 'rb') as f:
        regressor = pickle.load(f)

    with open('TP4/datos/grid_search.pkl', 'rb') as f:
        grid_search = pickle.load(f)

    # Predict
    # Regressor
    y_pred = regressor.predict(X)
    y_pred = pd.DataFrame(y_pred, index=y.index, columns=['pred'])
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred, squared=False)

    # Random Forest Regressor
    X = sm.add_constant(X)
    y2_pred = grid_search.predict(X)
    r2_rf = r2_score(y, y2_pred)
    mse_rf = mean_absolute_error(y, y2_pred)



    # --- MAIN PAGE ---
    st.title(":bar_chart: Tasas de Cambio")
    st.subheader("Predicción de los retornos de las divisas EUR/USD utilizando modelos de regresión lineal y random "
                 "forest con base en los retornos de los otros pares. Los datos fueron obtenidos de OANDA.")

    st.markdown("---")

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Regresión:")
        st.metric(label="El R2 para el modelo es:",
                  value=f"{r2:.4f}")
        st.metric(label="El MSE para el modelo es:",
                  value=f"{mse:.8f}")

    with right_column:
        st.subheader("Random Forest Regressor:")
        st.metric(label="El R2 para el modelo de random forest es:",
                  value=f"{r2_rf:.4f}")
        st.metric(label="El MSE para el modelo de random forest es:",
                  value=f"{mse_rf:.8f}")

    st.markdown("---")

    left_column, right_column = st.columns(2)
    with left_column:
        st.title("Retornos")
        st.dataframe(returns)
    with right_column:
        # make correlation matrix and plot it in streamlit
        corr = returns.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.title.set_text('Correlation Matrix')
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)



    # df = data[['open_bid', 'time']]
    # df['time'] = pd.to_datetime(df['time'])
    # df.index = df.time
    # df_ARIMA = df.drop(['time'], axis = 1)
    # exchange_rate = df_ARIMA.resample('M').agg({'open_bid':'mean'})
else:
    pair = st.sidebar.selectbox('Selecciona las divisas:', ("EUR_USD", "EUR_GBP", "GBP_USD",
                                                            "AUD_USD", "USD_JPY", "USD_CAD",
                                                            "USD_CHF"))
    # Prepare data
    data = pd.read_csv(f'TP4/datos/{pair}_H1.csv')
    data = data[['open_bid', 'time']]
    data['time'] = pd.to_datetime(data['time'])
    data.index = data.time
    data = data.drop(['time'], axis=1)
    data = data.dropna()


    # --- MAIN PAGE ---
    st.title(":chart_with_upwards_trend: Serie de Tiempo")
    st.subheader(f"Predicción de la variación de la relación del precio de apertura de las monedas "
                 f"{pair[0:3]}/{pair[4:7]} haciendo"
                 f"uso de series de tiempo (forecasting). Los datos fueron obtenidos de OANDA.")

    st.markdown("---")

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader(f"{pair[0:3]}/{pair[4:7]}")
        st.dataframe(data)
    data = data.reset_index()

    with right_column:
        plt.style.use("dark_background")
        # display histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.title.set_text('Histograma')
        sns.histplot(data['open_bid'], ax=ax, palette='pastel')
        # Hide grid lines
        ax.grid(False)
        st.pyplot(fig)

    # display line plot with plotly dark theme
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.title.set_text('Serie de Tiempo')
    sns.lineplot(data=data, x='time', y='open_bid', ax=ax, palette='pastel')
    # Hide grid lines
    ax.grid(False)
    st.pyplot(fig)


