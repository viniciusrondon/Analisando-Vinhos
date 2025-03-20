# imports

from contextlib import redirect_stdout
import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix

# Load Data
from ucimlrepo import fetch_ucirepo
# fetch dataset
wine_quality = fetch_ucirepo(id=186)
Z = wine_quality.data.original

sample_red = Z[Z["color"] == "red"]
sample_white = Z[Z["color"] == "white"]

X_red = sample_red.iloc[:, :-3].values
y_red = sample_red["quality"].values

X_white = sample_white.iloc[:, :-3].values
y_white = sample_white["quality"].values

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_white, y_white, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42)

# tinto:  X_train, y_train, X_valid, y_valid, X_test, y_test


Xw_train_full, Xw_test, yw_train_full, yw_test = train_test_split(
    X_white, y_white, test_size=0.1, random_state=42)
Xw_train, Xw_valid, yw_train, yw_valid = train_test_split(
    Xw_train_full, yw_train_full, test_size=0.1, random_state=42)
# Branco: Xw_train, yw_train, Xw_valid, y_valid, Xw_test, yw_test


# Funcoes


def plot_three_figures(Z):

    col1, col2, col3 = st.columns(3)

    # Figura 1: Heatmap da correlação das colunas de Z, exceto a última
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    sns.heatmap(Z.iloc[:, :-1].corr(), cmap='coolwarm', annot=True, ax=ax1)
    ax1.set_title("Heatmap Quality")
    col1.pyplot(fig1)
    plt.close(fig1)

    # Figura 2: Heatmap de W, onde a coluna "color" foi mapeada para valores numéricos
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    W = Z.copy()
    W["color"] = W["color"].map({"red": 0, "white": 1})
    sns.heatmap(W.corr(), cmap='coolwarm', annot=True, ax=ax2)
    ax2.set_title("Heatmap color")
    col2.pyplot(fig2)
    plt.close(fig2)

    # Figura 3: Gráfico de barras horizontal a partir de corr_sort (exceto a última linha)
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    corr_color = W.corr()
    corr_abs = corr_color["color"].abs()
    corr_sort = corr_abs.sort_values(ascending=True)
    corr_sort.iloc[:-1].plot(kind="barh", ax=ax3)
    ax3.set_title("Correlação de características x cor")
    col3.pyplot(fig3)
    plt.close(fig3)


def plot_jointplots(Z):
    # Cria duas colunas no layout do Streamlit
    col1, col2 = st.columns(2)

    # Figura 1: Jointplot com KDE e hue
    g1 = sns.jointplot(
        data=Z,
        x='total_sulfur_dioxide',
        y='volatile_acidity',
        kind='kde',
        hue='color'
    )
    fig1 = g1.fig  # Captura o objeto Figure do jointplot
    fig1.suptitle("KDE Jointplot", y=1.05)

    # Exibe a primeira figura na primeira coluna
    with col1:
        st.pyplot(fig1)
    plt.close(fig1)

    # Figura 2: Jointplot com scatter e sobreposição de KDE
    g2 = sns.jointplot(
        data=Z,
        x='total_sulfur_dioxide',
        y='volatile_acidity',
        kind='scatter',
        hue='color',
        alpha=0.2
    )
    # Sobrepõe o KDE no jointplot existente
    sns.kdeplot(
        data=Z,
        x='total_sulfur_dioxide',
        y='volatile_acidity',
        hue='color',
        ax=g2.ax_joint,
        alpha=1,
        linewidths=1
    )
    fig2 = g2.fig  # Captura o objeto Figure do jointplot
    fig2.suptitle("Scatter Jointplot com KDE", y=1.05)

    # Exibe a segunda figura na segunda coluna
    with col2:
        st.pyplot(fig2)
    plt.close(fig2)


def plot_barplot_table():
    """
    Transforma a tabela em um DataFrame (ignorando as linhas "total" e "proporção")
    e gera um barplot das notas (9 a 3) para os tipos 'tinto' e 'branco'.
    Retorna o objeto Figure do matplotlib.
    """
    # Dados referentes às notas (ignorando as linhas "total" e "proporção")
    data = {
        "rating": [9, 8, 7, 6, 5, 4, 3],
        "tinto": [0, 0.011, 0.124, 0.399, 0.426, 0.033, 0.006],
        "branco": [0.001, 0.036, 0.180, 0.449, 0.297, 0.033, 0.004]
    }

    # Cria o DataFrame
    df = pd.DataFrame(data)
    df["rating"] = df["rating"].astype(int)

    # Transforma o DataFrame para o formato longo para facilitar a plotagem
    df_melted = df.melt(id_vars="rating", value_vars=["tinto", "branco"],
                        var_name="tipo", value_name="valor")

    # Cria a figura e o eixo para o barplot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df_melted, x="rating", y="valor", hue="tipo", ax=ax)

    ax.set_title("Distribuição de Valores por Nota e Tipo de Vinho")
    ax.set_xlabel("Nota")
    ax.set_ylabel("Valor")
    plt.tight_layout()

    return fig


def display_regression_metrics(y_test, y_pred):
    """
    Calcula as métricas MAE, MSE, RMSE e R² a partir dos valores reais (y_test)
    e preditos (y_pred) e as exibe em quatro colunas usando st.metric().
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Cria quatro colunas para exibir as métricas lado a lado
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:.3f}")
    col1.metric("MSE", f"{mse:.3f}")
    col2.metric("RMSE", f"{rmse:.3f}")
    col2.metric("R²", f"{r2:.3f}")


def display_side_by_side_plots(y_test, y_pred, history):
    """
    Cria e exibe dois gráficos lado a lado:
    1. Scatter plot comparando os valores reais e previstos.
    2. Gráfico das métricas de treinamento ao longo das epochs.

    Parâmetros:
    - y_test: Valores reais.
    - y_pred: Valores previstos pelo modelo.
    - history: Objeto de histórico do treinamento, com atributos:
        - history.history: dicionário contendo as métricas por epoch.
        - history.epoch: lista ou array com os números das epochs.
    """
    # Figura 1: Scatter plot com linha de identidade
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(y_test, y_pred, alpha=0.7)
    ax1.set_xlabel("Valor Real")
    ax1.set_ylabel("Valor Previsto")
    ax1.set_title("Comparação entre valores reais e previstos")
    # Linha de identidade
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

    # Figura 2: Plot do histórico de treinamento
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    style_list = ["r--", "r--.", "b-", "b-*"]

    for key, style in zip(history.history, style_list):
        # Ajusta as epochs: se a métrica não for de validação, desloca em -0.5
        epochs = np.array(history.epoch) + \
            (0 if key.startswith("val_") else -0.5)
        ax2.plot(epochs, history.history[key], style, label=key)

    ax2.set_xlabel("Epoch")
    ax2.set_xlim(-0.5, 29)
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right")
    ax2.grid(True)

    # Exibe os gráficos lado a lado em duas colunas
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)

    # Fecha as figuras para liberar memória
    plt.close(fig1)
    plt.close(fig2)


def compute_accuracy_tolerance(y_true, y_pred, tolerance=1):

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    error = np.abs(y_true - y_pred)
    correct = np.sum(error <= tolerance)
    accuracy = correct / len(y_true)
    return accuracy


def display_confusion_matrix(y_test, y_pred, sample):
    labels = sorted(sample.unique())
    # Calcula a matriz de confusão
    cm = confusion_matrix(y_test, np.round(y_pred).astype(int), labels=labels)

    # Cria o plot da matriz de confusão
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Valor Previsto")
    ax.set_ylabel("Valor Real")
    ax.set_title("Matriz de Confusão")
    ax.invert_yaxis()
    st.pyplot(fig)


# Heading
st.write("# Analisando Vinho")

st.divider()
st.write("## Análise Exploratória dos Dados\n")
st.write("### Dataset")
st.dataframe(Z)

st.write("### Método `describe()`")

st.dataframe(Z.describe())

st.write('''
- **X (Features):** 11 características dos vinhos com número equilibrado de amostras.
- **y (Target):** Notas variam de 3 a 9, com poucos extremos (poucas abaixo de 5 e acima de 7).
- **Contagem e Desvio:** Consistência na coleta e baixa dispersão, indicando dados homogêneos.
- **Distribuição:** Algumas variáveis apresentam outliers, com valores máximos distantes do 75º percentil.
- **Dimensões e Tipos:** 6497 amostras por 11 características, todas em formato float64.
- **Integridade:** Dados completos, sem valores ausentes ou colunas vazias.
''')

st.write("### Análise de variáveis\n")

plot_three_figures(Z)

st.write('''**O gráfico indica que as características com maior correlação com o tipo de vinho são:**

- `total_sulfur_dioxide`: 0.700357  
- `volatile_acidity`: -0.653036  
- `chlorides`: -0.512678
''')

plot_jointplots(Z)

st.write("### Avaliacão")
vinhos_bem = Z[Z["quality"] > 7]
vinhos_mal = Z[Z["quality"] <= 3]


prevalencia_bem = vinhos_bem["color"].value_counts(normalize=True)
st.write("Prevalência em vinhos bem avaliados:\n", prevalencia_bem)

prevalencia_mal = vinhos_mal["color"].value_counts(normalize=True)
st.write("Prevalência em vinhos mal avaliados:\n", prevalencia_mal)

st.write(f'''total de amostras:{Z.shape[0]}.\n
      total de amostras vinho tinto: {sample_red.shape[0]}, {round(sample_red.shape[0] * 100/Z.shape[0], 2)}%\n
      total de amostras vinho branco: {sample_white.shape[0]}, {round(sample_white.shape[0] * 100/Z.shape[0], 2)}%''')

st.pyplot(plot_barplot_table())

st.write('''

As amostras de vinho tinto correspondem a 24,61%, enquanto as de vinho branco correspondem a 75,38%. Esse desbalanceamento dificulta a avaliação para saber se, de fato, os vinhos brancos apresentam qualidade superior, considerando que recebem notas maiores em comparação aos tintos.

**Análise de Proporcionalidade**

- **Bem avaliados (nota ≥ 7):**  
  - Tinto: 13,5%  
  - Branco: 21,7%

- **Mal avaliados (nota ≤ 4):**  
  - Tinto: 3,9%  
  - Branco: 3,7%

- **Avaliação média:**  
  - Tinto: 82,5%  
  - Branco: 74,6%

Para obter uma avaliação mais assertiva — tanto para identificar as características mais relevantes para classificar um vinho quanto para a predição de qualidade — é fundamental que os dados estejam melhor estratificados, com um número maior de amostras de vinho tinto e um balanceamento aprimorado das notas de qualidade.

''')

st.write("## Treinando modelo MLP")

st.divider()


col1, col2 = st.columns(2)

with col1:
    st.write("## Vinho Tinto")
    tf.random.set_seed(42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(5, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    history = model.fit(X_train, y_train, epochs=30,
                        validation_data=(X_valid, y_valid))

    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    summary_str = stream.getvalue()

    st.code(summary_str)

    y_pred = model.predict(X_test)

    display_regression_metrics(y_test, y_pred)

    # Calcula e exibe a acurácia baseada em tolerância
    tolerance_value = 1  # ajuste a tolerância conforme desejado
    accuracy_tol = compute_accuracy_tolerance(
        y_test, y_pred, tolerance=tolerance_value)
    st.write(
        f"Acurácia com tolerância de {tolerance_value}: {accuracy_tol*100:.2f}%")

    display_side_by_side_plots(y_test, y_pred, history)
    display_confusion_matrix(y_test, y_pred, sample_red["quality"])

with col2:
    st.write("## Vinho Branco")
    tf.random.set_seed(42)

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(5, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    optimizer2 = tf.keras.optimizers.SGD(learning_rate=1e-1)
    model2.compile(loss="mse", optimizer=optimizer2, metrics=["mae"])

    history2 = model2.fit(Xw_train, yw_train, epochs=20,
                          validation_data=(Xw_valid, yw_valid))

    stream = io.StringIO()
    with redirect_stdout(stream):
        model2.summary()
    summary_str = stream.getvalue()

    st.code(summary_str)

    yw_pred = model2.predict(Xw_test)

    display_regression_metrics(yw_test, yw_pred)

    # Calcula e exibe a acurácia baseada em tolerância
    tolerance_value2 = 1  # ajuste a tolerância conforme desejado
    accuracy_tol2 = compute_accuracy_tolerance(
        yw_test, yw_pred, tolerance=tolerance_value2)
    st.write(
        f"Acurácia com tolerância de {tolerance_value2}: {accuracy_tol2*100:.2f}%")

    display_side_by_side_plots(yw_test, yw_pred, history2)
    display_confusion_matrix(yw_test, yw_pred, sample_white["quality"])
