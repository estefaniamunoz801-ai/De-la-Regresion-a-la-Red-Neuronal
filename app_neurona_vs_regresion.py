"""
App Streamlit: De la Regresión a la Red Neuronal (companion interactivo)
Curso: Machine Learning (ET0178)

Objetivo: reforzar de forma interactiva los conceptos del Colab
"Redes_Neuronales_vs_Regresion.ipynb": qué es una neurona, cómo se relaciona
con la regresión lineal/logística, y por qué a veces se necesita más de una.

Solo exploración interactiva — sin quiz ni evaluación embebida.
App autocontenida: no depende de ningún otro archivo del curso.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import make_regression, make_classification, make_moons, make_circles, make_blobs
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Neurona vs Regresión", layout="wide")

# ----------------------------------------------------------------------
# Utilidades compartidas
# ----------------------------------------------------------------------
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

def identidad(z):
    return z

ACTIVACIONES = {
    "Sigmoide (σ)": sigmoide,
    "ReLU": relu,
    "Tanh": tanh,
    "Identidad (regresión lineal)": identidad,
}


def graficar_frontera(modelo, X, y, titulo, ax):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    acc = accuracy_score(y, modelo.predict(X))
    ax.set_title(f"{titulo}\nAccuracy: {acc:.2f}")
    return ax


def graficar_frontera_proba(modelo, X, y, titulo, ax, umbral=0.5):
    """Frontera basada en probabilidad continua + umbral ajustable."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    proba = modelo.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    cf = ax.contourf(xx, yy, proba, levels=20, cmap="RdBu_r", alpha=0.7, vmin=0, vmax=1)
    ax.contour(xx, yy, proba, levels=[umbral], colors="black", linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")

    y_pred_umbral = (modelo.predict_proba(X)[:, 1] >= umbral).astype(int)
    acc = accuracy_score(y, y_pred_umbral)
    ax.set_title(f"{titulo}\nAccuracy (umbral={umbral:.2f}): {acc:.2f}")
    return cf


st.title("🧠 De la Regresión a la Red Neuronal")
st.caption("Companion interactivo del notebook — Machine Learning (ET0178)")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ La Neurona",
    "2️⃣ Regresión Lineal",
    "3️⃣ Regresión Logística",
    "4️⃣ Cuando una neurona no basta",
])

# ----------------------------------------------------------------------
# TAB 1: La neurona
# ----------------------------------------------------------------------
with tab1:
    st.header("¿Qué hace una neurona artificial?")
    st.markdown(
        "Una neurona combina sus entradas de forma ponderada y aplica una "
        "**función de activación**: **z = w1·x1 + w2·x2 + b** → **a = f(z)**. "
        "Mueve los controles y observa cómo cambia la salida.\n\n"
        "**Ejemplo contextualizado:** entradas de un sensor real (temperatura "
        "y hora del día) para estimar la probabilidad de un **pico de "
        "consumo eléctrico**."
    )

    col_izq, col_der = st.columns([1, 1.3])

    with col_izq:
        st.subheader("Entradas y parámetros")
        activacion_nombre = st.selectbox(
            "Función de activación", list(ACTIVACIONES.keys()), index=0
        )
        f_activacion = ACTIVACIONES[activacion_nombre]

        x1 = st.slider("x1: Temperatura (°C)", 15.0, 40.0, 30.0, 0.5)
        x2 = st.slider("x2: Hora del día", 0.0, 23.0, 18.0, 1.0)
        w1 = st.slider("w1 (peso temperatura)", -1.0, 1.0, 0.4, 0.05)
        w2 = st.slider("w2 (peso hora)", -1.0, 1.0, 0.1, 0.05)
        b = st.slider("b (sesgo)", -20.0, 5.0, -12.0, 0.5)

        z = w1 * x1 + w2 * x2 + b
        a = f_activacion(z)

        st.metric("z = w1·x1 + w2·x2 + b", f"{z:.3f}")
        st.metric(f"a = f(z)  [{activacion_nombre}]", f"{a:.3f}")

        if activacion_nombre == "Sigmoide (σ)":
            if a > 0.5:
                st.error(f"⚠️ Alerta: {a*100:.1f}% de probabilidad de **pico de consumo**")
            else:
                st.success(f"✅ Consumo normal esperado ({a*100:.1f}% de probabilidad de pico)")
        elif activacion_nombre == "Identidad (regresión lineal)":
            st.info(f"Con activación identidad, a = z. Esta neurona se comporta "
                    f"exactamente como una **regresión lineal** — salida = {a:.3f}.")
        else:
            st.info(f"Salida de la neurona con {activacion_nombre}: {a:.3f}")

    with col_der:
        st.subheader(f"Función de activación: {activacion_nombre}")
        z_rango = np.linspace(-10, 10, 200)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(z_rango, f_activacion(z_rango), color="steelblue")
        if activacion_nombre == "Sigmoide (σ)":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0, color="lightgray", linewidth=0.8)
        ax.axvline(0, color="lightgray", linewidth=0.8)
        ax.scatter([z], [a], color="red", zorder=5, s=80, label="Tu neurona ahora")
        ax.set_xlabel("z"); ax.set_ylabel("f(z)")
        ax.legend()
        st.pyplot(fig)

        st.markdown(
            "**Comparando activaciones:**\n"
            "- **Sigmoide:** aplasta a [0,1] → clasificación binaria.\n"
            "- **ReLU:** deja pasar positivos, corta negativos a 0 → la más usada en capas ocultas.\n"
            "- **Tanh:** parecida a sigmoide pero en rango [-1,1].\n"
            "- **Identidad:** no transforma nada → **así es una regresión lineal**."
        )

    st.markdown("🔗 Explora lo mismo de forma visual y completa en "
                "[TensorFlow Playground](https://playground.tensorflow.org).")

# ----------------------------------------------------------------------
# TAB 2: Regresión lineal
# ----------------------------------------------------------------------
with tab2:
    st.header("Regresión Lineal: una neurona sin activación")
    st.markdown(
        "Aquí el modelo se queda en **z = w·x + b**, sin aplicar activación "
        "no lineal (activación identidad).\n\n"
        "**Ejemplo:** predecir el **consumo eléctrico (kWh)** a partir de la "
        "**temperatura ambiente (°C)** — como en tu dashboard de consumo."
    )

    col_izq, col_der = st.columns([1, 1.3])
    with col_izq:
        n_muestras = st.slider("Número de muestras", 30, 300, 150, 10, key="lin_n")
        ruido = st.slider("Ruido (dificultad)", 0, 50, 15, 1, key="lin_noise")

        X_reg, y_reg = make_regression(
            n_samples=n_muestras, n_features=1, noise=ruido, random_state=42
        )
        modelo = LinearRegression().fit(X_reg, y_reg)
        y_pred = modelo.predict(X_reg)
        residuales = y_reg - y_pred

        st.metric("Peso (w)", f"{modelo.coef_[0]:.3f}")
        st.metric("Sesgo (b)", f"{modelo.intercept_:.3f}")
        st.caption(
            "En este ejemplo, `x` simula la temperatura y `y` el consumo "
            "eléctrico en kWh (datos sintéticos, normalizados)."
        )

        vista = st.radio("Vista", ["Ajuste del modelo", "Residuales (error por punto)"], key="lin_vista")

    with col_der:
        if vista == "Ajuste del modelo":
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.scatter(X_reg, y_reg, alpha=0.6, label="Datos")
            orden = np.argsort(X_reg[:, 0])
            ax.plot(X_reg[orden], y_pred[orden], color="red", linewidth=2, label="Modelo")
            ax.set_xlabel("Temperatura (°C, normalizada)")
            ax.set_ylabel("Consumo eléctrico (kWh, normalizado)")
            ax.legend()
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.scatter(X_reg, residuales, alpha=0.6, color="darkorange")
            ax.axhline(0, color="black", linestyle="--")
            ax.set_xlabel("Temperatura (°C, normalizada)")
            ax.set_ylabel("Residual (real − predicho)")
            ax.set_title("Distancia entre lo real y lo predicho")
            st.pyplot(fig)
            st.caption(
                "Cada punto es el error de una predicción individual. "
                "Residuales dispersos alrededor de 0 sin patrón = buen ajuste."
            )

    st.info("💡 Sube el ruido y observa cómo el modelo sigue encontrando la "
            "tendencia general aunque los datos individuales varíen más.")

# ----------------------------------------------------------------------
# TAB 3: Regresión logística
# ----------------------------------------------------------------------
with tab3:
    st.header("Regresión Logística: la neurona completa")
    st.markdown(
        "Mismo **z = w·x + b**, pero ahora sí pasa por la sigmoide → es una "
        "neurona.\n\n"
        "**Ejemplo:** clasificar si va a haber un **pico de consumo** (sí/no) "
        "en vez de predecir un número — mismo tipo de entradas del sensor, "
        "pregunta de negocio distinta."
    )

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        separacion = st.slider(
            "Separación entre clases", 0.5, 3.0, 1.8, 0.1, key="log_sep",
            help="Valores bajos = clases mezcladas y más difíciles de separar."
        )
    with col_ctrl2:
        umbral = st.slider(
            "Umbral de decisión", 0.05, 0.95, 0.5, 0.05, key="log_umbral",
            help="Por defecto se usa 0.5, pero se puede mover (ej. para reducir falsos negativos)."
        )

    X_clas, y_clas = make_classification(
        n_samples=200, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, class_sep=separacion, random_state=42
    )
    modelo_log = LogisticRegression().fit(X_clas, y_clas)

    col_izq, col_der = st.columns([1.3, 1])
    with col_izq:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cf = graficar_frontera_proba(modelo_log, X_clas, y_clas, "Mapa de probabilidad", ax, umbral)
        fig.colorbar(cf, ax=ax, label="Probabilidad de clase 1")
        st.pyplot(fig)
        st.caption("La línea negra es la frontera de decisión al umbral elegido. "
                   "El color de fondo es la probabilidad continua, no solo la clase final.")

    with col_der:
        w = modelo_log.coef_[0]
        b = modelo_log.intercept_[0]
        x_ejemplo = X_clas[0]
        z = np.dot(w, x_ejemplo) + b
        a = sigmoide(z)
        st.markdown("**Verificación: es una neurona**")
        st.code(
            f"x = {np.round(x_ejemplo, 2)}\n"
            f"w = {np.round(w, 2)}\n"
            f"b = {b:.2f}\n"
            f"z = w·x+b = {z:.2f}\n"
            f"a = σ(z)  = {a:.3f}\n"
            f"predict_proba: {modelo_log.predict_proba([x_ejemplo])[0][1]:.3f}"
        )
        st.caption("Nota: la frontera siempre es una **línea recta** — ese es su límite. "
                    "Mover el umbral desplaza *dónde* se traza esa línea sobre el mapa de probabilidad.")

# ----------------------------------------------------------------------
# TAB 4: Cuando una neurona no basta
# ----------------------------------------------------------------------
with tab4:
    st.header("¿Cuándo una sola neurona no es suficiente?")
    st.markdown(
        "Elige un dataset y una arquitectura, y observa cómo cambia la "
        "frontera de decisión y el proceso de entrenamiento."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        dataset_nombre = st.selectbox(
            "Dataset", ["Lunas (no lineal)", "Círculos (no lineal)", "Separable (lineal)"]
        )
    with col_b:
        capa1 = st.slider("Neuronas capa oculta 1", 0, 20, 0, 1, key="capa1",
                           help="0 = sin red, se usa regresión logística pura")
    with col_c:
        capa2 = st.slider("Neuronas capa oculta 2 (opcional)", 0, 20, 0, 1, key="capa2",
                           help="Deja en 0 para usar una sola capa oculta")

    ruido_moons = st.slider("Ruido del dataset", 0.0, 0.4, 0.2, 0.02, key="moons_noise")

    if dataset_nombre == "Lunas (no lineal)":
        X_data, y_data = make_moons(n_samples=300, noise=ruido_moons, random_state=42)
    elif dataset_nombre == "Círculos (no lineal)":
        X_data, y_data = make_circles(n_samples=300, noise=ruido_moons, factor=0.5, random_state=42)
    else:
        X_data, y_data = make_blobs(n_samples=300, centers=2, cluster_std=1.0 + ruido_moons * 3, random_state=42)

    capas = tuple(n for n in (capa1, capa2) if n > 0)

    if len(capas) == 0:
        modelo_actual = LogisticRegression().fit(X_data, y_data)
        nombre_modelo = "Regresión Logística (0 neuronas ocultas)"
        tiene_curva_perdida = False
    else:
        modelo_actual = MLPClassifier(
            hidden_layer_sizes=capas, max_iter=3000, random_state=42
        ).fit(X_data, y_data)
        nombre_modelo = f"Red Neuronal (capas ocultas: {capas})"
        tiene_curva_perdida = True

    col_izq, col_der = st.columns([1.4, 1])
    with col_izq:
        fig, ax = plt.subplots(figsize=(6, 5))
        graficar_frontera(modelo_actual, X_data, y_data, nombre_modelo, ax)
        st.pyplot(fig)

    with col_der:
        acc = accuracy_score(y_data, modelo_actual.predict(X_data))
        if len(capas) == 0:
            st.warning(f"Con 0 neuronas ocultas (solo regresión logística), la frontera "
                       f"es una recta y el accuracy queda limitado a {acc:.2f}.")
        else:
            st.success(f"Con capas {capas}, la frontera ya puede curvarse "
                       f"y el accuracy sube a {acc:.2f}.")

        if tiene_curva_perdida and hasattr(modelo_actual, "loss_curve_"):
            st.markdown("**Curva de pérdida durante el entrenamiento**")
            fig2, ax2 = plt.subplots(figsize=(4.5, 3))
            ax2.plot(modelo_actual.loss_curve_, color="purple")
            ax2.set_xlabel("Iteración"); ax2.set_ylabel("Pérdida (loss)")
            st.pyplot(fig2)
            st.caption("Así se ve el descenso de gradiente 'aprendiendo': "
                       "la pérdida baja iteración a iteración.")

    st.info(
        "💡 Compara esto con TensorFlow Playground: elegir dataset y agregar "
        "neuronas/capas aquí es exactamente lo mismo que hacías allá."
    )

st.divider()
st.caption(
    "App educativa — Machine Learning (ET0178). "
    "Basada en el notebook 'Redes_Neuronales_vs_Regresion.ipynb'."
)
