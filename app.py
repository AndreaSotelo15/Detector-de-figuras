import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Clasificador de Figuras", layout="centered")

# Cargar modelo entrenado
modelo_path = "modelo_entrenado_figuras.h5"
if not os.path.exists(modelo_path):
    st.error("❌ No se encuentra el modelo entrenado. Ejecuta 'entrenar_y_guardar.py' primero.")
    st.stop()

modelo = load_model(modelo_path)
clases = ['CIRCULO', 'CUADRADO', 'TRIANGULO']

# Interfaz Streamlit
st.title("🧠 Clasificador de Figuras Geométricas")
archivo = st.file_uploader("Sube una imagen (círculo, cuadrado o triángulo)", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = Image.open(archivo).convert("L").resize((128, 128))
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    entrada = np.expand_dims(np.array(imagen).astype("float32") / 255.0, axis=(0, -1))
    pred = modelo.predict(entrada)[0]
    resultado = clases[np.argmax(pred)]

    st.markdown("## 🔍 Resultado")
    st.success(f"Figura detectada: **{resultado}**")

    st.markdown("### 🔢 Confianza:")
    for i, clase in enumerate(clases):
        st.write(f"- {clase}: {pred[i]:.4f}")

    st.markdown("### 📐 Fórmula del área:")
    if resultado == "CIRCULO":
        st.latex(r"A = \pi \cdot r^2")
    elif resultado == "CUADRADO":
        st.latex(r"A = l^2")
    elif resultado == "TRIANGULO":
        st.latex(r"A = \\frac{1}{2} \\cdot b \\cdot h")