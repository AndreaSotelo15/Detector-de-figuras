import numpy as np
import cv2
import streamlit as st
import math
from PIL import Image
from tensorflow.keras.models import load_model

# Cargar el modelo
modelo = load_model("modelo_entrenado_con_imagenes.h5")

st.set_page_config(page_title="Clasificador de Figuras", layout="centered")
st.title("Clasificador de Figuras")

archivo = st.file_uploader("Sube una imagen (círculo, cuadrado o triángulo)", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = Image.open(archivo).convert("RGB")
    st.image(imagen, caption="Imagen cargada", use_container_width=True)

    img_np = np.array(imagen)
    img_resized = cv2.resize(img_np, (128, 128))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    entrada = img_gray.astype("float32") / 255.0
    entrada = np.expand_dims(entrada, axis=(0, -1))

    pred = modelo.predict(entrada)[0]
    clases = ["CIRCULO", "CUADRADO", "TRIANGULO"]
    resultado = clases[np.argmax(pred)]

    st.markdown("### Resultado")
    st.success(f"Figura detectada: **{resultado}**")

    # Calcular área estimada (número de píxeles oscuros)
    A_val = np.sum(img_gray < 200)
    pi_val = 3.1416

    st.markdown("## Área y perímetro:")

    if resultado == "CIRCULO":
        r_val = math.sqrt(A_val / pi_val) if A_val > 0 else 0
        P = 2 * pi_val * r_val
        st.latex(r"A = \pi \cdot r^2")
        st.write(f"π ≈ {pi_val}")
        st.write(f"r ≈ {r_val:.2f} px") 
        st.write(f"A ≈ {A_val:.2f} px²")
        st.latex(r"P = 2 \cdot \pi \cdot r")
        st.write(f"P ≈ {P:.2f} px")

    elif resultado == "CUADRADO":
        l_val = math.sqrt(A_val) if A_val > 0 else 0
        P = 4 * l_val
        st.latex(r"A = l^2")
        st.write(f"l ≈ {l_val:.2f} px")
        st.write(f"A ≈ {A_val:.2f} px²")
        st.latex(r"P = 4 \cdot l")
        st.write(f"P ≈ {P:.2f} px")

    elif resultado == "TRIANGULO":
        b_val = math.sqrt(2 * A_val) if A_val > 0 else 0
        h_val = b_val
        P = 3 * b_val  # Aproximación como triángulo equilátero
        st.latex(r"A = \frac{1}{2} \cdot b \cdot h")
        st.write(f"b ≈ {b_val:.2f} px")
        st.write(f"h ≈ {h_val:.2f} px")
        st.write(f"A ≈ {A_val:.2f} px²")
        st.latex(r"P = b + b + b")
        st.write(f"P ≈ {P:.2f} px")