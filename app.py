import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Configuración inicial
st.set_page_config(page_title="Detector de Figuras", layout="centered")

# Estilos
st.markdown("""
    <style>
        .stApp {
            background-color: #e6ccff;
        }
        button[kind="primary"] {
            background-color: #800080;
            color: white;
        }
        button[title="Stop"] {
            background-color: #800080;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Detector de Figuras Geometricas")
st.write("Sube una imagen o usa tu cámara para detectar triángulos, cuadrados y círculos.")
st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------- DETECCIÓN GENERAL -----------------------------
def detectar_figuras(imagen, para_video=False):
    figuras_detectadas = []
    salida = imagen.copy()

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    borrosa = cv2.GaussianBlur(gris, (9, 9), 2)

    # ------------------ CÍRCULO (NO TOCAR, excepto param2 para video) ------------------
    param2_val = 60 if not para_video else 80  # Mayor valor = menos falsos círculos
    circulos = cv2.HoughCircles(
        borrosa, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=100, param2=param2_val, minRadius=40, maxRadius=300
    )
    if circulos is not None:
        circulos = np.uint16(np.around(circulos[0]))
        mayor = max(circulos, key=lambda c: c[2])
        x, y, r = mayor
        cv2.circle(salida, (x, y), r, (0, 255, 0), 2)
        cv2.putText(salida, "Circulo", (x - 40, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        figuras_detectadas.append("Circulo")

    # ------------------ TRIÁNGULO Y CUADRADO ------------------
    _, binaria = cv2.threshold(gris, 180, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < 500:
            continue

        epsilon = 0.02 * cv2.arcLength(contorno, True)
        approx = cv2.approxPolyDP(contorno, epsilon, True)
        lados = len(approx)

        figura = None
        if lados == 3:
            figura = "Triangulo"
        elif lados == 4:
            figura = "Cuadrado"
        else:
            continue

        figuras_detectadas.append(figura)
        cv2.drawContours(salida, [approx], 0, (0, 255, 0), 2)
        x, y = approx[0][0]
        cv2.putText(salida, figura, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return salida, figuras_detectadas

# ----------------------------- SUBIR IMAGEN -----------------------------
archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if archivo:
    imagen_pil = Image.open(archivo).convert("RGB")
    imagen_np = np.array(imagen_pil)
    imagen_cv2 = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)

    procesada, figuras = detectar_figuras(imagen_cv2)
    st.image(cv2.cvtColor(procesada, cv2.COLOR_BGR2RGB), caption="Resultado", use_container_width=True)

    st.subheader("Figuras detectadas:")
    if figuras:
        for i, figura in enumerate(figuras, 1):
            st.write(f"{i}. {figura}")
    else:
        st.warning("No se detectaron figuras.")

# ----------------------------- CÁMARA -----------------------------
st.subheader("Captura desde cámara en tiempo real")

class Detector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        procesada, _ = detectar_figuras(img, para_video=True)
        return procesada

webrtc_streamer(
    key="figuras",
    video_transformer_factory=Detector,
    media_stream_constraints={"video": True, "audio": False}
)
