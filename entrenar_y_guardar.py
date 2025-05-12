import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
labels = ["circulo", "cuadrado", "triangulo"]
data_dir = r"C:\Users\andre\figuras_reales"

X, y = [], []

for idx, label in enumerate(labels):
    carpeta = os.path.join(data_dir, label)
    for nombre in os.listdir(carpeta):
        path = os.path.join(carpeta, nombre)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(idx)

X = np.array(X).astype("float32") / 255.0
X = np.expand_dims(X, axis=-1)
y = to_categorical(np.array(y), num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save("modelo_entrenado_con_imagenes.h5", include_optimizer=False)
print("Modelo entrenado y guardado como modelo_entrenado_con_imagenes.h5")
