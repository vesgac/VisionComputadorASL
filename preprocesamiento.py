import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen
img = cv2.imread('sena.png')

# Paso 1: Convertir la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Paso 2: Desenfoque Gaussiano
blur = cv2.GaussianBlur(gray, (5, 5), 2)

# Paso 3: Umbralización Adaptativa (Binarización)
th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Paso 4: Umbralización con método Otsu
ret, otsu = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Paso 5: Redimensionar la imagen
img_size = 128
resized = cv2.resize(otsu, (img_size, img_size))

# Paso 6: Normalización de la imagen
normalized = resized / 255.0

# Mostrar todas las imágenes al tiempo con Matplotlib
titles = ['Paso 1: Imagen Original', 
          'Paso 2: Escala de Grises', 
          'Paso 3: Desenfoque Gaussiano', 
          'Paso 4: Umbralización Adaptativa', 
          'Paso 5: Método Otsu', 
          'Paso 6: Redimensionada', 
          'Paso 7: Normalizada']

images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gray, blur, th3, otsu, resized, normalized]

# Crear una cuadrícula de 2 filas y 4 columnas (puedes ajustar el tamaño si es necesario)
plt.figure(figsize=(12, 8))

for i in range(len(images)):
    plt.subplot(2, 4, i + 1)  # Subplot con 2 filas y 4 columnas
    plt.imshow(images[i], cmap='gray' if i > 0 else None)  # Mostrar en escala de grises a partir de la imagen 1
    plt.title(titles[i])
    plt.axis('off')  # Ocultar los ejes

# Mostrar la cuadrícula de imágenes
plt.tight_layout()
plt.show()

