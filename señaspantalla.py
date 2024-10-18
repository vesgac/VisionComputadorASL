import cv2
import numpy as np
import tensorflow as tf

# Parámetros
img_size = 128
minValue = 70

# Diccionario de etiquetas (A-Z y 0)
labels_dict = {
    0: '0',
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'E',
    6: 'F',
    7: 'G',
    8: 'H',
    9: 'I',
    10: 'J',
    11: 'K',
    12: 'L',
    13: 'M',
    14: 'N',
    15: 'O',
    16: 'P',
    17: 'Q',
    18: 'R',
    19: 'S',
    20: 'T',
    21: 'U',
    22: 'V',
    23: 'W',
    24: 'X',
    25: 'Y',
    26: 'Z'
}

# Definir color del rectángulo y coordenadas de la ROI (región de interés)
color_dict = (0, 255, 0)
roi_x1, roi_y1, roi_x2, roi_y2 = 24, 24, 250, 250  # Coordenadas del rectángulo original del ROI

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('asl_classifier_1.h5', compile=False)

# Inicialización de captura de video desde la cámara
source = cv2.VideoCapture(0)
count = 0
string = ""
prev = ""
prev_val = 0

# Bucle principal
while True:
    ret, img = source.read()
    if not ret:
        break  # Si no hay más frames, salir del bucle

    # ---- EVITAR EFECTO ESPEJO ----
    # Invertir la imagen horizontalmente para corregir el efecto espejo
    img = cv2.flip(img, 1)

    # ---- AJUSTAR POSICIÓN DEL ROI ----
    # Obtener las dimensiones de la imagen
    img_width = img.shape[1]

    # Calcular la nueva posición del ROI reflejado horizontalmente
    # ROI nuevo se calculará a partir del ancho de la imagen menos las posiciones originales
    new_roi_x1 = img_width - roi_x2
    new_roi_x2 = img_width - roi_x1

    # Dibujar el rectángulo en la nueva posición
    cv2.rectangle(img, (new_roi_x1, roi_y1), (new_roi_x2, roi_y2), color_dict, 2)

    # Recortar la imagen en la nueva región de interés (ROI)
    crop_img = img[roi_y1:roi_y2, new_roi_x1:new_roi_x2]

    # Incrementar el contador
    count += 1
    if count % 100 == 0:
        prev_val = count

    # Mostrar el contador en la imagen
    cv2.putText(img, str(prev_val // 100), (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # Procesamiento de la imagen (desenfoque, umbralización)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Redimensionar la imagen a 128x128 y normalizar
    resized = cv2.resize(res, (img_size, img_size))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, img_size, img_size, 1))

    # Realizar la predicción con el modelo
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]

    # Actualizar el texto acumulado cada 300 frames
    if count == 300:
        count = 99
        prev = labels_dict[label]  # Usar el diccionario para convertir el label en una letra o número
        if label == 0:
            string += " "
        else:
            string += prev

    # Mostrar la etiqueta y el texto acumulado en la imagen
    cv2.putText(img, prev, (24, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, string, (275, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Colocar la imagen preprocesada en la ROI dentro de la imagen principal
    preprocessed_roi = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    img[roi_y1:roi_y2, new_roi_x1:new_roi_x2] = preprocessed_roi

    # Mostrar la imagen procesada y la imagen original
    cv2.imshow('LIVE', img)

    # Verificar si se presiona la tecla Esc para salir
    key = cv2.waitKey(1)
    if key == 27:  # Tecla Esc para salir
        break

# Imprimir el texto final
print(string)

# Liberar recursos
cv2.destroyAllWindows()
source.release()



