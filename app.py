from flask import Flask, request, jsonify,render_template, send_file
from flask_cors import CORS
from PIL import Image
import requests
from io import BytesIO
import io
import base64
import joblib
import numpy as np
import requests

app = Flask(__name__)
CORS(app)
CORS(app, origins=['http://localhost:8000', 'http://localhost:80'])

modelo= joblib.load('modeloVocales.pkl')
vocales = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}

def predecir_imagen(imagen):
    img = imagen.convert('L')
    img = img.resize((240, 240))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    resultado = modelo.predict(img_array)
    valor_max = np.max(resultado)
    if valor_max>=0.7:
      vocal_detectada = list(vocales.keys())[np.argmax(resultado)]
    else:
      vocal_detectada = 'vocal_no_detectada'
    return vocal_detectada

def resumen_vocales(predicciones):
    vocales_tot = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0, 'vocal_no_detectada': 0}

    for pred in predicciones:
        if pred in vocales_tot:
            vocales_tot[pred] += 1
    return vocales_tot

def segment_image(file,num_horizontal_segments,num_vertical_segments ):
    # Cargar la imagen
    img = Image.open(file)

    # Obtener el tamaño de la imagen
    width, height = img.size
    # Calcular el ancho y alto de cada segmento
    segment_width = width // num_horizontal_segments
    segment_height = height // num_vertical_segments
    # Lista para almacenar los segmentos
    segments = []
    for i in range(num_horizontal_segments):
        for j in range(num_vertical_segments):
         # Calcular las coordenadas del segmento actual
            left = i * segment_width
            upper = j * segment_height
            right = (i + 1) * segment_width
            lower = (j + 1) * segment_height

            # Extraer el segmento actual de la imagen
            segment = img.crop((left, upper, right, lower))

            # Agregar el segmento a la lista
            segments.append(segment)
    return segments


@app.route('/processimage', methods=['POST'])
def contadorVocales():
    img = request.files['image']
    horizontal = int(request.form['horizontal'])
    vertical = int(request.form['vertical'])
    img_segmentada = segment_image(img, horizontal, vertical)

    predicciones = []
    imagenes = []  # Lista para almacenar las imágenes procesadas

    for img in img_segmentada:
        vocal_detectada_prueba = predecir_imagen(img)
        predicciones.append(vocal_detectada_prueba)

        # Convierte la imagen a base64 y agrega a la lista de imágenes
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        imagenes.append(img_base64)

    resumen = resumen_vocales(predicciones)

    # Enviar la solicitud POST del resumen a microservcio baseDatos
    response = requests.post("http://localhost:5001/baseDatos", json=resumen)

    # Agregar las imágenes procesadas al JSON de respuesta
    response_data = {
        "resumen": resumen,
        "imagenes": imagenes
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)