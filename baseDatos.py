from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import datetime

app = Flask(__name__)
CORS(app)
CORS(app, origins=['http://localhost:8000', 'http://localhost:80'])

def crear_base_de_datos():
    # Verificar si la base de datos ya existe
    if not os.path.exists('resultados.db'):
        # Conexión a la base de datos (o crear si no existe)
        conn = sqlite3.connect('resultados.db')
        cursor = conn.cursor()

        # Crear la tabla si no existe
        cursor.execute('''CREATE TABLE resultados
                          (id INTEGER PRIMARY KEY, fecha TEXT, vocal TEXT, cantidad INTEGER)''')

        # Guardar cambios y cerrar la conexión al finalizar
        conn.commit()
        conn.close()

@app.route('/baseDatos', methods=['POST'])
def guardar_resultado_en_base_de_datos():

    resultado = request.json

    # Crear la base de datos si no existe
    crear_base_de_datos()

    # Conexión a la base de datos
    conn = sqlite3.connect('resultados.db')
    cursor = conn.cursor()

    # Obtener la marca de tiempo actual
    fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for vocal, cantidad in resultado.items():
        # Insertar el nuevo documento con la marca de tiempo
        cursor.execute("INSERT INTO resultados (fecha, vocal, cantidad) VALUES (?, ?, ?)", (fecha_actual, vocal, cantidad))

    # Guardar cambios y cerrar la conexión
    conn.commit()
    conn.close()
    return jsonify({"message": "Datos guardados en base de datos"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)