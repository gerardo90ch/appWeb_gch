Esta app web permite identificar vocales en una imagen, de preferencia manuscritas. Se ingresa la imagen y se selecciona en base a la cantidad de letras presente en la imagen, igualmente en la carpeta /imagenes_prueba hay muestras para realizar pruebas.
Utiliza los puertos 8000, 5000 y 5001
Igualmente, esta en la carpeta /modelo, el modelo con el set de entrenamiento, si es necesario generar el modelo entrenado.


GCH

# Instrucciones

## Instalar dependencias

```bash
sudo apt update
sudo apt install python3-pip
python3 -m pip install -r requirements.txt
```
## Generar modelo primero

```bash
cd modelo/ 
python3 modelo.py
cd .. 
```

## Correr un servidor web y el API

```bash
python3 -m http.server &
python3 app.py &
python3 baseDatos.py
```


## Otras instrucciones


