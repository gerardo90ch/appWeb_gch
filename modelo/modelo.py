import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import os
from PIL import Image
import joblib

# Función para cargar las imágenes y etiquetas de entrenamiento
def cargar_imagenes_y_etiquetas(ruta):
    imagenes = []
    etiquetas = []
    for vocal in os.listdir(ruta):
        vocal_ruta = os.path.join(ruta, vocal)
        for imagen in os.listdir(vocal_ruta):
            imagen_ruta = os.path.join(vocal_ruta, imagen)
            img = Image.open(imagen_ruta).convert('L')  # Convertir a escala de grises
            img = img.resize((240, 240))
            img_array = np.array(img)
            imagenes.append(img_array)
            etiquetas.append(vocal)
    return np.array(imagenes), np.array(etiquetas)

ruta_entrenamiento = 'vocales'

# Cargar las imágenes de entrenamiento
imagenes_entrenamiento, etiquetas_entrenamiento = cargar_imagenes_y_etiquetas(ruta_entrenamiento)

# Convertir etiquetas de texto a números
vocales = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
etiquetas_entrenamiento_numeros = [vocales[vocal] for vocal in etiquetas_entrenamiento]
etiquetas_entrenamiento_onehot = to_categorical(etiquetas_entrenamiento_numeros, num_classes=5)

# Normalizar las imágenes
imagenes_entrenamiento = imagenes_entrenamiento.astype('float32') / 255.0

# Division de la data en entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(imagenes_entrenamiento, etiquetas_entrenamiento_onehot, test_size=0.1)
print(len(X_train))
print(len(X_test))

# Construir el modelo CNN
modelo = Sequential()
modelo.add(Conv2D(32, (24, 24), activation='relu', input_shape=(240, 240, 1)))
modelo.add(MaxPooling2D((20, 20)))
modelo.add(Flatten())
modelo.add(Dense(128, activation='relu'))
modelo.add(Dense(5, activation='softmax'))

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modelo.summary()

# Entrenar el modelo
history = modelo.fit(X_train.reshape(-1, 240, 240, 1), y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# Obtener las predicciones del modelo en el conjunto de prueba
y_pred = modelo.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calcular matriz de confusión
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
print(confusion_mtx)

#Se obtiene los valores de las métricas
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='weighted')
print("Precisión:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)

accuracy = accuracy_score(y_true, y_pred_classes)
print("Accuracy:", accuracy)
'''
# Graficar la accuracy durante las epochs
def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy del Modelo')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
    plt.show()

# Llamar a la función para graficar
plot_accuracy(history)

# Obtener índices de las imágenes mal clasificadas
indices_incorrecto = np.where(y_pred_classes != y_true)[0]
print(indices_incorrecto)
# Elegir un número limitado de imágenes para visualizar
imagenes_a_visualizar = min(10, len(indices_incorrecto))

# Configurar subplots
fig, axes = plt.subplots(1, imagenes_a_visualizar, figsize=(15, 15))


for i in range(imagenes_a_visualizar):
    index = indices_incorrecto[i]
    predicted_class = y_pred_classes[index]
    true_class = y_true[index]
    image = X_test[index].reshape(240, 240)
    predicted_vowel = [vocal for vocal, idx in vocales.items() if idx == predicted_class][0]
    true_vowel = [vocal for vocal, idx in vocales.items() if idx == true_class][0]
    image_label = f"Orig: {true_vowel}, Pred: {predicted_vowel}"

    # Mostrar las imagenes
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(image_label)
    axes[i].axis('off')

plt.show()
'''
#Guardar modelo
joblib.dump(modelo, '../modeloVocales.pkl')
