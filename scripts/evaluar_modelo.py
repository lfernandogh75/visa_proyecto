import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
def evaluar_modelo():
    # cargar los conjuntos de datos
    X_test= pd.read_csv('../dataset/X_test.csv')
    y_test = pd.read_csv('../dataset/y_test.csv')
    # Cargar el modelo
    #modelo = load_model('../models/visa_model.h5')model_entrenado.h5
    modelo = load_model('../models/model_entrenado.h5')
    # evaluar el modelo en el conjunto de prueba
    loss, accuracy = modelo.evaluate(X_test, y_test)
    print(f'Pérdida en el conjunto de prueba: {loss}')
    print(f'Exactitud en el conjunto de prueba: {accuracy}')
    #Cargar el historial de entrenamiento
    #history = pd.read_csv('../models/historial_entrenamiemto.csv')
    history = pd.read_csv('../models/history.csv')
    # visualización del rendimiento durante el estrenamiento
    plt.figure(figsize=(12,5))
    #precisíon
    plt.subplot(1,2,1)
    plt.plot(history['accuracy'],label='Entrenamiento')
    plt.plot(history['val_accuracy'],label='Validación')
    plt.title('Precisión del modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    # pérdida
    plt.subplot(1,2,2)
    plt.plot(history['loss'],label='Entrenamiento')
    plt.plot(history['val_loss'],label='Validación')
    plt.title('Pérdida del modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #realizar predicciones en el conjunto de prueba
    y_pred = (modelo.predict(X_test)>0.5).astype("int32")
    # generar la matriz de confusión
    cm= confusion_matrix(y_test, y_pred)
    print('Matriz de confusión:')
    print(cm)
    # Visualización de la matriz de confusión
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True,fmt='d', cmap='Blues',
                xticklabels=['Rechazada', 'Aprobada'],yticklabels=['Rechazada', 'Aprobada'])
    plt.xlabel('Predición')
    plt.ylabel('Realidad')
    plt.title('Matriz de confusión')
    plt.show()
    # generar el informe de clasificación
    reporte = classification_report(y_test, y_pred,target_names=['Rechazada', 'Aprobada'])
    print('Informe de clasificación:')
    print(reporte)
if __name__ == '__main__':
    evaluar_modelo()
    
    
