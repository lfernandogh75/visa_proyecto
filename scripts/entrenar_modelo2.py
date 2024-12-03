import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import kerastuner as kt

def entrenar_modelo_tuning(max_trials=10,executions_per_trial=1):
    #Cargar los conjuntos de datos
    X_train = pd.read_csv('../dataset/X_train.csv')
    y_train = pd.read_csv('../dataset/y_train.csv')
    X_val = pd.read_csv('../dataset/X_val.csv')
    y_val = pd.read_csv('../dataset/y_val.csv')
    X_test = pd.read_csv('../dataset/X_test.csv')
    y_test = pd.read_csv('../dataset/y_test.csv')
    
    # Definir el espacio de búsqueda para la búsqueda de mejores hiperparámetros
    def model_builder(hp):
        model = Sequential()
        # Heperparámetros: número de neuronas en la primera capa oculta
        hp_units =hp.Int('units_layer_1',min_value=16,max_value=128,step=16)
           
        model.add(Dense(units=hp_units,activation='relu',input_dim=X_train.shape[1]))
        # Hiperparámetro: incluir o no segunda capa oculta
        if hp.Boolean('use_layer_2'):
             hp_units_2 =hp.Int('units_layer_2',min_value=16,max_value=128,step=16)
             model.add(Dense(units=hp_units_2,activation='relu'))
        #capa de salida con activación sigmoide
        model.add(Dense(1,activation='sigmoid'))
        #hiperparámetro: Tasa de aprendizaje del optimizador Adam
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
        return model
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='../models',
        project_name='visa_proyecto')
    # ejecutar la búsqueda de hiperparámetros
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    # Obtener los mejores hiperparámetros encontrados
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Números de neuronas en la primera capa oculta: {best_hps.get('units_layer_1')}")
    print(f"usar segunda capa oculta: {best_hps.get('use_layer_2')}")
    if best_hps.get('use_layer_2'):
         print(f"Números de neuronas en la segunda capa oculta: {best_hps.get('units_layer_2')}")
    print(f"Tasa de aprendizaje: {best_hps.get('learning_rate')}")
    
    # costruir el modelo con los mejores hiperparámetros
    model =tuner.hypermodel.build(best_hps)
    #mostra resumen de la arquitectura 
    model.summary()
   #combinar los datos de entrenamiento y validación para entrenar el modelo final
    X_final_train = pd.concat([X_train, X_val])
    y_final_train = pd.concat([y_train, y_val])
    #entrenar el modelo final
    history = model.fit(
        X_final_train,
        y_final_train,
        epochs=50,
        validation_data=(X_test, y_test)
    )
    ##guardar el modelo entrenado
    model.save('../models/model_entrenado.h5')
    print('Modelo entrenado y guardado en ../models/model_entrenado.h5')
    #Guardar el historico de entrenamiento en un archivo csv
    pd.DataFrame(history.history).to_csv('../models/history.csv', index=False)
    print('Historico de entrenamiento guardado en ../models/history.csv')
    #guardar la arquitectura del modelo en un archivo json
    model_json=model.to_json()
    with open('../models/model_architecture.json', 'w') as json_file:
        json_file.write(model_json)
    print('Arquitectura del modelo guardada en ../models/model_architecture.json')

if __name__ == '__main__':
    entrenar_modelo_tuning()