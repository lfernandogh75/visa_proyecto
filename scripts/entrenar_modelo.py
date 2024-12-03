import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def entrenar_modelo(epochs=50,batch_size=32):
    X_train = pd.read_csv('../dataset/X_train.csv')
    y_train = pd.read_csv('../dataset/y_train.csv')
    X_val = pd.read_csv('../dataset/X_val.csv')
    y_val = pd.read_csv('../dataset/y_val.csv')
    #definir el modelo
    model = Sequential()
    model.add(Dense(64,input_dim=X_train.shape[1],activation='relu',))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #compilar el modelo
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #entrenar el modelo
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    #guardar el modelo
    model.save('../models/visa_model.h5')
    pd.DataFrame(history.history).to_csv('../models/historial_entrenamiemto.csv',index=False)
    print('Modelo entrenado y guardado')

if __name__ == "__main__":
    entrenar_modelo()
    
