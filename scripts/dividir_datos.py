import pandas as pd
from sklearn.model_selection import train_test_split 
def dividir_datos(archivo_entrada='../dataset/datos_preprocesados.csv'):
    # Leer el dataset
    datos = pd.read_csv(archivo_entrada)
    
    # Dividir los datos en características (X) y etiquetas (y)
    target_col='Resultado'
    X = datos.drop(target_col, axis=1)
    y = datos[target_col]
    
    # Dividir en conjunto de entrenamiento y conjunto temporal
    X_train, X_temp, y_train, y_temp = train_test_split(X, y,test_size=0.3,random_state=42)
    
   # Dividir el conjunto de entrenamiento en conjunto de entrenamiento y conjunto de validación
    X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42)
    #Guardar el conjunto de entrenamiento
    X_train.to_csv('../dataset/X_train.csv',index=False)
    X_val.to_csv('../dataset/X_val.csv',index=False)
    X_test.to_csv('../dataset/X_test.csv',index=False)
    y_train.to_csv('../dataset/y_train.csv',index=False)
    y_val.to_csv('../dataset/y_val.csv',index=False)
    y_test.to_csv('../dataset/y_test.csv',index=False)
    print('Datos divididos correctamente.')

if __name__ == '__main__':
    dividir_datos()