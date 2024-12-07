import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
def preprocesar_datos(archivo_entrada='../datasetoriginal/datos_visa.csv',
                      archivo_salida='../dataset/datos_preprocesados.csv'):
    data= pd.read_csv(archivo_entrada)
    categorical_cols=['País de origen', 'Motivo del viaje','Historial laboral',
                      'Nivel educación', 'Antecedentes penales',
                      'Dominio del idioma', 'Tipo de visa']
    
    target_col='Resultado'
    data[target_col]=data[target_col].map({'Rechazada':0,'Aprobada':1})
    
    data=pd.get_dummies(data,columns=categorical_cols)
    numerical_cols=['Edad','Ingresos anuales (USD)','Duración (días)']
    scaler=StandardScaler()
    data[numerical_cols]=scaler.fit_transform(data[numerical_cols])
    
    data.to_csv(archivo_salida, index=False)
    print(f'Datos preprocesados y guardados en {archivo_salida}')
    
    joblib.dump(scaler,'../models/scaler.joblib')
    columnas_modelo=data.drop(target_col, axis=1).columns
    with open('../models/columnas_modelo.joblib','wb') as f:
        joblib.dump(columnas_modelo, f)
if __name__=="__main__":
    preprocesar_datos()