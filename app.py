from flask import Flask, render_template, request
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Cargar el modelo
model = load_model('models/visa_model.h5')
# Cargar el scaler

scaler = joblib.load('models/scaler.joblib')
with open('models/columnas_modelo.joblib', 'rb') as f:
    columnas_modelo = joblib.load(f)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        inputs_data ={
            "Edad": float(request.form["Edad"]),
            "País de origen":request.form["País de origen"],
            "Motivo del viaje":request.form["Motivo del viaje"],
            "Historial laboral":request.form["Historial laboral"],
            "Nivel educación":request.form["Nivel educación"],
            "Antecedentes penales":request.form["Antecedentes penales"],
            "Ingresos anuales (USD)":float(request.form["Ingresos anuales (USD)"]),
            "Dominio del idioma":request.form["Dominio del idioma"],
            "Tipo de visa":request.form["Tipo de visa"],
            "Duración (días)":float(request.form["Duración (días)"]),
    
        }
    # convertir a DataFrame
    input_df = pd.DataFrame([inputs_data])
    # preprocesar datos
    #codificar variables categóricas
    input_df = pd.get_dummies(input_df)
    #Alinear las columnas con las del modelo
    input_df = input_df.reindex(columns=columnas_modelo,fill_value=0)
    
    #Escalar las variables numéricas
    numerical_cols=['Edad','Ingresos anuales (USD)','Duración (días)']
    input_df[numerical_cols]= scaler.transform(input_df[numerical_cols])
    # predecir
    prediction = model.predict(input_df)[0][0]
    if prediction >= 0.5:
        resultado = "Aprobada"
    else:
        resultado = "Rechazada"
    return render_template('result.html', prediction=resultado)

if __name__ == '__main__':
    app.run(debug=True)
    
    
                      