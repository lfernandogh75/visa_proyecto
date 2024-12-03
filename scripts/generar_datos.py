#1 
import pandas as pd
import numpy as np
def generar_datos(n=1000, archivo_salida='../datasetoriginal/datos_visa.csv'):
    paises = ['País A', 'País B', 'País C', 'País D', 'País E', 'País F']
    motivos_viaje=['Turismo','Negocios','Estudios','Trabajo']
    historial_laborar=['Empleado','Desempleado','Estudiante']
    nivel_educacion=['Secundaria','Licenciatura','Maestría','Doctorado']
    antecedentes_penales=['Sí','No']
    dominio_idioma=['Bajo','Medio','Alto']
    tipo_visa=['Turista','Negocios','Estudiante','Trabajo']
    np.random.seed(42)
    data = pd.DataFrame({
        'Edad':np.random.randint(18,65,size=n),
        'País de origen':np.random.choice(paises,size=n),
        'Motivo del viaje':np.random.choice(motivos_viaje,size=n),
        'Historial laboral':np.random.choice(historial_laborar,size=n),
        'Nivel educación':np.random.choice(nivel_educacion,size=n),
        'Antecedentes penales':np.random.choice(antecedentes_penales,size=n,p=[0.1,0.9]),
        'Ingresos anuales (USD)':np.random.randint(5000,150000,size=n),
        'Dominio del idioma':np.random.choice(dominio_idioma,size=n,p=[0.3,0.5,0.2]),
        'Tipo de visa':np.random.choice(tipo_visa,size=n),
        'Duración (días)':np.random.randint(7,730,size=n)   
    })
    def determinar_resultado(row):
        if row['Antecedentes penales'] == 'Sí':
            return 'Rechazada'
        elif row['Ingresos anuales (USD)'] < 10000 and row['Motivo del viaje'] != 'Estudios':
            return 'Rechazada'
        elif row['Dominio del idioma']=='Bajo' and row['Tipo de visa'] in['Trabajo','Estudiante']:
             return 'Rechazada'
        else:
            return 'Aprobada'
    data['Resultado'] = data.apply(determinar_resultado, axis=1)
    data.to_csv(archivo_salida, index=False)
    print(f'Datos generados y guardados en {archivo_salida}')
    
if __name__ == '__main__':
    generar_datos()
        
       
    