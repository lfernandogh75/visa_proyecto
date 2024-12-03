import generar_datos
import preprocesar_datos
import dividir_datos
import entrenar_modelo
import entrenar_modelo2
import evaluar_modelo

def main():
    # Generar datos
    generar_datos.generar_datos()
    # Preprocesar datos
    preprocesar_datos.preprocesar_datos()
    # Dividir datos
    dividir_datos.dividir_datos()
    # Entrenar modelo
    entrenar_modelo.entrenar_modelo()
    # Entrenar modelo 2
    entrenar_modelo2.entrenar_modelo_tuning()
    # Evaluar modelo
    evaluar_modelo.evaluar_modelo()

if __name__ == '__main__':
    main()