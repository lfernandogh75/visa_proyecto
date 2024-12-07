Python 3.12.1
virtualenv visa_env
activar
visa_env\Scripts\activate
1 generar datos
pip install numpy pandas
2 preprocesar datos y dividir datos
pip install scikit-learn
3 entrenar modelo
pip install tensorflow
pip install keras-tuner
pip install keras-tuner --upgrade

4 evaluar mdelo
evaluar_modelo.py
pip install matplotlib
pip install seaborn

crear el archivo requirements.txt
pip freeze > requirements.txt
.gitignore




categorical_cols = data.select_dtypes(include=['object', 'category']).columns
git init  
git add .
git commit -m "proyecto visas v1"
git remote add origin https://github.com/lfernandogh75/visa_proyecto.git

crear app.py
pip install flask joblib