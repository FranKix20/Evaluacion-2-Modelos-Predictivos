# modelo_regresion.py
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def ejecutar_regresion(df):
    X = df[['Trimestre_Global', 'Pacientes_Consulta', 'Pacientes_Cirugia']]
    y = df['Promedio_dias_Consulta']

    X_entrenar, X_prueba, y_entrenar, y_prueba = train_test_split(X, y, test_size=0.25, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_entrenar, y_entrenar)
    y_predicho = modelo.predict(X_prueba)

    print("\n=== REGRESIÓN LINEAL ===")
    print("Coeficientes:", modelo.coef_)
    print("Error cuadrático medio (RMSE):", np.sqrt(mean_squared_error(y_prueba, y_predicho)))

    plt.figure()
    plt.plot(y_prueba.values, label="Real", marker='o')
    plt.plot(y_predicho, label="Predicho", marker='x')
    plt.title("Regresión Lineal - Promedio de días de espera en consultas")
    plt.xlabel("Muestras de prueba")
    plt.ylabel("Promedio de días")
    plt.legend()
    plt.show()