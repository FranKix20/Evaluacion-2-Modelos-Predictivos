# modelo_clasificacion.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def ejecutar_clasificacion(df):
    df['Categoria_Espera'] = np.where(df['Promedio_dias_Consulta'] > 400, 1, 0)

    X = df[['Trimestre_Global', 'Pacientes_Consulta', 'Pacientes_Cirugia']]
    y = df['Categoria_Espera']

    X_entrenar, X_prueba, y_entrenar, y_prueba = train_test_split(X, y, test_size=0.25, random_state=42)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_entrenar, y_entrenar)
    y_predicho = modelo.predict(X_prueba)

    print("\n=== CLASIFICACIÓN (Regresión Logística) ===")
    print("Exactitud:", accuracy_score(y_prueba, y_predicho))
    print(classification_report(y_prueba, y_predicho))

    matriz = confusion_matrix(y_prueba, y_predicho)
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=['Baja', 'Alta'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión - Nivel de espera")
    plt.show()
