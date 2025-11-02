import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def ejecutar_arbol_y_bosque(df):
    """
    Aplica un Árbol de Decisión y un Bosque Aleatorio para clasificar
    el nivel de espera (Alta o Baja) según los pacientes y el tiempo.
    Genera gráficos interpretativos de ambos modelos.
    """

    print("\n=== MODELOS DE ÁRBOL DE DECISIÓN Y BOSQUE ALEATORIO ===")

    # Crear variable categórica binaria (Alta/Baja espera)
    df['Categoria_Espera'] = np.where(df['Promedio_dias_Consulta'] > 400, 'Alta', 'Baja')

    # Variables predictoras y objetivo
    X = df[['Trimestre_Global', 'Pacientes_Consulta', 'Pacientes_Cirugia']]
    y = df['Categoria_Espera']

    # División de datos
    X_entrenar, X_prueba, y_entrenar, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)

    # ==============================================
    #  1. Árbol de Decisión
    # ==============================================
    modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo_arbol.fit(X_entrenar, y_entrenar)

    # Visualización del árbol
    plt.figure(figsize=(14, 8))
    plot_tree(
        modelo_arbol,
        feature_names=X.columns,
        class_names=modelo_arbol.classes_,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Árbol de Decisión - Clasificación de Nivel de Espera", fontsize=14)
    plt.show()

    # Evaluación del árbol
    y_pred_arbol = modelo_arbol.predict(X_prueba)
    exactitud_arbol = accuracy_score(y_prueba, y_pred_arbol)
    print(f" Exactitud del Árbol de Decisión: {exactitud_arbol:.2f}")

    # ==============================================
    #  2. Bosque Aleatorio
    # ==============================================
    modelo_bosque = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_bosque.fit(X_entrenar, y_entrenar)

    y_pred_bosque = modelo_bosque.predict(X_prueba)
    exactitud_bosque = accuracy_score(y_prueba, y_pred_bosque)
    print(f" Exactitud del Bosque Aleatorio: {exactitud_bosque:.2f}")
    print(f"Comparación: Árbol Simple = {exactitud_arbol:.2f} | Bosque = {exactitud_bosque:.2f}")

    # ==============================================
    # 3. Importancia de las variables (gráfico)
    # ==============================================
    importancias = modelo_bosque.feature_importances_
    plt.figure(figsize=(7, 5))
    plt.bar(X.columns, importancias, color='seagreen')
    plt.title("Importancia de Variables - Bosque Aleatorio", fontsize=14)
    plt.ylabel("Importancia Relativa")
    plt.xlabel("Variables Predictoras")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # ==============================================
    # 4. Resumen final
    # ==============================================
    print("\n=== RESUMEN DE RESULTADOS ===")
    print(f"Exactitud Árbol: {exactitud_arbol:.2f}")
    print(f"Exactitud Bosque: {exactitud_bosque:.2f}")
    print("Variables más influyentes:", list(X.columns[np.argsort(importancias)[::-1]]))
