# limpieza_datos.py
import pandas as pd
import numpy as np

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza, validación y normalización de los datos de listas de espera.
    Corrige valores nulos, negativos, extremos y relaciones incoherentes.
    No elimina duplicados automáticamente, solo los reporta.
    """

    print("\n=== INICIANDO LIMPIEZA Y VALIDACIÓN DE DATOS ===")

    # ------------------------------------------
    # 1. Verificación de duplicados
    # ------------------------------------------

    duplicados = df[df.duplicated()]
    if not duplicados.empty:
        print(f"Se detectaron {len(duplicados)} registros potencialmente duplicados.")
        print("Sugerencia: revisar columnas que puedan diferenciar estos registros.")
    else:
        print("No se detectaron duplicados exactos.")

    # ------------------------------------------
    # 2. Manejo de valores nulos
    # ------------------------------------------

    if df.isnull().sum().sum() > 0:
        print("Se detectaron valores nulos. Serán reemplazados por la media de la columna.")
        df.fillna(df.mean(numeric_only=True), inplace=True)
