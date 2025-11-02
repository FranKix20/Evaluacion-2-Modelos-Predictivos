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

    # ------------------------------------------
    # 3. Corrección de valores negativos
    # ------------------------------------------

    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    for col in columnas_numericas:
        negativos = df[df[col] < 0].shape[0]
        if negativos > 0:
            print(f"{negativos} valores negativos detectados en '{col}'. Corrigiendo...")
            df.loc[df[col] < 0, col] = np.nan
            df[col].fillna(df[col].mean(), inplace=True)

    # ------------------------------------------
    # 4. Detección y corrección de valores extremos
    # ------------------------------------------

    for col in columnas_numericas:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        RIC = Q3 - Q1
        bajo, alto = Q1 - 1.5 * RIC, Q3 + 1.5 * RIC
        extremos = df[(df[col] < bajo) | (df[col] > alto)]
        if len(extremos) > 0:
            print(f"{len(extremos)} valores extremos detectados en '{col}'. Corrigiendo...")
            df[col] = np.where(df[col] < bajo, bajo,
                        np.where(df[col] > alto, alto, df[col]))

    # ------------------------------------------
    # 5. Normalización de escalas
    # ------------------------------------------

    for col in ['Pacientes_GES', 'Pacientes_Consulta', 'Pacientes_Cirugia']:
        if col in df.columns:
            df[col] = df[col] / 1_000_000  # transformar a millones

    # ------------------------------------------
    # 6. Validaciones lógicas
    # ------------------------------------------

    incoherentes = df[df['Pacientes_Consulta'] < df['Pacientes_GES']]
    if len(incoherentes) > 0:
        print(f"{len(incoherentes)} registros con incoherencias (Consulta < GES). Corrigiendo...")
        df.loc[df['Pacientes_Consulta'] < df['Pacientes_GES'], 'Pacientes_Consulta'] = \
            df['Pacientes_GES'] * 1.1

    # Años válidos
    df = df[(df['Año'] >= 2020) & (df['Año'] <= 2025)]

    # Días de espera positivos
    for col in ['Promedio_dias_GES', 'Promedio_dias_Consulta', 'Promedio_dias_Cirugia']:
        df[col] = df[col].apply(lambda x: x if x > 0 else df[col].mean())

    df.reset_index(drop=True, inplace=True)

    print("Limpieza y validación completadas correctamente.")
    print(f"Registros finales: {len(df)} | Columnas: {len(df.columns)}")
    return df
