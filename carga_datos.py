# carga_datos.py
import pandas as pd
import numpy as np
from limpieza_datos import limpiar_datos

def cargar_datos():
    """
    Carga los datos brutos de listas de espera y los envía al proceso de limpieza.
    """

    datos = {
        'Año': [2022]*4 + [2023]*4 + [2024]*4 + [2025],
        'Trimestre': [1,2,3,4]*3 + [1],
        'Promedio_dias_GES': [157,157,156,157,150,137,130,130,138,140,138,137,142],
        'Promedio_dias_Consulta': [504,478,455,426,405,386,371,353,356,360,356,359,359],
        'Promedio_dias_Cirugia': [604,600,584,544,515,484,464,449,430,426,425,422,422],
        'Pacientes_GES': [61059,66299,68128,60234,60378,69013,70590,68385,79846,83238,84288,74740,74956],
        'Pacientes_Consulta': [1707184,1764937,1813018,1851733,1890375,1923234,1955793,2006440,2094371,2129160,2182438,2165195,2239878],
        'Pacientes_Cirugia': [293109,294636,290822,267921,269907,278764,283274,294565,309034,322882,334969,343630,370265]
    }

    df = pd.DataFrame(datos)
    df['Trimestre_Global'] = np.arange(1, len(df) + 1)

    df_limpio = limpiar_datos(df)
    return df_limpio
