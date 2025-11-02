
from carga_datos import cargar_datos
from utilidades import mostrar_resumen
from modelo_regresion import ejecutar_regresion
from modelo_clasificacion import ejecutar_clasificacion
from modelo_arbol_bosque import ejecutar_arbol_y_bosque

def main():
    print("===============================================")
    print(" ANÁLISIS DE LISTAS DE ESPERA DEL MINSAL 2022-2025 ")
    print("===============================================")

    # 1. Carga y limpieza de datos
    datos = cargar_datos()
    mostrar_resumen(datos)

    # 2. Aplicación de modelos
    ejecutar_regresion(datos)
    ejecutar_clasificacion(datos)
    ejecutar_arbol_y_bosque(datos)

if __name__ == "__main__":
    main()
