# Tarea_03_mineria_datos script 4 mediad de tendecia central

# Punto 4: Análisis estadístico descriptivo básico
# Medidas de tendencia central, dispersión, cuartiles, sesgo y curtosis

import pandas as pd
from sklearn.datasets import load_diabetes

# 1. Cargar dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

# 2. Seleccionar solo variables numéricas
variables_numericas = df.select_dtypes(include=["int64", "float64"])

# 3. Calcular estadísticas descriptivas
estadisticas = pd.DataFrame({
    "Media": variables_numericas.mean(),
    "Mediana": variables_numericas.median(),
    "Moda": variables_numericas.mode().iloc[0],
    "Mínimo": variables_numericas.min(),
    "Máximo": variables_numericas.max(),
    "Rango": variables_numericas.max() - variables_numericas.min(),
    "Varianza": variables_numericas.var(),
    "Desviación estándar": variables_numericas.std(),
    "Coeficiente de variación": variables_numericas.std() / variables_numericas.mean(),
    "Q1": variables_numericas.quantile(0.25),
    "Q2": variables_numericas.quantile(0.50),
    "Q3": variables_numericas.quantile(0.75),
    "Rango intercuartílico": variables_numericas.quantile(0.75) - variables_numericas.quantile(0.25),
    "Sesgo": variables_numericas.skew(),
    "Curtosis": variables_numericas.kurtosis()
})

# 4. Mostrar resultados
print("ANÁLISIS ESTADÍSTICO DESCRIPTIVO")
print(estadisticas)

# 5. Guardar tabla en archivo CSV
estadisticas.to_csv("estadisticas_descriptivas_punto4.csv", encoding="utf-8-sig")

print("\nArchivo generado: estadisticas_descriptivas_punto4.csv")
