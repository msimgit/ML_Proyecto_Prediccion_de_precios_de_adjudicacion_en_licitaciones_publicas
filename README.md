# Proyecto ML - Predicción de precios de contratos públicos

Este repositorio forma parte del **segundo Project Break** del Bootcamp de Data Science en [The Bridge](https://www.thebridge.tech/), convocatoria de **febrero de 2025**.

El análisis y desarrollo del modelo ha sido realizado por **Mario Simarro**.

---

## Objetivo del proyecto

Desarrollar un modelo de **Machine Learning supervisado (regresión)** para predecir el precio final de adjudicación de contratos públicos, a partir de los datos disponibles en las licitaciones.

---

## Metodología

El flujo de trabajo se estructura en los siguientes pasos:

1. **Exploración y limpieza** de datos
2. **Tratamiento de fechas faltantes**, valores nulos, outliers y duplicados
3. **Feature Engineering**: descuentos, fechas, hashing para alta cardinalidad, codificaciones
4. **Escalado y transformación** de variables numéricas (Box-Cox, StandardScaler)
5. **Pipeline completo** con `ColumnTransformer` + `Pipeline`
6. **Entrenamiento con GridSearchCV** de 3 modelos:  
   - Logistic Regression  
   - Random Forest Regressor  
   - Gradient Boosting Regressor
7. **Evaluación y comparación** con métricas de regresión
8. **Exportación del mejor modelo entrenado**

---

## Modelos evaluados

| Modelo                         | Tipo                  | Pros                                                                 | Contras                                                             |
|-------------------------------|-----------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| Logistic Regression            | Lineal, interpretable | Simple, rápido, interpretable                                        | Solo relaciones lineales, sensible a outliers y NaNs               |
| Random Forest Regressor       | Ensamble, no lineal   | Robusto, captura relaciones complejas, tolera ruido                  | Más lento, no admite NaNs directamente                              |
| HistGradientBoostingRegressor | Ensamble, no lineal   | Excelente precisión, maneja NaNs y outliers                          | Modelo caja negra, sensible al tunning                             |

---

## Métricas de evaluación

Se han utilizado las siguientes métricas de regresión para evaluar los modelos:

- **R²** (coeficiente de determinación)
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

---

## Estructura del repositorio

```bash
📁 src/
│   ├── data/                  # Datos sin procesar
│   ├── data_sample/           # Subconjuntos de datos
│   ├── img/                   # Imágenes y visualizaciones
│   ├── models/                # Modelos entrenados serializados (.model)
│   ├── notebooks/             # Jupyter notebooks de desarrollo
│   ├── reports/               # Informes finales
│   └── utils/                 # Funciones auxiliares y scripts de preprocesamiento
📄 README.md                   # Este archivo
📄 main.ipynb                  # Notebook principal del proyecto

## Disclaimer

Este proyecto ha sido desarrollado con fines exclusivamente académicos como parte del Bootcamp de TheBridge. En ningún caso pretende confirmar o asegurar hechos pasados ni cuestionar a ninguna persona o entidad.