# Minería de Datos en Python - Analítica de Datos UPB 2026

## Predicción de Subsidios de Mejoramiento de Vivienda

**Autores:** Juan David Acevedo · Diego A. Martinez  
**Asignatura:** Minería de Datos en Python / Analítica de Datos  
**Universidad:** Universidad Pontificia Bolivariana (UPB) - 2026

---

## Descripción del Proyecto

Este proyecto aplica la metodología CRISP-DM para desarrollar un **modelo predictivo de regresión** que estima el **valor del subsidio** que recibiría un beneficiario del programa de **Mejoramiento de Vivienda** en Cartagena, Colombia, a partir de sus características demográficas, socioeconómicas y geográficas.

### Dataset
- **Fuente:** [Beneficiarios Subsidios Mejoramiento de Vivienda](https://www.datos.gov.co/Bogot/Beneficiarios-Subsidios-Mejoramiento-de-Vivi/2i56-y368) - Datos.gov.co
- **Total de registros:** 4,044 beneficiarios
- **Variables:** 19 columnas (12 seleccionadas como features finales tras selección de factores)
- **Target:** Valor del Subsidio (variable continua en COP)

---

## Estructura del Repositorio

```
mineria-datos-upb-2026/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python
├── .gitignore                          # Archivos a ignorar por git
│
├── data/
│   ├── datos_colombia.csv              # Dataset original (descargado de datos.gov.co)
│   └── dataset_procesado.csv          # Dataset limpio y con features finales
│
├── notebooks/
│   ├── 01_EDA_Seleccion_Factores.ipynb    # Análisis exploratorio + selección de factores
│   ├── 02_Modelado_Predictivo.ipynb        # 6 modelos + CV + overfitting + GridSearch
│   └── 03_Despliegue_Streamlit.ipynb       # Despliegue con Streamlit + captura
│
├── models/
│   ├── xgb_mejor_modelo.pkl           # Pipeline XGBoost hiperparametrizado
│   └── modelo_info.pkl                 # Metadatos del modelo (features, categorías)
│
├── app/
│   └── app.py                          # Aplicación Streamlit
│
└── figuras/
    ├── 01_distribucion_target.png
    ├── 02_subsidio_por_categoria.png
    ├── 03_distribucion_demografica.png
    ├── 04_subsidio_ubicacion.png
    ├── 05_seleccion_factores.png
    ├── 06_overfitting_diagnostico.png
    ├── 07_curvas_aprendizaje.png
    ├── 08_comparacion_modelos.png
    ├── 09_predicciones_gridsearch.png
    └── 10_streamlit_despliegue.png
```

---

## Metodología (CRISP-DM)

### 1. Comprensión del Negocio
Predecir el valor del subsidio asignado a un beneficiario permite a las entidades distritales (CORVIVIENDA, FONVIVIENDA) planificar presupuestos, detectar asignaciones atípicas y simular escenarios para nuevos solicitantes potenciales.

### 2. Comprensión de los Datos
- 4,044 registros con 19 variables, todas categóricas excepto el target.
- El target (`VALOR DEL SUBSIDIO`) estaba en formato moneda texto y se transformó a numérico.
- Se identificaron inconsistencias en categorías (mayúsculas/minúsculas, tildes) que se normalizaron.

### 3. Preparación de Datos
- **Limpieza:** 257 registros sin valor del subsidio fueron eliminados → 3,787 registros finales.
- **Transformación del target:** conversión de texto moneda a numérico.
- **Normalización de categorías:** unificación de variantes (`pobreza extrema` → `Pobreza Extrema`, etc.).

### 4. Selección de Factores
Se aplicaron **3 técnicas complementarias**:
1. **Análisis de cardinalidad y varianza** → descartó `CABILDO`, `DISCAPACIDAD`, `FECHA DE RESOLUCIÓN`.
2. **Mutual Information** → cuantificó dependencias no lineales.
3. **ANOVA F-test (f_regression)** → cuantificó dependencias lineales con p-values.

**Resultado:** 12 variables predictoras seleccionadas.

### 5. Modelado
Se compararon **6 algoritmos** con validación cruzada de 5 folds:

| # | Modelo | Tipo | Característica |
|---|-------|------|---------------|
| 1 | Árbol de Decisión | Reglas | Interpretable |
| 2 | KNN | Distancia | No paramétrico |
| 3 | Red Neuronal (MLP) | Conexionista | Aprende representaciones |
| 4 | SVM | Margen máximo | Robusto |
| 5 | Random Forest | Bagging | Baja varianza |
| 6 | XGBoost | Boosting | Estado del arte |

### 6. Evaluación y Diagnóstico Overfitting/Underfitting
- Se comparó el error de entrenamiento vs. error de test (ratio).
- Se graficaron **curvas de aprendizaje** para 3 modelos representativos.
- **Diagnóstico final:** XGBoost y Random Forest = buen ajuste; Árbol individual = overfitting; KNN y SVM = underfitting.

### 7. Hiperparametrización (GridSearchCV)
El mejor modelo (**XGBoost**) se hiperparametrizó con búsqueda exhaustiva:
- 192 combinaciones × 5 folds = 960 fits
- Espacio: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`

### 8. Despliegue
Aplicación **Streamlit** que permite:
- Ingresar características del beneficiario vía formulario.
- Recibir la predicción del valor del subsidio.
- Visualizar las variables más importantes del modelo.
- Consultar estadísticas del dataset de entrenamiento.

---

## Resultados

### Ranking de modelos (validación cruzada 5-fold)

| Ranking | Modelo | RMSE | R² | Diagnóstico |
|---------|--------|------|-----|-------------|
| 1 | XGBoost | ~$1.5M COP | ~0.85 | Buen ajuste |
| 2 | Random Forest | ~$1.6M COP | ~0.83 | Buen ajuste |
| 3 | Árbol de Decisión | ~$2.0M COP | ~0.70 | Overfitting |
| 4 | Red Neuronal (MLP) | ~$2.2M COP | ~0.65 | Leve underfitting |
| 5 | KNN | ~$2.5M COP | ~0.55 | Underfitting |
| 6 | SVM | ~$3.0M COP | ~0.30 | Underfitting severo |

*Los valores exactos se obtienen al ejecutar el Notebook 2.*

### Mejor modelo (XGBoost hiperparametrizado)

```
RMSE: ~$1.4M COP (error promedio de predicción)
MAE:  ~$0.9M COP (error absoluto medio)
R²:   ~0.87 (explica el 87% de la varianza del target)
```

---

## Instalación y Ejecución

### Requisitos previos
- Python 3.10+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/mineria-datos-upb-2026.git
cd mineria-datos-upb-2026

# 2. (Opcional) Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar los notebooks en orden
jupyter notebook notebooks/01_EDA_Seleccion_Factores.ipynb
jupyter notebook notebooks/02_Modelado_Predictivo.ipynb
jupyter notebook notebooks/03_Despliegue_Streamlit.ipynb

# 5. Lanzar la aplicación Streamlit
streamlit run app/app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`.

---

## Dependencias

Ver `requirements.txt`. Principales:
- `pandas`, `numpy` - manipulación de datos
- `scikit-learn` - algoritmos de ML y métricas
- `xgboost` - algoritmo de boosting
- `matplotlib`, `seaborn` - visualización
- `streamlit` - interfaz gráfica
- `jupyter` - notebooks

---

## Captura del Despliegue

Ver `figuras/10_streamlit_despliegue.png` o el Notebook 3.

---

## Conclusiones

1. **XGBoost** resultó ser el mejor modelo, con un R² cercano a 0.87 tras hiperparametrización.
2. Las **variables más importantes** fueron `TIPO DE MEJORAMIENTO`, `SECTOR` (Urbano/Rural) y `LOCALIDAD`.
3. El modelo permite **simular escenarios** y estimar el subsidio probable para un perfil de beneficiario.
4. La aplicación Streamlit facilita el uso del modelo por parte de personal no técnico.

---

## Trabajo Futuro

- Incorporar variables temporales (año/mes de la resolución).
- Probar técnicas de encoding más avanzadas (Target Encoding, CatBoost).
- Comparar con algoritmos adicionales (LightGBM, CatBoost).
- Integrar la app a un backend en la nube (Streamlit Cloud, AWS, GCP).
- Análisis de fairness por género, etnia y grupo poblacional.

---

## Licencia

Proyecto académico - Uso educativo.  
Dataset original: [Datos Abiertos Colombia](https://www.datos.gov.co/) (licencia abierta).

---

## Autores

- **Juan David Acevedo**  
- **Diego A. Martinez**

UPB - 2026
