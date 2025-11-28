# 📊 Análisis de Riesgo Crediticio con Machine Learning

Proyecto completo de análisis exploratorio de datos (EDA) y modelado de machine learning para la predicción de riesgo crediticio. El sistema utiliza algoritmos de aprendizaje supervisado para predecir la probabilidad de impago de préstamos basándose en características del solicitante y del préstamo.

## 🎯 Descripción del Proyecto

Este repositorio contiene un análisis exhaustivo de un dataset de riesgo crediticio que incluye:

- **Análisis Exploratorio de Datos (EDA)**: Exploración detallada de 32,581 registros con 12 características
- **Modelado de Machine Learning**: Implementación y comparación de múltiples algoritmos
- **Modelo de Producción**: Random Forest optimizado con 93.08% de precisión
- **Script de Uso**: Herramienta lista para realizar predicciones en nuevos datos

## 📁 Estructura del Repositorio

```
.
├── README.md                                    # Este archivo
├── analisis_credit_risk.ipynb                  # Notebook principal: EDA y modelado
├── modelo_para_usar.ipynb                      # Script para usar el modelo entrenado
├── credit_risk_dataset.csv                     # Dataset de riesgo crediticio
├── modelo_entrenado_guardado.zip               # Modelo guardado (backup)
└── modelos/
    └── modelo_riesgo_credito_random_forest.pkl # Modelo entrenado (Random Forest)
```

## 📊 Dataset

### Características del Dataset

- **Total de registros**: 32,581
- **Total de columnas**: 12
- **Variables numéricas**: 8
- **Variables categóricas**: 4
- **Tamaño**: ~9.62 MB

### Variables del Dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `person_age` | Numérica | Edad de la persona |
| `person_income` | Numérica | Ingreso anual de la persona |
| `person_home_ownership` | Categórica | Tipo de propiedad de vivienda (RENT, MORTGAGE, OWN, OTHER) |
| `person_emp_length` | Numérica | Años de empleo |
| `loan_intent` | Categórica | Propósito del préstamo (EDUCATION, MEDICAL, VENTURE, etc.) |
| `loan_grade` | Categórica | Grado/calificación del préstamo (A-G) |
| `loan_amnt` | Numérica | Monto del préstamo |
| `loan_int_rate` | Numérica | Tasa de interés del préstamo |
| `loan_status` | Numérica | **Variable objetivo**: Estado del préstamo (0=Pagado, 1=Impago) |
| `loan_percent_income` | Numérica | Porcentaje del préstamo respecto al ingreso |
| `cb_person_default_on_file` | Categórica | Historial de impago (Y/N) |
| `cb_person_cred_hist_length` | Numérica | Años de historial crediticio |

### Distribución de la Variable Objetivo

- **Pagado (0)**: 25,473 registros (78.18%)
- **Impago (1)**: 7,108 registros (21.82%)

## 🚀 Requisitos e Instalación

### Requisitos del Sistema

- Python 3.7 o superior
- Jupyter Notebook o JupyterLab

### Instalación de Dependencias

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

O instala todas las dependencias desde un archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dependencias Principales

- `pandas`: Manipulación y análisis de datos
- `numpy`: Operaciones numéricas
- `scikit-learn`: Algoritmos de machine learning
- `matplotlib`: Visualización de datos
- `seaborn`: Visualizaciones estadísticas avanzadas
- `joblib`: Guardado y carga de modelos

## 📖 Uso del Proyecto

### 1. Análisis Exploratorio y Modelado

Abre el notebook `analisis_credit_risk.ipynb` para:

- Explorar el dataset completo
- Realizar análisis estadísticos
- Visualizar distribuciones y relaciones
- Entrenar y comparar modelos de machine learning
- Evaluar el rendimiento de los modelos

```bash
jupyter notebook analisis_credit_risk.ipynb
```

### 2. Usar el Modelo Entrenado

Para realizar predicciones con el modelo entrenado, utiliza el notebook `modelo_para_usar.ipynb`:

```bash
jupyter notebook modelo_para_usar.ipynb
```

#### Ejemplo de Uso en Python

```python
import joblib
import pandas as pd

# Cargar el modelo
modelo = joblib.load('modelos/modelo_riesgo_credito_random_forest.pkl')

# Datos de un nuevo cliente
cliente = {
    'person_age': 25,
    'person_income': 50000,
    'person_home_ownership': 'RENT',
    'person_emp_length': 3.0,
    'loan_intent': 'EDUCATION',
    'loan_grade': 'B',
    'loan_amnt': 10000,
    'loan_int_rate': 10.5,
    'loan_percent_income': 0.20,
    'cb_person_default_on_file': 'N',
    'cb_person_cred_hist_length': 5
}

# Realizar predicción
datos = pd.DataFrame([cliente])
# ... (procesar datos según el pipeline del modelo)
prediccion = modelo['modelo'].predict(datos_procesados)
probabilidad = modelo['modelo'].predict_proba(datos_procesados)[:, 1]

print(f"Predicción: {'Impago' if prediccion[0] == 1 else 'Pagado'}")
print(f"Probabilidad de impago: {probabilidad[0]:.2%}")
```

## 🤖 Modelos Implementados

Se entrenaron y compararon tres algoritmos de machine learning:

1. **Regresión Logística**
   - Accuracy: 78.36%
   - ROC-AUC: 0.8530

2. **Random Forest** ⭐ (Modelo Seleccionado)
   - Accuracy: 93.08%
   - Precision: 97.27%
   - Recall: 70.25%
   - F1-Score: 0.8158
   - ROC-AUC: 0.9284

3. **Gradient Boosting**
   - Accuracy: 92.22%
   - Precision: 94.12%
   - ROC-AUC: 0.9254

### Modelo Final: Random Forest

El modelo **Random Forest** fue seleccionado como el mejor modelo basándose en el F1-Score, que balancea precisión y recall. Este modelo:

- Detecta correctamente el **93.08%** de los casos
- De los préstamos predichos como impago, el **97.27%** realmente fueron impago
- Detecta el **70.25%** de los impagos reales

### Importancia de Características

Las características más importantes según el modelo Random Forest son:

1. `loan_percent_income` (21.04%)
2. `person_income` (16.80%)
3. `loan_int_rate` (13.79%)
4. `loan_grade` (11.80%)
5. `loan_amnt` (7.86%)

## 📈 Métricas de Evaluación

El modelo fue evaluado usando las siguientes métricas:

- **Accuracy**: Porcentaje de predicciones correctas
- **Precision**: De los préstamos predichos como impago, cuántos realmente fueron impago
- **Recall**: De los préstamos que realmente fueron impago, cuántos fueron detectados
- **F1-Score**: Media armónica entre Precision y Recall
- **ROC-AUC**: Capacidad del modelo para distinguir entre clases

## 🔧 Preprocesamiento de Datos

El pipeline de preprocesamiento incluye:

1. **Manejo de valores nulos**: Imputación con mediana para variables numéricas
2. **Encoding categórico**: Label Encoding para variables categóricas
3. **Estandarización**: Escalado de variables numéricas (cuando es necesario)
4. **División de datos**: 80% entrenamiento, 20% prueba (con estratificación)

## 📝 Notas Importantes

- El modelo fue entrenado con `class_weight='balanced'` para manejar el desbalance de clases
- Se utilizó validación cruzada y división estratificada para mantener la proporción de clases
- El modelo guardado incluye todos los componentes necesarios para realizar predicciones (encoders, scaler, valores de imputación)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👤 Autor

**Martin**
