# 🫀 Predicción de Riesgo de Ataque Cardíaco

Proyecto de Machine Learning para clasificar pacientes según su probabilidad de sufrir un ataque cardíaco, a partir de indicadores médicos y fisiológicos.

---

## 📋 Descripción

El modelo analiza variables clínicas de cada paciente y predice si pertenece a la categoría de **bajo riesgo (0)** o **alto riesgo (1)** de ataque cardíaco. Se comparan tres algoritmos de clasificación con ajuste de hiperparámetros, priorizando el **Recall** como métrica principal dado el contexto médico del problema.

---

## 📁 Estructura del proyecto

```
├── heart_attack.ipynb   # Notebook principal con todo el análisis
├── heart_attack.csv     # Dataset de entrada
└── README.md
```

---

## 📊 Dataset

Fuente: `heart_attack.csv`  
Tamaño tras limpieza: **294 pacientes × 10 variables**  
Variable objetivo: `num` — 0 = bajo riesgo · 1 = alto riesgo

| Variable | Descripción |
|----------|-------------|
| `age` | Edad del paciente |
| `sex` | Género (1 = hombre, 0 = mujer) |
| `cp` | Tipo de dolor en el pecho (0–3) |
| `trestbps` | Presión arterial en reposo (mm Hg) |
| `chol` | Colesterol sérico (mg/dl) |
| `fbs` | Azúcar en sangre en ayunas (1 si > 120 mg/dl) |
| `restecg` | Resultados electrocardiográficos (0–2) |
| `thalach` | Frecuencia cardíaca máxima alcanzada |
| `exang` | Angina inducida por ejercicio (1 = sí) |
| `oldpeak` | Depresión del segmento ST en ejercicio |

> **Nota:** Las columnas `ca`, `thal` y `slope` fueron eliminadas por superar el 40% de valores nulos.

---

## ⚙️ Pipeline del proyecto

```
Carga de datos
     │
     ▼
Limpieza y tratamiento de nulos
  · Eliminación de columnas con > 40% de nulos
  · Imputación KNN (k=5) para el resto
     │
     ▼
Análisis Exploratorio (EDA)
  · Balance de clases
  · Distribuciones KDE por clase
  · Variables categóricas
  · Mapa de correlación
     │
     ▼
Preparación del modelado
  · Split estratificado 80/20
  · StratifiedKFold (5 folds)
     │
     ▼
Entrenamiento con GridSearchCV
  · k-NN (en Pipeline con StandardScaler)
  · Random Forest
  · Gradient Boosting
     │
     ▼
Evaluación en test
  · Classification report por modelo
  · Tabla comparativa de métricas
```

---

## 🏆 Resultados

| Modelo | ROC-AUC | Accuracy | Recall (alto riesgo) | Precision | F1 |
|--------|:-------:|:--------:|:--------------------:|:---------:|:--:|
| **k-NN** | **0.899** | **88%** | **76%** | — | — |
| Random Forest | 0.887 | 76% | 52% | — | — |
| Gradient Boosting | 0.865 | 83% | 67% | — | — |

**Modelo ganador: k-NN con k=25 y pesos por distancia.**

Aunque es el modelo más simple de los tres, obtiene el mejor ROC-AUC y Accuracy. Con un dataset de ~294 filas, los modelos más complejos tienden a sobreajustarse, lo que explica el rendimiento inferior de Random Forest y Gradient Boosting en test.

---

## 💡 Hallazgos del EDA

- **`cp` (tipo de dolor)** es el predictor individual más fuerte; su distribución por clase es la más diferenciada.
- **`exang` y `oldpeak`** también muestran alta correlación con el riesgo.
- **`oldpeak = 0`** concentra la mayoría de los pacientes de bajo riesgo.
- **`thalach` bajo** (ritmo cardíaco máximo reducido) es paradójicamente más alarmante.
- **`fbs`** (azúcar en ayunas) y `restecg` tienen escaso poder predictivo.
- El dataset presenta un desbalance leve: 188 bajo riesgo vs. 106 alto riesgo.

---

## 🔧 Instalación y uso

### Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Ejecución

1. Clona el repositorio y asegúrate de tener `heart_attack.csv` en la misma carpeta que el notebook.
2. Abre `heart_attack.ipynb` en Jupyter Lab, Jupyter Notebook o VS Code.
3. Ejecuta todas las celdas en orden (`Run All`).

---

## ⚠️ Consideraciones médicas

En un entorno clínico real, minimizar los **falsos negativos** (pacientes de alto riesgo clasificados como sanos) es prioritario sobre la precisión global. Se recomienda:

- Ajustar el **umbral de decisión** del k-NN por debajo de 0.5 para aumentar el Recall a costa de algo de Precision.
- Validar con un dataset externo antes de cualquier uso asistencial.
- Este proyecto tiene carácter **educativo y exploratorio**, no constituye herramienta de diagnóstico médico.

---

## 🛠️ Tecnologías

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-teal)
