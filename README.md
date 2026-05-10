# 👗 Clasificación de Ropa con Fashion MNIST

Proyecto de Deep Learning para reconocer prendas de ropa a partir de imágenes en escala de grises, usando una Red Neuronal Convolucional (CNN) entrenada sobre el dataset Fashion MNIST de Zalando.

---

## 📋 Descripción

El modelo recibe imágenes de 28×28 píxeles en escala de grises y las clasifica en una de las **10 categorías de prendas** del dataset. Se parte de una arquitectura densa básica (`Flatten → Dense → Dense`) y se evoluciona hacia una CNN con regularización, data augmentation y callbacks adaptativos.

---

## 📁 Estructura del proyecto

```
├── clasi_ropa.ipynb     # Notebook principal con todo el desarrollo
├── mejor_modelo.keras   # Mejor modelo guardado por ModelCheckpoint
└── README.md
```

> El dataset se descarga automáticamente desde `tensorflow_datasets` al ejecutar el notebook, no requiere archivo CSV externo.

---

## 🏷️ Categorías del dataset

| ID | Clase | ID | Clase |
|----|-------|----|-------|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

**70 000 imágenes en total** — 60 000 entrenamiento · 10 000 prueba  
Resolución: 28×28 px · Escala de grises · 10 clases balanceadas

---

## ⚙️ Pipeline del proyecto

```
Descarga automática (tensorflow_datasets)
         │
         ▼
Carga y exploración
  · Visualización de 25 muestras
  · Estadísticas del dataset
         │
         ▼
Preprocesamiento
  · Normalización: [0, 255] → [0.0, 1.0]
  · Data Augmentation (solo entrenamiento):
      - Espejo horizontal aleatorio
      - Variación de brillo (±0.1)
      - Variación de contraste (0.9–1.1)
  · Pipeline optimizado con .cache() + .prefetch(AUTOTUNE)
         │
         ▼
Arquitectura CNN
  · Bloque Conv 1: Conv2D(32) × 2 + BN + ReLU + MaxPool + Dropout(0.25)
  · Bloque Conv 2: Conv2D(64) × 2 + BN + ReLU + MaxPool + Dropout(0.25)
  · Clasificador:  Flatten + Dense(256) + BN + Dropout(0.5) + Softmax(10)
         │
         ▼
Entrenamiento
  · Optimizer: Adam (lr=1e-3)
  · Loss: SparseCategoricalCrossentropy
  · Epochs: hasta 10 (con EarlyStopping)
         │
         ▼
Evaluación
  · Curvas de pérdida y accuracy (train vs. validación)
  · Accuracy final en test
  · Matriz de confusión
```

---

## 🏗️ Arquitectura del modelo

```
Input (28, 28, 1)
       │
┌──────▼──────────────────────────┐
│  Conv2D(32, 3×3) → BN → ReLU   │  Bloque
│  Conv2D(32, 3×3) → BN → ReLU   │  Convolucional 1
│  MaxPooling2D(2×2)              │
│  Dropout(0.25)                  │
└──────┬──────────────────────────┘
       │
┌──────▼──────────────────────────┐
│  Conv2D(64, 3×3) → BN → ReLU   │  Bloque
│  Conv2D(64, 3×3) → BN → ReLU   │  Convolucional 2
│  MaxPooling2D(2×2)              │
│  Dropout(0.25)                  │
└──────┬──────────────────────────┘
       │
┌──────▼──────────────────────────┐
│  Flatten                        │
│  Dense(256) → BN                │  Clasificador
│  Dropout(0.5)                   │
│  Dense(10, softmax)             │
└─────────────────────────────────┘
```

**BatchNormalization** estabiliza el entrenamiento. **Dropout** previene el sobreajuste.

---

## ⚡ Callbacks

| Callback | Configuración | Función |
|----------|--------------|---------|
| `EarlyStopping` | `patience=5`, `restore_best_weights=True` | Detiene si `val_loss` no mejora |
| `ReduceLROnPlateau` | `factor=0.5`, `patience=3`, `min_lr=1e-6` | Reduce el LR a la mitad al estancarse |
| `ModelCheckpoint` | `monitor=val_accuracy`, `save_best_only=True` | Guarda el mejor modelo en disco |

---

## 🔧 Instalación y uso

### Requisitos

```bash
pip install tensorflow tensorflow-datasets matplotlib seaborn scikit-learn numpy
```

### Ejecución

1. Clona el repositorio.
2. Abre `clasi_ropa.ipynb` en Jupyter Lab, Jupyter Notebook o VS Code.
3. Ejecuta todas las celdas en orden (`Run All`). El dataset se descarga automáticamente.

> **GPU recomendada.** El entrenamiento es funcional en CPU pero significativamente más lento. Consulta la sección de configuración de GPU si usas una RTX local.

---

## 📈 Resultados esperados

Con la arquitectura CNN descrita y 10 épocas de entrenamiento:

| Métrica | Valor aproximado |
|---------|:----------------:|
| Accuracy en test | ~93–94% |
| Loss en test | ~0.18–0.22 |

Las clases con más confusiones suelen ser **Shirt ↔ T-shirt/top** y **Coat ↔ Pullover**, visible en la matriz de confusión.

---

## 🛠️ Tecnologías

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-teal)
