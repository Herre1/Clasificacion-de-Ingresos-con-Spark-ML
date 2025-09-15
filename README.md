# Clasificación de Ingresos con Spark ML

## Descripción del Proyecto

Este proyecto implementa un modelo de clasificación binaria utilizando Apache Spark ML para predecir si una persona gana más de 50K al año basándose en características demográficas y laborales.

## Objetivo

Construir un modelo de **Logistic Regression** con Spark ML que pueda predecir si una persona pertenece a la clase `>50K` o `<=50K` de ingresos anuales.

## Dataset

- **Archivo**: `Data/adult_income_sample.csv`
- **Registros**: 2000 registros simulados
- **Características**:
  - `age`: Edad de la persona (años)
  - `sex`: Género (Male, Female)
  - `workclass`: Tipo de empleo (Private, Self-emp, Gov)
  - `fnlwgt`: Peso estadístico asociado al registro
  - `education`: Nivel educativo (Bachelors, HS-grad, 11th, Masters, etc.)
  - `hours_per_week`: Horas trabajadas por semana
- **Variable objetivo**: `label` (>50K o <=50K)

## Tecnologías Utilizadas

- **Apache Spark**: Framework de procesamiento distribuido
- **PySpark**: API de Python para Spark
- **Spark ML**: Librería de Machine Learning de Spark
- **Python**: Lenguaje de programación principal

## Estructura del Proyecto

```
Clasificacion-de-Ingresos-con-Spark-ML/
├── Data/
│   └── adult_income_sample.csv          # Dataset principal
├── clasificacion_ingresos_spark_ml.py   # Script principal
├── README.md                            # Este archivo
└── requirements.txt                     # Dependencias del proyecto
```

## Instalación y Configuración

### 1. Requisitos Previos

- Python 3.7 o superior
- Java 8 o 11 (requerido por Spark)
- pip (gestor de paquetes de Python)

### 2. Instalación de Dependencias

```bash
# Crear entorno virtual (recomendado)
python -m venv spark_ml_env
source spark_ml_env/bin/activate  # En Windows: spark_ml_env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Verificar Instalación de Spark

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Test").getOrCreate()
print(f"Spark Version: {spark.version}")
spark.stop()
```

## Ejecución del Proyecto

### Opción 1: Script de Python

```bash
python clasificacion_ingresos_spark_ml.py
```

## Flujo del Proyecto

### 1. **Carga de Datos**
- Lectura del archivo CSV en un DataFrame de Spark
- Inspección del esquema y exploración inicial

### 2. **Preprocesamiento**
- **StringIndexer**: Transformación de variables categóricas a índices numéricos
- **OneHotEncoder**: Conversión a vectores binarios para evitar interpretaciones de orden

### 3. **Ensamblaje de Características**
- **VectorAssembler**: Combinación de variables numéricas y categóricas codificadas

### 4. **Modelo de Machine Learning**
- **Logistic Regression**: Algoritmo de clasificación binaria
- **Pipeline**: Encadenamiento de todas las transformaciones y entrenamiento

### 5. **Evaluación**
- Métricas de evaluación (AUC, Accuracy)
- Matriz de confusión
- Análisis de predicciones

### 6. **Predicción con Datos Nuevos**
- Creación de 9 registros de prueba
- Aplicación del modelo entrenado
- Interpretación de resultados

## Resultados Esperados

El modelo proporciona:
- **Predicciones binarias**: >50K o <=50K
- **Probabilidades**: Confianza de cada predicción
- **Métricas de evaluación**: AUC y Accuracy
- **Matriz de confusión**: Análisis de aciertos y errores

## Componentes Técnicos

### Clases de Spark ML Utilizadas
- `StringIndexer`: Indexación de variables categóricas
- `OneHotEncoder`: Codificación one-hot
- `VectorAssembler`: Ensamblaje de vectores de características
- `LogisticRegression`: Modelo de regresión logística
- `Pipeline`: Flujo de trabajo de ML
- `BinaryClassificationEvaluator`: Evaluación de modelos binarios

### Métodos Principales
- `.fit()`: Entrenamiento del modelo
- `.transform()`: Aplicación de transformaciones y predicciones
- `.show()`: Visualización de resultados

## Características del Modelo

- **Algoritmo**: Regresión Logística
- **Tipo**: Clasificación Binaria Supervisada
- **Características**: 6 variables (3 numéricas + 3 categóricas codificadas)
- **Regularización**: L2 (Ridge) con parámetro 0.01
- **Iteraciones máximas**: 100

## Ejemplos de Predicción

El modelo predice ingresos altos (>50K) para perfiles como:
- Profesionales con educación superior (Bachelors, Masters, Doctorate)
- Personas de mediana edad con experiencia
- Trabajadores con muchas horas semanales
- Empleados en ciertos sectores (gobierno, empresa propia)

## Limitaciones

- **Datos simulados**: Los resultados pueden no reflejar patrones reales
- **Sin validación cruzada**: Posible sobreajuste al dataset
- **Muestra pequeña**: 2000 registros pueden ser insuficientes para generalización
- **Variables limitadas**: Faltan características importantes como experiencia, ubicación, etc.

## Objetivos de Aprendizaje

Este proyecto enseña:
- Uso de Apache Spark para Machine Learning
- Preprocesamiento de datos categóricos
- Implementación de pipelines de ML
- Evaluación de modelos de clasificación
- Interpretación de resultados de ML

---

**Proyecto académico - Clasificación de Ingresos con Spark ML**

