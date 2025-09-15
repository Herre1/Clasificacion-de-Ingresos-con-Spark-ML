"""
Clasificación de Ingresos con Spark ML

Collin Gonzalez 
Manuel Herrera
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when
import pandas as pd

def crear_sesion_spark():
    """Crear y configurar sesión de Spark"""
    spark = SparkSession.builder \
        .appName("ClasificacionIngresos") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("Sesión de Spark creada exitosamente")
    return spark

def cargar_y_explorar_datos(spark, ruta_archivo):
    """
    Carga el dataset y realiza exploración inicial
    """
    print("\n" + "="*60)
    print("1. CARGA Y EXPLORACIÓN DE DATOS")
    print("="*60)
    
    # Cargar datos
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(ruta_archivo)
    
    print(f"Dataset cargado: {df.count()} registros, {len(df.columns)} columnas")
    
    # Mostrar esquema
    print("\nEsquema del dataset:")
    df.printSchema()
    
    # Mostrar primeros registros
    print("\nPrimeros 10 registros:")
    df.show(10, truncate=False)
    
    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas de variables numéricas:")
    df.select("age", "fnlwgt", "hours_per_week").describe().show()
    
    # Distribución de la variable objetivo
    print("\nDistribución de la variable objetivo:")
    df.groupBy("label").count().show()
    
    # Verificar valores nulos
    print("\nVerificación de valores nulos:")
    for col_name in df.columns:
        null_count = df.filter(col(col_name).isNull()).count()
        print(f"  {col_name}: {null_count} valores nulos")
    
    return df

def crear_pipeline_ml(df):
    """
    Crear pipeline de Machine Learning con preprocesamiento y modelo
    """
    print("\n" + "="*60)
    print("2. PREPROCESAMIENTO DE VARIABLES CATEGÓRICAS")
    print("="*60)
    
    # Columnas categóricas que necesitan transformación
    categorical_cols = ["sex", "workclass", "education"]
    
    # Crear StringIndexers para cada columna categórica
    indexers = []
    for col_name in categorical_cols:
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed")
        indexers.append(indexer)
        print(f"StringIndexer creado para: {col_name}")
    
    # Crear StringIndexer para la variable objetivo
    label_indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
    indexers.append(label_indexer)
    print("StringIndexer creado para: label")
    
    # Crear OneHotEncoders para las variables categóricas (no para label)
    encoders = []
    encoded_cols = []
    for col_name in categorical_cols:
        encoder = OneHotEncoder(inputCol=f"{col_name}_indexed", outputCol=f"{col_name}_encoded")
        encoders.append(encoder)
        encoded_cols.append(f"{col_name}_encoded")
        print(f"OneHotEncoder creado para: {col_name}")
    
    print("\n" + "="*60)
    print("3. ENSAMBLAJE DE CARACTERÍSTICAS")
    print("="*60)
    
    # Columnas numéricas
    numeric_cols = ["age", "fnlwgt", "hours_per_week"]
    
    # Todas las columnas para el vector de características
    feature_cols = numeric_cols + encoded_cols
    
    # VectorAssembler para combinar todas las características
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    print(f"VectorAssembler creado con {len(feature_cols)} características:")
    for col in feature_cols:
        print(f"   - {col}")
    
    print("\n" + "="*60)
    print("4. CONFIGURACIÓN DEL MODELO")
    print("="*60)
    
    # Configurar Logistic Regression
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label_indexed",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0
    )
    print("Logistic Regression configurado")
    print(f"   - Máximo de iteraciones: {lr.getMaxIter()}")
    print(f"   - Parámetro de regularización: {lr.getRegParam()}")
    
    # Crear Pipeline
    pipeline_stages = indexers + encoders + [assembler, lr]
    pipeline = Pipeline(stages=pipeline_stages)
    print(f"\nPipeline creado con {len(pipeline_stages)} etapas")
    
    return pipeline

def entrenar_modelo(pipeline, df):
    """
    Entrenar el modelo usando todo el dataset
    """
    print("\n" + "="*60)
    print("5. ENTRENAMIENTO DEL MODELO")
    print("="*60)
    
    print("Iniciando entrenamiento del modelo...")
    modelo = pipeline.fit(df)
    print("Modelo entrenado exitosamente")
    
    return modelo

def evaluar_modelo(modelo, df):
    """
    Evaluar el modelo y mostrar predicciones
    """
    print("\n" + "="*60)
    print("6. EVALUACIÓN DEL MODELO")
    print("="*60)
    
    # Hacer predicciones
    predicciones = modelo.transform(df)
    
    # Mostrar predicciones con probabilidades
    print("\nPredicciones del modelo (primeras 20):")
    predicciones.select(
        "age", "sex", "workclass", "education", "hours_per_week",
        "label", "prediction", "probability"
    ).show(20, truncate=False)
    
    # Evaluación con métricas
    evaluator_binary = BinaryClassificationEvaluator(
        labelCol="label_indexed",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    evaluator_multiclass = MulticlassClassificationEvaluator(
        labelCol="label_indexed",
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    auc = evaluator_binary.evaluate(predicciones)
    accuracy = evaluator_multiclass.evaluate(predicciones)
    
    print(f"\nMétricas de evaluación:")
    print(f"   - AUC (Area Under ROC): {auc:.4f}")
    print(f"   - Exactitud (Accuracy): {accuracy:.4f}")
    
    # Matriz de confusión manual
    print(f"\nMatriz de confusión:")
    confusion_matrix = predicciones.groupBy("label_indexed", "prediction").count().orderBy("label_indexed", "prediction")
    confusion_matrix.show()
    
    # Estadísticas por clase
    print(f"\nDistribución de predicciones:")
    predicciones.groupBy("prediction").count().show()
    
    return predicciones

def crear_datos_nuevos(spark):
    """
    Crear DataFrame con 9 registros nuevos para predicción
    """
    print("\n" + "="*60)
    print("7. PREDICCIÓN CON DATOS NUEVOS")
    print("="*60)
    
    # Crear 9 registros nuevos con diferentes perfiles
    nuevos_datos = [
        # Perfil 1: Joven profesional con educación universitaria
        (28, "Male", "Private", 150000, "Bachelors", 45),
        
        # Perfil 2: Mujer de mediana edad con maestría
        (42, "Female", "Gov", 180000, "Masters", 40),
        
        # Perfil 3: Hombre mayor con poca educación
        (65, "Male", "Self-emp", 120000, "HS-grad", 30),
        
        # Perfil 4: Mujer joven estudiante
        (22, "Female", "Private", 90000, "Some-college", 20),
        
        # Perfil 5: Profesional con maestría avanzada
        (45, "Male", "Gov", 200000, "Masters", 50),
        
        # Perfil 6: Trabajador de mediana edad sin educación superior
        (38, "Female", "Private", 160000, "11th", 35),
        
        # Perfil 7: Emprendedor exitoso
        (50, "Male", "Self-emp", 250000, "Bachelors", 60),
        
        # Perfil 8: Trabajadora gubernamental con experiencia
        (55, "Female", "Gov", 190000, "Masters", 42),
        
        # Perfil 9: Joven trabajador de tiempo parcial
        (25, "Male", "Private", 110000, "HS-grad", 25)
    ]
    
    # Definir esquema
    schema = StructType([
        StructField("age", IntegerType(), True),
        StructField("sex", StringType(), True),
        StructField("workclass", StringType(), True),
        StructField("fnlwgt", IntegerType(), True),
        StructField("education", StringType(), True),
        StructField("hours_per_week", IntegerType(), True)
    ])
    
    # Crear DataFrame
    df_nuevos = spark.createDataFrame(nuevos_datos, schema)
    
    print("Datos nuevos creados:")
    # Alternativa más robusta para mostrar datos
    try:
        df_nuevos.show(truncate=False)
    except Exception as e:
        print(f"Error al mostrar DataFrame (problema conocido en Windows): {type(e).__name__}")
        print(f"DataFrame creado exitosamente con {len(nuevos_datos)} registros nuevos")
        # Mostrar datos manualmente
        print("\nRegistros creados:")
        for i, record in enumerate(nuevos_datos, 1):
            print(f"  {i}. Edad: {record[0]}, Género: {record[1]}, Trabajo: {record[2]}, Educación: {record[4]}, Horas: {record[5]}")
        print()
    
    return df_nuevos

def predecir_nuevos_datos(modelo, df_nuevos):
    """
    Aplicar modelo a datos nuevos y mostrar predicciones
    """
    print("\nAplicando modelo a datos nuevos...")
    
    # Hacer predicciones
    predicciones_nuevas = modelo.transform(df_nuevos)
    
    # Crear columna con interpretación textual
    predicciones_con_texto = predicciones_nuevas.withColumn(
        "prediccion_texto",
        when(col("prediction") == 1.0, ">50K").otherwise("<=50K")
    )
    
    print("\nPredicciones para datos nuevos:")
    predicciones_con_texto.select(
        "age", "sex", "workclass", "education", "hours_per_week",
        "prediccion_texto", "probability"
    ).show(truncate=False)
    
    # Análisis de resultados
    print("\nResumen de predicciones:")
    resumen = predicciones_con_texto.groupBy("prediccion_texto").count()
    resumen.show()
    
    return predicciones_con_texto

def main():
    """
    Función principal que ejecuta todo el flujo
    """
    print("INICIANDO PROYECTO: CLASIFICACIÓN DE INGRESOS CON SPARK ML")
    print("="*70)
    
    # Crear sesión Spark
    spark = crear_sesion_spark()
    
    try:
        # 1. Cargar y explorar datos
        df = cargar_y_explorar_datos(spark, "Data/adult_income_sample.csv")
        
        # 2. Crear pipeline
        pipeline = crear_pipeline_ml(df)
        
        # 3. Entrenar modelo
        modelo = entrenar_modelo(pipeline, df)
        
        # 4. Evaluar modelo
        predicciones_entrenamiento = evaluar_modelo(modelo, df)
        
        # 5. Crear datos nuevos
        df_nuevos = crear_datos_nuevos(spark)
        
        # 6. Predecir datos nuevos
        predicciones_nuevas = predecir_nuevos_datos(modelo, df_nuevos)
        
        # 7. Reflexión
        # en documento
        
        print("\nPROYECTO COMPLETADO EXITOSAMENTE!")
        
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        raise
    finally:
        # Cerrar sesión Spark
        spark.stop()
        print("Sesión de Spark cerrada")

if __name__ == "__main__":
    main()

