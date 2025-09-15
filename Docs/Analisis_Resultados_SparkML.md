# Análisis de resultados:

---

## 1. Evaluación del Modelo

### Métricas de Rendimiento
| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **AUC (Area Under ROC)** | 0.5456 | Capacidad discriminativa limitada |
| **Exactitud (Accuracy)** | 0.5280 | 52.8% de predicciones correctas |

### Matriz de Confusión
|  | Pred: <=50K | Pred: >50K | Total |
|--|-------------|------------|-------|
| **Real: <=50K** | 597 | 429 | 1,026 |
| **Real: >50K** | 515 | 459 | 974 |
| **Total** | 1,112 | 888 | 2,000 |

### Métricas Calculadas
- **Sensibilidad (Recall para >50K)**: 459/974 = 47.1%
- **Especificidad (Recall para <=50K)**: 597/1,026 = 58.2%
- **Precisión para >50K**: 459/888 = 51.7%
- **Precisión para <=50K**: 597/1,112 = 53.7%

---

## 2. Predicciones en Datos Nuevos

### Perfiles de Prueba Creados

### Resumen de Predicciones
- **<=50K**: 6 personas (66.7%)
- **>50K**: 3 personas (33.3%)

---

## 3. Análisis e Interpretación

### Patrones Identificados

#### ✅ Factores Positivos para Ingresos >50K:
1. **Educación Superior**: Personas con Masters tienen mayor probabilidad
2. **Empleo Gubernamental**: Trabajadores del gobierno tienden a ganar más
3. **Experiencia + Educación**: Combinación de edad media y educación superior
4. **Horas de Trabajo**: Más horas trabajadas correlacionan con mayores ingresos

#### ❌ Factores Negativos para Ingresos >50K:
1. **Baja Educación**: HS-grad o 11th grado reducen probabilidades
2. **Trabajo de Medio Tiempo**: Pocas horas semanales (20-30)
3. **Edad Extrema**: Muy jóvenes (22-25) o muy mayores (65+)
4. **Trabajo Independiente**: Self-employed mostró resultados mixtos

### Observaciones Interesantes

1. **Caso Sorprendente**: El emprendedor de 50 años con Bachelors trabajando 60 horas fue clasificado como <=50K, sugiriendo que el modelo favorece empleos estables sobre emprendimiento.

2. **Sesgo de Género**: Las mujeres con educación superior (Masters) y empleo gubernamental tienen buenas probabilidades de >50K.

3. **Umbral de Decisión**: Las probabilidades están muy cerca del 50%, indicando incertidumbre del modelo.

---

## 4. Limitaciones del Modelo

### Rendimiento Limitado
- **AUC = 0.5456**: Apenas mejor que clasificación aleatoria (0.5)
- **Accuracy = 52.8%**: Rendimiento modesto
- **Probabilidades cercanas al 50%**: Alta incertidumbre en predicciones

### Posibles Causas
1. **Características Insuficientes**: Faltan variables importantes como:
   - Experiencia laboral específica
   - Industria o sector detallado
   - Ubicación geográfica
   - Estado civil
   - Número de dependientes

2. **Datos Simulados**: Los datos no reflejan patrones reales complejos

