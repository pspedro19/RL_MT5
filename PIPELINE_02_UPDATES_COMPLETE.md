# Pipeline 02 - Actualizaciones Completas

## Resumen de Actualizaciones

Se han actualizado los pipelines 01 y 02 para cubrir completamente la funcionalidad del código de referencia `dataset_generation_03.py`. Las actualizaciones incluyen validaciones avanzadas de mercado, normalización específica por tipo de feature, selección automática de features y checklist de calidad avanzado.

## Funcionalidades Agregadas al Pipeline 02

### 1. Validaciones de Mercado Avanzadas (`validation/market_validator.py`)

**Nuevas funcionalidades:**
- **Validación de horarios M5 específicos**: Verifica que todos los datos estén en horarios oficiales de mercado (9:30 AM - 4:00 PM EST)
- **Validación de consistencia M5**: Verifica que los intervalos sean exactamente de 5 minutos (±0.1 min de tolerancia)
- **Validación de flags de datos**: Analiza la proporción de datos reales vs imputados (mínimo 80% datos reales)
- **Validación de fines de semana**: Detecta y reporta registros en fines de semana

**Métodos agregados:**
- `validate_m5_consistency()`: Valida intervalos de 5 minutos
- `validate_data_flags()`: Valida flags de datos reales vs imputados
- `validate_all()`: Ejecuta todas las validaciones de mercado

### 2. Normalización Específica por Tipo de Feature (`preprocessing/data_normalizer.py`)

**Nuevas funcionalidades:**
- **Normalización específica por categoría de feature**:
  - **Precios**: Normalización con respecto al precio de cierre promedio
  - **Volumen**: Normalización con respecto al volumen promedio
  - **Features técnicos**: Verificación de rangos (RSI 0-100, Stochastic 0-100)
  - **Features de retorno**: Clip de outliers extremos
  - **Features temporales**: Ya normalizados (sin/cos)
  - **Features de sesión**: Verificación de rangos (0-1)

**Métodos agregados:**
- `apply_robust_normalization()`: Normalización específica por tipo
- `get_normalization_info()`: Información de normalización aplicada

### 3. Selección Automática de Features (`features/feature_selector.py`)

**Nuevas funcionalidades:**
- **Selección automática con heurística**: Lista de features recomendados basada en el código de referencia
- **Eliminación de features constantes**: Usando VarianceThreshold
- **Eliminación de features correlacionadas**: Correlación > 0.95
- **Cálculo de importancia**: Random Forest para features importantes
- **Features obligatorios**: Siempre incluye 'time' y 'data_flag'

**Features recomendados agregados:**
```python
[
    'open', 'high', 'low', 'close', 'log_return',
    'ema_21', 'ema_55', 'sma_200', 'atr_14', 'volatility_20',
    'rsi_14', 'macd_histogram', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'vwap', 'volume_ratio', 'doji', 'bullish_engulfing', 'bearish_engulfing',
    'minutes_since_open', 'minutes_to_close', 'session_progress'
]
```

### 4. Checklist de Calidad Avanzado (`validation/quality_checklist.py`)

**Nuevas funcionalidades:**
- **Análisis de cobertura temporal**: Verifica días de trading y continuidad
- **Análisis de integridad de mercado**: Fines de semana, horarios, flags de datos
- **Análisis de NaN**: Detección y reporte de valores faltantes
- **Análisis de coherencia numérica**: OHLC válido, precios positivos, volumen no negativo
- **Análisis de gaps**: Detección de gaps mayores a 5 minutos
- **Análisis estadístico**: Outliers en retornos, análisis de volatilidad
- **Análisis de features**: Categorización y conteo de features
- **Análisis de splits**: Verificación de splits de entrenamiento/validación
- **Score de calidad**: Cálculo automático de score (0-100) con pesos por categoría

**Categorías de análisis:**
- Cobertura temporal (25%)
- Integridad de mercado (25%)
- Coherencia numérica (20%)
- Calidad de datos (15%)
- Análisis de features (10%)
- Análisis de splits (5%)

### 5. Pipeline Principal Actualizado (`main.py`)

**Nuevas funcionalidades integradas:**
- **Flujo de procesamiento mejorado**: Integración de todas las nuevas validaciones
- **Validaciones de mercado avanzadas**: Ejecución automática de todas las validaciones M5
- **Selección automática de features**: Con heurística y cálculo de importancia
- **Normalización específica**: Por tipo de feature
- **Checklist de calidad**: Generación automática de reporte completo
- **Reportes avanzados**: JSON y Markdown con información detallada

**Nuevo flujo de procesamiento:**
1. Carga de datos
2. Validaciones de mercado avanzadas (M5, flags, horarios)
3. Validaciones de integridad
4. Validaciones temporales
5. Limpieza de datos
6. Selección automática de features con heurística
7. Normalización específica por tipo de feature
8. Tracking de calidad avanzado
9. Análisis avanzado de datos
10. Generación de reporte de calidad
11. Creación de splits temporales
12. Generación de checklist de calidad avanzado
13. Guardado de resultados
14. Guardado de información de selección de features
15. Guardado de checklist de calidad
16. Generación de reportes avanzados
17. Generación de reporte básico

### 6. Configuración Actualizada (`config.py`)

**Nuevas configuraciones:**
- **Configuración de mercado**: Tolerancia M5, consistencia mínima, porcentaje de datos reales
- **Configuración de checklist**: Cobertura mínima, outliers máximos, barras esperadas por día
- **Configuración de features**: Umbrales de varianza y correlación, uso de heurística
- **Configuración de normalización**: Método específico por feature, ventanas de rolling
- **Configuración de rendimiento**: Procesamiento paralelo, límites de memoria

## Archivos Eliminados

- `dataset_generation_03.py`: Código de referencia ya integrado en los pipelines

## Resultados Esperados

Con estas actualizaciones, el Pipeline 02 ahora proporciona:

1. **Validaciones completas de mercado M5** con verificación de horarios oficiales
2. **Normalización robusta específica** por tipo de feature
3. **Selección automática de features** con heurística basada en mejores prácticas
4. **Checklist de calidad avanzado** con score automático y análisis detallado
5. **Reportes completos** en JSON y Markdown con información detallada
6. **Integración completa** con el Pipeline 01 para procesamiento end-to-end

## Uso

El pipeline actualizado se ejecuta con la misma interfaz:

```python
from pipelines.02_validate.main import ValidationPipeline
from pipelines.02_validate.config import get_config

config = get_config()
pipeline = ValidationPipeline(config)
results = pipeline.run('data/historical_data_m5.csv', 'data/validated/')
```

Los resultados incluyen:
- Dataset validado y normalizado
- Splits de entrenamiento/validación
- Reporte de calidad con score
- Información de selección de features
- Checklist de calidad detallado
- Reportes avanzados en múltiples formatos 