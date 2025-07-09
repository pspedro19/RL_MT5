# RL_SP500 - Pipeline de Trading con Machine Learning

Pipeline optimizado para captura y procesamiento de datos del S&P 500 usando MetaTrader 5 con capacidades HPC.

## �� Ejecución Rápida

### Pipeline 02: Captura de Datos
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar Pipeline 02 (captura de datos)
python run_pipeline.py

# Ejecutar con opciones específicas
python main_pipeline.py --symbol US500 --start 2023-01-01 --end 2023-12-31
```

### Pipeline 03: Validación y Preparación
```bash
# Ejecutar Pipeline 03 (validación y preparación para ML)
python run_pipeline03.py

# Ejecutar con archivo específico del Pipeline 02
python run_pipeline03.py --input data/sp500_m5_hpc_2023-01-01_2023-12-31.parquet
```

### Pipeline Completo (02 + 03)
```bash
# Ejecutar ambos pipelines en secuencia
python run_full_pipeline.py
```

## 📁 Estructura del Proyecto

```
RL_SP500/
├── main_pipeline.py          # Pipeline 02 principal
├── run_pipeline.py           # Script Pipeline 02
├── run_pipeline03.py         # Script Pipeline 03
├── run_full_pipeline.py      # Script Pipeline Completo (02+03)
├── requirements.txt          # Dependencias
├── README.md                # Este archivo
├── EJECUCION.md             # Guía detallada de ejecución
├── .gitignore               # Archivos a ignorar en git
├── config/                  # Configuración Pipeline 02
│   ├── constants.py         # Constantes globales
│   └── settings.py          # Configuración adicional
├── data/                    # Datos y conectores Pipeline 02
│   ├── connectors/
│   │   └── mt5_connector.py # Conector MetaTrader 5
│   ├── capture/
│   │   └── hpc_capture.py   # Motor de captura HPC
│   ├── processing.py        # Procesamiento de datos
│   └── quality/
│       └── quality_tracker.py # Tracking de calidad
├── features/                # Generación de features Pipeline 02
│   └── technical/
│       └── indicators.py    # Indicadores técnicos
├── utils/                   # Utilidades Pipeline 02
│   ├── market_calendar.py   # Calendario de mercado
│   └── report_generator.py  # Generación de reportes
├── 02_validate/            # Pipeline 02 - Validación y Preparación
│   ├── main.py              # Pipeline principal
│   ├── config.py            # Configuración
│   ├── validation/          # Validadores
│   ├── preprocessing/       # Preprocesamiento
│   ├── features/            # Selección de features
│   ├── data/                # División de datos
│   └── reports/             # Reportes de calidad
├── data/                    # Datos generados
├── logs/                    # Logs de ejecución
└── reports/                 # Reportes generados
```

## 🎯 Características Principales

### Pipeline 02: Captura de Datos
- **Captura HPC**: Paralelización con Ray para captura masiva de datos
- **Optimización GPU**: Soporte para cálculos en GPU con cuDF/CuPy
- **Tracking de Calidad**: Sistema avanzado de monitoreo de calidad de datos
- **Imputación Inteligente**: Brownian Bridge para llenar gaps de datos
- **Reportes Detallados**: Análisis completo de calidad y completitud

### Pipeline 03: Validación y Preparación
- **Validación de Integridad**: Verificación de coherencia OHLCV
- **Validación de Mercado**: Filtrado de horarios y días festivos
- **Validación Temporal**: Verificación de intervalos M5 consistentes
- **Selección de Features**: Selección automática de features relevantes
- **Preparación para ML**: Normalización y división train/validation

## 📊 Flujo de Trabajo Completo

```
Pipeline 02: Captura de Datos
    ↓
    ├── Conexión MT5
    ├── Captura multi-timeframe
    ├── Consolidación a M5
    ├── Imputación de gaps
    ├── Generación de features
    └── Guardado con tracking de calidad
    
Pipeline 03: Validación y Preparación
    ↓
    ├── Carga del dataset de Pipeline 02
    ├── Validaciones de integridad
    ├── Filtrado de horarios válidos
    ├── Selección de features
    ├── Normalización
    └── División train/validation
    
Resultado Final:
    ↓
    Dataset listo para entrenamiento de modelos
    con total transparencia sobre origen y calidad
```

## 📊 Salidas del Pipeline

### Pipeline 02
- **Datos procesados**: `data/sp500_m5_hpc_YYYY-MM-DD_YYYY-MM-DD.parquet`
- **Reportes de calidad**: `reports/quality_*.md`
- **Métricas JSON**: `reports/metrics_*.json`
- **Logs detallados**: `logs/`

### Pipeline 03
- **Train dataset**: `data/validated/train_quality_m5.csv`
- **Validation dataset**: `data/validated/validation_quality_m5.csv`
- **Reporte de validación**: `data/validated/quality_checklist_report.json`
- **Features seleccionados**: `data/validated/selected_features.json`

## ⚙️ Configuración

### Pipeline 02
Editar `config/constants.py` para:
- Símbolos de trading
- Timeframes
- Parámetros de mercado
- Configuración HPC

### Pipeline 03
Editar `pipelines/02_validate/config.py` para:
- Umbrales de validación
- Métodos de normalización
- Criterios de selección de features
- Configuración de división temporal

## 🔧 Dependencias Opcionales

- **Ray**: Paralelización masiva
- **Numba**: Aceleración JIT
- **Polars**: Procesamiento rápido
- **GPU**: Cálculos acelerados

## 📈 Monitoreo

Los pipelines generan reportes detallados incluyendo:
- Completitud de datos por período
- Análisis de gaps y calidad
- Métricas de rendimiento
- Distribución de fuentes de datos
- Validaciones de integridad
- Scores de calidad final

## 🚨 Requisitos

- Python 3.8+
- MetaTrader 5 instalado y configurado
- Conexión a internet para datos de mercado
