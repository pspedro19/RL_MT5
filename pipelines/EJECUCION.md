# Guía de Ejecución del Pipeline RL_SP500

## 🚀 Ejecución Rápida

### Opción 1: Script Simple
```bash
python run_pipeline.py
```
Este script ejecuta el pipeline con parámetros por defecto (último año de datos del US500).

### Opción 2: Pipeline Principal
```bash
python main_pipeline.py --symbol US500 --start-date 2023-01-01 --end-date 2023-12-31
```

## 📋 Parámetros Disponibles

### Parámetros Básicos
- `--symbol`: Símbolo a capturar (default: US500)
- `--start-date`: Fecha de inicio (YYYY-MM-DD)
- `--end-date`: Fecha de fin (YYYY-MM-DD)

### Parámetros de Rendimiento
- `--use-ray`: Activar paralelización con Ray (default: True)
- `--no-gpu`: Desactivar procesamiento GPU
- `--workers`: Número de workers Ray (default: auto)

### Parámetros de Salida
- `--output-format`: Formatos de salida (parquet, csv, feather)
- `--output-dir`: Directorio de salida (default: data/)

### Parámetros de Calidad
- `--max-gap-minutes`: Gaps máximos para imputación (default: 30)
- `--quality-threshold`: Umbral mínimo de calidad (default: 0.7)

## 🔧 Configuración de MetaTrader 5

### Configuración Automática
El pipeline intentará conectarse automáticamente a MT5. Si tienes credenciales específicas:

```bash
python main_pipeline.py --symbol US500 --login 12345 --password "tu_password" --server "tu_broker"
```

### Verificar Conexión
```python
from data.connectors.mt5_connector import connect_mt5, get_broker_info

# Conectar
if connect_mt5():
    print("✅ MT5 conectado")
    
    # Obtener información del broker
    info = get_broker_info()
    print(f"Broker: {info['broker']}")
    print(f"Server: {info['server']}")
    print(f"Ping: {info['ping_ms']} ms")
else:
    print("❌ Error conectando a MT5")
```

## 📊 Salidas del Pipeline

### Archivos de Datos
- `data/sp500_m5_hpc_YYYY_YYYY.parquet` - Dataset principal
- `data/sp500_m5_hpc_YYYY_YYYY.csv` - Dataset en CSV

### Reportes de Calidad
- `reports/quality_analysis_YYYY_YYYY.md` - Análisis detallado
- `reports/executive_summary_YYYY_YYYY.md` - Resumen ejecutivo
- `reports/metrics_YYYY_YYYY.json` - Métricas en JSON

### Logs
- `logs/pipeline_YYYYMMDD_HHMMSS.log` - Logs detallados

## 🎯 Ejemplos de Uso

### 1. Captura del Último Año
```bash
python main_pipeline.py --symbol US500 --start-date 2023-01-01 --end-date 2023-12-31
```

### 2. Captura con Configuración Mínima
```bash
python main_pipeline.py --symbol US500 --start-date 2024-01-01 --no-ray --output-format csv
```

### 3. Captura con Alta Calidad
```bash
python main_pipeline.py --symbol US500 --start-date 2020-01-01 --end-date 2024-01-01 --quality-threshold 0.8
```

### 4. Captura de Múltiples Períodos
```bash
# 2020-2021
python main_pipeline.py --symbol US500 --start-date 2020-01-01 --end-date 2021-12-31

# 2022-2023
python main_pipeline.py --symbol US500 --start-date 2022-01-01 --end-date 2023-12-31

# 2024
python main_pipeline.py --symbol US500 --start-date 2024-01-01 --end-date 2024-12-31
```

## 🔍 Monitoreo de Progreso

### Durante la Ejecución
El pipeline muestra:
- Progreso de captura por período
- Estadísticas de calidad en tiempo real
- Uso de recursos (CPU, memoria)
- Gaps detectados y llenados

### Después de la Ejecución
Revisar los reportes generados:
```bash
# Ver reporte de calidad
cat reports/quality_analysis_*.md

# Ver métricas JSON
cat reports/metrics_*.json
```

## 🚨 Solución de Problemas

### Error: "No se pudo conectar a MT5"
1. Verificar que MT5 esté abierto
2. Confirmar que el símbolo esté disponible
3. Probar con credenciales específicas

### Error: "Ray no se pudo inicializar"
1. En Windows, verificar permisos de escritura
2. Reducir workers: `--workers 4`
3. Usar `--no-ray` como alternativa

### Error: "Memoria insuficiente"
1. Reducir período de captura
2. Usar `--no-ray` para menor uso de memoria
3. Procesar por chunks más pequeños

### Error: "GPU no detectada"
1. Verificar instalación de CUDA
2. Instalar cuDF: `pip install cudf-cu11`
3. Usar `--no-gpu` si persisten problemas

## 📈 Optimización de Rendimiento

### Para Sistemas con Mucha Memoria (>16GB)
```bash
python main_pipeline.py --symbol US500 --start-date 2020-01-01 --end-date 2024-01-01 --use-ray --workers 16
```

### Para Sistemas con Poca Memoria (<8GB)
```bash
python main_pipeline.py --symbol US500 --start-date 2023-01-01 --end-date 2023-12-31 --no-ray --output-format csv
```

### Para Procesamiento Rápido
```bash
python main_pipeline.py --symbol US500 --start-date 2024-01-01 --end-date 2024-12-31 --use-ray --workers 8 --output-format parquet
```

## 🔄 Automatización

### Script de Ejecución Automática
```bash
#!/bin/bash
# run_daily.sh

DATE=$(date +%Y-%m-%d)
python main_pipeline.py --symbol US500 --start-date 2024-01-01 --end-date $DATE --use-ray
```

### Programación con Cron (Linux/Mac)
```bash
# Ejecutar diariamente a las 6 PM
0 18 * * * cd /path/to/RL_SP500 && python run_pipeline.py
```

### Programación con Task Scheduler (Windows)
1. Abrir Task Scheduler
2. Crear tarea básica
3. Programar ejecución diaria
4. Acción: `python run_pipeline.py` 