#!/usr/bin/env python3
"""
Configuración de ejecución del pipeline
"""
import os
from datetime import datetime, timedelta

# ===============================================================================
# CONFIGURACIÓN DE EJECUCIÓN
# ===============================================================================

# Configuración por defecto
DEFAULT_SYMBOL = "US500"
DEFAULT_START_DATE = datetime.now() - timedelta(days=365)  # Último año
DEFAULT_END_DATE = datetime.now()

# Configuración de salida
OUTPUT_FORMATS = ['parquet', 'csv']  # Formatos de salida por defecto
COMPRESSION = 'snappy'  # Compresión para Parquet

# Configuración de logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Configuración de rendimiento
USE_RAY = True  # Usar Ray para paralelización
USE_GPU = True  # Usar GPU si está disponible
CHUNK_SIZE = 10000  # Tamaño de chunks para procesamiento

# Configuración de calidad
MAX_GAP_MINUTES = 30  # Gaps máximos para imputación
QUALITY_THRESHOLD = 0.7  # Umbral mínimo de calidad

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Crear directorios si no existen
for directory in [DATA_DIR, LOGS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configuración de MetaTrader 5
MT5_CONFIG = {
    'login': None,  # Configurar si es necesario
    'password': None,  # Configurar si es necesario
    'server': None,  # Configurar si es necesario
    'max_attempts': 5
}

# Configuración de reportes
REPORT_CONFIG = {
    'generate_quality_report': True,
    'generate_executive_summary': True,
    'generate_json_metrics': True,
    'include_sparklines': True,
    'max_worst_records': 100
} 