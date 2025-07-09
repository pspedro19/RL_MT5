#!/usr/bin/env python3
"""
PIPELINE 02 MASTER - VERSION HPC OPTIMIZADA v2.4 CON REPORTES DE CALIDAD AVANZADOS
==================================================================================
Version con sistema de tracking detallado para reportar completitud y fuentes de datos.
"""
import os
import sys
import time
import json
import logging
import argparse
import warnings
import hashlib
import platform
import subprocess
import socket
from datetime import datetime
import pytz
import pandas as pd
import MetaTrader5 as mt5
import functools
from typing import Dict, List, Optional, Any
import shutil
import numpy as np
import re

# ===============================================================================
# CONFIGURACIÓN DE LOGGING
# ===============================================================================
# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)

# Reemplazar símbolos Unicode en los mensajes de logging por ASCII seguro
class SafeLogger(logging.Logger):
    def safe_info(self, msg, *args, **kwargs):
        msg = re.sub(r'[✓✔]', '[OK]', msg)
        msg = re.sub(r'[✗✘]', '[ERROR]', msg)
        super().info(msg, *args, **kwargs)
    def safe_error(self, msg, *args, **kwargs):
        msg = re.sub(r'[✓✔]', '[OK]', msg)
        msg = re.sub(r'[✗✘]', '[ERROR]', msg)
        super().error(msg, *args, **kwargs)

logging.setLoggerClass(SafeLogger)
logger = logging.getLogger(__name__)

# ===============================================================================
# CONFIGURACIÓN ESPECÍFICA PARA WINDOWS
# ===============================================================================
if platform.system() == 'Windows':
    # Configuraciones específicas de Windows para mejor rendimiento
    os.environ['NUMBA_NUM_THREADS'] = str(os.cpu_count())
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    os.environ['RAY_USE_MULTIPROCESSING_CPU_COUNT'] = '1'
    
    # Configurar para evitar problemas con Ray en Windows
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    
    logger.info("Configuración específica de Windows aplicada")

# Asegurar que los módulos estén en el path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importaciones locales
from config.constants import (
    SYMBOL_ALIASES, TIMEFRAMES, CAPTURE_METHODS, SEGMENT_SIZES,
    MARKET_CONFIGS, MARKET_EVENTS, OUTPUT_DIR, LOG_DIR, DASK_WORKERS,
    RESAMPLING_CONFIG
)
from data.connectors.mt5_connector import MT5Connector
from data.capture.hpc_capture import HpcCaptureEngine
from data.processing import (
    filter_market_hours, detect_gaps_optimized, clean_data_optimized,
    validate_data_integrity, get_market_config,
    process_ticks_to_ohlc, resample_timeframe_to_m5,
    combine_and_prioritize_dataframes,
    brownian_bridge_imputation_numba_tracked,
    analyze_data_parallel_dask, optimize_dataframe_memory
)
from data.quality.quality_tracker import DataQualityTracker
from features.technical.indicators import (
    generate_features_gpu, 
    generate_features_cpu_optimized, 
    warmup_numba,
    GPU_AVAILABLE,
    GPU_INFO,
    NUMBA_AVAILABLE,
    BOTTLENECK_AVAILABLE
)
from utils.market_calendar import validate_no_holiday_data, get_market_config as get_market_calendar_config
from utils.report_generator import (
    generate_json_report_enhanced,
    generate_quality_markdown_report,
    generate_markdown_report_enhanced
)

# ===============================================================================
# FUNCIONES DE I/O OPTIMIZADAS
# ===============================================================================

def save_dataset_optimized(df: pd.DataFrame, base_path: str, formats: List[str] = ['parquet', 'csv']) -> Dict[str, str]:
    """Guardar dataset en múltiples formatos optimizados"""
    saved_files = {}
    
    # Timer
    io_start_time = time.time()
    
    # Parquet (más eficiente)
    if 'parquet' in formats:
        parquet_path = f"{base_path}.parquet"
        df.to_parquet(
            parquet_path,
            engine='pyarrow',
            compression='zstd',
            index=False
        )
        saved_files['parquet'] = parquet_path
        logger.info(f"Dataset guardado en Parquet: {parquet_path}")
    
    # CSV (compatibilidad)
    if 'csv' in formats:
        csv_path = f"{base_path}.csv"
        df.to_csv(
            csv_path,
            index=False,
            compression='gzip' if len(df) > 100000 else None
        )
        saved_files['csv'] = csv_path
        logger.info(f"Dataset guardado en CSV: {csv_path}")
    
    # Feather (ultra-rápido para lectura)
    if 'feather' in formats:
        feather_path = f"{base_path}.feather"
        df.to_feather(feather_path)
        saved_files['feather'] = feather_path
        logger.info(f"Dataset guardado en Feather: {feather_path}")
    
    return saved_files

def analyze_data_by_period(df: pd.DataFrame) -> Dict[str, Any]:
    """Análisis básico de datos por período"""
    logger.info("Realizando análisis básico de datos...")
    
    # Análisis temporal
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.dayofweek
    
    # Estadísticas por año
    yearly_stats = df.groupby('year').agg({
        'close': ['count', 'mean', 'std', 'min', 'max'],
        'tick_volume': 'sum',
        'quality_score': 'mean'
    }).round(4)
    
    # Estadísticas por mes
    monthly_stats = df.groupby(['year', 'month']).agg({
        'close': ['count', 'mean', 'std'],
        'tick_volume': 'sum',
        'quality_score': 'mean'
    }).round(4)
    
    # Estadísticas generales
    general_stats = {
        'total_records': len(df),
        'date_range': {
            'start': df['time'].min().isoformat(),
            'end': df['time'].max().isoformat()
        },
        'price_stats': {
            'mean': float(df['close'].mean()),
            'std': float(df['close'].std()),
            'min': float(df['close'].min()),
            'max': float(df['close'].max())
        },
        'volume_stats': {
            'total': int(df['tick_volume'].sum()),
            'mean': float(df['tick_volume'].mean()),
            'max': int(df['tick_volume'].max())
        },
        'quality_stats': {
            'mean_score': float(df['quality_score'].mean()),
            'min_score': float(df['quality_score'].min()),
            'max_score': float(df['quality_score'].max())
        }
    }
    
    return {
        'general_stats': general_stats,
        'yearly_analysis': yearly_stats.to_dict(),
        'monthly_analysis': monthly_stats.to_dict()
    }

def analyze_data_parallel_dask(df: pd.DataFrame, dask_workers: int = 8) -> Dict:
    """Analisis paralelo de datos usando Dask"""
    if not DASK_AVAILABLE or len(df) < 100000:
        # Fallback a analisis estandar para datasets pequenos
        logger.info("Usando análisis estándar (Dask no disponible o dataset pequeño)")
        return analyze_data_by_period(df)
    
    logger.info(f"Realizando analisis paralelo con Dask ({dask_workers} workers)")
    
    try:
        # Convertir a Dask DataFrame
        ddf = dd.from_pandas(df, npartitions=dask_workers)
        
        # Computar estadisticas en paralelo
        with dask.distributed.LocalCluster(n_workers=dask_workers, threads_per_worker=2) as cluster:
            with dask.distributed.Client(cluster) as client:
                
                # Análisis general
                total_records = len(df)
                date_range = {
                    'start': df['time'].min().isoformat(),
                    'end': df['time'].max().isoformat()
                }
                
                # Analisis por year (paralelo)
                yearly_stats = (ddf
                    .assign(year=ddf['time'].dt.year)
                    .groupby('year')
                    .agg({
                        'time': 'count',
                        'high': 'max',
                        'low': 'min',
                        'close': ['mean', 'std'],
                        'tick_volume': 'sum'
                    })
                    .compute()
                )
                
                # Analisis por mes (paralelo)
                monthly_stats = (ddf
                    .assign(
                        year=ddf['time'].dt.year,
                        month=ddf['time'].dt.month
                    )
                    .groupby(['year', 'month'])
                    .agg({
                        'time': 'count',
                        'close': ['mean', 'std'],
                        'tick_volume': 'sum'
                    })
                    .compute()
                )
                
                # Gaps analysis (paralelo)
                time_diffs = ddf['time'].diff()
                gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)].compute()
        
        # Construir resultado
        analysis = {
            'summary': {
                'total_records': total_records,
                'date_range': date_range,
                'trading_days': df['time'].dt.date.nunique()
            },
            'yearly_analysis': yearly_stats.to_dict(),
            'monthly_analysis': monthly_stats.to_dict(),
            'parallel_processing': {
                'method': 'Dask',
                'workers': dask_workers,
                'partitions': ddf.npartitions
            },
            'gap_analysis': {
                'total_gaps': len(gaps),
                'avg_gap_minutes': gaps.mean().total_seconds() / 60 if len(gaps) > 0 else 0
            }
        }
        
        return analysis
        
    except Exception as e:
        logger.warning(f"Error en análisis paralelo: {e}. Usando análisis estándar.")
        return analyze_data_by_period(df)

# Detectar Ray
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    warnings.warn("Ray no disponible. Instalarlo mejorará significativamente el rendimiento: pip install ray")

# Detectar Polars
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Detectar Dask
try:
    import dask
    import dask.dataframe as dd
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# Configuración de logging
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'capture_run.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger('pipeline_hpc')

# Configurar encoding UTF-8 para Windows
if sys.platform == "win32":
    import locale
    try:
        # Intentar configurar UTF-8
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# ===============================================================================
# FUNCIONES AUXILIARES PARA METADATOS Y PROVENIENCIA
# ===============================================================================

def get_git_info() -> dict:
    """Obtener información de git del repositorio actual"""
    git_info = {
        'commit': 'unknown',
        'branch': 'unknown',
        'tag': 'unknown',
        'dirty': False,
        'remote': 'unknown'
    }
    
    try:
        # Commit actual
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            git_info['commit'] = result.stdout.strip()[:8]  # Primeros 8 caracteres
        
        # Rama actual
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()
        
        # Tag más cercano
        result = subprocess.run(['git', 'describe', '--tags', '--abbrev=0'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            git_info['tag'] = result.stdout.strip()
        
        # Estado del repositorio
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            git_info['dirty'] = bool(result.stdout.strip())
            
        # Remote origin
        result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            git_info['remote'] = result.stdout.strip()
            
    except Exception as e:
        logger.debug(f"Error obteniendo información de git: {e}")
    
    return git_info

def get_system_info() -> dict:
    """Obtener información detallada del sistema"""
    return {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'cpu_count': os.cpu_count(),
        'memory_gb': get_available_memory_gb(),
        'os_version': platform.version(),
        'architecture': platform.machine()
    }

def get_available_memory_gb() -> float:
    """Obtener memoria disponible en GB"""
    try:
        if sys.platform == "win32":
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        else:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        return int(line.split()[1]) / (1024**2)
    except:
        return 0.0
    return 0.0

def calculate_file_checksum(filepath: str, algorithm: str = 'sha256') -> str:
    """Calcular checksum de un archivo"""
    if not os.path.exists(filepath):
        return 'file_not_found'
    
    hash_func = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def format_duration(seconds: float) -> str:
    """Formatear duración en formato legible"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def parse_year_or_date(value):
    """Permite año (YYYY) o fecha (YYYY-MM-DD) y retorna datetime"""
    try:
        # Intentar parsear como año
        if len(value) == 4 and value.isdigit():
            return datetime(int(value), 1, 1)
        # Intentar parsear como fecha completa
        return datetime.strptime(value, '%Y-%m-%d')
    except Exception:
        raise argparse.ArgumentTypeError(f"Formato inválido para fecha/año: {value}. Usa YYYY o YYYY-MM-DD.")

def filter_market_hours_modular(df, instrument='US500'):
    """Filtrar horarios de mercado de forma modular según el instrumento"""
    # Validar que df sea un DataFrame válido
    if not isinstance(df, pd.DataFrame):
        logger.warning(f"filter_market_hours_modular: df no es un DataFrame, es {type(df)}")
        return df
    
    # Verificar si el DataFrame está vacío de forma segura
    if df.empty:
        return df
    
    try:
        # Obtener configuración de mercado
        market_config = get_market_config(instrument)
        market_type = market_config.get('market_type', 'stock_market')
        
        if POLARS_AVAILABLE:
            df_pl = pl.from_pandas(df)
            if market_type == 'forex':
                # Lógica específica para Forex (24/5)
                df_filtered = df_pl.with_columns([
                    pl.col('time').dt.weekday().alias('weekday'),
                    pl.col('time').dt.hour().alias('hour')
                ]).filter(
                    ~((pl.col('weekday') == 5) |  # Sábados
                      (pl.col('weekday') == 6) & (pl.col('hour') < 22))  # Domingos antes de 22:00
                ).drop(['weekday', 'hour'])
            else:
                # Lógica para mercado de acciones (USA)
                df_filtered = filter_market_hours(df_pl, instrument)
            
            return df_filtered.to_pandas()
        else:
            # Fallback a pandas
            return filter_market_hours(df, instrument)
                
    except Exception as e:
        logger.warning(f"Error en filtrado modular: {e}")
        return df

def validate_column_types(df: pd.DataFrame):
    """Validar y convertir tipos de columnas problemáticas para serialización"""
    logger.info("Validando y corrigiendo tipos de columnas para serialización...")
    
    # Primero, eliminar cualquier columna duplicada
    if df.columns.duplicated().any():
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        logger.warning(f"Columnas duplicadas detectadas y eliminadas: {duplicated_cols}")
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    df_clean = df.copy()
    
    # Manejar caso especial de 'return' que es palabra reservada
    if 'return' in df_clean.columns:
        df_clean = df_clean.rename(columns={'return': 'returns'})
        logger.info("Columna 'return' renombrada a 'returns'")
    
    # Procesar columnas de tipo objeto
    for col in df_clean.columns:
        try:
            # Verificar si la columna existe y obtener su tipo de forma segura
            if col not in df_clean.columns:
                continue
                
            col_dtype = str(df_clean[col].dtype)
            
            if col_dtype == 'object':
                # Intentar conversión a numérico primero
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    if df_clean[col].notna().any():  # Si hay valores válidos
                        logger.info(f"Columna '{col}' convertida a numérico")
                    else:
                        # Si todos son NaN después de conversión, eliminar
                        df_clean = df_clean.drop(columns=[col])
                        logger.warning(f"Columna '{col}' eliminada (todos valores inválidos)")
                except Exception as conv_error:
                    logger.debug(f"No se pudo convertir '{col}' a numérico: {conv_error}")
                    # Si no es numérico, verificar si es string válido
                    try:
                        if df_clean[col].apply(lambda x: isinstance(x, str) if pd.notna(x) else True).all():
                            logger.info(f"Columna '{col}' mantenida como string")
                        else:
                            # Convertir a string como último recurso
                            df_clean[col] = df_clean[col].astype(str)
                            logger.warning(f"Columna '{col}' convertida a string como fallback")
                    except Exception as str_error:
                        logger.error(f"Error procesando columna '{col}' como string: {str_error}")
                        # Si hay error, intentar eliminar la columna problemática
                        if col in df_clean.columns:
                            df_clean = df_clean.drop(columns=[col])
                            logger.warning(f"Columna '{col}' eliminada debido a errores")
        except Exception as e:
            logger.error(f"Error procesando columna '{col}': {e}")
            # Si hay error, intentar eliminar la columna problemática
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])
                logger.warning(f"Columna '{col}' eliminada debido a errores")
    
    # Verificación final de duplicados
    if df_clean.columns.duplicated().any():
        duplicated_cols = df_clean.columns[df_clean.columns.duplicated()].tolist()
        logger.warning(f"Columnas duplicadas detectadas y eliminadas: {duplicated_cols}")
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated(keep='first')]
    
    logger.info("Validación y corrección de tipos de columnas completada exitosamente")
    return df_clean

def check_data_before_save(df: pd.DataFrame):
    """Verifica el DataFrame antes de guardar con validaciones robustas"""
    logger.info("Verificando integridad de datos antes de guardar...")
    
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'summary': {}
    }
    
    try:
        # Verificar columnas duplicadas
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicated_cols:
            validation_results['warnings'].append(f"Columnas duplicadas encontradas: {duplicated_cols}")
            logger.warning(f"Columnas duplicadas encontradas: {duplicated_cols}")
        
        # Verificar tipos de datos problemáticos
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            logger.info(f"Columnas con objetos: {object_cols}")
            validation_results['summary']['object_columns'] = object_cols
            for col in object_cols:
                try:
                    unique_vals = df[col].nunique()
                    logger.info(f"  - {col}: {unique_vals} valores únicos")
                    if unique_vals < 10:
                        logger.info(f"    Valores: {df[col].unique()[:5]}")
                except Exception as e:
                    logger.warning(f"Error analizando columna '{col}': {e}")
                    validation_results['warnings'].append(f"Error analizando columna '{col}': {e}")
        
        # Verificar valores nulos
        try:
            null_counts = df.isnull().sum()
            if null_counts.any():
                logger.info("Columnas con valores nulos:")
                validation_results['summary']['null_counts'] = {}
                for col, count in null_counts[null_counts > 0].items():
                    logger.info(f"  - {col}: {count} nulos")
                    validation_results['summary']['null_counts'][col] = int(count)
        except Exception as e:
            logger.warning(f"Error verificando valores nulos: {e}")
            validation_results['warnings'].append(f"Error verificando valores nulos: {e}")
        
        # Verificar columnas requeridas
        required_columns = ['time', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['errors'].append(f"Columnas requeridas faltantes: {missing_columns}")
            validation_results['is_valid'] = False
            logger.error(f"Columnas requeridas faltantes: {missing_columns}")
        
        # Verificar que el DataFrame no esté vacío
        if df.empty:
            validation_results['errors'].append("DataFrame está vacío")
            validation_results['is_valid'] = False
            logger.error("DataFrame está vacío")
        
        # Verificar tipos de datos críticos
        if 'time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                validation_results['warnings'].append("Columna 'time' no es datetime")
                logger.warning("Columna 'time' no es datetime")
        
        # Verificar precios positivos
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                if (df[col] < 0).any():
                    validation_results['warnings'].append(f"Valores negativos en {col}")
                    logger.warning(f"Valores negativos en {col}")
        
        # Resumen final
        validation_results['summary']['total_rows'] = len(df)
        validation_results['summary']['total_columns'] = len(df.columns)
        validation_results['summary']['memory_mb'] = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        logger.info(f"Verificación completada: {len(df)} filas, {len(df.columns)} columnas")
        
        if validation_results['warnings']:
            logger.info(f"Advertencias encontradas: {len(validation_results['warnings'])}")
        if validation_results['errors']:
            logger.error(f"Errores encontrados: {len(validation_results['errors'])}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error crítico en verificación de datos: {e}")
        validation_results['errors'].append(f"Error crítico en verificación: {e}")
        validation_results['is_valid'] = False
        return validation_results

def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Asegurar que existan todas las columnas requeridas con valores por defecto si faltan"""
    logger.info("Verificando y creando columnas requeridas...")
    
    df_robust = df.copy()
    
    # Columnas requeridas básicas
    required_columns = {
        'time': pd.Timestamp.now(),  # Valor por defecto temporal
        'open': 0.0,
        'high': 0.0,
        'low': 0.0,
        'close': 0.0,
        'tick_volume': 0,
        'spread': 0.0,
        'real_volume': 0,
        'data_flag': 'unknown',
        'quality_score': 0.0
    }
    
    # Columnas adicionales que pueden faltar
    optional_columns = {
        'source_timeframe': 'unknown',
        'capture_method': 'unknown',
        'returns': 0.0,  # En caso de que 'return' se haya renombrado
        'log_return': 0.0
    }
    
    # Crear columnas requeridas faltantes
    for col, default_value in required_columns.items():
        if col not in df_robust.columns:
            logger.warning(f"Columna requerida '{col}' no encontrada. Creando con valor por defecto: {default_value}")
            if col == 'time':
                # Para time, usar el primer timestamp disponible o ahora
                if len(df_robust) > 0 and 'time' in df_robust.columns:
                    df_robust[col] = df_robust['time']
                else:
                    df_robust[col] = pd.Timestamp.now()
            else:
                df_robust[col] = default_value
    
    # Crear columnas opcionales faltantes
    for col, default_value in optional_columns.items():
        if col not in df_robust.columns:
            logger.info(f"Columna opcional '{col}' no encontrada. Creando con valor por defecto: {default_value}")
            df_robust[col] = default_value
    
    # Asegurar tipos de datos correctos
    type_mapping = {
        'time': 'datetime64[ns]',
        'open': 'float32',
        'high': 'float32', 
        'low': 'float32',
        'close': 'float32',
        'tick_volume': 'int32',
        'spread': 'float32',
        'real_volume': 'int32',
        'data_flag': 'object',
        'quality_score': 'float32',
        'source_timeframe': 'object',
        'capture_method': 'object',
        'returns': 'float32',
        'log_return': 'float32'
    }
    
    for col, expected_type in type_mapping.items():
        if col in df_robust.columns:
            try:
                if expected_type == 'datetime64[ns]':
                    df_robust[col] = pd.to_datetime(df_robust[col])
                elif expected_type == 'object':
                    df_robust[col] = df_robust[col].astype(str)
                else:
                    df_robust[col] = df_robust[col].astype(expected_type)
            except Exception as e:
                logger.warning(f"No se pudo convertir columna '{col}' a {expected_type}: {e}")
    
    logger.info(f"Columnas requeridas verificadas. DataFrame final: {len(df_robust)} filas, {len(df_robust.columns)} columnas")
    return df_robust

def robust_dataframe_validation(df: pd.DataFrame) -> Dict:
    """Validación robusta del DataFrame con correcciones automáticas"""
    logger.info("Ejecutando validación robusta del DataFrame...")
    
    validation_report = {
        'original_shape': df.shape,
        'corrections_applied': [],
        'warnings': [],
        'errors': [],
        'final_shape': None,
        'is_valid': True
    }
    
    try:
        df_robust = df.copy()
        
        # 1. Asegurar columnas requeridas
        df_robust = ensure_required_columns(df_robust)
        validation_report['corrections_applied'].append('required_columns_ensured')
        
        # 2. Eliminar duplicados de columnas
        if df_robust.columns.duplicated().any():
            duplicated_cols = df_robust.columns[df_robust.columns.duplicated()].tolist()
            df_robust = df_robust.loc[:, ~df_robust.columns.duplicated(keep='first')]
            validation_report['corrections_applied'].append(f'removed_duplicate_columns: {duplicated_cols}')
            validation_report['warnings'].append(f'Columnas duplicadas eliminadas: {duplicated_cols}')
        
        # 3. Eliminar duplicados de filas
        initial_rows = len(df_robust)
        df_robust = df_robust.drop_duplicates()
        if len(df_robust) < initial_rows:
            removed_rows = initial_rows - len(df_robust)
            validation_report['corrections_applied'].append(f'removed_duplicate_rows: {removed_rows}')
            validation_report['warnings'].append(f'Filas duplicadas eliminadas: {removed_rows}')
        
        # 4. Validar y corregir tipos de columnas
        df_robust = validate_column_types(df_robust)
        validation_report['corrections_applied'].append('column_types_validated')
        
        # 5. Verificar integridad OHLC
        if all(col in df_robust.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df_robust['high'] < df_robust['low']) |
                (df_robust['open'] < df_robust['low']) | (df_robust['open'] > df_robust['high']) |
                (df_robust['close'] < df_robust['low']) | (df_robust['close'] > df_robust['high'])
            )
            if invalid_ohlc.any():
                invalid_count = invalid_ohlc.sum()
                validation_report['warnings'].append(f'Registros con lógica OHLC inválida: {invalid_count}')
                # Corregir automáticamente
                df_robust.loc[invalid_ohlc, 'high'] = df_robust.loc[invalid_ohlc, ['open', 'close']].max(axis=1)
                df_robust.loc[invalid_ohlc, 'low'] = df_robust.loc[invalid_ohlc, ['open', 'close']].min(axis=1)
                validation_report['corrections_applied'].append(f'corrected_ohlc_logic: {invalid_count}')
        
        # 6. Verificar precios negativos
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df_robust.columns:
                negative_prices = (df_robust[col] < 0).sum()
                if negative_prices > 0:
                    validation_report['warnings'].append(f'Precios negativos en {col}: {negative_prices}')
                    # Corregir automáticamente
                    df_robust.loc[df_robust[col] < 0, col] = 0.0
                    validation_report['corrections_applied'].append(f'corrected_negative_prices_{col}: {negative_prices}')
        
        # 7. Verificar orden temporal
        if 'time' in df_robust.columns:
            df_robust = df_robust.sort_values('time').reset_index(drop=True)
            validation_report['corrections_applied'].append('temporal_order_fixed')
        
        # 8. Verificación final
        final_validation = check_data_before_save(df_robust)
        if not final_validation['is_valid']:
            validation_report['errors'].extend(final_validation['errors'])
            validation_report['is_valid'] = False
        
        validation_report['warnings'].extend(final_validation['warnings'])
        validation_report['final_shape'] = df_robust.shape
        
        logger.info(f"Validación robusta completada. Forma final: {df_robust.shape}")
        logger.info(f"Correcciones aplicadas: {len(validation_report['corrections_applied'])}")
        logger.info(f"Advertencias: {len(validation_report['warnings'])}")
        logger.info(f"Errores: {len(validation_report['errors'])}")
        
        return validation_report, df_robust
        
    except Exception as e:
        logger.error(f"Error en validación robusta: {e}")
        validation_report['errors'].append(f"Error crítico en validación: {e}")
        validation_report['is_valid'] = False
        return validation_report, df

# ===============================================================================
# FUNCIÓN PRINCIPAL OPTIMIZADA
# ===============================================================================

def main():
    logger.info("[DEBUG] INICIO DE LA FUNCIÓN MAIN DEL PIPELINE")
    parser = argparse.ArgumentParser(description='Pipeline 02 MASTER HPC v2.4 - Captura optimizada con tracking de calidad avanzado')
    parser.add_argument('--symbol', default='US500', help='Simbolo o grupo de simbolos')
    parser.add_argument('--instrument', default='sp500', choices=['sp500', 'usdcop'], help='Instrumento a descargar (sp500 o usdcop)')
    parser.add_argument('--start', type=str, default='2020', help='Año o fecha de inicio (YYYY o YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y'), help='Año o fecha de fin (YYYY o YYYY-MM-DD)')
    parser.add_argument('--login', type=int, help='MT5 login (opcional)')
    parser.add_argument('--server', help='MT5 server (opcional)')
    parser.add_argument('--password', help='MT5 password (opcional)')
    parser.add_argument('--output', default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'), help='Directorio de salida (por defecto: data en raíz)')
    parser.add_argument('--no-ray', action='store_true', help='Desactivar Ray para captura')
    parser.add_argument('--no-gpu', action='store_true', help='Desactivar procesamiento GPU')
    parser.add_argument('--formats', nargs='+', default=['parquet', 'csv'], 
                       choices=['parquet', 'csv', 'feather'], help='Formatos de salida')
    
    args = parser.parse_args()
    
    # Parsear fechas de inicio y fin
    start_dt = parse_year_or_date(args.start)
    end_dt = parse_year_or_date(args.end)
    
    if end_dt < start_dt:
        logger.error('La fecha/año de fin debe ser posterior a la de inicio.')
        logger.info("[DEBUG] RETURN 1: La fecha/año de fin debe ser posterior a la de inicio.")
        return 1
    
    # Selección de instrumento y configuración de mercado
    if args.instrument == 'usdcop':
        instrument = 'USDCOP'
        symbol = args.symbol if args.symbol != 'US500' else 'USDCOP'
    else:
        instrument = 'US500'
        symbol = args.symbol
    
    # Timer total
    pipeline_start_time = time.time()
    
    # Crear directorios
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # En Windows, crear directorios temporales para Ray si es necesario
    if sys.platform == "win32" and not args.no_ray:
        from config.constants import RAY_TEMP_DIR, RAY_SPILL_DIR
        try:
            os.makedirs(os.path.dirname(RAY_TEMP_DIR), exist_ok=True)
            os.makedirs(RAY_TEMP_DIR, exist_ok=True)
            os.makedirs(RAY_SPILL_DIR, exist_ok=True)
            logger.info(f"Directorios temporales de Ray creados: {RAY_TEMP_DIR}")
        except Exception as e:
            logger.warning(f"No se pudieron crear directorios temporales para Ray: {e}")
            logger.warning("Use --no-ray para desactivar Ray")
    
    logger.info("="*80)
    logger.info("INICIANDO PIPELINE 02 MASTER - VERSION HPC OPTIMIZADA v2.4")
    logger.info("CON TRACKING DE CALIDAD AVANZADO")
    logger.info("="*80)
    logger.info(f"Hardware detectado: {os.cpu_count()} cores CPU, GPU: {GPU_AVAILABLE}")
    if GPU_INFO:
        logger.info(f"GPU Info: {GPU_INFO}")
    
    # Mensaje especial para GPU no disponible
    if not GPU_AVAILABLE and GPU_INFO:
        logger.info("GPU detectada por nvidia-smi pero cuDF no está instalado.")
        logger.info("Para habilitar GPU, instale: pip install cudf-cu11 cupy-cuda11x")
    
    # Warmup de Numba
    warmup_numba()
    
    try:
        # Conectar a MT5 usando el conector OO
        mt5_connector = MT5Connector(symbol)
        if not mt5_connector.connected:
            logger.error("No se pudo conectar a MT5")
            logger.info("[DEBUG] RETURN 1: No se pudo conectar a MT5")
            return 1
        
        # El símbolo ya fue validado por el conector MT5Connector
        logger.info(f"\nSímbolo validado: {symbol}")
        
        logger.info(f"Configuracion:")
        logger.info(f"  - Simbolo: {symbol}")
        logger.info(f"  - Periodo: {start_dt} - {end_dt}")
        logger.info(f"  - Ray: {'Desactivado por usuario' if args.no_ray else 'Intentando activar' if RAY_AVAILABLE else 'No disponible'}")
        logger.info(f"  - GPU: {'Desactivado por usuario' if args.no_gpu else 'Activo' if GPU_AVAILABLE else 'No disponible (instale cuDF)'}")
        logger.info(f"  - Formatos salida: {args.formats}")
        logger.info(f"  - Directorio salida: {args.output}")
        
        # Resumen de optimizaciones disponibles
        logger.info("\nOptimizaciones disponibles:")
        optimizations = []
        if NUMBA_AVAILABLE:
            optimizations.append("Numba JIT")
        if RAY_AVAILABLE and not args.no_ray:
            optimizations.append("Ray (paralelización)")
        if GPU_AVAILABLE and not args.no_gpu:
            optimizations.append("GPU/CUDA")
        if POLARS_AVAILABLE:
            optimizations.append("Polars (procesamiento)")
        if DASK_AVAILABLE:
            optimizations.append("Dask (análisis)")
        if BOTTLENECK_AVAILABLE:
            optimizations.append("Bottleneck")
        
        if optimizations:
            logger.info(f"  [OK] {', '.join(optimizations)}")
        else:
            logger.info("  [!] Ejecutando sin optimizaciones HPC")
        
        # Determinar si usar GPU
        use_gpu = GPU_AVAILABLE and not args.no_gpu
        
        # Inicializar motor de captura HPC con tracking
        logger.info("\nInicializando motor de captura con tracking de calidad...")
        engine = HpcCaptureEngine(symbol, use_ray=not args.no_ray)
        
        # Verificar si Ray se inicializó correctamente
        if not args.no_ray and not engine.use_ray:
            logger.warning("Ray no se pudo inicializar - continuando con ThreadPool")
            logger.info("Para forzar desactivacion de Ray use: --no-ray")
        
        # ✅ NUEVA LÓGICA: Capturar datos por años con fallback correcto como en los scripts de referencia
        all_data = []
        years_without_data = []
        
        for year in range(start_dt.year, end_dt.year + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"[LOG] INICIO PROCESO AÑO {year} - {symbol}")
            logger.info(f"{'='*60}")
            year_start = datetime(year, 1, 1, tzinfo=pytz.UTC)
            year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=pytz.UTC)
            if year == datetime.now().year:
                year_end = datetime.now(pytz.UTC)
            logger.info(f"[LOG] Intentando captura de datos para {symbol} desde {year_start} hasta {year_end}")
            df_year = engine.capture_with_all_methods(year_start, year_end)
            logger.info(f"[LOG] Registros capturados brutos para {year}: {len(df_year)}")
            if df_year.empty:
                years_without_data.append(year)
                logger.warning(f"[LOG] Año {year}: No se encontraron datos disponibles en MT5")
                continue
            # ✅ LÓGICA DE PRIORIZACIÓN CORREGIDA: Los datos ya vienen resampleados a M5
            # Verificar si hay datos válidos (ya procesados por el motor de captura)
            if not df_year.empty:
                logger.info(f"[LOG] Año {year}: Datos capturados y procesados ({len(df_year)} barras M5).")
                logger.info(f"[LOG] Año {year}: Fuentes de datos: {df_year['source_timeframe'].value_counts().to_dict()}")
                all_data.append(df_year)
            else:
                logger.warning(f"[LOG] Año {year}: No se encontraron datos válidos.")
                years_without_data.append(year)
                continue
            logger.info(f"[LOG] Año {year}: {len(all_data[-1])} registros capturados tras priorización.")
            logger.info(f"[LOG] FIN PROCESO AÑO {year} - {symbol}")
        
        # ✅ SIEMPRE CONTINUAR, incluso si hay años sin datos
        if not all_data:
            logger.error("No se capturaron datos para ningún año")
            logger.info("[DEBUG] RETURN 1: No se capturaron datos para ningún año")
            logger.info("Generando reportes con dataset vacío para documentar el intento...")
            
            # Crear dataset vacío pero con estructura correcta
            df_clean = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'data_flag', 'quality_score'])
            df_clean['time'] = pd.to_datetime([])
            
            # Generar reportes de error
            error_report = {
                'error': 'No se capturaron datos',
                'years_attempted': list(range(start_dt.year, end_dt.year + 1)),
                'years_without_data': years_without_data,
                'symbol': symbol,
                'period': f"{start_dt} - {end_dt}",
                'timestamp': datetime.now().isoformat()
            }
            
            # Guardar reporte de error
            error_report_path = os.path.join(args.output, f"ERROR_REPORT_{symbol}_{start_dt.year}_{end_dt.year}.json")
            with open(error_report_path, 'w') as f:
                json.dump(error_report, f, indent=2)
            
            logger.info(f"Reporte de error guardado: {error_report_path}")
            return 1
        
        # Combinar todos los años
        logger.info("\nCombinando datos de todos los años...")
        df_all = pd.concat(all_data, ignore_index=True)
        df_all = df_all.sort_values('time').reset_index(drop=True)
        
        # Limpiar memoria después de combinar
        del all_data
        import gc
        gc.collect()
        logger.info(f"Memoria liberada después de combinar años. Memoria actual: {df_all.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        logger.info(f"Total inicial: {len(df_all):,} registros")
        
        # ✅ Validación y corrección automática de datos
        logger.info("Realizando validación y corrección automática de datos...")
        
        # 1. Eliminar duplicados de timestamp
        initial_count = len(df_all)
        df_all = df_all.drop_duplicates(subset=['time'], keep='last')
        if len(df_all) < initial_count:
            logger.info(f"Eliminados {initial_count - len(df_all)} registros duplicados")
        
        # 2. Asegurar tipos correctos para columnas numéricas
        numeric_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'real_volume', 'spread']
        for col in numeric_cols:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
        
        # 3. Eliminar filas con todos valores NaN
        initial_count = len(df_all)
        df_all = df_all.dropna(subset=['open', 'high', 'low', 'close'], how='all')
        if len(df_all) < initial_count:
            logger.info(f"Eliminadas {initial_count - len(df_all)} filas con todos valores NaN")
        
        # 4. Verificar orden temporal
        df_all = df_all.sort_values('time').reset_index(drop=True)
        
        logger.info(f"Validación completada. Registros finales: {len(df_all):,}")
        
        # ✅ MOSTRAR AÑOS SIN DATOS
        if years_without_data:
            logger.warning(f"Años sin datos disponibles: {years_without_data}")
            logger.info(f"Continuando con {len(all_data)} años que sí tienen datos")
        
        # Pipeline de procesamiento HPC con tracking y optimizaciones
        logger.info("\nAplicando pipeline de procesamiento HPC con tracking de calidad y optimizaciones...")
        mem0 = df_all.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"Memoria antes de procesamiento: {mem0:.2f} MB")
        
        # 1. Validación de integridad inicial
        validation_start = time.time()
        validation_results = validate_data_integrity(df_all, instrument)
        if not validation_results['valid']:
            logger.warning(f"Problemas de integridad detectados: {validation_results['issues']}")
        if validation_results['outliers']:
            logger.info(f"Outliers detectados: {validation_results['outliers']}")
        engine.quality_tracker.execution_times['validation'] = time.time() - validation_start
        
        # 2. Filtrar horarios de mercado (Polars si está disponible)
        filter_start = time.time()
        df_filtered = filter_market_hours_modular(df_all, instrument=instrument)
        logger.info(f"Memoria tras filtrado: {df_filtered.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        engine.quality_tracker.execution_times['filtering'] = time.time() - filter_start
        
        # 3. Detectar e imputar gaps con tracking
        imputation_start = time.time()
        df_imputed = brownian_bridge_imputation_numba_tracked(df_filtered, engine.quality_tracker)
        logger.info(f"Memoria tras imputación: {df_imputed.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        engine.quality_tracker.execution_times['imputation'] = time.time() - imputation_start
        
        # 4. Generar features (GPU si está disponible)
        features_start = time.time()
        if use_gpu:
            df_features = generate_features_gpu(df_imputed)
        else:
            df_features = generate_features_cpu_optimized(df_imputed)
        logger.info(f"Memoria tras features: {df_features.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        engine.quality_tracker.execution_times['features'] = time.time() - features_start
        
        # 5. Limpiar datos (optimizado)
        cleaning_start = time.time()
        df_clean = clean_data_optimized(df_features)
        logger.info(f"Memoria tras limpieza: {df_clean.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        engine.quality_tracker.execution_times['cleaning'] = time.time() - cleaning_start
        
        # 6. Optimizar memoria
        memory_start = time.time()
        df_clean = optimize_dataframe_memory(df_clean)
        engine.quality_tracker.execution_times['memory_optimization'] = time.time() - memory_start
        
        # 7. Validación robusta del DataFrame con correcciones automáticas
        logger.info("Aplicando validación robusta del DataFrame...")
        validation_report, df_clean = robust_dataframe_validation(df_clean)
        
        if not validation_report['is_valid']:
            logger.error("Problemas críticos detectados en la validación robusta:")
            for error in validation_report['errors']:
                logger.error(f"  - {error}")
            logger.error("Continuando con datos originales...")
            # Usar datos originales si la validación falla críticamente
            df_clean = df.copy()
        
        # Mostrar advertencias si las hay
        if validation_report['warnings']:
            logger.warning("Advertencias en validación robusta:")
            for warning in validation_report['warnings']:
                logger.warning(f"  - {warning}")
        
        # Mostrar correcciones aplicadas
        if validation_report['corrections_applied']:
            logger.info("Correcciones aplicadas:")
            for correction in validation_report['corrections_applied']:
                logger.info(f"  - {correction}")
        
        # 8. Aplicar trazabilidad estandarizada
        logger.info("Aplicando trazabilidad estandarizada...")
        try:
            from utils.data_traceability import DataTraceabilityManager

            traceability_manager = DataTraceabilityManager()

            # Aplicar asignación de origen a todos los registros
            if 'imputed' in df_clean.columns and df_clean['imputed'].any():
                imputed_mask = df_clean['imputed'] == True
                df_imputed = traceability_manager.assign_data_origin(
                    df_clean.loc[imputed_mask],
                    capture_method='brownian_bridge',
                    source_timeframe='M5',
                    is_imputed=True,
                    imputation_method='brownian_bridge'
                )
                df_real = traceability_manager.assign_data_origin(
                    df_clean.loc[~imputed_mask],
                    capture_method='aggregation',
                    source_timeframe='M5',
                    is_imputed=False
                )
                df_clean = pd.concat([df_real, df_imputed]).sort_values('time').reset_index(drop=True)
            else:
                df_clean = traceability_manager.assign_data_origin(
                    df_clean,
                    capture_method='aggregation',
                    source_timeframe='M5',
                    is_imputed=False
                )

            logger.info(f"Trazabilidad aplicada: {df_clean['data_origin'].value_counts().to_dict()}")
                
        except Exception as e:
            logger.warning(f"Error aplicando trazabilidad: {e}")
            logger.info("Continuando sin trazabilidad...")
        
        # 9. Aplicar tracking final a todos los records
        logger.info("Aplicando tracking final de calidad...")
        try:
            df_clean.apply(engine.quality_tracker.track_record, axis=1)
        except Exception as e:
            logger.warning(f"Error en tracking de calidad: {e}")
            logger.info("Continuando sin tracking de calidad...")
        
        # Logging mejorado para debugging
        logger.info(f"Tipos de datos finales: {df_clean.dtypes.value_counts()}")
        object_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            logger.info(f"Columnas con objetos: {object_cols}")
        else:
            logger.info("No hay columnas con objetos")
        
        logger.info(f"Total final: {len(df_clean):,} registros con {len(df_clean.columns)} columnas")
        
        # Tiempo total de ejecución
        engine.quality_tracker.execution_times['total'] = time.time() - pipeline_start_time
        
        # Calcular completitud diaria usando configuracion de mercado
        try:
            start_range = df_clean['time'].min()
            end_range = df_clean['time'].max()
            expected_map = engine.quality_tracker.calculate_expected_bars(start_range, end_range)
            df_clean['date'] = df_clean['time'].dt.floor('D')
            for day, expected in expected_map.items():
                captured = (df_clean['date'] == day).sum()
                engine.quality_tracker.update_daily_completeness(day, int(captured), expected)
            df_clean.drop(columns=['date'], inplace=True)
        except Exception as e:
            logger.warning(f"Error calculando completitud diaria: {e}")

        # Generar reporte de calidad
        logger.info("\nGenerando reporte de calidad...")
        quality_report = engine.get_quality_report()
        
        # Analisis paralelo (Dask si esta disponible)
        if DASK_AVAILABLE and len(df_clean) > 100000:
            analysis = analyze_data_parallel_dask(df_clean, DASK_WORKERS)
        else:
            analysis = analyze_data_by_period(df_clean)
        
        # ✅ GUARDAR DATASET EN TODOS LOS FORMATOS COMO EN LOS SCRIPTS DE REFERENCIA
        io_start = time.time()
        base_path = os.path.join(args.output, f"{symbol.lower()}_m5_hpc_{start_dt.year}_{end_dt.year}")
        
        try:
            saved_files = save_dataset_optimized(df_clean, base_path, args.formats)
            engine.quality_tracker.execution_times['io'] = time.time() - io_start
            logger.info("Dataset guardado exitosamente en todos los formatos")
        except Exception as e:
            logger.error(f"Error guardando dataset: {e}")
            logger.info("Intentando guardar en formato CSV como fallback...")
            
            # Fallback a CSV simple
            try:
                csv_path = f"{base_path}.csv"
                df_clean.to_csv(csv_path, index=False)
                saved_files = {'csv': csv_path}
                logger.info(f"Dataset guardado en CSV como fallback: {csv_path}")
            except Exception as csv_error:
                logger.error(f"Error en fallback CSV: {csv_error}")
                logger.error("No se pudo guardar el dataset")
                return 1
        
        # Calcular checksums
        metadata = {
            'file_checksums': {}
        }
        for format_type, filepath in saved_files.items():
            if os.path.exists(filepath):
                checksum = calculate_file_checksum(filepath)
                metadata['file_checksums'][os.path.basename(filepath)] = checksum
                logger.info(f"SHA-256 {format_type}: {checksum[:16]}...")
                logger.info(f"[LOG] Archivo guardado: {filepath}")
        
        # ✅ GENERAR TODOS LOS REPORTES COMO EN LOS SCRIPTS DE REFERENCIA
        logger.info("Generando reportes...")
        
        # Reporte JSON principal
        try:
            json_report_path = os.path.join(args.output, f"report_{symbol.lower()}_hpc_{start_dt.year}_{end_dt.year}.json")
            generate_json_report_enhanced(df_clean, analysis, {}, quality_report, 
                                        symbol, json_report_path, metadata)
            logger.info(f"Reporte JSON generado: {json_report_path}")
        except Exception as e:
            logger.error(f"Error generando reporte JSON: {e}")
            json_report_path = None
        
        # Reporte de calidad JSON
        try:
            quality_report_path = os.path.join(args.output, f"quality_report_{symbol.lower()}_{start_dt.year}_{end_dt.year}.json")
            engine.save_quality_report(quality_report_path)
            logger.info(f"Reporte de calidad JSON generado: {quality_report_path}")
        except Exception as e:
            logger.error(f"Error generando reporte de calidad JSON: {e}")
            quality_report_path = None
        
        # Reporte de calidad Markdown
        try:
            quality_md_path = os.path.join(args.output, f"QUALITY_ANALYSIS_{symbol.upper()}_{start_dt.year}_{end_dt.year}.md")
            generate_quality_markdown_report(quality_report, quality_md_path, metadata, 
                                           engine.quality_tracker.execution_times)
            logger.info(f"Análisis de calidad MD generado: {quality_md_path}")
        except Exception as e:
            logger.error(f"Error generando análisis de calidad MD: {e}")
            quality_md_path = None
        
        # Resumen ejecutivo Markdown
        try:
            md_report_path = os.path.join(args.output, f"EXECUTIVE_SUMMARY_{symbol.upper()}_{start_dt.year}_{end_dt.year}.md")
            generate_markdown_report_enhanced(df_clean, analysis, {}, 
                                            quality_report, symbol, md_report_path)
            logger.info(f"Resumen ejecutivo generado: {md_report_path}")
        except Exception as e:
            logger.error(f"Error generando resumen ejecutivo: {e}")
            md_report_path = None
        
        # ✅ GENERAR ARCHIVOS CSV DE ANÁLISIS COMO EN LOS SCRIPTS DE REFERENCIA
        logger.info("\nGenerando archivos CSV de análisis...")
        
        # Generar CSV de completitud
        try:
            completeness_csv = f"{args.output}/{symbol.lower()}_m5_hpc_{start_dt.year}_{end_dt.year}_completeness.csv"
            engine.quality_tracker.generate_completeness_heatmap(completeness_csv)
            logger.info(f"CSV de completitud guardado: {completeness_csv}")
        except Exception as e:
            logger.error(f"Error generando CSV de completitud: {e}")
            completeness_csv = None
        
        # Generar CSV de resumen anual
        try:
            yearly_csv = f"{args.output}/{symbol.lower()}_m5_hpc_{start_dt.year}_{end_dt.year}_yearly_summary.csv"
            yearly_data = []
            for year in sorted(engine.quality_tracker.yearly_tracking.keys()):
                data = engine.quality_tracker.yearly_tracking[year]
                total = data.get('total_records', 0)
                if total > 0:
                    yearly_data.append({
                        'year': year,
                        'total_records': total,
                        'native_m5': data.get('native_m5', 0),
                        'native_m5_pct': (data.get('native_m5', 0) / total * 100) if total else 0,
                        'aggregated_m1': data.get('aggregated_m1', 0),
                        'aggregated_m1_pct': (data.get('aggregated_m1', 0) / total * 100) if total else 0,
                        'imputed': data.get('imputed_records', 0),
                        'imputed_pct': (data.get('imputed_records', 0) / total * 100) if total else 0,
                        'real_captured': data.get('real_captured', 0),
                        'real_captured_pct': (data.get('real_captured', 0) / total * 100) if total else 0
                    })
            
            if yearly_data:
                df_yearly = pd.DataFrame(yearly_data)
                df_yearly.to_csv(yearly_csv, index=False)
                logger.info(f"CSV de resumen anual guardado: {yearly_csv}")
            else:
                logger.warning("No hay datos anuales para generar CSV")
                yearly_csv = None
        except Exception as e:
            logger.error(f"Error generando CSV de resumen anual: {e}")
            yearly_csv = None
        
        # Generar CSV de resumen mensual
        try:
            monthly_csv = f"{args.output}/{symbol.lower()}_m5_hpc_{start_dt.year}_{end_dt.year}_monthly_summary.csv"
            monthly_data = []
            for year in sorted(engine.quality_tracker.monthly_tracking.keys()):
                for month in sorted(engine.quality_tracker.monthly_tracking[year].keys()):
                    data = engine.quality_tracker.monthly_tracking[year][month]
                    total = data.get('total_records', 0)
                    if total > 0:
                        monthly_data.append({
                            'year': year,
                            'month': month,
                            'month_name': datetime(year, month, 1).strftime('%B'),
                            'total_records': total,
                            'expected_bars': data.get('expected_bars', 0),
                            'completeness_pct': (total / data.get('expected_bars', 1) * 100) if data.get('expected_bars', 0) > 0 else 0,
                            'native_m5': data.get('native_m5', 0),
                            'native_m5_pct': (data.get('native_m5', 0) / total * 100) if total else 0,
                            'aggregated_m1': data.get('aggregated_m1', 0),
                            'aggregated_m1_pct': (data.get('aggregated_m1', 0) / total * 100) if total else 0,
                            'imputed': data.get('imputed_records', 0),
                            'imputed_pct': (data.get('imputed_records', 0) / total * 100) if total else 0
                        })
            
            if monthly_data:
                df_monthly = pd.DataFrame(monthly_data)
                df_monthly.to_csv(monthly_csv, index=False)
                logger.info(f"CSV de resumen mensual guardado: {monthly_csv}")
            else:
                logger.warning("No hay datos mensuales para generar CSV")
                monthly_csv = None
        except Exception as e:
            logger.error(f"Error generando CSV de resumen mensual: {e}")
            monthly_csv = None
        
        # Generar CSV de gaps detectados
        try:
            gaps_csv = f"{args.output}/{symbol.lower()}_m5_hpc_{start_dt.year}_{end_dt.year}_gaps_detected.csv"
            gaps_data = []
            for date, gaps in engine.quality_tracker.gap_tracking.items():
                for gap in gaps:
                    gaps_data.append({
                        'date': getattr(date, 'isoformat', lambda: str(date))(),
                        'gap_start': gap.get('start').isoformat() if gap.get('start') else '',
                        'gap_end': gap.get('end').isoformat() if gap.get('end') else '',
                        'gap_minutes': gap.get('minutes', 0),
                        'bars_missing': gap.get('bars_missing', 0),
                        'filled': gap.get('filled_by') is not None,
                        'filled_by': gap.get('filled_by', 'N/A') or 'N/A',
                        'variance': gap.get('variance', 0.0)
                    })
            
            if gaps_data:
                df_gaps = pd.DataFrame(gaps_data)
                df_gaps.to_csv(gaps_csv, index=False)
                logger.info(f"CSV de gaps guardado: {gaps_csv}")
            else:
                logger.info("No hay gaps detectados para generar CSV")
                gaps_csv = None
        except Exception as e:
            logger.error(f"Error generando CSV de gaps: {e}")
            gaps_csv = None
        
        # ✅ RESUMEN FINAL MEJORADO COMO EN LOS SCRIPTS DE REFERENCIA
        logger.info("\n" + "="*80)
        logger.info("PIPELINE HPC v2.4 COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info(f"Símbolo: {symbol}")
        logger.info(f"Registros totales: {len(df_clean):,}")
        logger.info(f"Período: {df_clean['time'].min()} - {df_clean['time'].max()}")
        logger.info(f"Calidad promedio: {df_clean['quality_score'].mean():.3f}")
        logger.info(f"Features generados: {len([c for c in df_clean.columns if c not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'data_flag', 'quality_score']])}")
        logger.info(f"Memoria utilizada: {df_clean.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Métricas de calidad
        logger.info("\nMétricas de Calidad:")
        try:
            overall_completeness = quality_report.get('summary', {}).get('overall_completeness', None)
            if overall_completeness is not None:
                logger.info(f"  - Completitud global: {overall_completeness:.2f}%")
            else:
                logger.warning("  - Completitud global: No disponible en quality_report")
        except Exception as e:
            logger.warning(f"  - Completitud global: Error accediendo a la métrica: {e}")
        try:
            gap_summary = quality_report.get('gap_analysis', {}).get('summary', {})
            total_gaps = gap_summary.get('total_gaps', 'N/A')
            gaps_filled = gap_summary.get('gaps_filled', 'N/A')
            gap_fill_rate = quality_report.get('gap_analysis', {}).get('gap_fill_rate', 'N/A')
            logger.info(f"  - Gaps detectados: {total_gaps}")
            logger.info(f"  - Gaps llenados: {gaps_filled} ({gap_fill_rate}%)")
        except Exception as e:
            logger.warning(f"  - Gaps: Error accediendo a métricas de gaps: {e}")
        
        # Años sin datos
        if years_without_data:
            logger.warning(f"\nAños sin datos disponibles: {years_without_data}")
            logger.info("Esto es normal para cuentas demo o brokers con histórico limitado")
        
        logger.info("\nArchivos generados:")
        for format_type, path in saved_files.items():
            logger.info(f"  - {format_type.upper()}: {path}")
        
        # Reportes generados
        if json_report_path:
            logger.info(f"[LOG] Reporte JSON: {json_report_path}")
        if quality_report_path:
            logger.info(f"[LOG] Reporte de calidad JSON: {quality_report_path}")
        if quality_md_path:
            logger.info(f"[LOG] Análisis de calidad MD: {quality_md_path}")
        if md_report_path:
            logger.info(f"[LOG] Resumen ejecutivo: {md_report_path}")
        
        # CSVs generados
        if completeness_csv:
            logger.info(f"[LOG] CSV de completitud: {completeness_csv}")
        if yearly_csv:
            logger.info(f"[LOG] CSV resumen anual: {yearly_csv}")
        if monthly_csv:
            logger.info(f"[LOG] CSV resumen mensual: {monthly_csv}")
        if gaps_csv:
            logger.info(f"[LOG] CSV gaps detectados: {gaps_csv}")
        
        # Estadísticas de rendimiento HPC
        logger.info("\nEstadísticas HPC:")
        if hasattr(engine, 'use_ray'):
            logger.info(f"  - Ray workers utilizados: {engine.use_ray and engine.workers or 0} {'(fallback a ThreadPool)' if not engine.use_ray and not args.no_ray else ''}")
        else:
            logger.info(f"  - Ray workers utilizados: 0")
        logger.info(f"  - GPU utilizada: {'Si' if use_gpu else 'No'}")
        logger.info(f"  - Funciones Numba compiladas: {'Si' if NUMBA_AVAILABLE else 'No'}")
        logger.info(f"  - Análisis Dask: {'Si' if DASK_AVAILABLE and len(df_clean) > 100000 else 'No'}")
        
        # Validar archivos generados
        logger.info("\nValidando archivos generados...")
        for format_type, filepath in saved_files.items():
            try:
                if format_type == 'parquet':
                    df_test = pd.read_parquet(filepath)
                elif format_type == 'csv':
                    # Intentar con compresión primero, luego sin compresión
                    try:
                        df_test = pd.read_csv(filepath, compression='gzip', nrows=1000)
                    except:
                        df_test = pd.read_csv(filepath, nrows=1000)
                elif format_type == 'feather':
                    df_test = pd.read_feather(filepath)
                logger.info(f"[OK] Validación OK: {filepath} ({len(df_test)} filas, {len(df_test.columns)} columnas)")
            except Exception as e:
                logger.error(f"[ERROR] Error al validar archivo {filepath}: {e}")
                logger.warning(f"El archivo {filepath} puede estar corrupto o ser inaccesible")
        
        logger.info("[DEBUG] FIN DE LA FUNCIÓN MAIN DEL PIPELINE - ÉXITO")
        return 0
        
    except Exception as e:
        logger.error(f"Error crítico en el pipeline: {e}", exc_info=True)
        logger.info("[DEBUG] RETURN 1: Excepción capturada en el pipeline")
        return 1
        
    finally:
        # Cerrar MT5
        try:
            mt5.shutdown()
            logger.info("\nConexión MT5 cerrada")
        except Exception as e:
            logger.debug(f"Error cerrando MT5: {e}")
        
        # Cerrar Ray si está activo
        if RAY_AVAILABLE and ray.is_initialized():
            try:
                ray.shutdown()
                logger.info("Ray shutdown completado")
            except Exception as e:
                logger.debug(f"Error cerrando Ray: {e}")
        logger.info("[DEBUG] FINALLY ejecutado en main()")

def process_chunk_multiprocessing(chunk_data):
    """Procesar un chunk de datos usando multiprocessing nativo"""
    try:
        df_chunk, year, symbol = chunk_data
        
        # Simular procesamiento de chunk
        if not df_chunk.empty:
            # Aplicar filtrado básico
            df_filtered = filter_market_hours_modular(df_chunk, instrument='US500')
            
            # Generar features básicas
            if len(df_filtered) > 0:
                df_features = generate_features_cpu_optimized(df_filtered)
                return df_features
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error procesando chunk {year}: {e}")
        return pd.DataFrame()

def process_data_multiprocessing(df: pd.DataFrame, symbol: str, max_workers: int = None) -> pd.DataFrame:
    """Procesar datos usando multiprocessing nativo como alternativa a Ray"""
    if df.empty:
        return df
    
    if max_workers is None:
        max_workers = min(os.cpu_count(), 8)  # Máximo 8 workers
    
    logger.info(f"Procesando datos con multiprocessing nativo ({max_workers} workers)")
    
    try:
        from multiprocessing import Pool
        
        # Dividir datos por años para procesamiento paralelo
        df['year'] = df['time'].dt.year
        chunks = []
        
        for year in df['year'].unique():
            year_data = df[df['year'] == year].copy()
            if not year_data.empty:
                chunks.append((year_data, year, symbol))
        
        # Procesar chunks en paralelo
        with Pool(processes=max_workers) as pool:
            results = pool.map(process_chunk_multiprocessing, chunks)
        
        # Combinar resultados
        valid_results = [r for r in results if not r.empty]
        
        if valid_results:
            combined_df = pd.concat(valid_results, ignore_index=True)
            combined_df = combined_df.sort_values('time').reset_index(drop=True)
            
            # Remover columna auxiliar
            if 'year' in combined_df.columns:
                combined_df = combined_df.drop('year', axis=1)
            
            logger.info(f"Procesamiento multiprocessing completado: {len(combined_df)} registros")
            return combined_df
        else:
            logger.warning("No se obtuvieron resultados válidos del procesamiento multiprocessing")
            return df
            
    except Exception as e:
        logger.error(f"Error en multiprocessing: {e}")
        logger.info("Fallback a procesamiento secuencial")
        return df

if __name__ == "__main__":
    sys.exit(main())
