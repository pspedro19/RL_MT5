#!/usr/bin/env python3
"""
Funciones de procesamiento, filtrado e imputación de datos
"""
import warnings
import logging
import gc
from typing import Dict, Union, List
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import pytz # Added for is_within_market_hours

from utils.market_calendar import get_us_market_holidays, is_forex_market_open
from features.technical.indicators import brownian_bridge_numba, NUMBA_AVAILABLE
from data.quality.quality_tracker import DataQualityTracker
from config.constants import MARKET_PARAMS, MARKET_CONFIGS, RESAMPLING_CONFIG, DATA_ORIGINS, DATA_ORIGIN_VALUES
from utils.data_traceability import DataTraceabilityManager, validate_dataframe_traceability

logger = logging.getLogger('data_processing')

# Detectar Polars si está disponible
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None
    warnings.warn("Polars no disponible. Instalarlo mejorará el rendimiento: pip install polars")

# Detectar Dask si está disponible
try:
    import dask.dataframe as dd
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("Dask no disponible. Instalarlo mejorará el análisis paralelo: pip install dask[complete]")

# ===============================================================================
# PROCESAMIENTO Y CALIDAD DE DATOS OPTIMIZADOS
# ===============================================================================

def filter_forex_hours_polars(df: Union['pl.DataFrame', pd.DataFrame]) -> Union['pl.DataFrame', pd.DataFrame]:
    """Filtrar horarios de mercado Forex usando Polars (más rápido)"""
    if not POLARS_AVAILABLE or not isinstance(df, pl.DataFrame):
        return filter_forex_hours_pandas(df)
    
    try:
        logger.info("Aplicando filtro de horarios de mercado Forex con Polars")
        
        # Agregar columnas de día de la semana y hora
        df_filtered = df.with_columns([
            pl.col('time').dt.weekday().alias('weekday'),
            pl.col('time').dt.hour().alias('hour')
        ])
        
        # Filtrar: excluir sábados y domingos antes de las 22:00 UTC
        df_filtered = df_filtered.filter(
            ~((pl.col('weekday') == 5) |  # Sábados
              (pl.col('weekday') == 6) & (pl.col('hour') < 22))  # Domingos antes de 22:00
        )
        
        # Remover columnas auxiliares
        df_filtered = df_filtered.drop(['weekday', 'hour'])
        
        logger.info(f"Filtrado Forex completado: {len(df_filtered)} registros")
        return df_filtered
        
    except Exception as e:
        logger.warning(f"Error en filtrado Polars, usando Pandas: {e}")
        return filter_forex_hours_pandas(df)

def filter_forex_hours_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Filtrar horarios de mercado Forex usando pandas (fallback)"""
    if df.empty:
        return df
    
    logger.info("Aplicando filtro de horarios de mercado Forex con pandas")
    
    # Crear máscara para horarios de mercado Forex
    forex_mask = df['time'].apply(is_forex_market_open)
    
    df_filtered = df[forex_mask].copy()
    
    logger.info(f"Filtrado Forex completado: {len(df_filtered)} registros")
    return df_filtered

def filter_market_hours(df: pd.DataFrame, instrument: str = 'US500') -> pd.DataFrame:
    """Filtrar horarios de mercado según el instrumento - PRIMERA ETAPA CRÍTICA"""
    if df.empty:
        return df
    
    logger.info(f"Aplicando filtro de horarios de mercado para {instrument} - PRIMERA ETAPA")
    
    # Importar aquí para evitar dependencias circulares
    from utils.market_hours_filter import apply_market_hours_filter
    
    # Aplicar filtro robusto de horarios de mercado
    df_filtered, market_analysis = apply_market_hours_filter(df, instrument)
    
    logger.info(f"Filtrado de horarios completado: {len(df_filtered)} registros después del filtro")
    
    return df_filtered

def detect_gaps_optimized(df: pd.DataFrame, instrument: str = 'US500') -> pd.DataFrame:
    """Detectar gaps en los datos y retornar DataFrame de gaps con clasificación detallada"""
    if df.empty:
        return pd.DataFrame(columns=['gap_start', 'gap_end', 'gap_minutes', 'market_hours', 'gap_type', 'imputation_status', 'reason'])
    
    logger.info("Detectando gaps en los datos con clasificación detallada...")
    
    # Ordenar por tiempo
    df = df.sort_values('time').reset_index(drop=True)
    
    # Calcular diferencias de tiempo
    df['time_diff'] = df['time'].diff()
    
    # Identificar gaps (diferencia mayor a 5 minutos para M5)
    market_config = MARKET_CONFIGS.get(instrument, MARKET_CONFIGS['US500'])
    expected_interval = pd.Timedelta(minutes=5)  # M5 timeframe
    
    # Para Forex, considerar gaps más grandes debido a operación 24/5
    if instrument == 'USDCOP':
        gap_threshold = pd.Timedelta(minutes=10)  # Mayor tolerancia para Forex
    else:
        gap_threshold = pd.Timedelta(minutes=7)   # Tolerancia para mercado USA
    
    # Encuentra los índices donde hay gaps
    gap_indices = df.index[df['time_diff'] > gap_threshold].tolist()
    
    gaps = []
    for idx in gap_indices:
        gap_start = df.loc[idx - 1, 'time']
        gap_end = df.loc[idx, 'time']
        gap_minutes = int((gap_end - gap_start).total_seconds() // 60)
        
        # Determinar si el gap está dentro del horario de mercado
        is_market_hours = is_within_market_hours(gap_start, gap_end, instrument)
        
        # Clasificar el tipo de gap
        if is_market_hours:
            gap_type = 'market_hours'
            if gap_minutes <= 30:
                imputation_status = 'imputable'
                reason = 'Gap pequeño dentro de horario de mercado'
            else:
                imputation_status = 'not_imputable'
                reason = 'Gap muy grande para imputación segura'
        else:
            gap_type = 'outside_market_hours'
            imputation_status = 'ignored'
            reason = 'Fuera del horario de mercado (normal)'
        
        gaps.append({
            'gap_start': gap_start,
            'gap_end': gap_end,
            'gap_minutes': gap_minutes,
            'market_hours': is_market_hours,
            'gap_type': gap_type,
            'imputation_status': imputation_status,
            'reason': reason
        })
    
    if gaps:
        # Estadísticas de gaps
        market_hours_gaps = [g for g in gaps if g['market_hours']]
        outside_market_gaps = [g for g in gaps if not g['market_hours']]
        imputable_gaps = [g for g in gaps if g['imputation_status'] == 'imputable']
        not_imputable_gaps = [g for g in gaps if g['imputation_status'] == 'not_imputable']
        ignored_gaps = [g for g in gaps if g['imputation_status'] == 'ignored']
        
        logger.info(f"Detectados {len(gaps)} gaps totales:")
        logger.info(f"  - Dentro de horario de mercado: {len(market_hours_gaps)}")
        logger.info(f"  - Fuera de horario de mercado: {len(outside_market_gaps)}")
        logger.info(f"  - Imputables: {len(imputable_gaps)}")
        logger.info(f"  - No imputables: {len(not_imputable_gaps)}")
        logger.info(f"  - Ignorados (fuera de horario): {len(ignored_gaps)}")
        
        if market_hours_gaps:
            max_market_gap = max(g['gap_minutes'] for g in market_hours_gaps)
            avg_market_gap = sum(g['gap_minutes'] for g in market_hours_gaps) / len(market_hours_gaps)
            logger.info(f"  - Gap más largo en horario de mercado: {max_market_gap} minutos")
            logger.info(f"  - Gap promedio en horario de mercado: {avg_market_gap:.1f} minutos")
    else:
        logger.info("No se detectaron gaps significativos")
    
    return pd.DataFrame(gaps, columns=['gap_start', 'gap_end', 'gap_minutes', 'market_hours', 'gap_type', 'imputation_status', 'reason'])

def is_within_market_hours(start_time: datetime, end_time: datetime, instrument: str = 'US500') -> bool:
    """Determinar si un período está dentro del horario de mercado (robusto a naive/aware)"""
    import pytz
    utc = pytz.UTC
    eastern = pytz.timezone('US/Eastern')

    def ensure_aware(dt):
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            return utc.localize(dt)
        return dt

    start_time = ensure_aware(start_time)
    end_time = ensure_aware(end_time)
    start_et = start_time.astimezone(eastern)
    end_et = end_time.astimezone(eastern)

    if instrument == 'USDCOP':
        # Forex opera 24/5, pero consideramos horario principal
        if start_et.weekday() >= 5 or end_et.weekday() >= 5:
            return False
        start_hour = start_et.hour
        end_hour = end_et.hour
        return 8 <= start_hour <= 17 and 8 <= end_hour <= 17
    else:
        # Mercado USA: 9:30 AM - 4:00 PM ET, Lunes a Viernes
        if start_et.weekday() >= 5 or end_et.weekday() >= 5:
            return False
        start_hour = start_et.hour + start_et.minute / 60
        end_hour = end_et.hour + end_et.minute / 60
        return 9.5 <= start_hour <= 16 and 9.5 <= end_hour <= 16

def brownian_bridge_imputation_numba_tracked(df: pd.DataFrame, quality_tracker: DataQualityTracker, 
                                            max_gap_minutes: int = 30) -> pd.DataFrame:
    """Imputacion optimizada usando Numba con tracking y clasificación de gaps"""
    if df.empty or not NUMBA_AVAILABLE:
        return df
    
    logger.info("Aplicando imputacion Brownian Bridge con Numba y tracking mejorado")
    
    # Inicializar gestor de trazabilidad
    traceability_manager = DataTraceabilityManager()
    
    gaps_df = detect_gaps_optimized(df)
    if gaps_df.empty:
        logger.info("No hay gaps para imputar")
        return df
    
    # Solo imputar gaps que están dentro de horario de mercado y son imputables
    imputable_gaps = gaps_df[
        (gaps_df['imputation_status'] == 'imputable') & 
        (gaps_df['gap_minutes'] <= max_gap_minutes)
    ]
    
    # Gaps que no se imputarán (para tracking)
    not_imputable_gaps = gaps_df[gaps_df['imputation_status'] == 'not_imputable']
    ignored_gaps = gaps_df[gaps_df['imputation_status'] == 'ignored']
    
    logger.info(f"Gaps clasificados:")
    logger.info(f"  - Imputables: {len(imputable_gaps)}")
    logger.info(f"  - No imputables: {len(not_imputable_gaps)}")
    logger.info(f"  - Ignorados (fuera de horario): {len(ignored_gaps)}")
    
    # Track gaps no imputados
    for _, gap in not_imputable_gaps.iterrows():
        quality_tracker.track_gap(
            gap['gap_start'], gap['gap_end'], gap['gap_minutes'],
            filled_by='not_imputed', reason=gap['reason']
        )
    
    for _, gap in ignored_gaps.iterrows():
        quality_tracker.track_gap(
            gap['gap_start'], gap['gap_end'], gap['gap_minutes'],
            filled_by='ignored', reason=gap['reason']
        )
    
    df = df.set_index('time').sort_index()
    imputed_records = []
    
    for _, gap in imputable_gaps.iterrows():
        gap_start = gap['gap_start']
        gap_end = gap['gap_end']
        
        gap_times = pd.date_range(
            start=gap_start + pd.Timedelta(minutes=5),
            end=gap_end - pd.Timedelta(minutes=5),
            freq='5min'
        )
        
        if len(gap_times) == 0:
            continue
        
        try:
            data_before = df.loc[gap_start]
            data_after = df.loc[gap_end]
        except KeyError:
            continue
        
        # Usar Numba para generar valores imputados
        steps = len(gap_times)
        
        # Calcular varianza para el gap
        gap_variance = 0.0
        
        for col in ['open', 'high', 'low', 'close']:
            if not pd.isna(data_before[col]) and not pd.isna(data_after[col]):
                values = brownian_bridge_numba(
                    float(data_before[col]), 
                    float(data_after[col]), 
                    steps
                )
                
                # Calcular varianza de los valores imputados
                if len(values) > 0:
                    gap_variance += np.var(values)
                
                for i, (t, val) in enumerate(zip(gap_times, values)):
                    if i >= len(imputed_records):
                        imputed_records.append({
                            'time': t,
                            'tick_volume': 0,
                            'spread': df['spread'].mean() if 'spread' in df.columns else 0,
                            'real_volume': 0
                        })
                    imputed_records[i][col] = val
                    
                    # Track la imputación
                    quality_tracker.track_imputation(
                        t, 'brownian_bridge', f"interpolated_{col}"
                    )
        
        # Track que el gap fue llenado con varianza estimada
        quality_tracker.track_gap(
            gap_start, gap_end, gap['gap_minutes'], 
            filled_by='brownian_bridge',
            variance=gap_variance / 4  # Promedio de las 4 columnas
        )
    
    if imputed_records:
        imputed_df = pd.DataFrame(imputed_records).set_index('time')
        
        # Ajustar high/low
        imputed_df['high'] = imputed_df[['high', 'open', 'close']].max(axis=1)
        imputed_df['low'] = imputed_df[['low', 'open', 'close']].min(axis=1)
        
        # Asignar data_origin estandarizado para datos imputados
        # Cada registro imputado debe tener su propio origen
        for idx in imputed_df.index:
            imputed_df.loc[idx, 'data_origin'] = 'M5_IMPUTADO_BROWNIAN'
            imputed_df.loc[idx, 'quality_score'] = DATA_ORIGINS['M5_IMPUTADO_BROWNIAN']['quality_score']
        
        df = pd.concat([df, imputed_df]).sort_index()
    
    df = df.reset_index()
    logger.info(f"Registros imputados: {len(imputed_records)}")
    
    return df

def clean_data_optimized(df: pd.DataFrame, instrument: str = 'US500') -> pd.DataFrame:
    """Limpiar datos de manera optimizada"""
    if df.empty:
        return df
    
    logger.info("Limpiando datos...")
    
    initial_rows = len(df)
    
    # Remover filas con valores nulos en columnas críticas
    critical_columns = ['time', 'open', 'high', 'low', 'close']
    df_clean = df.dropna(subset=critical_columns)
    
    # Verificar que high >= low
    df_clean = df_clean[df_clean['high'] >= df_clean['low']]
    
    # Verificar que open y close están entre high y low
    df_clean = df_clean[
        (df_clean['open'] >= df_clean['low']) & 
        (df_clean['open'] <= df_clean['high']) &
        (df_clean['close'] >= df_clean['low']) & 
        (df_clean['close'] <= df_clean['high'])
    ]
    
    # Remover duplicados de tiempo
    df_clean = df_clean.drop_duplicates(subset=['time'])
    
    # Ordenar por tiempo
    df_clean = df_clean.sort_values('time').reset_index(drop=True)
    
    rows_removed = initial_rows - len(df_clean)
    logger.info(f"Limpieza completada: {rows_removed} filas removidas")
    
    return df_clean

def validate_data_integrity(df: pd.DataFrame, instrument: str = 'US500') -> Dict:
    """Validar integridad de los datos con verificaciones avanzadas"""
    validation_results = {
        'valid': True,
        'issues': [],
        'summary': {},
        'outliers': {},
        'temporal_issues': {}
    }
    
    if df.empty:
        validation_results['valid'] = False
        validation_results['issues'].append('DataFrame vacío')
        return validation_results
    
    # Verificar columnas requeridas
    required_columns = ['time', 'open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_results['valid'] = False
        validation_results['issues'].append(f'Columnas faltantes: {missing_columns}')
    
    # Verificar tipos de datos
    if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
        validation_results['valid'] = False
        validation_results['issues'].append('Columna time no es datetime')
    
    # Verificar valores negativos en precios
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df.columns:
            if (df[col] < 0).any():
                validation_results['valid'] = False
                validation_results['issues'].append(f'Valores negativos en {col}')
            
            # Detectar outliers usando método IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                validation_results['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'min_outlier': outliers[col].min(),
                    'max_outlier': outliers[col].max()
                }
    
    # Verificar lógica de precios OHLC
    if all(col in df.columns for col in price_columns):
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['open'] < df['low']) | (df['open'] > df['high']) |
            (df['close'] < df['low']) | (df['close'] > df['high'])
        )
        if invalid_ohlc.any():
            validation_results['valid'] = False
            validation_results['issues'].append('Violaciones de lógica OHLC')
    
    # Verificar continuidad temporal
    if 'time' in df.columns:
        df_sorted = df.sort_values('time')
        time_diffs = df_sorted['time'].diff()
        
        # Verificar gaps inesperados (más de 10 minutos para M5)
        large_gaps = time_diffs[time_diffs > pd.Timedelta(minutes=10)]
        if len(large_gaps) > 0:
            validation_results['temporal_issues']['large_gaps'] = {
                'count': len(large_gaps),
                'max_gap_minutes': large_gaps.max().total_seconds() / 60,
                'avg_gap_minutes': large_gaps.mean().total_seconds() / 60
            }
        
        # Verificar duplicados temporales
        duplicates = df_sorted.duplicated(subset=['time'])
        if duplicates.any():
            validation_results['temporal_issues']['duplicates'] = {
                'count': duplicates.sum(),
                'percentage': duplicates.sum() / len(df) * 100
            }
    
    # Verificar rangos de precios razonables
    if all(col in df.columns for col in price_columns):
        price_range = df['high'].max() - df['low'].min()
        avg_price = df['close'].mean()
        
        # Si el rango es más del 50% del precio promedio, puede ser sospechoso
        if price_range > avg_price * 0.5:
            validation_results['issues'].append(f'Rango de precios sospechosamente grande: {price_range:.2f} vs precio promedio: {avg_price:.2f}')
    
    # Resumen estadístico
    validation_results['summary'] = {
        'total_records': len(df),
        'date_range': {
            'start': df['time'].min().isoformat() if not df.empty else None,
            'end': df['time'].max().isoformat() if not df.empty else None
        },
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated(subset=['time']).sum(),
        'price_stats': {
            'min_price': float(df['close'].min()) if 'close' in df.columns else None,
            'max_price': float(df['close'].max()) if 'close' in df.columns else None,
            'avg_price': float(df['close'].mean()) if 'close' in df.columns else None,
            'std_price': float(df['close'].std()) if 'close' in df.columns else None
        }
    }
    
    return validation_results

def get_market_config(instrument: str):
    """Obtener configuración de mercado para un instrumento específico"""
    return MARKET_CONFIGS.get(instrument, MARKET_CONFIGS['US500'])

# ===============================================================================
# ANALISIS DE DATOS
# ===============================================================================

def analyze_data_by_period(df: pd.DataFrame) -> Dict:
    """Analisis detallado de datos por periodo"""
    if df.empty:
        return {'summary': {'total_records': 0, 'error': 'DataFrame vacio'}}
    
    logger.info("Realizando analisis de datos...")
    
    analysis = {
        'summary': {
            'total_records': len(df),
            'date_range': {
                'start': df['time'].min().isoformat(),
                'end': df['time'].max().isoformat()
            },
            'trading_days': df['time'].dt.date.nunique(),
            'records_per_day': len(df) / df['time'].dt.date.nunique() if df['time'].dt.date.nunique() > 0 else 0
        }
    }
    
    return analysis

def analyze_data_parallel_dask(df: pd.DataFrame, dask_workers: int = 8) -> Dict:
    """Analisis paralelo de datos usando Dask"""
    if not DASK_AVAILABLE or len(df) < 100000:
        # Fallback a analisis estandar para datasets pequenos
        return analyze_data_by_period(df)
    
    logger.info("Realizando analisis paralelo con Dask")
    
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
                    'close': ['mean', 'std']
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
                    'close': ['mean', 'std']
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
        }
    }
    
    return analysis

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

def load_dataset_optimized(file_path: str) -> pd.DataFrame:
    """Cargar dataset de forma optimizada según formato"""
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path, engine='pyarrow')
    elif file_path.endswith('.feather'):
        return pd.read_feather(file_path)
    elif file_path.endswith('.csv') or file_path.endswith('.csv.gz'):
        # Leer CSV con tipos optimizados
        dtype_dict = {
            'open': np.float32,
            'high': np.float32,
            'low': np.float32,
            'close': np.float32,
            'tick_volume': np.int32,
            'spread': np.float32,
            'real_volume': np.int32,
            'quality_score': np.float32
        }
        return pd.read_csv(file_path, parse_dates=['time'], dtype=dtype_dict)
    else:
        raise ValueError(f"Formato no soportado: {file_path}")

# ===============================================================================
# FUNCIONES DE PROCESAMIENTO DE TICKS Y RESAMPLING
# ===============================================================================

def process_ticks_to_ohlc(ticks_df: pd.DataFrame, target_timeframe: str = 'M5') -> pd.DataFrame:
    """Convertir datos de ticks a OHLC en el timeframe especificado"""
    if ticks_df.empty:
        return pd.DataFrame()
    
    logger.info(f"Procesando {len(ticks_df)} ticks a OHLC {target_timeframe}")
    
    # Inicializar gestor de trazabilidad
    traceability_manager = DataTraceabilityManager()
    
    # Obtener configuración de resampling
    resampling_config = RESAMPLING_CONFIG
    target_minutes = resampling_config['target_minutes']
    ohlc_config = resampling_config['ohlc_aggregation']
    
    try:
        # Asegurar que tenemos las columnas necesarias
        required_columns = ['time', 'bid', 'ask']
        if not all(col in ticks_df.columns for col in required_columns):
            logger.error(f"Columnas requeridas faltantes: {required_columns}")
            return pd.DataFrame()
        
        # Calcular precio medio si no existe
        if 'price' not in ticks_df.columns:
            ticks_df['price'] = (ticks_df['bid'] + ticks_df['ask']) / 2
        
        # Crear timestamp redondeado al timeframe objetivo
        ticks_df = ticks_df.copy()
        ticks_df.loc[:, 'time_rounded'] = ticks_df['time'].dt.floor(f'{target_minutes}min')
        
        # Agregar a OHLC
        ohlc_data = []
        
        for time_group, group in ticks_df.groupby('time_rounded'):
            if len(group) == 0:
                continue
            
            # Calcular OHLC
            ohlc_bar = {
                'time': time_group,
                'open': group['price'].iloc[0],
                'high': group['price'].max(),
                'low': group['price'].min(),
                'close': group['price'].iloc[-1],
                'tick_volume': len(group),
                'real_volume': group.get('volume', 0).sum() if 'volume' in group.columns else 0,
                'spread': (group['ask'] - group['bid']).mean() if 'ask' in group.columns and 'bid' in group.columns else 0
            }
            
            ohlc_data.append(ohlc_bar)
        
        result_df = pd.DataFrame(ohlc_data)
        
        # Asignar data_origin estandarizado para datos agregados desde ticks
        result_df = traceability_manager.assign_data_origin(
            result_df,
            capture_method='aggregation',
            source_timeframe='ticks',
            is_imputed=False
        )
        
        # Ordenar por tiempo
        result_df = result_df.sort_values('time').reset_index(drop=True)
        
        logger.info(f"Resampling completado: {len(result_df)} barras OHLC generadas")
        return result_df
        
    except Exception as e:
        logger.error(f"Error procesando ticks a OHLC: {e}")
        return pd.DataFrame()

def resample_timeframe_to_m5(df: pd.DataFrame, source_timeframe: str) -> pd.DataFrame:
    """Resamplear datos de cualquier timeframe a M5"""
    if df.empty:
        return df
    
    logger.info(f"Resampleando {source_timeframe} a M5")
    
    # Inicializar gestor de trazabilidad
    traceability_manager = DataTraceabilityManager()
    
    # Si ya es M5, no hacer nada
    if source_timeframe == 'M5':
        return df
    
    # Obtener configuración de resampling
    resampling_config = RESAMPLING_CONFIG
    target_minutes = resampling_config['target_minutes']
    ohlc_config = resampling_config['ohlc_aggregation']
    
    try:
        # Crear timestamp redondeado a M5
        df = df.copy()
        df.loc[:, 'time_rounded'] = df['time'].dt.floor(f'{target_minutes}min')
        
        # Agregar a OHLC M5
        ohlc_data = []
        
        for time_group, group in df.groupby('time_rounded'):
            if len(group) == 0:
                continue
            
            # Calcular OHLC
            ohlc_bar = {
                'time': time_group,
                'open': group['open'].iloc[0],
                'high': group['high'].max(),
                'low': group['low'].min(),
                'close': group['close'].iloc[-1],
                'tick_volume': group['tick_volume'].sum(),
                'real_volume': group.get('real_volume', 0).sum(),
                'spread': group.get('spread', 0).mean()
            }
            
            ohlc_data.append(ohlc_bar)
        
        result_df = pd.DataFrame(ohlc_data)
        
        # Asignar data_origin estandarizado para datos resampleados
        result_df = traceability_manager.assign_data_origin(
            result_df,
            capture_method='resampling',
            source_timeframe=source_timeframe,
            is_imputed=False
        )
        
        # Ordenar por tiempo
        result_df = result_df.sort_values('time').reset_index(drop=True)
        
        logger.info(f"Resampling completado: {len(result_df)} barras M5 generadas desde {source_timeframe}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error resampleando {source_timeframe} a M5: {e}")
        return df

def combine_and_prioritize_dataframes(dataframes: List[pd.DataFrame], 
                                    source_timeframes: List[str],
                                    instrument: str = 'US500') -> pd.DataFrame:
    """Combinar múltiples DataFrames priorizando por calidad de datos"""
    if not dataframes:
        return pd.DataFrame()
    
    logger.info("Combinando y priorizando múltiples fuentes de datos")
    
    # Obtener configuración de mercado
    market_config = get_market_config(instrument)
    capture_prefs = market_config.get('capture_preferences', {})
    
    # Definir prioridades de calidad
    quality_priorities = {
        'M5': 1.0,    # Máxima prioridad
        'M1': 0.95,   # Muy alta prioridad
        'M10': 0.85,  # Alta prioridad
        'M15': 0.80,  # Media-alta prioridad
        'M20': 0.75,  # Media prioridad
        'M30': 0.70,  # Media-baja prioridad
        'H1': 0.65,   # Baja prioridad
        'ticks': 0.98 # Muy alta prioridad (datos granulares)
    }
    
    # Preparar DataFrames con información de prioridad
    prioritized_dfs = []
    
    for df, timeframe in zip(dataframes, source_timeframes):
        if df.empty:
            continue
        
        # Asignar prioridad
        priority = quality_priorities.get(timeframe, 0.5)
        
        # Añadir columnas de metadatos
        df_copy = df.copy()
        df_copy['source_priority'] = priority
        df_copy['source_timeframe'] = timeframe
        df_copy['quality_score'] = priority
        
        prioritized_dfs.append(df_copy)
    
    if not prioritized_dfs:
        return pd.DataFrame()
    
    # Combinar todos los DataFrames
    combined_df = pd.concat(prioritized_dfs, ignore_index=True)
    
    # Ordenar por tiempo y prioridad
    combined_df = combined_df.sort_values(['time', 'source_priority'], 
                                         ascending=[True, False])
    
    # Eliminar duplicados manteniendo el de mayor prioridad
    combined_df = combined_df.drop_duplicates(subset=['time'], keep='first')
    
    # Ordenar final por tiempo
    combined_df = combined_df.sort_values('time').reset_index(drop=True)
    
    # Remover columnas auxiliares
    if 'source_priority' in combined_df.columns:
        combined_df = combined_df.drop('source_priority', axis=1)
    
    logger.info(f"Combinación completada: {len(combined_df)} registros únicos")
    
    return combined_df

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizar tipos de datos para reducir uso de memoria"""
    if df.empty:
        return df
    
    logger.info("Optimizando tipos de datos para memoria...")
    initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    df_optimized = df.copy()
    
    # Optimizar tipos numéricos
    for col in df_optimized.columns:
        col_dtype = str(df_optimized.dtypes[col])
        if col_dtype == 'float64':
            # Usar float32 para precios (suficiente precisión)
            if col in ['open', 'high', 'low', 'close', 'spread']:
                df_optimized[col] = df_optimized[col].astype('float32')
            # Usar float32 para features calculadas
            elif 'ema_' in col or 'sma_' in col or 'rsi_' in col or 'bb_' in col or 'atr_' in col:
                df_optimized[col] = df_optimized[col].astype('float32')
            # Usar float32 para volatilidad y otros indicadores
            elif 'volatility_' in col or 'stoch_' in col or 'macd' in col:
                df_optimized[col] = df_optimized[col].astype('float32')
        
        elif col_dtype == 'int64':
            # Usar int32 para volúmenes (suficiente rango)
            if col in ['tick_volume', 'real_volume']:
                df_optimized[col] = df_optimized[col].astype('int32')
            # Usar int16 para features binarias
            elif col in ['is_first_hour', 'is_last_hour', 'is_lunch_time', 'is_morning', 'is_afternoon']:
                df_optimized[col] = df_optimized[col].astype('int16')
            # Usar int8 para features pequeñas
            elif col in ['hour', 'minute', 'day_of_week', 'day_of_month', 'month', 'quarter']:
                df_optimized[col] = df_optimized[col].astype('int8')
    
    final_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
    memory_saved = initial_memory - final_memory
    
    logger.info(f"Memoria optimizada: {initial_memory:.2f}MB -> {final_memory:.2f}MB (ahorro: {memory_saved:.2f}MB, {memory_saved/initial_memory*100:.1f}%)")
    
    return df_optimized
