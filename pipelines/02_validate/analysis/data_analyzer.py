"""
Análisis avanzado de datos para Pipeline 02
Basado en el código de referencia data_extraction_02_simple.py
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional
import warnings

logger = logging.getLogger(__name__)

# Detectar Dask
try:
    import dask.dataframe as dd
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("Dask no disponible. Instalarlo mejorará el análisis paralelo: pip install dask[complete]")

# Configuración de workers
DASK_WORKERS = 4  # Ajustar según CPU


def analyze_data_by_period(df: pd.DataFrame) -> Dict:
    """Análisis detallado de datos por período"""
    if df.empty:
        return {'summary': {'total_records': 0, 'error': 'DataFrame vacío'}}
    
    logger.info("Realizando análisis de datos...")
    
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
    
    # Análisis por año
    analysis['yearly_analysis'] = _analyze_yearly_data(df)
    
    # Análisis por mes
    analysis['monthly_analysis'] = _analyze_monthly_data(df)
    
    # Análisis de volatilidad
    analysis['volatility_analysis'] = _analyze_volatility(df)
    
    # Análisis de volumen
    analysis['volume_analysis'] = _analyze_volume(df)
    
    # Análisis de gaps temporales
    analysis['temporal_analysis'] = _analyze_temporal_patterns(df)
    
    return analysis


def analyze_data_parallel_dask(df: pd.DataFrame, workers: int = None) -> Dict:
    """Análisis paralelo de datos usando Dask"""
    if not DASK_AVAILABLE or len(df) < 100000:
        # Fallback a análisis estándar para datasets pequeños
        logger.info("Usando análisis estándar (Dask no disponible o dataset pequeño)")
        return analyze_data_by_period(df)
    
    workers = workers or DASK_WORKERS
    logger.info(f"Realizando análisis paralelo con Dask ({workers} workers)")
    
    try:
        # Convertir a Dask DataFrame
        ddf = dd.from_pandas(df, npartitions=workers)
        
        # Computar estadísticas en paralelo
        with dask.distributed.LocalCluster(n_workers=workers, threads_per_worker=2) as cluster:
            with dask.distributed.Client(cluster) as client:
                
                # Análisis general
                total_records = len(df)
                date_range = {
                    'start': df['time'].min().isoformat(),
                    'end': df['time'].max().isoformat()
                }
                
                # Análisis por año (paralelo)
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
                
                # Análisis por mes (paralelo)
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
                'workers': workers,
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


def _analyze_yearly_data(df: pd.DataFrame) -> Dict:
    """Análisis detallado por año"""
    yearly_stats = {}
    
    for year in df['time'].dt.year.unique():
        year_data = df[df['time'].dt.year == year]
        
        yearly_stats[year] = {
            'total_records': len(year_data),
            'trading_days': year_data['time'].dt.date.nunique(),
            'price_stats': {
                'high': year_data['high'].max(),
                'low': year_data['low'].min(),
                'close_mean': year_data['close'].mean(),
                'close_std': year_data['close'].std(),
                'price_range': year_data['high'].max() - year_data['low'].min()
            },
            'volume_stats': {
                'total_volume': year_data['tick_volume'].sum(),
                'avg_volume': year_data['tick_volume'].mean(),
                'max_volume': year_data['tick_volume'].max()
            },
            'volatility': {
                'daily_returns_std': year_data['close'].pct_change().std(),
                'price_volatility': year_data['close'].std() / year_data['close'].mean()
            }
        }
    
    return yearly_stats


def _analyze_monthly_data(df: pd.DataFrame) -> Dict:
    """Análisis detallado por mes"""
    monthly_stats = {}
    
    # Agrupar por año y mes
    df_grouped = df.groupby([df['time'].dt.year, df['time'].dt.month])
    
    for (year, month), month_data in df_grouped:
        month_key = f"{year}-{month:02d}"
        
        monthly_stats[month_key] = {
            'total_records': len(month_data),
            'trading_days': month_data['time'].dt.date.nunique(),
            'avg_records_per_day': len(month_data) / month_data['time'].dt.date.nunique(),
            'price_stats': {
                'high': month_data['high'].max(),
                'low': month_data['low'].min(),
                'close_mean': month_data['close'].mean(),
                'close_std': month_data['close'].std()
            },
            'volume_stats': {
                'total_volume': month_data['tick_volume'].sum(),
                'avg_volume': month_data['tick_volume'].mean()
            }
        }
    
    return monthly_stats


def _analyze_volatility(df: pd.DataFrame) -> Dict:
    """Análisis de volatilidad"""
    # Calcular retornos
    returns = df['close'].pct_change().dropna()
    
    volatility_analysis = {
        'overall_volatility': returns.std(),
        'annualized_volatility': returns.std() * np.sqrt(252 * 78),  # 78 barras por día
        'volatility_by_year': {},
        'volatility_by_month': {},
        'extreme_moves': {
            'max_daily_return': returns.max(),
            'min_daily_return': returns.min(),
            'returns_above_2pct': (returns > 0.02).sum(),
            'returns_below_minus_2pct': (returns < -0.02).sum()
        }
    }
    
    # Volatilidad por año
    for year in df['time'].dt.year.unique():
        year_returns = returns[df['time'].dt.year == year]
        volatility_analysis['volatility_by_year'][year] = year_returns.std()
    
    # Volatilidad por mes
    for month in range(1, 13):
        month_returns = returns[df['time'].dt.month == month]
        if len(month_returns) > 0:
            volatility_analysis['volatility_by_month'][month] = month_returns.std()
    
    return volatility_analysis


def _analyze_volume(df: pd.DataFrame) -> Dict:
    """Análisis de volumen"""
    volume_analysis = {
        'overall_stats': {
            'total_volume': df['tick_volume'].sum(),
            'avg_volume': df['tick_volume'].mean(),
            'max_volume': df['tick_volume'].max(),
            'volume_std': df['tick_volume'].std()
        },
        'volume_by_year': {},
        'volume_by_month': {},
        'volume_patterns': {
            'high_volume_days': (df.groupby(df['time'].dt.date)['tick_volume'].sum() > 
                               df.groupby(df['time'].dt.date)['tick_volume'].sum().quantile(0.9)).sum(),
            'low_volume_days': (df.groupby(df['time'].dt.date)['tick_volume'].sum() < 
                              df.groupby(df['time'].dt.date)['tick_volume'].sum().quantile(0.1)).sum()
        }
    }
    
    # Volumen por año
    for year in df['time'].dt.year.unique():
        year_data = df[df['time'].dt.year == year]
        volume_analysis['volume_by_year'][year] = {
            'total_volume': year_data['tick_volume'].sum(),
            'avg_volume': year_data['tick_volume'].mean(),
            'max_volume': year_data['tick_volume'].max()
        }
    
    # Volumen por mes
    for month in range(1, 13):
        month_data = df[df['time'].dt.month == month]
        if len(month_data) > 0:
            volume_analysis['volume_by_month'][month] = {
                'total_volume': month_data['tick_volume'].sum(),
                'avg_volume': month_data['tick_volume'].mean()
            }
    
    return volume_analysis


def _analyze_temporal_patterns(df: pd.DataFrame) -> Dict:
    """Análisis de patrones temporales"""
    temporal_analysis = {
        'hourly_patterns': {},
        'daily_patterns': {},
        'monthly_patterns': {},
        'gaps_analysis': {}
    }
    
    # Patrones por hora
    for hour in range(24):
        hour_data = df[df['time'].dt.hour == hour]
        if len(hour_data) > 0:
            temporal_analysis['hourly_patterns'][f"{hour:02d}:00"] = {
                'records': len(hour_data),
                'avg_volume': hour_data['tick_volume'].mean(),
                'avg_volatility': hour_data['close'].pct_change().std()
            }
    
    # Patrones por día de la semana
    for day in range(7):
        day_data = df[df['time'].dt.weekday == day]
        if len(day_data) > 0:
            day_name = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][day]
            temporal_analysis['daily_patterns'][day_name] = {
                'records': len(day_data),
                'avg_volume': day_data['tick_volume'].mean(),
                'avg_volatility': day_data['close'].pct_change().std()
            }
    
    # Patrones por mes
    for month in range(1, 13):
        month_data = df[df['time'].dt.month == month]
        if len(month_data) > 0:
            month_name = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][month-1]
            temporal_analysis['monthly_patterns'][month_name] = {
                'records': len(month_data),
                'avg_volume': month_data['tick_volume'].mean(),
                'avg_volatility': month_data['close'].pct_change().std()
            }
    
    # Análisis de gaps
    time_diffs = df['time'].diff()
    gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)]
    
    temporal_analysis['gaps_analysis'] = {
        'total_gaps': len(gaps),
        'avg_gap_minutes': gaps.mean().total_seconds() / 60 if len(gaps) > 0 else 0,
        'max_gap_minutes': gaps.max().total_seconds() / 60 if len(gaps) > 0 else 0,
        'gaps_by_size': {
            'small (5-15 min)': len(gaps[(gaps > pd.Timedelta(minutes=5)) & (gaps <= pd.Timedelta(minutes=15))]),
            'medium (15-60 min)': len(gaps[(gaps > pd.Timedelta(minutes=15)) & (gaps <= pd.Timedelta(minutes=60))]),
            'large (>60 min)': len(gaps[gaps > pd.Timedelta(minutes=60)])
        }
    }
    
    return temporal_analysis


def generate_analysis_summary(analysis: Dict) -> str:
    """Generar resumen textual del análisis"""
    summary = []
    
    # Resumen general
    summary.append("=== RESUMEN DE ANÁLISIS DE DATOS ===")
    summary.append(f"Total de registros: {analysis['summary']['total_records']:,}")
    summary.append(f"Días de trading: {analysis['summary']['trading_days']}")
    summary.append(f"Período: {analysis['summary']['date_range']['start']} - {analysis['summary']['date_range']['end']}")
    summary.append(f"Promedio registros/día: {analysis['summary']['records_per_day']:.1f}")
    
    # Análisis de volatilidad
    if 'volatility_analysis' in analysis:
        vol = analysis['volatility_analysis']
        summary.append(f"\n=== VOLATILIDAD ===")
        summary.append(f"Volatilidad general: {vol['overall_volatility']:.4f}")
        summary.append(f"Volatilidad anualizada: {vol['annualized_volatility']:.2%}")
        summary.append(f"Retorno máximo diario: {vol['extreme_moves']['max_daily_return']:.2%}")
        summary.append(f"Retorno mínimo diario: {vol['extreme_moves']['min_daily_return']:.2%}")
    
    # Análisis de volumen
    if 'volume_analysis' in analysis:
        vol = analysis['volume_analysis']
        summary.append(f"\n=== VOLUMEN ===")
        summary.append(f"Volumen total: {vol['overall_stats']['total_volume']:,}")
        summary.append(f"Volumen promedio: {vol['overall_stats']['avg_volume']:.1f}")
        summary.append(f"Volumen máximo: {vol['overall_stats']['max_volume']:,}")
    
    # Análisis temporal
    if 'temporal_analysis' in analysis:
        temp = analysis['temporal_analysis']
        summary.append(f"\n=== PATRONES TEMPORALES ===")
        summary.append(f"Gaps totales: {temp['gaps_analysis']['total_gaps']}")
        summary.append(f"Gap promedio: {temp['gaps_analysis']['avg_gap_minutes']:.1f} minutos")
    
    return "\n".join(summary) 