#!/usr/bin/env python3
"""
Sistema de trazabilidad granular por registro
Analiza cada registro individualmente para determinar su origen correcto
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pytz

from config.constants import DATA_ORIGINS, DATA_ORIGIN_VALUES
from utils.market_hours_filter import MarketHoursFilter

logger = logging.getLogger(__name__)


class GranularTraceabilityAnalyzer:
    """Analizador de trazabilidad granular por registro"""
    
    def __init__(self, instrument: str = 'US500'):
        """
        Inicializar analizador
        
        Args:
            instrument: Instrumento ('US500', 'USDCOP', etc.)
        """
        self.instrument = instrument
        self.market_filter = MarketHoursFilter(instrument)
        
    def analyze_record_origin(self, row: pd.Series, context: Dict[str, Any] = None) -> str:
        """
        Analizar el origen de un registro individual
        
        Args:
            row: Serie de pandas con los datos del registro
            context: Contexto adicional (opcional)
        
        Returns:
            Valor de data_origin para este registro
        """
        timestamp = row['time']
        
        # Asegurar que timestamp sea un objeto datetime
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # 1. Verificar si está dentro del horario de mercado
        is_market_hours = self.market_filter.is_market_open(timestamp)
        
        if not is_market_hours:
            return 'M5_FUERA_DE_HORARIO'
        
        # 2. Analizar intervalos de tiempo para determinar origen
        origin = self._determine_timeframe_origin(row, context)
        
        return origin
    
    def _determine_timeframe_origin(self, row: pd.Series, context: Dict[str, Any] = None) -> str:
        """
        Determinar el timeframe de origen basado en análisis temporal
        """
        if context is None or 'time_series' not in context:
            return 'M5_NATIVO'

        time_series = context['time_series']
        current_idx = context.get('current_idx', 0)

        # Forzar conversión a datetime
        if not pd.api.types.is_datetime64_any_dtype(time_series):
            time_series = pd.to_datetime(time_series)

        intervals = self._calculate_intervals_around_index(time_series, current_idx)
        if len(intervals) == 0:
            return 'M5_NATIVO'

        avg_interval = np.mean(intervals)

        # Ampliar mapeo de intervalos
        timeframe_mapping = {
            0: 'TICK_NATIVO',
            1: 'M1_NATIVO',
            5: 'M5_NATIVO',
            10: 'M10_NATIVO',
            15: 'M15_NATIVO',
            20: 'M20_NATIVO',
            30: 'M30_NATIVO',
            60: 'H1_NATIVO'
        }

        # Lógica para ticks imputados (si hay intervalos muy pequeños pero con imputación)
        if 'imputed' in row.get('data_origin', '').lower() or 'imputado' in row.get('data_origin', '').lower():
            return 'TICK_IMPUTADO'

        # Si el intervalo promedio es menor a 0.75 min, consideramos tick nativo
        if avg_interval < 0.75:
            return 'TICK_NATIVO'

        # Encontrar el timeframe más cercano
        closest_timeframe = min(timeframe_mapping.keys(), key=lambda x: abs(x - avg_interval))
        return timeframe_mapping[closest_timeframe]
    
    def _calculate_intervals_around_index(self, time_series: pd.Series, idx: int, window: int = 5) -> List[float]:
        """
        Calcular intervalos alrededor de un índice específico
        """
        # Forzar conversión a datetime
        if not pd.api.types.is_datetime64_any_dtype(time_series):
            time_series = pd.to_datetime(time_series)

        intervals = []
        # Calcular intervalos hacia atrás
        for i in range(max(0, idx - window), idx):
            if i + 1 < len(time_series):
                # Forzar conversión de ambos valores a datetime antes de la resta
                val1 = pd.to_datetime(time_series.iloc[i + 1])
                val2 = pd.to_datetime(time_series.iloc[i])
                interval = (val1 - val2).total_seconds() / 60
                intervals.append(interval)
        # Calcular intervalos hacia adelante
        for i in range(idx, min(len(time_series) - 1, idx + window)):
            # Forzar conversión de ambos valores a datetime antes de la resta
            val1 = pd.to_datetime(time_series.iloc[i + 1])
            val2 = pd.to_datetime(time_series.iloc[i])
            interval = (val1 - val2).total_seconds() / 60
            intervals.append(interval)
        return intervals
    
    def analyze_dataframe_granular(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analizar DataFrame completo con trazabilidad granular
        
        Args:
            df: DataFrame a analizar
        
        Returns:
            DataFrame con data_origin asignado por registro
        """
        if df.empty:
            return df
        
        logger.info(f"Aplicando análisis granular de trazabilidad a {len(df)} registros")
        
        # Ordenar por tiempo
        df_sorted = df.sort_values('time').reset_index(drop=True)

        # Forzar conversión de 'time' a datetime
        if not pd.api.types.is_datetime64_any_dtype(df_sorted['time']):
            df_sorted['time'] = pd.to_datetime(df_sorted['time'])

        # Preparar contexto
        time_series = df_sorted['time']
        if not pd.api.types.is_datetime64_any_dtype(time_series):
            time_series = pd.to_datetime(time_series)
        context = {
            'time_series': time_series,
            'total_records': len(df_sorted)
        }
        
        # Analizar cada registro
        data_origins = []
        quality_scores = []
        
        for idx, row in df_sorted.iterrows():
            context['current_idx'] = idx
            
            # Determinar origen
            origin = self.analyze_record_origin(row, context)
            data_origins.append(origin)
            
            # Asignar quality_score
            quality_score = DATA_ORIGINS[origin]['quality_score']
            quality_scores.append(quality_score)
        
        # Asignar columnas al DataFrame
        df_sorted['data_origin'] = data_origins
        df_sorted['quality_score'] = quality_scores
        
        # Generar estadísticas
        origin_distribution = df_sorted['data_origin'].value_counts()
        logger.info("Distribución de orígenes:")
        for origin, count in origin_distribution.items():
            percentage = (count / len(df_sorted)) * 100
            logger.info(f"  - {origin}: {count} registros ({percentage:.2f}%)")
        
        return df_sorted
    
    def detect_irregular_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detectar patrones irregulares en los datos
        
        Args:
            df: DataFrame con columna 'time'
        
        Returns:
            Dict con patrones detectados
        """
        if df.empty:
            return {'error': 'DataFrame vacío'}
        
        df_sorted = df.sort_values('time')
        time_diffs = df_sorted['time'].diff().dropna()
        
        # Convertir a minutos
        intervals_minutes = time_diffs.dt.total_seconds() / 60
        
        # Detectar patrones
        patterns = {
            'expected_interval': 5,  # minutos para M5
            'actual_intervals': {
                'mean': float(intervals_minutes.mean()),
                'median': float(intervals_minutes.median()),
                'std': float(intervals_minutes.std()),
                'min': float(intervals_minutes.min()),
                'max': float(intervals_minutes.max())
            },
            'irregular_intervals': [],
            'gaps_detected': [],
            'duplicates_detected': []
        }
        
        # Detectar intervalos irregulares (>1 minuto de diferencia del esperado)
        irregular_mask = abs(intervals_minutes - patterns['expected_interval']) > 1
        irregular_indices = irregular_mask[irregular_mask].index.tolist()
        
        for idx in irregular_indices[:10]:  # Solo los primeros 10
            patterns['irregular_intervals'].append({
                'index': int(idx),
                'timestamp': df_sorted.loc[idx, 'time'].isoformat(),
                'interval_minutes': float(intervals_minutes.loc[idx])
            })
        
        # Detectar gaps grandes (>30 minutos)
        gap_mask = intervals_minutes > 30
        gap_indices = gap_mask[gap_mask].index.tolist()
        
        for idx in gap_indices[:5]:  # Solo los primeros 5
            patterns['gaps_detected'].append({
                'index': int(idx),
                'timestamp': df_sorted.loc[idx, 'time'].isoformat(),
                'gap_minutes': float(intervals_minutes.loc[idx])
            })
        
        # Detectar duplicados
        duplicates = df_sorted[df_sorted.duplicated(subset=['time'], keep=False)]
        if not duplicates.empty:
            patterns['duplicates_detected'] = duplicates['time'].dt.strftime('%Y-%m-%d %H:%M:%S').head(5).tolist()
        
        return patterns


def apply_granular_traceability(df: pd.DataFrame, instrument: str = 'US500') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Función de conveniencia para aplicar trazabilidad granular
    
    Args:
        df: DataFrame a analizar
        instrument: Instrumento
    
    Returns:
        Tuple con (DataFrame con trazabilidad, análisis completo)
    """
    analyzer = GranularTraceabilityAnalyzer(instrument)
    
    # Aplicar análisis granular
    df_traced = analyzer.analyze_dataframe_granular(df)
    
    # Detectar patrones irregulares
    patterns = analyzer.detect_irregular_patterns(df)
    
    # Generar estadísticas
    origin_stats = df_traced['data_origin'].value_counts().to_dict()
    quality_stats = {
        'mean': float(df_traced['quality_score'].mean()),
        'min': float(df_traced['quality_score'].min()),
        'max': float(df_traced['quality_score'].max()),
        'std': float(df_traced['quality_score'].std())
    }
    
    analysis = {
        'origin_distribution': origin_stats,
        'quality_stats': quality_stats,
        'irregular_patterns': patterns,
        'instrument': instrument,
        'total_records': len(df_traced)
    }
    
    return df_traced, analysis


def validate_granular_traceability(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validar trazabilidad granular
    
    Args:
        df: DataFrame con data_origin
    
    Returns:
        Dict con resultados de validación
    """
    if 'data_origin' not in df.columns:
        return {
            'valid': False,
            'error': 'Columna data_origin no encontrada'
        }
    
    # Verificar valores válidos
    invalid_origins = df[~df['data_origin'].isin(DATA_ORIGIN_VALUES)]
    
    validation_result = {
        'valid': len(invalid_origins) == 0,
        'total_records': len(df),
        'valid_origins': len(df) - len(invalid_origins),
        'invalid_origins': len(invalid_origins),
        'invalid_examples': invalid_origins['data_origin'].unique().tolist() if len(invalid_origins) > 0 else [],
        'origin_distribution': df['data_origin'].value_counts().to_dict()
    }
    
    return validation_result 