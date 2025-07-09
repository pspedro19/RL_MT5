#!/usr/bin/env python3
"""
Filtro robusto de horarios de mercado
Primera etapa del procesamiento para asegurar solo datos dentro del horario de mercado
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional, Any
import pytz

from config.constants import MARKET_CONFIGS, TIME_FORMAT_CONFIG

logger = logging.getLogger(__name__)


class MarketHoursFilter:
    """Filtro avanzado de horarios de mercado con trazabilidad"""
    
    def __init__(self, instrument: str = 'US500'):
        """
        Inicializar filtro para un instrumento específico
        
        Args:
            instrument: Instrumento ('US500', 'USDCOP', etc.)
        """
        self.instrument = instrument
        self.market_config = MARKET_CONFIGS.get(instrument, MARKET_CONFIGS['US500'])
        self.timezone = self.market_config['timezone']
        
        # Configuración específica por instrumento
        if instrument == 'US500':
            self.market_hours = {
                'open_time': time(9, 30),  # 9:30 AM ET
                'close_time': time(16, 0),  # 4:00 PM ET
                'timezone': pytz.timezone('US/Eastern'),
                'trading_days': [0, 1, 2, 3, 4],  # Lunes a Viernes
                'holidays': self._get_us_market_holidays()
            }
        elif instrument == 'USDCOP':
            self.market_hours = {
                'open_time': time(0, 0),   # 00:00 UTC (Forex 24/5)
                'close_time': time(23, 59), # 23:59 UTC
                'timezone': pytz.UTC,
                'trading_days': [0, 1, 2, 3, 4],  # Lunes a Viernes
                'holidays': []  # Forex no tiene feriados tradicionales
            }
        else:
            # Configuración por defecto
            self.market_hours = {
                'open_time': time(9, 30),
                'close_time': time(16, 0),
                'timezone': pytz.timezone('US/Eastern'),
                'trading_days': [0, 1, 2, 3, 4],
                'holidays': []
            }
    
    def _get_us_market_holidays(self) -> List[str]:
        """Obtener feriados del mercado US"""
        # Lista de feriados principales del mercado US
        holidays = [
            '2020-01-01', '2020-01-20', '2020-02-17', '2020-04-10', '2020-05-25',
            '2020-07-03', '2020-09-07', '2020-11-26', '2020-12-25',
            '2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02', '2021-05-31',
            '2021-07-05', '2021-09-06', '2021-11-25', '2021-12-24',
            '2022-01-17', '2022-02-21', '2022-04-15', '2022-05-30',
            '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-26',
            '2023-01-02', '2023-01-16', '2023-02-20', '2023-04-07', '2023-05-29',
            '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25',
            '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27',
            '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
            '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
            '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'
        ]
        return holidays
    
    def is_market_open(self, timestamp: pd.Timestamp) -> bool:
        """
        Verificar si un timestamp está dentro del horario de mercado
        
        Args:
            timestamp: Timestamp a verificar
        
        Returns:
            True si el mercado está abierto, False en caso contrario
        """
        # Asegurar que timestamp sea un objeto datetime
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Asegurar que el timestamp sea timezone-aware
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        
        # Convertir a zona horaria del mercado
        market_time = timestamp.tz_convert(self.market_hours['timezone'])
        
        # Verificar día de la semana
        if market_time.weekday() not in self.market_hours['trading_days']:
            return False
        
        # Verificar feriados
        date_str = market_time.strftime('%Y-%m-%d')
        if date_str in self.market_hours['holidays']:
            return False
        
        # Verificar horario
        current_time = market_time.time()
        
        if self.instrument == 'USDCOP':
            # Forex: 24 horas al día, pero verificamos horario principal
            # Horario principal: 8:00 AM - 5:00 PM ET
            et_time = market_time.astimezone(pytz.timezone('US/Eastern'))
            et_current_time = et_time.time()
            return time(8, 0) <= et_current_time <= time(17, 0)
        else:
            # Mercado US: 9:30 AM - 4:00 PM ET
            return self.market_hours['open_time'] <= current_time <= self.market_hours['close_time']
    
    def filter_market_hours(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Filtrar DataFrame para mantener solo datos dentro del horario de mercado
        
        Args:
            df: DataFrame con columna 'time'
        
        Returns:
            Tuple con (DataFrame filtrado, estadísticas de filtrado)
        """
        if df.empty:
            return df, {'total_records': 0, 'filtered_records': 0, 'removed_records': 0}
        
        logger.info(f"Aplicando filtro de horarios de mercado para {self.instrument}")
        logger.info(f"Total de registros iniciales: {len(df)}")
        
        # Crear máscara de horarios de mercado
        market_mask = df['time'].apply(self.is_market_open)
        
        # Aplicar filtro
        df_filtered = df[market_mask].copy()
        
        # Calcular estadísticas
        total_records = len(df)
        filtered_records = len(df_filtered)
        removed_records = total_records - filtered_records
        
        stats = {
            'total_records': total_records,
            'filtered_records': filtered_records,
            'removed_records': removed_records,
            'removal_percentage': (removed_records / total_records * 100) if total_records > 0 else 0
        }
        
        logger.info(f"Filtrado completado:")
        logger.info(f"  - Registros totales: {total_records}")
        logger.info(f"  - Registros filtrados: {filtered_records}")
        logger.info(f"  - Registros removidos: {removed_records} ({stats['removal_percentage']:.2f}%)")
        
        return df_filtered, stats
    
    def analyze_time_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizar distribución temporal de los datos
        
        Args:
            df: DataFrame con columna 'time'
        
        Returns:
            Dict con análisis temporal
        """
        if df.empty:
            return {'error': 'DataFrame vacío'}
        
        # Asegurar timezone-aware
        df_time = df['time'].copy()
        if df_time.dt.tz is None:
            df_time = df_time.dt.tz_localize('UTC')
        
        # Convertir a zona horaria del mercado
        market_time = df_time.dt.tz_convert(self.market_hours['timezone'])
        
        # Análisis por hora
        hourly_distribution = market_time.dt.hour.value_counts().sort_index()
        
        # Análisis por día de la semana
        daily_distribution = market_time.dt.dayofweek.value_counts().sort_index()
        
        # Análisis de intervalos
        time_diffs = df_time.diff().dropna()
        interval_stats = {
            'mean_interval_minutes': time_diffs.mean().total_seconds() / 60,
            'median_interval_minutes': time_diffs.median().total_seconds() / 60,
            'min_interval_minutes': time_diffs.min().total_seconds() / 60,
            'max_interval_minutes': time_diffs.max().total_seconds() / 60,
            'std_interval_minutes': time_diffs.std().total_seconds() / 60
        }
        
        # Detectar intervalos irregulares
        expected_interval = 5  # minutos para M5
        irregular_intervals = time_diffs[abs(time_diffs.dt.total_seconds() / 60 - expected_interval) > 1]
        
        analysis = {
            'time_range': {
                'start': df_time.min().isoformat(),
                'end': df_time.max().isoformat(),
                'total_days': (df_time.max() - df_time.min()).days
            },
            'hourly_distribution': hourly_distribution.to_dict(),
            'daily_distribution': daily_distribution.to_dict(),
            'interval_analysis': interval_stats,
            'irregular_intervals_count': len(irregular_intervals),
            'expected_interval_minutes': expected_interval,
            'data_quality_issues': []
        }
        
        # Detectar problemas de calidad
        if interval_stats['mean_interval_minutes'] > expected_interval * 1.5:
            analysis['data_quality_issues'].append('Intervalos promedio mayores a lo esperado')
        
        if len(irregular_intervals) > len(df) * 0.1:  # Más del 10% irregular
            analysis['data_quality_issues'].append('Alto porcentaje de intervalos irregulares')
        
        if len(analysis['data_quality_issues']) == 0:
            analysis['data_quality_issues'].append('Sin problemas detectados')
        
        return analysis


def apply_market_hours_filter(df: pd.DataFrame, instrument: str = 'US500') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Función de conveniencia para aplicar filtro de horarios de mercado
    
    Args:
        df: DataFrame a filtrar
        instrument: Instrumento ('US500', 'USDCOP', etc.)
    
    Returns:
        Tuple con (DataFrame filtrado, análisis completo)
    """
    filter_obj = MarketHoursFilter(instrument)
    
    # Aplicar filtro
    df_filtered, filter_stats = filter_obj.filter_market_hours(df)
    
    # Analizar distribución temporal
    time_analysis = filter_obj.analyze_time_distribution(df_filtered)
    
    # Combinar resultados
    complete_analysis = {
        'filter_stats': filter_stats,
        'time_analysis': time_analysis,
        'instrument': instrument,
        'market_config': {
            'timezone': str(filter_obj.market_hours['timezone']),
            'trading_days': filter_obj.market_hours['trading_days'],
            'open_time': filter_obj.market_hours['open_time'].isoformat(),
            'close_time': filter_obj.market_hours['close_time'].isoformat()
        }
    }
    
    return df_filtered, complete_analysis


def validate_market_hours_compliance(df: pd.DataFrame, instrument: str = 'US500') -> Dict[str, Any]:
    """
    Validar que un DataFrame cumple con los horarios de mercado
    
    Args:
        df: DataFrame a validar
        instrument: Instrumento
    
    Returns:
        Dict con resultados de validación
    """
    filter_obj = MarketHoursFilter(instrument)
    
    # Verificar cada registro
    market_mask = df['time'].apply(filter_obj.is_market_open)
    non_compliant_records = df[~market_mask]
    
    validation_result = {
        'total_records': len(df),
        'compliant_records': market_mask.sum(),
        'non_compliant_records': len(non_compliant_records),
        'compliance_percentage': (market_mask.sum() / len(df) * 100) if len(df) > 0 else 0,
        'is_compliant': len(non_compliant_records) == 0,
        'non_compliant_examples': []
    }
    
    # Mostrar ejemplos de registros no conformes
    if len(non_compliant_records) > 0:
        sample_size = min(5, len(non_compliant_records))
        validation_result['non_compliant_examples'] = non_compliant_records['time'].head(sample_size).dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    
    return validation_result 