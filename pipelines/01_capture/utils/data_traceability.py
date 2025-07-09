#!/usr/bin/env python3
"""
Utilidades para trazabilidad y auditoría de datos
Sistema estandarizado para tracking del origen y calidad de datos
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pytz

from config.constants import (
    DATA_ORIGINS, DATA_ORIGIN_VALUES, CAPTURE_METHOD_TO_DATA_ORIGIN,
    TIME_FORMAT_CONFIG, DATA_VALIDATION_CONFIG
)

logger = logging.getLogger(__name__)


class DataTraceabilityManager:
    """Gestor de trazabilidad de datos estandarizado"""
    
    def __init__(self):
        """Inicializar gestor de trazabilidad"""
        self.validation_errors = []
        self.traceability_stats = {
            'total_records': 0,
            'valid_origins': 0,
            'invalid_origins': 0,
            'origin_distribution': {},
            'quality_distribution': {}
        }
    
    def assign_data_origin(self, df: pd.DataFrame, capture_method: str, 
                          source_timeframe: str, is_imputed: bool = False,
                          imputation_method: str = None) -> pd.DataFrame:
        """
        Asignar data_origin estandarizado a un DataFrame
        
        Args:
            df: DataFrame con datos
            capture_method: Método de captura usado
            source_timeframe: Timeframe original de los datos
            is_imputed: Si los datos son imputados
            imputation_method: Método de imputación usado (si aplica)
        
        Returns:
            DataFrame con columna data_origin asignada
        """
        if df.empty:
            return df
        
        df_copy = df.copy()
        
        # Determinar data_origin basado en los parámetros
        data_origin = self._determine_data_origin(
            capture_method, source_timeframe, is_imputed, imputation_method
        )
        
        # Asignar data_origin
        df_copy['data_origin'] = data_origin
        
        # Asignar quality_score basado en data_origin
        quality_score = DATA_ORIGINS[data_origin]['quality_score']
        df_copy['quality_score'] = quality_score
        
        # Formatear tiempo de manera estándar
        df_copy['time'] = self._format_time_standard(df_copy['time'])
        
        logger.info(f"Data origin asignado: {data_origin} para {len(df_copy)} registros")
        
        return df_copy
    
    def _determine_data_origin(self, capture_method: str, source_timeframe: str,
                              is_imputed: bool, imputation_method: str = None) -> str:
        """Determinar el data_origin correcto basado en los parámetros"""
        
        # Si es imputado, usar método de imputación
        if is_imputed and imputation_method:
            if imputation_method in CAPTURE_METHOD_TO_DATA_ORIGIN:
                return CAPTURE_METHOD_TO_DATA_ORIGIN[imputation_method].get('imputed', 'M5_IMPUTADO_SIMPLE')
            else:
                return 'M5_IMPUTADO_SIMPLE'
        
        # Si es agregación/resampling
        if capture_method in ['aggregation', 'resampling']:
            if source_timeframe in CAPTURE_METHOD_TO_DATA_ORIGIN['aggregation']:
                return CAPTURE_METHOD_TO_DATA_ORIGIN['aggregation'][source_timeframe]
            else:
                return f'M5_AGREGADO_{source_timeframe}'
        
        # Si es captura directa
        if capture_method in CAPTURE_METHOD_TO_DATA_ORIGIN:
            if source_timeframe in CAPTURE_METHOD_TO_DATA_ORIGIN[capture_method]:
                return CAPTURE_METHOD_TO_DATA_ORIGIN[capture_method][source_timeframe]
        
        # Fallback: construir basado en timeframe
        if source_timeframe == 'ticks':
            return 'TICKS_NATIVO'
        elif source_timeframe in ['M1', 'M5', 'M10', 'M15', 'M20', 'M30', 'H1']:
            return f'{source_timeframe}_NATIVO'
        else:
            return 'DESCONOCIDO'
    
    def _format_time_standard(self, time_series: pd.Series) -> pd.Series:
        """Formatear serie de tiempo de manera estándar"""
        # Asegurar que sea timezone-aware
        if time_series.dt.tz is None:
            time_series = time_series.dt.tz_localize(TIME_FORMAT_CONFIG['default_timezone'])
        
        # Convertir a UTC si no lo está
        if time_series.dt.tz != pytz.UTC:
            time_series = time_series.dt.tz_convert('UTC')
        
        return time_series
    
    def validate_data_origin(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validar que todos los valores de data_origin sean correctos
        
        Returns:
            (is_valid, error_messages)
        """
        if df.empty:
            return True, []
        
        errors = []
        
        # Verificar que existe la columna
        if 'data_origin' not in df.columns:
            errors.append("Columna 'data_origin' no encontrada")
            return False, errors
        
        # Verificar valores válidos
        invalid_origins = df[~df['data_origin'].isin(DATA_ORIGIN_VALUES)]
        if not invalid_origins.empty:
            unique_invalid = invalid_origins['data_origin'].unique()
            errors.append(f"Valores inválidos en data_origin: {list(unique_invalid)}")
        
        # Verificar que no hay valores nulos
        null_origins = df['data_origin'].isnull().sum()
        if null_origins > 0:
            errors.append(f"Hay {null_origins} valores nulos en data_origin")
        
        # Verificar calidad de datos
        if 'quality_score' in df.columns:
            invalid_quality = df[
                (df['quality_score'] < DATA_VALIDATION_CONFIG['min_quality_score']) |
                (df['quality_score'] > DATA_VALIDATION_CONFIG['max_quality_score'])
            ]
            if not invalid_quality.empty:
                errors.append(f"Hay {len(invalid_quality)} registros con quality_score inválido")
        
        # Actualizar estadísticas
        self._update_traceability_stats(df)
        
        return len(errors) == 0, errors
    
    def _update_traceability_stats(self, df: pd.DataFrame):
        """Actualizar estadísticas de trazabilidad"""
        self.traceability_stats['total_records'] += len(df)
        
        if 'data_origin' in df.columns:
            origin_counts = df['data_origin'].value_counts()
            for origin, count in origin_counts.items():
                if origin in DATA_ORIGIN_VALUES:
                    self.traceability_stats['valid_origins'] += count
                    self.traceability_stats['origin_distribution'][origin] = \
                        self.traceability_stats['origin_distribution'].get(origin, 0) + count
                else:
                    self.traceability_stats['invalid_origins'] += count
        
        if 'quality_score' in df.columns:
            quality_ranges = {
                'excellent': (0.9, 1.0),
                'good': (0.7, 0.9),
                'fair': (0.5, 0.7),
                'poor': (0.0, 0.5)
            }
            
            for range_name, (min_val, max_val) in quality_ranges.items():
                count = len(df[(df['quality_score'] >= min_val) & (df['quality_score'] < max_val)])
                self.traceability_stats['quality_distribution'][range_name] = \
                    self.traceability_stats['quality_distribution'].get(range_name, 0) + count
    
    def generate_traceability_report(self) -> Dict[str, Any]:
        """Generar reporte de trazabilidad"""
        total = self.traceability_stats['total_records']
        
        if total == 0:
            return {'error': 'No hay datos para analizar'}
        
        # Calcular porcentajes
        origin_percentages = {}
        for origin, count in self.traceability_stats['origin_distribution'].items():
            origin_percentages[origin] = {
                'count': count,
                'percentage': (count / total) * 100,
                'description': DATA_ORIGINS[origin]['description'],
                'category': DATA_ORIGINS[origin]['category']
            }
        
        quality_percentages = {}
        for quality, count in self.traceability_stats['quality_distribution'].items():
            quality_percentages[quality] = {
                'count': count,
                'percentage': (count / total) * 100
            }
        
        # Análisis por categoría
        category_analysis = {}
        for origin_info in origin_percentages.values():
            category = origin_info['category']
            if category not in category_analysis:
                category_analysis[category] = {'count': 0, 'percentage': 0}
            category_analysis[category]['count'] += origin_info['count']
            category_analysis[category]['percentage'] += origin_info['percentage']
        
        return {
            'summary': {
                'total_records': total,
                'valid_origins': self.traceability_stats['valid_origins'],
                'invalid_origins': self.traceability_stats['invalid_origins'],
                'validity_rate': (self.traceability_stats['valid_origins'] / total) * 100
            },
            'origin_breakdown': origin_percentages,
            'quality_breakdown': quality_percentages,
            'category_analysis': category_analysis,
            'validation_errors': self.validation_errors
        }
    
    def convert_legacy_data_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convertir columnas legacy (data_flag, source_timeframe, capture_method) 
        a data_origin estandarizado
        """
        if df.empty:
            return df
        
        df_copy = df.copy()
        
        # Si ya tiene data_origin, no hacer nada
        if 'data_origin' in df_copy.columns:
            return df_copy
        
        # Crear data_origin basado en columnas legacy
        data_origins = []
        
        for _, row in df_copy.iterrows():
            data_flag = row.get('data_flag', 'unknown')
            source_timeframe = row.get('source_timeframe', 'unknown')
            capture_method = row.get('capture_method', 'unknown')
            
            # Mapear a data_origin
            data_origin = self._map_legacy_to_data_origin(
                data_flag, source_timeframe, capture_method
            )
            data_origins.append(data_origin)
        
        df_copy['data_origin'] = data_origins
        
        # Asignar quality_score
        df_copy['quality_score'] = df_copy['data_origin'].map(
            lambda x: DATA_ORIGINS.get(x, {}).get('quality_score', 0.0)
        )
        
        logger.info(f"Convertidos {len(df_copy)} registros legacy a data_origin")
        
        return df_copy
    
    def _map_legacy_to_data_origin(self, data_flag: str, source_timeframe: str, 
                                  capture_method: str) -> str:
        """Mapear valores legacy a data_origin estandarizado"""
        
        # Casos específicos de data_flag
        if 'real_m5' in data_flag or 'rates' in data_flag:
            if source_timeframe == 'M5':
                return 'M5_NATIVO'
            elif source_timeframe in ['M1', 'M10', 'M15', 'M20', 'M30', 'H1']:
                return f'{source_timeframe}_NATIVO'
        
        elif 'aggregated' in data_flag or 'from_m1' in data_flag:
            if source_timeframe == 'M1':
                return 'M5_AGREGADO_M1'
            elif source_timeframe in ['M10', 'M15', 'M20', 'M30', 'H1']:
                return f'M5_AGREGADO_{source_timeframe}'
        
        elif 'imputed' in data_flag.lower():
            if 'brownian' in data_flag.lower():
                return 'M5_IMPUTADO_BROWNIAN'
            elif 'interpolation' in data_flag.lower():
                return 'M5_IMPUTADO_INTERPOLADO'
            else:
                return 'M5_IMPUTADO_SIMPLE'
        
        elif 'ticks' in data_flag.lower():
            return 'TICKS_NATIVO'
        
        # Fallback basado en source_timeframe
        if source_timeframe in ['M1', 'M5', 'M10', 'M15', 'M20', 'M30', 'H1']:
            return f'{source_timeframe}_NATIVO'
        elif source_timeframe == 'ticks':
            return 'TICKS_NATIVO'
        
        return 'DESCONOCIDO'


def validate_dataframe_traceability(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Función de conveniencia para validar trazabilidad de un DataFrame
    
    Returns:
        Dict con resultados de validación
    """
    manager = DataTraceabilityManager()
    
    # Convertir legacy si es necesario
    if 'data_origin' not in df.columns and any(col in df.columns for col in ['data_flag', 'source_timeframe']):
        df = manager.convert_legacy_data_flags(df)
    
    # Validar
    is_valid, errors = manager.validate_data_origin(df)
    
    # Generar reporte
    report = manager.generate_traceability_report()
    report['validation'] = {
        'is_valid': is_valid,
        'errors': errors
    }
    
    return report


def format_time_standard(time_series: pd.Series, include_timezone: bool = True) -> pd.Series:
    """
    Formatear serie de tiempo de manera estándar
    
    Args:
        time_series: Serie de tiempo a formatear
        include_timezone: Si incluir zona horaria en el formato
    
    Returns:
        Serie formateada
    """
    # Asegurar que sea timezone-aware
    if time_series.dt.tz is None:
        time_series = time_series.dt.tz_localize(TIME_FORMAT_CONFIG['default_timezone'])
    
    # Convertir a UTC
    if time_series.dt.tz != pytz.UTC:
        time_series = time_series.dt.tz_convert('UTC')
    
    # Formatear según configuración
    if include_timezone:
        return time_series.dt.strftime(TIME_FORMAT_CONFIG['display_format_with_tz'])
    else:
        return time_series.dt.strftime(TIME_FORMAT_CONFIG['display_format'])


def get_data_origin_info(data_origin: str) -> Dict[str, Any]:
    """
    Obtener información detallada de un data_origin específico
    
    Args:
        data_origin: Valor de data_origin
    
    Returns:
        Dict con información del data_origin
    """
    if data_origin not in DATA_ORIGINS:
        return {
            'error': f'Data origin "{data_origin}" no encontrado',
            'valid_origins': DATA_ORIGIN_VALUES
        }
    
    return DATA_ORIGINS[data_origin]


def list_data_origins_by_category(category: str = None) -> Dict[str, Any]:
    """
    Listar data_origins por categoría
    
    Args:
        category: Categoría específica (native, aggregated, imputed, etc.)
    
    Returns:
        Dict con data_origins organizados por categoría
    """
    if category:
        return {
            origin: info for origin, info in DATA_ORIGINS.items()
            if info['category'] == category
        }
    
    # Agrupar por categoría
    categorized = {}
    for origin, info in DATA_ORIGINS.items():
        cat = info['category']
        if cat not in categorized:
            categorized[cat] = {}
        categorized[cat][origin] = info
    
    return categorized 