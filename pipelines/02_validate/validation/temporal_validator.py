"""
Validación de consistencia temporal
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
from datetime import timedelta

logger = logging.getLogger(__name__)


class TemporalValidator:
    """Validador de consistencia temporal para datos M5"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializar validador temporal
        
        Args:
            config: Configuración opcional
        """
        config = config or {}
        self.expected_interval = config.get('expected_interval', 5)  # minutos
        self.tolerance = config.get('tolerance', 0.1)  # 10% tolerancia
        self.max_gap_minutes = config.get('max_gap_minutes', 30)  # máximo gap permitido
        
    def validate_m5_consistency(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validar consistencia de intervalos M5
        
        Args:
            df: DataFrame con columna 'time'
            
        Returns:
            Tuple[bool, Dict]: (es_válido, detalles_validación)
        """
        logger.info("Validando consistencia de intervalos M5...")
        
        # Asegurar orden temporal
        df_sorted = df.sort_values('time').reset_index(drop=True)
        
        # Calcular diferencias temporales
        time_diffs = df_sorted['time'].diff().dropna()
        time_diffs_minutes = time_diffs.dt.total_seconds() / 60
        
        # Clasificar intervalos
        min_interval = self.expected_interval * (1 - self.tolerance)
        max_interval = self.expected_interval * (1 + self.tolerance)
        
        correct_intervals = ((time_diffs_minutes >= min_interval) & 
                           (time_diffs_minutes <= max_interval))
        
        # Detectar diferentes tipos de problemas
        too_short = time_diffs_minutes < min_interval
        too_long = (time_diffs_minutes > max_interval) & (time_diffs_minutes <= self.max_gap_minutes)
        gaps = time_diffs_minutes > self.max_gap_minutes
        
        # Calcular métricas
        total_intervals = len(time_diffs_minutes)
        consistency = correct_intervals.sum() / total_intervals if total_intervals > 0 else 0
        
        # Analizar distribución de intervalos
        interval_stats = {
            'mean': time_diffs_minutes.mean(),
            'std': time_diffs_minutes.std(),
            'min': time_diffs_minutes.min(),
            'max': time_diffs_minutes.max(),
            'median': time_diffs_minutes.median()
        }
        
        validation_passed = consistency >= 0.95  # 95% consistencia mínima
        
        details = {
            'total_intervals': total_intervals,
            'correct_intervals': correct_intervals.sum(),
            'consistency_percentage': consistency * 100,
            'too_short_intervals': too_short.sum(),
            'too_long_intervals': too_long.sum(),
            'gap_count': gaps.sum(),
            'interval_statistics': interval_stats,
            'validation_passed': validation_passed
        }
        
        if validation_passed:
            logger.info(f"✓ Consistencia M5: {consistency:.2%}")
        else:
            logger.error(f"✗ Consistencia M5 insuficiente: {consistency:.2%}")
            if gaps.sum() > 0:
                logger.warning(f"  - Gaps detectados: {gaps.sum()}")
                
        return validation_passed, details
    
    def detect_gaps(self, df: pd.DataFrame) -> Tuple[List[Dict], Dict]:
        """
        Detectar y analizar gaps temporales
        
        Args:
            df: DataFrame con columna 'time'
            
        Returns:
            Tuple[List[Dict], Dict]: (lista_de_gaps, estadísticas)
        """
        logger.info("Detectando gaps temporales...")
        
        df_sorted = df.sort_values('time').reset_index(drop=True)
        
        # Calcular diferencias
        df_sorted['time_diff'] = df_sorted['time'].diff()
        df_sorted['time_diff_minutes'] = df_sorted['time_diff'].dt.total_seconds() / 60
        
        # Detectar gaps (más del máximo permitido)
        gap_mask = df_sorted['time_diff_minutes'] > self.max_gap_minutes
        gaps = df_sorted[gap_mask]
        
        gap_list = []
        for idx, row in gaps.iterrows():
            gap_info = {
                'start_time': df_sorted.loc[idx-1, 'time'] if idx > 0 else None,
                'end_time': row['time'],
                'duration_minutes': row['time_diff_minutes'],
                'duration_hours': row['time_diff_minutes'] / 60,
                'index': idx
            }
            gap_list.append(gap_info)
        
        # Estadísticas de gaps
        if len(gap_list) > 0:
            gap_durations = [g['duration_minutes'] for g in gap_list]
            stats = {
                'total_gaps': len(gap_list),
                'total_gap_time_minutes': sum(gap_durations),
                'avg_gap_duration': np.mean(gap_durations),
                'max_gap_duration': max(gap_durations),
                'min_gap_duration': min(gap_durations)
            }
        else:
            stats = {
                'total_gaps': 0,
                'total_gap_time_minutes': 0,
                'avg_gap_duration': 0,
                'max_gap_duration': 0,
                'min_gap_duration': 0
            }
        
        logger.info(f"Gaps detectados: {stats['total_gaps']}")
        if stats['total_gaps'] > 0:
            logger.info(f"  - Duración total: {stats['total_gap_time_minutes']:.1f} minutos")
            logger.info(f"  - Gap máximo: {stats['max_gap_duration']:.1f} minutos")
        
        return gap_list, stats
    
    def validate_monotonicity(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validar orden temporal (monotonicidad)
        
        Args:
            df: DataFrame con columna 'time'
            
        Returns:
            Tuple[bool, Dict]: (es_válido, detalles_validación)
        """
        logger.info("Validando orden temporal...")
        
        # Verificar si está ordenado
        is_monotonic = df['time'].is_monotonic_increasing
        
        # Encontrar registros desordenados
        if not is_monotonic:
            df_sorted = df.sort_values('time').reset_index(drop=True)
            original_indices = df.index.tolist()
            sorted_indices = df_sorted.index.tolist()
            
            # Encontrar posiciones que cambiaron
            out_of_order = []
            for i, (orig, sorted_idx) in enumerate(zip(original_indices, sorted_indices)):
                if orig != sorted_idx:
                    out_of_order.append({
                        'original_position': orig,
                        'correct_position': sorted_idx,
                        'time': df.loc[orig, 'time']
                    })
        else:
            out_of_order = []
        
        # Verificar duplicados temporales
        duplicates = df['time'].duplicated()
        duplicate_times = df[duplicates]['time'].unique()
        
        validation_passed = is_monotonic and len(duplicate_times) == 0
        
        details = {
            'is_monotonic': is_monotonic,
            'out_of_order_count': len(out_of_order),
            'duplicate_timestamps': len(duplicate_times),
            'duplicate_records': duplicates.sum(),
            'validation_passed': validation_passed
        }
        
        if validation_passed:
            logger.info("✓ Orden temporal correcto")
        else:
            if not is_monotonic:
                logger.error(f"✗ Registros desordenados: {len(out_of_order)}")
            if len(duplicate_times) > 0:
                logger.error(f"✗ Timestamps duplicados: {len(duplicate_times)}")
                
        return validation_passed, details
    
    def analyze_temporal_coverage(self, df: pd.DataFrame) -> Dict:
        """
        Analizar cobertura temporal del dataset
        
        Args:
            df: DataFrame con columna 'time'
            
        Returns:
            Dict: Análisis de cobertura temporal
        """
        logger.info("Analizando cobertura temporal...")
        
        # Rango temporal
        time_range = {
            'start': df['time'].min(),
            'end': df['time'].max(),
            'total_days': (df['time'].max() - df['time'].min()).days
        }
        
        # Días únicos
        unique_dates = df['time'].dt.date.unique()
        
        # Distribución por día de la semana
        weekday_dist = df['time'].dt.weekday.value_counts().sort_index()
        
        # Distribución por hora
        hour_dist = df['time'].dt.hour.value_counts().sort_index()
        
        # Registros por día
        daily_counts = df.groupby(df['time'].dt.date).size()
        expected_daily = 78  # 6.5 horas * 12 registros/hora para M5
        
        # Clasificar días por cobertura
        perfect_days = (daily_counts == expected_daily).sum()
        good_days = ((daily_counts >= expected_daily * 0.9) & 
                    (daily_counts < expected_daily)).sum()
        acceptable_days = ((daily_counts >= expected_daily * 0.7) & 
                         (daily_counts < expected_daily * 0.9)).sum()
        poor_days = (daily_counts < expected_daily * 0.7).sum()
        
        coverage_analysis = {
            'time_range': time_range,
            'unique_trading_days': len(unique_dates),
            'weekday_distribution': weekday_dist.to_dict(),
            'hour_distribution': hour_dist.to_dict(),
            'daily_coverage': {
                'perfect_days': perfect_days,
                'good_days': good_days,
                'acceptable_days': acceptable_days,
                'poor_days': poor_days,
                'avg_records_per_day': daily_counts.mean(),
                'min_records_per_day': daily_counts.min(),
                'max_records_per_day': daily_counts.max()
            }
        }
        
        logger.info(f"Período: {time_range['start']} a {time_range['end']}")
        logger.info(f"Días de trading: {len(unique_dates)}")
        logger.info(f"Cobertura diaria - Perfecta: {perfect_days}, Buena: {good_days}, "
                   f"Aceptable: {acceptable_days}, Pobre: {poor_days}")
        
        return coverage_analysis
    
    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Ejecutar todas las validaciones temporales
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Tuple[bool, Dict]: (todas_válidas, resumen_detalles)
        """
        results = {}
        all_valid = True
        
        # Consistencia M5
        valid, details = self.validate_m5_consistency(df)
        results['m5_consistency'] = details
        all_valid &= valid
        
        # Monotonicidad
        valid, details = self.validate_monotonicity(df)
        results['monotonicity'] = details
        all_valid &= valid
        
        # Gaps
        gaps, gap_stats = self.detect_gaps(df)
        results['gaps'] = {
            'gap_list': gaps[:10],  # Primeros 10 gaps
            'statistics': gap_stats
        }
        
        # Cobertura temporal
        coverage = self.analyze_temporal_coverage(df)
        results['temporal_coverage'] = coverage
        
        results['all_validations_passed'] = all_valid
        
        return all_valid, results