"""
Validador de horarios de mercado y consistencia temporal
"""
import pandas as pd
import numpy as np
import logging
import pytz
from datetime import datetime, time
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class MarketValidator:
    """Validador de horarios de mercado y consistencia temporal"""
    
    def __init__(self, config: Dict):
        """
        Inicializar validador
        
        Args:
            config: Configuración del validador
        """
        self.config = config
        
        # Horarios oficiales del mercado de valores (EST/EDT)
        self.MARKET_OPEN_HOUR = 9
        self.MARKET_OPEN_MINUTE = 30
        self.MARKET_CLOSE_HOUR = 16
        self.MARKET_CLOSE_MINUTE = 0
        
        # Configuración de consistencia M5
        self.m5_tolerance = config.get('m5_tolerance', 0.1)  # ±0.1 minutos
        self.min_consistency = config.get('min_consistency', 0.95)  # 95%
        
    def is_market_open(self, dt: pd.Timestamp) -> bool:
        """
        Verificar si es horario de mercado abierto
        
        Args:
            dt: Timestamp a verificar
            
        Returns:
            bool: True si es horario de mercado
        """
        # Convertir a EST/EDT
        est_tz = pytz.timezone('US/Eastern')
        dt_est = dt.astimezone(est_tz)
        
        # Verificar si es fin de semana
        if dt_est.weekday() >= 5:  # Sábado = 5, Domingo = 6
            return False
        
        # Verificar horario de mercado (9:30 AM - 4:00 PM EST)
        market_start = dt_est.replace(
            hour=self.MARKET_OPEN_HOUR, 
            minute=self.MARKET_OPEN_MINUTE, 
            second=0, 
            microsecond=0
        )
        market_end = dt_est.replace(
            hour=self.MARKET_CLOSE_HOUR, 
            minute=self.MARKET_CLOSE_MINUTE, 
            second=0, 
            microsecond=0
        )
        
        return market_start <= dt_est <= market_end
    
    def validate_market_hours(self, df: pd.DataFrame) -> Dict:
        """
        Validar que todos los datos estén en horarios de mercado
        
        Args:
            df: DataFrame con columna 'time'
            
        Returns:
            Dict: Resultados de la validación
        """
        logger.info("Validando horarios oficiales de mercado...")
        
        # Convertir timezone si es necesario
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC')
        
        # Verificar fines de semana
        weekend_records = df[df['time'].dt.weekday >= 5]
        weekend_count = len(weekend_records)
        
        # Verificar horarios de mercado
        market_mask = df['time'].apply(self.is_market_open)
        outside_market = df[~market_mask]
        outside_market_count = len(outside_market)
        
        # Calcular métricas
        total_records = len(df)
        valid_market_records = len(df[market_mask])
        market_percentage = valid_market_records / total_records * 100
        
        # Determinar si pasa la validación
        is_valid = weekend_count == 0 and outside_market_count == 0
        
        if is_valid:
            logger.info("OK: Todos los registros están en horarios oficiales de mercado")
        else:
            if weekend_count > 0:
                logger.error(f"ERROR: {weekend_count} registros en fines de semana")
            if outside_market_count > 0:
                logger.error(f"ERROR: {outside_market_count} registros fuera de horarios de mercado")
        
        return {
            'is_valid': is_valid,
            'total_records': total_records,
            'valid_market_records': valid_market_records,
            'market_percentage': market_percentage,
            'weekend_records': weekend_count,
            'outside_market_records': outside_market_count,
            'weekend_records_data': weekend_records if weekend_count > 0 else None,
            'outside_market_data': outside_market if outside_market_count > 0 else None
        }
    
    def validate_m5_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Validar consistencia de intervalos M5
        
        Args:
            df: DataFrame con columna 'time'
            
        Returns:
            Dict: Resultados de la validación
        """
        logger.info("Validando consistencia de intervalos M5...")
        
        df_sorted = df.sort_values('time').reset_index(drop=True)
        time_diffs = df_sorted['time'].diff().dropna().dt.total_seconds() / 60
        
        # Verificar que todos los intervalos sean de 5 minutos
        min_interval = 5 - self.m5_tolerance
        max_interval = 5 + self.m5_tolerance
        
        correct_intervals = ((time_diffs >= min_interval) & (time_diffs <= max_interval)).sum()
        total_intervals = len(time_diffs)
        consistency = correct_intervals / total_intervals if total_intervals > 0 else 0
        
        # Encontrar intervalos incorrectos
        incorrect_intervals = time_diffs[~((time_diffs >= min_interval) & (time_diffs <= max_interval))]
        
        # Determinar si pasa la validación
        is_valid = consistency >= self.min_consistency
        
        if is_valid:
            logger.info(f"OK: Consistencia M5: {consistency:.2%}")
        else:
            logger.error(f"ERROR: Consistencia M5 insuficiente: {consistency:.2%}")
        
        return {
            'is_valid': is_valid,
            'consistency_percentage': consistency * 100,
            'correct_intervals': int(correct_intervals),
            'total_intervals': int(total_intervals),
            'min_interval': min_interval,
            'max_interval': max_interval,
            'incorrect_intervals_count': len(incorrect_intervals),
            'incorrect_intervals_data': incorrect_intervals.tolist() if len(incorrect_intervals) > 0 else None,
            'time_diffs_stats': {
                'mean': float(time_diffs.mean()),
                'std': float(time_diffs.std()),
                'min': float(time_diffs.min()),
                'max': float(time_diffs.max())
            }
        }
    
    def validate_data_flags(self, df: pd.DataFrame) -> Dict:
        """
        Validar flags de datos reales vs imputados
        
        Args:
            df: DataFrame con columna 'data_flag'
            
        Returns:
            Dict: Resultados de la validación
        """
        logger.info("Validando flags de datos...")
        
        if 'data_flag' not in df.columns:
            logger.error("ERROR: Columna 'data_flag' no encontrada")
            return {
                'is_valid': False,
                'error': 'data_flag column not found',
                'real_data_count': 0,
                'imputed_data_count': 0,
                'real_percentage': 0,
                'imputed_percentage': 0
            }
        
        real_data = df[df['data_flag'] == 'real']
        imputed_data = df[df['data_flag'] == 'imputed_brownian']
        
        total_records = len(df)
        real_count = len(real_data)
        imputed_count = len(imputed_data)
        real_percentage = real_count / total_records * 100
        imputed_percentage = imputed_count / total_records * 100
        
        logger.info(f"Datos reales: {real_count:,} ({real_percentage:.1f}%)")
        logger.info(f"Datos imputados: {imputed_count:,} ({imputed_percentage:.1f}%)")
        
        # Determinar si pasa la validación (mínimo 80% datos reales)
        min_real_percentage = self.config.get('min_real_percentage', 80)
        is_valid = real_percentage >= min_real_percentage
        
        if is_valid:
            logger.info("OK: Proporción de datos reales aceptable")
        else:
            logger.warning(f"ADVERTENCIA: Proporción de datos reales baja: {real_percentage:.1f}%")
        
        return {
            'is_valid': is_valid,
            'real_data_count': real_count,
            'imputed_data_count': imputed_count,
            'real_percentage': real_percentage,
            'imputed_percentage': imputed_percentage,
            'min_real_percentage': min_real_percentage,
            'total_records': total_records
        }
    
    def validate_all(self, df: pd.DataFrame) -> Dict:
        """
        Ejecutar todas las validaciones de mercado
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Dict: Resultados de todas las validaciones
        """
        results = {
            'market_hours': self.validate_market_hours(df),
            'm5_consistency': self.validate_m5_consistency(df),
            'data_flags': self.validate_data_flags(df)
        }
        
        # Resumen general
        all_valid = all([
            results['market_hours']['is_valid'],
            results['m5_consistency']['is_valid'],
            results['data_flags']['is_valid']
        ])
        
        results['overall_valid'] = all_valid
        
        return results