"""
Validación de integridad de datos OHLCV
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
from scipy import stats

logger = logging.getLogger(__name__)


class DataIntegrityValidator:
    """Validador de integridad de datos OHLCV y flags"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializar validador
        
        Args:
            config: Configuración opcional
        """
        config = config or {}
        self.max_price_change = config.get('max_price_change', 0.20)  # 20% max cambio
        self.min_volume = config.get('min_volume', 0)
        self.outlier_std = config.get('outlier_std', 5)  # 5 desviaciones estándar
        
    def validate_ohlcv(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validar coherencia OHLCV
        
        Args:
            df: DataFrame con columnas OHLCV
            
        Returns:
            Tuple[bool, Dict]: (es_válido, detalles_validación)
        """
        logger.info("Validando integridad OHLCV...")
        
        # Verificar columnas requeridas
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Columnas OHLC faltantes: {missing_cols}")
            return False, {'missing_columns': missing_cols}
        
        # Validar coherencia OHLC
        ohlc_invalid = df[
            (df['low'] > df['high']) | 
            (df['open'] > df['high']) |
            (df['open'] < df['low']) | 
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        ]
        
        # Validar precios positivos
        negative_prices = df[
            (df['open'] <= 0) | 
            (df['high'] <= 0) | 
            (df['low'] <= 0) | 
            (df['close'] <= 0)
        ]
        
        # Detectar cambios extremos de precio
        price_changes = df['close'].pct_change().abs()
        extreme_changes = df[price_changes > self.max_price_change]
        
        # Calcular métricas
        total_records = len(df)
        validation_passed = len(ohlc_invalid) == 0 and len(negative_prices) == 0
        
        details = {
            'total_records': total_records,
            'ohlc_invalid_count': len(ohlc_invalid),
            'negative_prices_count': len(negative_prices),
            'extreme_changes_count': len(extreme_changes),
            'ohlc_validity_rate': (total_records - len(ohlc_invalid)) / total_records * 100,
            'max_price_change': price_changes.max() * 100 if len(price_changes) > 0 else 0,
            'validation_passed': validation_passed
        }
        
        if validation_passed:
            logger.info(f"✓ OHLCV válido: {total_records} registros")
        else:
            logger.error(f"✗ OHLCV inválido: {len(ohlc_invalid)} registros")
            if len(negative_prices) > 0:
                logger.error(f"✗ Precios negativos: {len(negative_prices)} registros")
                
        return validation_passed, details
    
    def validate_volume(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validar volumen y spread
        
        Args:
            df: DataFrame con columnas de volumen
            
        Returns:
            Tuple[bool, Dict]: (es_válido, detalles_validación)
        """
        logger.info("Validando volumen y spread...")
        
        details = {}
        validation_passed = True
        
        # Validar volumen
        if 'tick_volume' in df.columns:
            negative_volume = df[df['tick_volume'] < self.min_volume]
            zero_volume = df[df['tick_volume'] == 0]
            
            details['negative_volume_count'] = len(negative_volume)
            details['zero_volume_count'] = len(zero_volume)
            details['zero_volume_percentage'] = len(zero_volume) / len(df) * 100
            
            if len(negative_volume) > 0:
                validation_passed = False
                logger.error(f"✗ Volumen negativo: {len(negative_volume)} registros")
            else:
                logger.info("✓ Volumen válido")
                
            # Detectar outliers de volumen
            if len(df) > 100:
                volume_z_scores = np.abs(stats.zscore(df['tick_volume'].fillna(0)))
                volume_outliers = df[volume_z_scores > self.outlier_std]
                details['volume_outliers_count'] = len(volume_outliers)
        
        # Validar spread
        if 'spread' in df.columns:
            negative_spread = df[df['spread'] < 0]
            details['negative_spread_count'] = len(negative_spread)
            
            if len(negative_spread) > 0:
                validation_passed = False
                logger.error(f"✗ Spread negativo: {len(negative_spread)} registros")
            else:
                logger.info("✓ Spread válido")
                
            # Estadísticas de spread
            details['avg_spread'] = df['spread'].mean()
            details['max_spread'] = df['spread'].max()
            
        details['validation_passed'] = validation_passed
        return validation_passed, details
    
    def validate_price_continuity(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validar continuidad de precios
        
        Args:
            df: DataFrame ordenado por tiempo
            
        Returns:
            Tuple[bool, Dict]: (es_válido, detalles_validación)
        """
        logger.info("Validando continuidad de precios...")
        
        # Asegurar orden temporal
        df_sorted = df.sort_values('time').reset_index(drop=True)
        
        # Calcular gaps de precio (close to open)
        price_gaps = abs(df_sorted['open'].iloc[1:].values - 
                        df_sorted['close'].iloc[:-1].values)
        price_gap_pct = price_gaps / df_sorted['close'].iloc[:-1].values * 100
        
        # Detectar gaps significativos (> 2%)
        significant_gaps = price_gap_pct[price_gap_pct > 2]
        extreme_gaps = price_gap_pct[price_gap_pct > 5]
        
        # Calcular autocorrelación de retornos
        returns = df_sorted['close'].pct_change().dropna()
        if len(returns) > 10:
            autocorr = returns.autocorr(lag=1)
        else:
            autocorr = np.nan
            
        validation_passed = len(extreme_gaps) == 0
        
        details = {
            'total_gaps': len(price_gaps),
            'significant_gaps_count': len(significant_gaps),
            'extreme_gaps_count': len(extreme_gaps),
            'max_gap_percentage': price_gap_pct.max() if len(price_gap_pct) > 0 else 0,
            'avg_gap_percentage': price_gap_pct.mean() if len(price_gap_pct) > 0 else 0,
            'returns_autocorrelation': autocorr,
            'validation_passed': validation_passed
        }
        
        if validation_passed:
            logger.info("✓ Continuidad de precios aceptable")
        else:
            logger.warning(f"✗ Gaps extremos detectados: {len(extreme_gaps)}")
            
        return validation_passed, details
    
    def validate_data_flags(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validar flags de datos reales vs imputados
        
        Args:
            df: DataFrame con columna 'data_flag'
            
        Returns:
            Tuple[bool, Dict]: (es_válido, detalles_validación)
        """
        logger.info("Validando flags de datos...")
        
        if 'data_flag' not in df.columns:
            logger.error("ERROR: Columna 'data_flag' no encontrada")
            return False, {'error': 'data_flag column missing'}
        
        # Contar tipos de datos
        flag_counts = df['data_flag'].value_counts()
        total_records = len(df)
        
        # Calcular porcentajes
        real_data = df[df['data_flag'] == 'real']
        imputed_data = df[df['data_flag'].str.contains('imputed', na=False)]
        
        real_percentage = len(real_data) / total_records * 100
        imputed_percentage = len(imputed_data) / total_records * 100
        
        # Validar secuencias largas de datos imputados
        df['is_imputed'] = df['data_flag'].str.contains('imputed', na=False)
        imputed_sequences = self._find_sequences(df['is_imputed'])
        max_imputed_sequence = max(imputed_sequences) if imputed_sequences else 0
        
        # Criterio: al menos 80% datos reales y no más de 50 registros seguidos imputados
        validation_passed = real_percentage >= 80 and max_imputed_sequence <= 50
        
        details = {
            'total_records': total_records,
            'real_data_count': len(real_data),
            'imputed_data_count': len(imputed_data),
            'real_percentage': real_percentage,
            'imputed_percentage': imputed_percentage,
            'flag_distribution': flag_counts.to_dict(),
            'max_imputed_sequence': max_imputed_sequence,
            'validation_passed': validation_passed
        }
        
        logger.info(f"Datos reales: {len(real_data):,} ({real_percentage:.1f}%)")
        logger.info(f"Datos imputados: {len(imputed_data):,} ({imputed_percentage:.1f}%)")
        
        if validation_passed:
            logger.info("✓ Proporción de datos reales aceptable")
        else:
            logger.warning(f"✗ Proporción de datos reales baja o secuencias largas imputadas")
            
        return validation_passed, details
    
    def _find_sequences(self, boolean_series: pd.Series) -> List[int]:
        """
        Encontrar longitudes de secuencias True consecutivas
        
        Args:
            boolean_series: Serie booleana
            
        Returns:
            List[int]: Lista de longitudes de secuencias
        """
        sequences = []
        current_length = 0
        
        for value in boolean_series:
            if value:
                current_length += 1
            else:
                if current_length > 0:
                    sequences.append(current_length)
                current_length = 0
                
        if current_length > 0:
            sequences.append(current_length)
            
        return sequences
    
    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Ejecutar todas las validaciones de integridad
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Tuple[bool, Dict]: (todas_válidas, resumen_detalles)
        """
        results = {}
        all_valid = True

        # Comprobar timezone de la columna 'time'
        tz_result: Dict[str, any] = {}
        tz_valid = True
        try:
            tzinfo = df['time'].dt.tz
            if tzinfo is None:
                df['time'] = df['time'].dt.tz_localize('America/New_York')
                tz_result.update({'tzinfo_present': False, 'converted': True})
            else:
                tz_result['tzinfo_present'] = True
                if str(tzinfo) != 'America/New_York':
                    df['time'] = df['time'].dt.tz_convert('America/New_York')
                    tz_result.update({'converted': True, 'original_timezone': str(tzinfo)})
                else:
                    tz_result['converted'] = False
            tz_result['timezone'] = str(df['time'].dt.tz)
            tz_valid = tz_result['timezone'] == 'America/New_York'
        except Exception as e:
            tz_valid = False
            tz_result['error'] = str(e)
        tz_result['validation_passed'] = tz_valid
        results['timezone_check'] = tz_result
        all_valid &= tz_valid

        # Verificar columnas duplicadas
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        dup_result = {
            'has_duplicates': len(duplicate_cols) > 0,
            'duplicates': duplicate_cols,
            'validation_passed': len(duplicate_cols) == 0
        }
        if duplicate_cols:
            logger.warning(f"Columnas duplicadas detectadas: {duplicate_cols}")
        results['column_duplicates'] = dup_result
        all_valid &= dup_result['validation_passed']
        
        # OHLCV
        valid, details = self.validate_ohlcv(df)
        results['ohlcv'] = details
        all_valid &= valid
        
        # Volume
        valid, details = self.validate_volume(df)
        results['volume'] = details
        all_valid &= valid
        
        # Price continuity
        valid, details = self.validate_price_continuity(df)
        results['price_continuity'] = details
        all_valid &= valid
        
        # Data flags
        valid, details = self.validate_data_flags(df)
        results['data_flags'] = details
        all_valid &= valid
        
        results['all_validations_passed'] = all_valid
                return all_valid, results