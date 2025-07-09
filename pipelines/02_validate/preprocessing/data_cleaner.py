"""
Limpieza inteligente de datos para trading
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


class DataCleaner:
    """Limpiador inteligente de datos de trading"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializar limpiador
        
        Args:
            config: Configuración del limpiador
        """
        self.config = config or {}
        self.max_nan_percentage = self.config.get('max_nan_percentage', 5.0)
        self.outlier_std = self.config.get('outlier_std', 5)
        self.outlier_method = self.config.get('outlier_method', 'iqr')
        
        # Estrategias de limpieza por tipo de feature
        self.cleaning_strategies = {
            'returns': {
                'method': 'zero_fill',
                'outlier_treatment': 'winsorize'
            },
            'technical_indicators': {
                'method': 'interpolate',
                'outlier_treatment': 'clip'
            },
            'volume': {
                'method': 'forward_fill',
                'outlier_treatment': 'log_transform'
            },
            'price': {
                'method': 'interpolate',
                'outlier_treatment': 'none'
            },
            'binary': {
                'method': 'zero_fill',
                'outlier_treatment': 'none'
            },
            'temporal': {
                'method': 'none',
                'outlier_treatment': 'none'
            },
            'market_status': {
                'method': 'forward_fill',
                'outlier_treatment': 'none'
            }
        }
        
        # Mapeo de features a categorías
        self.feature_categories = {
            'returns': ['log_return', 'return_5m', 'volume_change', 'volume_delta'],
            'technical_indicators': [
                'rsi_14', 'rsi_28', 'stochastic_k', 'stochastic_d',
                'macd', 'macd_signal', 'macd_histogram', 'atr_14',
                'sma_20', 'sma_20_slope', 'realized_vol_20',
                'cb_level'
            ],
            'volume': ['tick_volume', 'volume_sma_20', 'volume_ratio'],
            'price': ['open', 'high', 'low', 'close', 'vwap', 'adj_factor'],
            'binary': ['doji', 'hammer', 'shooting_star',
                       'bullish_engulfing', 'bearish_engulfing'],
            'temporal': ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos'],
            'market_status': ['halt_flag']
        }
        
    def clean_nan_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Limpieza inteligente de valores NaN por tipo de feature
        
        Args:
            df: DataFrame con NaN
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (DataFrame limpio, estadísticas)
        """
        logger.info("Limpiando valores NaN...")
        
        df_clean = df.copy()
        nan_stats = {}
        
        # Analizar NaN inicial
        initial_nans = df_clean.isna().sum()
        nan_cols = initial_nans[initial_nans > 0]
        
        if len(nan_cols) == 0:
            logger.info("No hay valores NaN en el dataset")
            return df_clean, {'total_nans': 0}
        
        logger.info(f"Columnas con NaN: {len(nan_cols)}")
        
        # Limpiar por categoría
        for category, features in self.feature_categories.items():
            category_features = [f for f in features if f in df_clean.columns and f in nan_cols.index]
            
            if not category_features:
                continue
                
            strategy = self.cleaning_strategies.get(category, {})
            method = strategy.get('method', 'forward_fill')
            
            logger.info(f"  - Limpiando {category} con método: {method}")
            
            for feature in category_features:
                nan_count = df_clean[feature].isna().sum()
                nan_percentage = nan_count / len(df_clean) * 100
                
                if nan_percentage > self.max_nan_percentage:
                    logger.warning(f"    * {feature}: {nan_percentage:.1f}% NaN (alto)")
                
                # Aplicar método de limpieza
                if method == 'zero_fill':
                    df_clean[feature] = df_clean[feature].fillna(0)
                    
                elif method == 'forward_fill':
                    df_clean[feature] = df_clean[feature].ffill().bfill()
                    
                elif method == 'interpolate':
                    df_clean[feature] = df_clean[feature].interpolate(
                        method='linear', 
                        limit_direction='both'
                    )
                    
                elif method == 'mean_fill':
                    df_clean[feature] = df_clean[feature].fillna(
                        df_clean[feature].mean()
                    )
                
                # Verificar si quedan NaN
                remaining_nans = df_clean[feature].isna().sum()
                if remaining_nans > 0:
                    # Último recurso: forward fill + backward fill
                    df_clean[feature] = df_clean[feature].ffill().bfill()
                    
                    # Si aún hay NaN, llenar con 0
                    if df_clean[feature].isna().sum() > 0:
                        df_clean[feature] = df_clean[feature].fillna(0)
                
                nan_stats[feature] = {
                    'original_nans': nan_count,
                    'percentage': nan_percentage,
                    'method_used': method
                }
        
        # Limpiar columnas no categorizadas
        other_cols = [col for col in nan_cols.index 
                     if col not in sum(self.feature_categories.values(), [])]
        
        for col in other_cols:
            # Default: forward fill + backward fill
            df_clean[col] = df_clean[col].ffill().bfill()
            if df_clean[col].isna().sum() > 0:
                df_clean[col] = df_clean[col].fillna(0)
        
        # Estadísticas finales
        final_nans = df_clean.isna().sum().sum()
        nan_stats['summary'] = {
            'initial_total_nans': initial_nans.sum(),
            'final_total_nans': final_nans,
            'columns_affected': len(nan_cols),
            'cleaning_success_rate': (1 - final_nans / initial_nans.sum()) * 100 if initial_nans.sum() > 0 else 100
        }
        
        logger.info(f"Limpieza de NaN completada: {initial_nans.sum()} → {final_nans}")
        
        return df_clean, nan_stats
    
    def handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Manejo inteligente de outliers por tipo de feature
        
        Args:
            df: DataFrame con posibles outliers
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (DataFrame tratado, estadísticas)
        """
        logger.info("Manejando outliers...")
        
        df_clean = df.copy()
        outlier_stats = {}
        
        for category, features in self.feature_categories.items():
            category_features = [f for f in features if f in df_clean.columns]
            
            if not category_features:
                continue
                
            strategy = self.cleaning_strategies.get(category, {})
            outlier_treatment = strategy.get('outlier_treatment', 'none')
            
            if outlier_treatment == 'none':
                continue
                
            logger.info(f"  - Tratando outliers en {category} con método: {outlier_treatment}")
            
            for feature in category_features:
                if outlier_treatment == 'winsorize':
                    # Winsorización en percentiles
                    lower = df_clean[feature].quantile(0.001)
                    upper = df_clean[feature].quantile(0.999)
                    outliers_count = ((df_clean[feature] < lower) | 
                                    (df_clean[feature] > upper)).sum()
                    df_clean[feature] = df_clean[feature].clip(lower=lower, upper=upper)
                    
                elif outlier_treatment == 'clip':
                    # Clip por desviaciones estándar
                    mean = df_clean[feature].mean()
                    std = df_clean[feature].std()
                    lower = mean - self.outlier_std * std
                    upper = mean + self.outlier_std * std
                    outliers_count = ((df_clean[feature] < lower) | 
                                    (df_clean[feature] > upper)).sum()
                    df_clean[feature] = df_clean[feature].clip(lower=lower, upper=upper)
                    
                elif outlier_treatment == 'iqr':
                    # Método IQR
                    Q1 = df_clean[feature].quantile(0.25)
                    Q3 = df_clean[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outliers_count = ((df_clean[feature] < lower) | 
                                    (df_clean[feature] > upper)).sum()
                    df_clean[feature] = df_clean[feature].clip(lower=lower, upper=upper)
                    
                elif outlier_treatment == 'log_transform':
                    # Transformación logarítmica para volumen
                    if (df_clean[feature] >= 0).all():
                        df_clean[feature] = np.log1p(df_clean[feature])
                        outliers_count = 0  # No contamos después de transformación
                    else:
                        outliers_count = 0
                else:
                    outliers_count = 0
                
                if outliers_count > 0:
                    outlier_stats[feature] = {
                        'outliers_count': outliers_count,
                        'percentage': outliers_count / len(df_clean) * 100,
                        'treatment': outlier_treatment
                    }
        
        total_outliers = sum(stat.get('outliers_count', 0) 
                           for stat in outlier_stats.values())
        
        outlier_stats['summary'] = {
            'total_outliers_treated': total_outliers,
            'features_with_outliers': len(outlier_stats) - 1
        }
        
        logger.info(f"Tratamiento de outliers completado: {total_outliers} outliers tratados")
        
        return df_clean, outlier_stats
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Validar calidad de datos después de limpieza
        
        Args:
            df: DataFrame limpio
            
        Returns:
            Dict: Métricas de calidad
        """
        logger.info("Validando calidad de datos...")
        
        quality_metrics = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'nan_count': df.isna().sum().sum(),
            'nan_percentage': df.isna().sum().sum() / (len(df) * len(df.columns)) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'features_with_zero_variance': [],
            'features_with_high_correlation': [],
            'data_types': df.dtypes.value_counts().to_dict()
        }
        
        # Verificar varianza cero
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].var() < 1e-10:
                quality_metrics['features_with_zero_variance'].append(col)
        
        # Verificar correlaciones altas
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_pairs = []
            for col in upper_triangle.columns:
                high_corr = upper_triangle[col][upper_triangle[col] > 0.95].index.tolist()
                for corr_col in high_corr:
                    high_corr_pairs.append((col, corr_col))
            
            quality_metrics['features_with_high_correlation'] = high_corr_pairs
        
        # Verificar integridad OHLCV si aplica
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_valid = ((df['high'] >= df[['open', 'close']].max(axis=1)) & 
                         (df['low'] <= df[['open', 'close']].min(axis=1)) & 
                         (df['low'] <= df['high'])).all()
            quality_metrics['ohlcv_integrity'] = bool(ohlc_valid)
        
        return quality_metrics
    
    def final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza final del dataset
        
        Args:
            df: DataFrame casi limpio
            
        Returns:
            pd.DataFrame: DataFrame completamente limpio
        """
        logger.info("Aplicando limpieza final...")
        
        df_final = df.copy()
        
        # 1. Eliminar duplicados si existen
        duplicates_before = df_final.duplicated().sum()
        if duplicates_before > 0:
            df_final = df_final.drop_duplicates()
            logger.info(f"  - Eliminados {duplicates_before} duplicados")
        
        # 2. Asegurar tipos de datos correctos
        # Fechas
        if 'time' in df_final.columns:
            df_final['time'] = pd.to_datetime(df_final['time'])
        
        # Binarios
        for feature in self.feature_categories.get('binary', []):
            if feature in df_final.columns:
                df_final[feature] = df_final[feature].astype(int)
        for feature in self.feature_categories.get('market_status', []):
            if feature in df_final.columns:
                df_final[feature] = df_final[feature].astype(int)
        
        # 3. Ordenar por tiempo si existe
        if 'time' in df_final.columns:
            df_final = df_final.sort_values('time').reset_index(drop=True)
        
        # 4. Verificación final de NaN
        final_nans = df_final.isna().sum().sum()
        if final_nans > 0:
            logger.warning(f"  - Aún quedan {final_nans} NaN después de limpieza")
            # Último recurso: eliminar filas con NaN
            df_final = df_final.dropna()
            logger.info(f"  - Eliminadas filas con NaN. Registros finales: {len(df_final)}")
        
        # 5. Resetear índice
        df_final = df_final.reset_index(drop=True)
        
        logger.info("Limpieza final completada")
        
        return df_final
    
    def clean_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Proceso completo de limpieza
        
        Args:
            df: DataFrame a limpiar
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (DataFrame limpio, resumen de limpieza)
        """
        logger.info("="*50)
        logger.info("INICIANDO PROCESO DE LIMPIEZA DE DATOS")
        logger.info("="*50)
        
        initial_shape = df.shape
        cleaning_summary = {
            'initial_shape': initial_shape,
            'steps': {}
        }
        
        # 1. Limpiar NaN
        df_clean, nan_stats = self.clean_nan_values(df)
        cleaning_summary['steps']['nan_cleaning'] = nan_stats
        
        # 2. Manejar outliers
        df_clean, outlier_stats = self.handle_outliers(df_clean)
        cleaning_summary['steps']['outlier_treatment'] = outlier_stats
        
        # 3. Limpieza final
        df_clean = self.final_cleanup(df_clean)
        
        # 4. Validar calidad
        quality_metrics = self.validate_data_quality(df_clean)
        cleaning_summary['quality_metrics'] = quality_metrics
        
        # Resumen final
        cleaning_summary['final_shape'] = df_clean.shape
        cleaning_summary['records_removed'] = initial_shape[0] - df_clean.shape[0]
        cleaning_summary['removal_percentage'] = (
            cleaning_summary['records_removed'] / initial_shape[0] * 100
        )
        
        logger.info("="*50)
        logger.info("LIMPIEZA COMPLETADA")
        logger.info(f"Shape inicial: {initial_shape}")
        logger.info(f"Shape final: {df_clean.shape}")
        logger.info(f"Registros eliminados: {cleaning_summary['records_removed']} "
                   f"({cleaning_summary['removal_percentage']:.1f}%)")
        logger.info("="*50)
        
        return df_clean, cleaning_summary