"""
Checklist de calidad avanzado para validación de datasets
"""
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class QualityChecklist:
    """Checklist de calidad avanzado para validación de datasets"""
    
    def __init__(self, config: Dict = None):
        """
        Inicializar checklist de calidad
        
        Args:
            config: Configuración del checklist
        """
        self.config = config or {}
        
        # Configuración de validaciones
        self.min_coverage_percentage = config.get('min_coverage_percentage', 0.7)
        self.max_outlier_percentage = config.get('max_outlier_percentage', 0.01)
        self.expected_daily_bars = config.get('expected_daily_bars', 78)  # 6.5 horas * 12 registros/hora
        
    def generate_checklist(self, df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """
        Generar checklist completo de calidad
        
        Args:
            df: DataFrame completo
            train_df: DataFrame de entrenamiento
            val_df: DataFrame de validación
            
        Returns:
            Dict: Checklist de calidad
        """
        logger.info("Generando checklist de calidad avanzado...")
        
        checklist = {}
        
        # 1. Cobertura y continuidad temporal
        checklist.update(self._check_temporal_coverage(df))
        
        # 2. Integridad de mercado y calendario
        checklist.update(self._check_market_integrity(df))
        
        # 3. Análisis de NaN
        checklist.update(self._check_nan_analysis(df))
        
        # 4. Coherencia numérica
        checklist.update(self._check_numerical_coherence(df))
        
        # 5. Análisis de gaps
        checklist.update(self._check_gaps(df))
        
        # 6. Análisis estadístico
        checklist.update(self._check_statistical_analysis(df))
        
        # 7. Análisis de features
        checklist.update(self._check_features_analysis(df))
        
        # 8. Análisis de splits
        checklist.update(self._check_splits_analysis(train_df, val_df))
        
        # 9. Calcular score general
        checklist['overall_score'] = self._calculate_overall_score(checklist)
        
        # 10. Estado general
        checklist['overall_status'] = self._determine_overall_status(checklist)
        
        logger.info(f"Checklist completado. Score: {checklist['overall_score']:.1f}/100")
        
        return checklist
    
    def _check_temporal_coverage(self, df: pd.DataFrame) -> Dict:
        """Verificar cobertura y continuidad temporal"""
        logger.info("Verificando cobertura temporal...")
        
        # Cobertura de días de trading
        expected_days = pd.date_range(df['time'].min().date(), df['time'].max().date(), freq='B')
        actual_days = pd.to_datetime(df['time'].dt.date.unique())
        trading_days_coverage = len(actual_days) / len(expected_days) if len(expected_days) > 0 else 0
        
        # Verificar duplicados
        duplicates = int(df.duplicated('time').sum())
        
        # Verificar monotonicidad
        is_monotonic = bool(df['time'].is_monotonic_increasing)
        
        # Cobertura diaria
        daily_counts = df.groupby(df['time'].dt.date).size()
        perfect_days = (daily_counts == self.expected_daily_bars).sum()
        good_days = ((daily_counts >= self.expected_daily_bars * 0.9) & 
                    (daily_counts < self.expected_daily_bars)).sum()
        acceptable_days = ((daily_counts >= self.expected_daily_bars * 0.7) & 
                          (daily_counts < self.expected_daily_bars * 0.9)).sum()
        poor_days = (daily_counts < self.expected_daily_bars * 0.7).sum()
        
        return {
            'temporal_coverage': {
                'trading_days_coverage': trading_days_coverage,
                'duplicates': duplicates,
                'is_monotonic': is_monotonic,
                'perfect_days': int(perfect_days),
                'good_days': int(good_days),
                'acceptable_days': int(acceptable_days),
                'poor_days': int(poor_days),
                'total_trading_days': len(actual_days),
                'expected_trading_days': len(expected_days)
            }
        }
    
    def _check_market_integrity(self, df: pd.DataFrame) -> Dict:
        """Verificar integridad de mercado y calendario"""
        logger.info("Verificando integridad de mercado...")
        
        # Verificar fines de semana
        weekend_records = int((df['time'].dt.weekday >= 5).sum())
        
        # Verificar horarios de mercado (9:30 AM - 4:00 PM EST)
        market_hours_mask = (
            (df['time'].dt.hour >= 9) & (df['time'].dt.hour < 16) |
            ((df['time'].dt.hour == 16) & (df['time'].dt.minute == 0))
        )
        outside_market_hours = int((~market_hours_mask).sum())
        
        # Verificar flags de datos
        real_data_count = int((df['data_flag'] == 'real').sum()) if 'data_flag' in df.columns else 0
        imputed_data_count = int((df['data_flag'] == 'imputed_brownian').sum()) if 'data_flag' in df.columns else 0
        total_records = len(df)
        real_percentage = real_data_count / total_records * 100 if total_records > 0 else 0
        
        return {
            'market_integrity': {
                'weekend_records': weekend_records,
                'outside_market_hours': outside_market_hours,
                'real_data_count': real_data_count,
                'imputed_data_count': imputed_data_count,
                'real_percentage': real_percentage,
                'total_records': total_records
            }
        }
    
    def _check_nan_analysis(self, df: pd.DataFrame) -> Dict:
        """Analizar valores NaN"""
        logger.info("Analizando valores NaN...")
        
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        total_nan = nan_counts.sum()
        nan_percentage = total_nan / (len(df) * len(df.columns)) * 100 if len(df) > 0 else 0
        
        return {
            'nan_analysis': {
                'total_nan': int(total_nan),
                'nan_percentage': nan_percentage,
                'columns_with_nan': int(len(nan_cols)),
                'nan_by_column': nan_cols.to_dict() if len(nan_cols) > 0 else {}
            }
        }
    
    def _check_numerical_coherence(self, df: pd.DataFrame) -> Dict:
        """Verificar coherencia numérica"""
        logger.info("Verificando coherencia numérica...")
        
        # Verificar OHLC
        ohlc_valid = True
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_valid = int((
                (df['high'] >= df[['open', 'close']].max(axis=1)) & 
                (df['low'] <= df[['open', 'close']].min(axis=1)) & 
                (df['low'] <= df['high'])
            ).all())
        
        # Verificar precios positivos
        price_cols = ['open', 'high', 'low', 'close']
        price_cols_present = [col for col in price_cols if col in df.columns]
        positive_prices = True
        if price_cols_present:
            positive_prices = float(df[price_cols_present].min().min()) > 0
        
        # Verificar volumen no negativo
        nonnegative_volume = True
        if 'tick_volume' in df.columns:
            nonnegative_volume = float(df['tick_volume'].min()) >= 0
        
        # Verificar spread no negativo
        nonnegative_spread = True
        if 'spread' in df.columns:
            nonnegative_spread = float(df['spread'].min()) >= 0
        
        return {
            'numerical_coherence': {
                'ohlc_valid': bool(ohlc_valid),
                'positive_prices': bool(positive_prices),
                'nonnegative_volume': bool(nonnegative_volume),
                'nonnegative_spread': bool(nonnegative_spread)
            }
        }
    
    def _check_gaps(self, df: pd.DataFrame) -> Dict:
        """Verificar gaps en los datos"""
        logger.info("Verificando gaps en los datos...")
        
        # Calcular diferencias de tiempo
        time_diffs = df['time'].sort_values().diff().dt.total_seconds().div(60).fillna(5)
        
        # Contar gaps mayores a 5 minutos
        gaps_over_5min = int((time_diffs > 5).sum())
        
        # Análisis de distribución de gaps
        gap_stats = {
            'mean': float(time_diffs.mean()),
            'std': float(time_diffs.std()),
            'min': float(time_diffs.min()),
            'max': float(time_diffs.max()),
            'median': float(time_diffs.median())
        }
        
        return {
            'gaps_analysis': {
                'gaps_over_5min': gaps_over_5min,
                'gap_statistics': gap_stats,
                'total_intervals': len(time_diffs)
            }
        }
    
    def _check_statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """Análisis estadístico de los datos"""
        logger.info("Realizando análisis estadístico...")
        
        # Análisis de outliers en retornos
        return_outliers = 0.0
        if 'log_return' in df.columns:
            std = df['log_return'].std()
            if std > 0:
                outliers = (df['log_return'].abs() > 20 * std).sum()
                return_outliers = float(outliers / len(df) * 100)
        
        # Análisis de volatilidad
        volatility_analysis = {}
        if 'log_return' in df.columns:
            volatility_analysis = {
                'mean_return': float(df['log_return'].mean()),
                'std_return': float(df['log_return'].std()),
                'skewness': float(df['log_return'].skew()),
                'kurtosis': float(df['log_return'].kurtosis())
            }
        
        return {
            'statistical_analysis': {
                'return_outliers_percentage': return_outliers,
                'volatility_analysis': volatility_analysis
            }
        }
    
    def _check_features_analysis(self, df: pd.DataFrame) -> Dict:
        """Análisis de features"""
        logger.info("Analizando features...")
        
        # Contar features
        feature_cols = [c for c in df.columns if c not in ['time', 'data_flag', 'quality_score']]
        features_count = len(feature_cols)
        
        # Categorizar features
        price_features = ['open', 'high', 'low', 'close']
        volume_features = ['tick_volume', 'volume_sma_20', 'volume_relative']
        technical_features = ['ema_9', 'ema_21', 'ema_55', 'macd', 'rsi_14', 'rsi_28']
        temporal_features = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']
        session_features = ['session_progress', 'is_first_30min', 'is_last_30min']
        
        feature_categories = {
            'price_features': [f for f in price_features if f in df.columns],
            'volume_features': [f for f in volume_features if f in df.columns],
            'technical_features': [f for f in technical_features if f in df.columns],
            'temporal_features': [f for f in temporal_features if f in df.columns],
            'session_features': [f for f in session_features if f in df.columns]
        }
        
        return {
            'features_analysis': {
                'features_count': features_count,
                'feature_categories': feature_categories,
                'total_categories': len(feature_categories)
            }
        }
    
    def _check_splits_analysis(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """Análisis de splits de entrenamiento/validación"""
        logger.info("Analizando splits de entrenamiento/validación...")
        
        train_size = len(train_df)
        val_size = len(val_df)
        total_size = train_size + val_size
        
        # Verificar que no haya overlap temporal
        train_max_time = train_df['time'].max() if len(train_df) > 0 else None
        val_min_time = val_df['time'].min() if len(val_df) > 0 else None
        
        temporal_overlap = False
        if train_max_time and val_min_time:
            temporal_overlap = train_max_time >= val_min_time
        
        # Verificar proporción
        train_ratio = train_size / total_size if total_size > 0 else 0
        val_ratio = val_size / total_size if total_size > 0 else 0
        
        return {
            'splits_analysis': {
                'train_size': train_size,
                'val_size': val_size,
                'total_size': total_size,
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'temporal_overlap': temporal_overlap,
                'train_start': str(train_df['time'].min()) if len(train_df) > 0 else None,
                'train_end': str(train_df['time'].max()) if len(train_df) > 0 else None,
                'val_start': str(val_df['time'].min()) if len(val_df) > 0 else None,
                'val_end': str(val_df['time'].max()) if len(val_df) > 0 else None
            }
        }
    
    def _calculate_overall_score(self, checklist: Dict) -> float:
        """Calcular score general del checklist"""
        score = 100.0
        deductions = 0.0
        
        # Deducciones por problemas críticos
        if checklist['temporal_coverage']['duplicates'] > 0:
            deductions += 10.0
        
        if checklist['temporal_coverage']['weekend_records'] > 0:
            deductions += 15.0
        
        if checklist['market_integrity']['outside_market_hours'] > 0:
            deductions += 10.0
        
        if checklist['nan_analysis']['nan_percentage'] > 5.0:
            deductions += 10.0
        
        if not checklist['numerical_coherence']['ohlc_valid']:
            deductions += 20.0
        
        if checklist['gaps_analysis']['gaps_over_5min'] > 100:
            deductions += 5.0
        
        if checklist['statistical_analysis']['return_outliers_percentage'] > 1.0:
            deductions += 5.0
        
        if checklist['splits_analysis']['temporal_overlap']:
            deductions += 15.0
        
        # Deducciones por cobertura
        coverage = checklist['temporal_coverage']['trading_days_coverage']
        if coverage < 0.8:
            deductions += (0.8 - coverage) * 50
        
        # Deducciones por proporción de datos reales
        real_percentage = checklist['market_integrity']['real_percentage']
        if real_percentage < 80:
            deductions += (80 - real_percentage) * 0.5
        
        final_score = max(0.0, score - deductions)
        return final_score
    
    def _determine_overall_status(self, checklist: Dict) -> str:
        """Determinar estado general basado en el score"""
        score = checklist['overall_score']
        
        if score >= 90:
            return 'EXCELLENT'
        elif score >= 80:
            return 'GOOD'
        elif score >= 70:
            return 'ACCEPTABLE'
        elif score >= 60:
            return 'POOR'
        else:
            return 'FAILED'
    
    def save_checklist_report(self, checklist: Dict, output_dir: str):
        """
        Guardar reporte del checklist
        
        Args:
            checklist: Checklist generado
            output_dir: Directorio de salida
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar JSON
        with open(os.path.join(output_dir, 'quality_checklist_report.json'), 'w') as f:
            json.dump(checklist, f, indent=2)
        
        # Guardar Markdown
        with open(os.path.join(output_dir, 'quality_checklist_report.md'), 'w') as f:
            f.write("# Quality Checklist Report\n\n")
            f.write(f"**Overall Score:** {checklist['overall_score']:.1f}/100\n")
            f.write(f"**Status:** {checklist['overall_status']}\n\n")
            
            for category, data in checklist.items():
                if category not in ['overall_score', 'overall_status']:
                    f.write(f"## {category.replace('_', ' ').title()}\n\n")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                    else:
                        f.write(f"{data}\n")
                    f.write("\n")
        
        logger.info(f"Reporte del checklist guardado en {output_dir}") 