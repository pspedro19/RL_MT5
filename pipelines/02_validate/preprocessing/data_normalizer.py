"""
Normalización robusta de datos para trading
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import json

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalizador robusto para datos de trading"""
    
    def __init__(self, method: str = 'robust', config: Optional[Dict] = None):
        """
        Inicializar normalizador
        
        Args:
            method: Método de normalización ('robust', 'standard', 'minmax', 'feature_specific')
            config: Configuración adicional
        """
        self.method = method
        self.config = config or {}
        self.scalers = {}
        self.normalization_params = {}
        
        # Configuración de ventanas para normalización adaptativa
        self.rolling_window = self.config.get('rolling_window', 20)
        self.min_periods = self.config.get('min_periods', 10)
        
        # Features por categoría
        self.feature_categories = {
            'price': ['open', 'high', 'low', 'close', 'vwap'],
            'volume': ['tick_volume', 'volume_sma_20', 'volume_relative', 
                      'volume_ratio', 'volume_delta'],
            'technical_bounded': ['rsi_14', 'rsi_28', 'stochastic_k', 
                                'stochastic_d', 'percent_r'],
            'technical_unbounded': ['macd', 'macd_signal', 'macd_histogram', 
                                  'atr_14', 'bollinger_width'],
            'returns': ['log_return', 'return_5m', 'cumulative_return_session'],
            'temporal': ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
                        'dow_sin', 'dow_cos'],
            'binary': ['doji', 'hammer', 'shooting_star', 'bullish_engulfing',
                      'bearish_engulfing'],
            'exclude': ['time', 'symbol', 'data_flag', 'quality_score']
        }
        
    def normalize_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizar features de precio con método adaptativo
        
        Args:
            df: DataFrame con features de precio
            
        Returns:
            pd.DataFrame: DataFrame con precios normalizados
        """
        logger.info("Normalizando features de precio...")
        
        df_norm = df.copy()
        price_features = [f for f in self.feature_categories['price'] 
                         if f in df.columns]
        
        if not price_features:
            return df_norm
        
        if self.method == 'custom':
            # Normalización relativa al precio de cierre móvil
            rolling_close = df['close'].rolling(
                window=self.rolling_window,
                min_periods=self.min_periods
            ).mean()
            
            for feature in price_features:
                df_norm[f'{feature}_norm'] = df[feature] / rolling_close
                # Mantener original para referencia
                df_norm[f'{feature}_orig'] = df[feature]
                df_norm[feature] = df_norm[f'{feature}_norm']
                
            # Guardar parámetros
            self.normalization_params['price'] = {
                'method': 'rolling_close_relative',
                'window': self.rolling_window
            }
            
        else:
            # Usar scaler estándar
            scaler = self._get_scaler('price')
            df_norm[price_features] = scaler.fit_transform(df[price_features])
            self.scalers['price'] = scaler
        
        logger.info(f"  - Normalizados {len(price_features)} features de precio")
        
        return df_norm
    
    def normalize_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizar features de volumen
        
        Args:
            df: DataFrame con features de volumen
            
        Returns:
            pd.DataFrame: DataFrame con volumen normalizado
        """
        logger.info("Normalizando features de volumen...")
        
        df_norm = df.copy()
        volume_features = [f for f in self.feature_categories['volume'] 
                          if f in df.columns]
        
        if not volume_features:
            return df_norm
        
        if self.method == 'custom':
            # Normalización relativa al volumen móvil
            if 'tick_volume' in df.columns:
                rolling_volume = df['tick_volume'].rolling(
                    window=self.rolling_window,
                    min_periods=self.min_periods
                ).mean()
                
                for feature in volume_features:
                    if feature == 'tick_volume':
                        # Log transform para volumen
                        df_norm[f'{feature}_log'] = np.log1p(df[feature])
                        df_norm[feature] = df_norm[f'{feature}_log'] / np.log1p(rolling_volume)
                    else:
                        df_norm[feature] = df[feature] / rolling_volume.clip(lower=1)
                        
                self.normalization_params['volume'] = {
                    'method': 'rolling_volume_relative',
                    'window': self.rolling_window,
                    'log_transform': True
                }
        else:
            # Log transform primero
            df_log = df[volume_features].copy()
            for col in volume_features:
                df_log[col] = np.log1p(df_log[col])
                
            # Luego aplicar scaler
            scaler = self._get_scaler('volume')
            df_norm[volume_features] = scaler.fit_transform(df_log)
            self.scalers['volume'] = scaler
        
        logger.info(f"  - Normalizados {len(volume_features)} features de volumen")
        
        return df_norm
    
    def normalize_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizar indicadores técnicos según su naturaleza
        
        Args:
            df: DataFrame con indicadores técnicos
            
        Returns:
            pd.DataFrame: DataFrame con indicadores normalizados
        """
        logger.info("Normalizando indicadores técnicos...")
        
        df_norm = df.copy()
        
        # Indicadores acotados (0-100) - no necesitan normalización
        bounded_features = [f for f in self.feature_categories['technical_bounded'] 
                           if f in df.columns]
        logger.info(f"  - {len(bounded_features)} indicadores acotados (sin cambios)")
        
        # Indicadores no acotados
        unbounded_features = [f for f in self.feature_categories['technical_unbounded'] 
                             if f in df.columns]
        
        if unbounded_features:
            if self.method == 'custom':
                # Normalización específica por indicador
                for feature in unbounded_features:
                    if 'macd' in feature:
                        # MACD: normalizar por ATR
                        if 'atr_14' in df.columns:
                            df_norm[feature] = df[feature] / df['atr_14'].clip(lower=0.001)
                    elif feature == 'atr_14':
                        # ATR: normalizar por precio
                        df_norm[feature] = df[feature] / df['close']
                    elif feature == 'bollinger_width':
                        # Bollinger Width: ya está normalizado por precio
                        pass
                    else:
                        # Default: robust scaler
                        scaler = RobustScaler()
                        df_norm[feature] = scaler.fit_transform(df[[feature]])
                        
                self.normalization_params['technical'] = {
                    'method': 'indicator_specific'
                }
            else:
                # Scaler estándar para no acotados
                scaler = self._get_scaler('technical_unbounded')
                df_norm[unbounded_features] = scaler.fit_transform(df[unbounded_features])
                self.scalers['technical_unbounded'] = scaler
        
        logger.info(f"  - Normalizados {len(unbounded_features)} indicadores no acotados")
        
        return df_norm
    
    def normalize_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizar features de retorno
        
        Args:
            df: DataFrame con features de retorno
            
        Returns:
            pd.DataFrame: DataFrame con retornos normalizados
        """
        logger.info("Normalizando features de retorno...")
        
        df_norm = df.copy()
        return_features = [f for f in self.feature_categories['returns'] 
                          if f in df.columns]
        
        if not return_features:
            return df_norm
        
        # Los retornos ya están normalizados por naturaleza
        # Solo aplicar winsorización para outliers extremos
        for feature in return_features:
            # Winsorize en percentiles 0.1 y 99.9
            lower = df[feature].quantile(0.001)
            upper = df[feature].quantile(0.999)
            df_norm[feature] = df[feature].clip(lower=lower, upper=upper)
        
        self.normalization_params['returns'] = {
            'method': 'winsorization',
            'percentiles': [0.001, 0.999]
        }
        
        logger.info(f"  - Winsorized {len(return_features)} features de retorno")
        
        return df_norm
    
    def fit(self, df: pd.DataFrame) -> 'DataNormalizer':
        """
        Ajustar normalizador a los datos
        
        Args:
            df: DataFrame de entrenamiento
            
        Returns:
            self
        """
        logger.info("Ajustando normalizador a datos de entrenamiento...")
        
        # Ajustar scalers por categoría
        for category, features in self.feature_categories.items():
            if category == 'exclude':
                continue
                
            category_features = [f for f in features if f in df.columns]
            if not category_features:
                continue
                
            if category in ['technical_bounded', 'binary', 'temporal']:
                # Estos no necesitan scaler
                continue
                
            if self.method != 'custom':
                scaler = self._get_scaler(category)
                scaler.fit(df[category_features])
                self.scalers[category] = scaler
        
        logger.info("Normalizador ajustado")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transformar datos usando parámetros ajustados
        
        Args:
            df: DataFrame a transformar
            
        Returns:
            pd.DataFrame: DataFrame normalizado
        """
        logger.info("Aplicando normalización...")
        
        df_norm = df.copy()
        
        # Aplicar normalización por categoría
        df_norm = self.normalize_price_features(df_norm)
        df_norm = self.normalize_volume_features(df_norm)
        df_norm = self.normalize_technical_indicators(df_norm)
        df_norm = self.normalize_returns(df_norm)
        
        # Features temporales y binarios no necesitan normalización
        
        logger.info("Normalización completada")
        
        return df_norm
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajustar y transformar en un solo paso
        
        Args:
            df: DataFrame a normalizar
            
        Returns:
            pd.DataFrame: DataFrame normalizado
        """
        self.fit(df)
        return self.transform(df)
    
    def save_params(self, filepath: str):
        """
        Guardar parámetros de normalización
        
        Args:
            filepath: Ruta para guardar parámetros
        """
        params = {
            'method': self.method,
            'config': self.config,
            'normalization_params': self.normalization_params,
            'feature_categories': self.feature_categories
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
            
        logger.info(f"Parámetros de normalización guardados en {filepath}")
    
    def load_params(self, filepath: str):
        """
        Cargar parámetros de normalización
        
        Args:
            filepath: Ruta de parámetros
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
            
        self.method = params['method']
        self.config = params['config']
        self.normalization_params = params['normalization_params']
        self.feature_categories = params['feature_categories']
        
        logger.info(f"Parámetros de normalización cargados de {filepath}")
    
    def _get_scaler(self, category: str):
        """
        Obtener scaler apropiado para la categoría
        
        Args:
            category: Categoría de features
            
        Returns:
            Scaler de sklearn
        """
        if self.method == 'robust':
            return RobustScaler()
        elif self.method == 'standard':
            return StandardScaler()
        elif self.method == 'minmax':
            return MinMaxScaler()
        else:
            # Default
            return RobustScaler()
    
    def get_normalization_stats(self, df: pd.DataFrame) -> Dict:
        """
        Obtener estadísticas de normalización
        
        Args:
            df: DataFrame normalizado
            
        Returns:
            Dict: Estadísticas por categoría
        """
        stats = {}
        
        for category, features in self.feature_categories.items():
            if category == 'exclude':
                continue
                
            category_features = [f for f in features if f in df.columns]
            if not category_features:
                continue
                
            stats[category] = {
                'features': category_features,
                'count': len(category_features),
                'mean': df[category_features].mean().to_dict(),
                'std': df[category_features].std().to_dict(),
                'min': df[category_features].min().to_dict(),
                'max': df[category_features].max().to_dict()
            }
        
        return stats

    def apply_robust_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplicar normalización robusta específica por tipo de feature
        
        Args:
            df: DataFrame a normalizar
            
        Returns:
            pd.DataFrame: DataFrame normalizado
        """
        logger.info("Aplicando normalización robusta específica por tipo de feature...")
        
        df_norm = df.copy()
        
        # 1. Features de precio - normalizar con respecto al precio de cierre
        price_features_present = [col for col in self.feature_categories['price'] if col in df_norm.columns]
        if price_features_present:
            logger.info(f"Normalizando features de precio: {price_features_present}")
            for col in price_features_present:
                if col != 'close' and 'close' in df_norm.columns:
                    # Normalizar con respecto al precio de cierre promedio
                    price_ratio = df_norm['close'].rolling(20).mean()
                    if not price_ratio.isna().all():
                        df_norm[col] = df_norm[col] / price_ratio
                        df_norm[col] = df_norm[col].fillna(1.0)  # Fill NaN con 1.0
        
        # 2. Features de volumen - normalizar con respecto al volumen promedio
        volume_features_present = [col for col in self.feature_categories['volume'] if col in df_norm.columns]
        if volume_features_present:
            logger.info(f"Normalizando features de volumen: {volume_features_present}")
            for col in volume_features_present:
                if col != 'tick_volume' and 'tick_volume' in df_norm.columns:
                    # Normalizar con respecto al volumen promedio
                    volume_ratio = df_norm['tick_volume'].rolling(20).mean()
                    if not volume_ratio.isna().all():
                        df_norm[col] = df_norm[col] / volume_ratio
                        df_norm[col] = df_norm[col].fillna(1.0)
        
        # 3. Features técnicos - ya están en rangos específicos, verificar rangos
        tech_features_present = [col for col in self.feature_categories['technical_bounded'] if col in df_norm.columns]
        if tech_features_present:
            logger.info(f"Verificando rangos de features técnicos: {tech_features_present}")
            for col in tech_features_present:
                if col.startswith('rsi'):
                    # RSI debe estar entre 0 y 100
                    df_norm[col] = df_norm[col].clip(0, 100)
                elif col.startswith('stochastic'):
                    # Stochastic debe estar entre 0 y 100
                    df_norm[col] = df_norm[col].clip(0, 100)
        
        # 4. Features de retorno - ya están normalizados, verificar outliers
        return_features_present = [col for col in self.feature_categories['returns'] if col in df_norm.columns]
        if return_features_present:
            logger.info(f"Verificando outliers en features de retorno: {return_features_present}")
            for col in return_features_present:
                # Clip outliers extremos (más de 10 desviaciones estándar)
                std = df_norm[col].std()
                if std > 0:
                    df_norm[col] = df_norm[col].clip(-10 * std, 10 * std)
        
        # 5. Features temporales - ya están normalizados (sin/cos)
        temporal_features_present = [col for col in self.feature_categories['temporal'] if col in df_norm.columns]
        if temporal_features_present:
            logger.info(f"Features temporales ya normalizados: {temporal_features_present}")
        
        # 6. Features de sesión - verificar rangos
        session_features_present = [col for col in self.feature_categories['session'] if col in df_norm.columns]
        if session_features_present:
            logger.info(f"Verificando rangos de features de sesión: {session_features_present}")
            for col in session_features_present:
                if col == 'session_progress':
                    # Debe estar entre 0 y 1
                    df_norm[col] = df_norm[col].clip(0, 1)
        
        logger.info("Normalización robusta específica por feature completada")
        return df_norm
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizar datos según el método configurado
        
        Args:
            df: DataFrame a normalizar
            
        Returns:
            pd.DataFrame: DataFrame normalizado
        """
        if self.method == 'feature_specific':
            return self.apply_robust_normalization(df)
        elif self.method == 'robust':
            return self._apply_robust_scaling(df)
        elif self.method == 'standard':
            return self._apply_standard_scaling(df)
        elif self.method == 'minmax':
            return self._apply_minmax_scaling(df)
        else:
            logger.warning(f"Método de normalización '{self.method}' no reconocido, usando robusto")
            return self._apply_robust_scaling(df)
    
    def _apply_robust_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplicar escalado robusto"""
        logger.info("Aplicando escalado robusto...")
        
        # Identificar columnas numéricas para escalar
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [col for col in self.feature_categories['exclude'] if col in numeric_cols]
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not scale_cols:
            logger.warning("No hay columnas numéricas para escalar")
            return df
        
        # Aplicar RobustScaler
        scaler = RobustScaler()
        df_scaled = df.copy()
        df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
        
        # Guardar scaler para uso futuro
        self.scalers['robust'] = scaler
        
        logger.info(f"Escalado robusto aplicado a {len(scale_cols)} columnas")
        return df_scaled
    
    def _apply_standard_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplicar escalado estándar"""
        logger.info("Aplicando escalado estándar...")
        
        # Identificar columnas numéricas para escalar
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [col for col in self.feature_categories['exclude'] if col in numeric_cols]
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not scale_cols:
            logger.warning("No hay columnas numéricas para escalar")
            return df
        
        # Aplicar StandardScaler
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
        
        # Guardar scaler para uso futuro
        self.scalers['standard'] = scaler
        
        logger.info(f"Escalado estándar aplicado a {len(scale_cols)} columnas")
        return df_scaled
    
    def _apply_minmax_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplicar escalado MinMax"""
        logger.info("Aplicando escalado MinMax...")
        
        # Identificar columnas numéricas para escalar
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [col for col in self.feature_categories['exclude'] if col in numeric_cols]
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not scale_cols:
            logger.warning("No hay columnas numéricas para escalar")
            return df
        
        # Aplicar MinMaxScaler
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
        
        # Guardar scaler para uso futuro
        self.scalers['minmax'] = scaler
        
        logger.info(f"Escalado MinMax aplicado a {len(scale_cols)} columnas")
        return df_scaled
    
    def get_normalization_info(self) -> Dict:
        """
        Obtener información sobre la normalización aplicada
        
        Returns:
            Dict: Información de normalización
        """
        info = {
            'method': self.method,
            'scalers_used': list(self.scalers.keys()),
            'feature_types': {
                'price_features': self.feature_categories['price'],
                'volume_features': self.feature_categories['volume'],
                'tech_features': self.feature_categories['technical_bounded'],
                'return_features': self.feature_categories['returns'],
                'temporal_features': self.feature_categories['temporal'],
                'session_features': self.feature_categories['session'],
                'exclude_cols': self.feature_categories['exclude']
            }
        }
        
        return info