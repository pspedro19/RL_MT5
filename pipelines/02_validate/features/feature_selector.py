"""
Selección de features automática y heurística
"""
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Selector de features con métodos automáticos y heurísticos"""
    
    def __init__(self, config: Dict):
        """
        Inicializar selector de features
        
        Args:
            config: Configuración del selector
        """
        self.config = config
        
        # Configuración de selección
        self.variance_threshold = config.get('variance_threshold', 1e-5)
        self.correlation_threshold = config.get('correlation_threshold', 0.95)
        self.use_heuristic = config.get('use_heuristic', True)
        self.use_importance = config.get('use_importance', True)
        
        # Features recomendados por heurística
        self.recommended_features = [
            'open', 'high', 'low', 'close', 'log_return',
            'ema_21', 'ema_55', 'sma_200', 'atr_14', 'volatility_20',
            'rsi_14', 'macd_histogram', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'vwap', 'volume_ratio', 'doji', 'bullish_engulfing', 'bearish_engulfing',
            'minutes_since_open', 'minutes_to_close', 'session_progress'
        ]
        
        # Features obligatorios que siempre se incluyen
        self.mandatory_features = [
            'time', 'data_flag'
        ]
        
        # Resultados de selección
        self.selected_features = []
        self.feature_importances = {}
        self.selection_info = {}
        
    def select_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, List[str], Optional[Dict]]:
        """
        Selección automática y heurística de features
        
        Args:
            df: DataFrame con features
            target_col: Columna objetivo para calcular importancia (opcional)
            
        Returns:
            Tuple[pd.DataFrame, List[str], Optional[Dict]]: (DataFrame seleccionado, features seleccionados, importancias)
        """
        logger.info("Iniciando selección automática y heurística de features...")
        
        warnings.filterwarnings('ignore')
        
        # 1. Eliminar constantes/casi constantes
        logger.info("Eliminando features constantes...")
        X, constant_features_removed = self._remove_constant_features(df)
        
        # 2. Eliminar features altamente correlacionadas
        logger.info("Eliminando features altamente correlacionadas...")
        X, correlated_features_removed = self._remove_correlated_features(X)
        
        # 3. Aplicar heurística de features recomendados
        if self.use_heuristic:
            logger.info("Aplicando heurística de features recomendados...")
            X, heuristic_features_kept = self._apply_heuristic_selection(X)
        else:
            heuristic_features_kept = []
        
        # 4. Calcular importancia de features (si hay target)
        importances = None
        if target_col and target_col in df.columns and self.use_importance:
            logger.info("Calculando importancia de features...")
            importances = self._calculate_feature_importance(X, df[target_col])
        
        # 5. Asegurar que features obligatorios estén incluidos
        X = self._ensure_mandatory_features(X, df)
        
        # Guardar información de selección
        self.selected_features = X.columns.tolist()
        self.feature_importances = importances or {}
        self.selection_info = {
            'constant_features_removed': constant_features_removed,
            'correlated_features_removed': correlated_features_removed,
            'heuristic_features_kept': heuristic_features_kept,
            'total_features_selected': len(self.selected_features),
            'selection_methods_used': ['variance', 'correlation'] + (['heuristic'] if self.use_heuristic else []) + (['importance'] if importances else [])
        }
        
        logger.info(f"Selección completada: {len(self.selected_features)} features seleccionados")
        return X, self.selected_features, importances
    
    def _remove_constant_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Eliminar features constantes o casi constantes
        
        Args:
            df: DataFrame original
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: (DataFrame filtrado, features eliminados)
        """
        # Identificar features numéricos
        feature_cols = [c for c in df.columns if c not in self.mandatory_features]
        X = df[feature_cols].select_dtypes(include=[np.number]).copy()
        
        if X.empty:
            return df, []
        
        # Aplicar VarianceThreshold
        selector = VarianceThreshold(threshold=self.variance_threshold)
        X_transformed = pd.DataFrame(
            selector.fit_transform(X), 
            columns=np.array(feature_cols)[selector.get_support()]
        )
        
        # Identificar features eliminados
        removed_features = [col for col in feature_cols if col not in X_transformed.columns]
        
        # Reconstruir DataFrame con features no numéricos
        result_df = df.copy()
        for col in removed_features:
            if col in result_df.columns:
                result_df = result_df.drop(columns=[col])
        
        logger.info(f"Eliminadas {len(removed_features)} features constantes")
        return result_df, removed_features
    
    def _remove_correlated_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Eliminar features altamente correlacionadas
        
        Args:
            df: DataFrame original
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: (DataFrame filtrado, features eliminados)
        """
        # Identificar features numéricos
        feature_cols = [c for c in df.columns if c not in self.mandatory_features]
        X = df[feature_cols].select_dtypes(include=[np.number]).copy()
        
        if X.empty or len(X.columns) < 2:
            return df, []
        
        # Calcular correlaciones
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        
        # Identificar features a eliminar
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        
        # Eliminar features
        result_df = df.drop(columns=to_drop)
        
        logger.info(f"Eliminadas {len(to_drop)} features altamente correlacionadas")
        return result_df, to_drop
    
    def _apply_heuristic_selection(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Aplicar selección heurística basada en features recomendados
        
        Args:
            df: DataFrame original
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: (DataFrame filtrado, features mantenidos)
        """
        # Identificar features recomendados que están presentes
        available_features = df.columns.tolist()
        features_to_keep = [f for f in self.recommended_features if f in available_features]
        
        # Mantener también features obligatorios
        features_to_keep.extend([f for f in self.mandatory_features if f in available_features])
        
        # Filtrar DataFrame
        result_df = df[features_to_keep]
        
        logger.info(f"Mantenidas {len(features_to_keep)} features por heurística")
        return result_df, features_to_keep
    
    def _calculate_feature_importance(self, df: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """
        Calcular importancia de features usando Random Forest
        
        Args:
            df: DataFrame con features
            target: Serie objetivo
            
        Returns:
            Dict[str, float]: Importancia de cada feature
        """
        # Identificar features numéricos
        feature_cols = [c for c in df.columns if c not in self.mandatory_features]
        X = df[feature_cols].select_dtypes(include=[np.number]).copy()
        
        if X.empty:
            return {}
        
        # Alinear índices
        common_idx = X.index.intersection(target.index)
        X_aligned = X.loc[common_idx]
        y_aligned = target.loc[common_idx]
        
        if len(common_idx) < 100:  # Mínimo de datos para entrenar
            logger.warning("Datos insuficientes para calcular importancia de features")
            return {}
        
        # Entrenar Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_aligned.fillna(0), y_aligned)
        
        # Obtener importancias
        importances = dict(zip(X.columns, rf.feature_importances_))
        
        # Ordenar por importancia
        importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
        
        logger.info(f"Importancia calculada para {len(importances)} features")
        return importances
    
    def _ensure_mandatory_features(self, df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Asegurar que features obligatorios estén incluidos
        
        Args:
            df: DataFrame actual
            original_df: DataFrame original
            
        Returns:
            pd.DataFrame: DataFrame con features obligatorios
        """
        result_df = df.copy()
        
        for feature in self.mandatory_features:
            if feature in original_df.columns and feature not in result_df.columns:
                result_df[feature] = original_df[feature]
                logger.info(f"Agregado feature obligatorio: {feature}")
        
        return result_df
    
    def get_selection_summary(self) -> Dict:
        """
        Obtener resumen de la selección de features
        
        Returns:
            Dict: Resumen de selección
        """
        return {
            'selected_features': self.selected_features,
            'total_selected': len(self.selected_features),
            'selection_info': self.selection_info,
            'feature_importances': self.feature_importances,
            'recommended_features': self.recommended_features,
            'mandatory_features': self.mandatory_features
        }
    
    def save_selection_info(self, output_dir: str):
        """
        Guardar información de selección
        
        Args:
            output_dir: Directorio de salida
        """
        import json
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar features seleccionados
        with open(os.path.join(output_dir, 'selected_features.json'), 'w') as f:
            json.dump(self.selected_features, f, indent=2)
        
        # Guardar importancias
        if self.feature_importances:
            with open(os.path.join(output_dir, 'feature_importances.json'), 'w') as f:
                json.dump(self.feature_importances, f, indent=2)
        
        # Guardar resumen
        with open(os.path.join(output_dir, 'feature_selection_summary.json'), 'w') as f:
            json.dump(self.get_selection_summary(), f, indent=2)
        
        logger.info(f"Información de selección guardada en {output_dir}")