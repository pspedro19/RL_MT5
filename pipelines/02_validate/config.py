"""
Configuración del pipeline de validación
"""
import os
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """
    Obtener configuración del pipeline de validación
    
    Returns:
        Dict: Configuración completa
    """
    return {
        # Configuración de mercado
        'market': {
            'm5_tolerance': 0.1,  # ±0.1 minutos para consistencia M5
            'min_consistency': 0.95,  # 95% de consistencia mínima
            'min_real_percentage': 80,  # Mínimo 80% de datos reales
            'timezone': 'US/Eastern'
        },
        
        # Configuración de integridad
        'integrity': {
            'ohlc_tolerance': 0.001,  # Tolerancia para validación OHLC
            'volume_min': 0,  # Volumen mínimo
            'spread_min': 0,  # Spread mínimo
            'price_min': 0.01  # Precio mínimo
        },
        
        # Configuración temporal
        'temporal': {
            'min_continuity': 0.95,  # Continuidad temporal mínima
            'max_gap_minutes': 10,  # Gap máximo en minutos
            'expected_interval': 5  # Intervalo esperado en minutos
        },
        
        # Configuración de checklist de calidad
        'quality_checklist': {
            'min_coverage_percentage': 0.7,  # Cobertura mínima
            'max_outlier_percentage': 0.01,  # Porcentaje máximo de outliers
            'expected_daily_bars': 78  # Barras esperadas por día (6.5 horas * 12)
        },
        
        # Configuración de features
        'features': {
            'variance_threshold': 1e-5,  # Umbral de varianza
            'correlation_threshold': 0.95,  # Umbral de correlación
            'use_heuristic': True,  # Usar selección heurística
            'use_importance': True,  # Usar importancia de features
            'selection_method': 'hybrid'  # Método de selección
        },
        
        # Configuración de normalización
        'normalization': {
            'method': 'feature_specific',  # 'feature_specific', 'robust', 'standard', 'minmax'
            'price_rolling_window': 20,  # Ventana para normalización de precios
            'volume_rolling_window': 20,  # Ventana para normalización de volumen
            'outlier_threshold': 10  # Umbral de outliers (desviaciones estándar)
        },
        
        # Configuración de limpieza
        'cleaning': {
            'remove_duplicates': True,
            'fill_nan_method': 'forward',  # 'forward', 'backward', 'interpolate'
            'remove_outliers': True,
            'outlier_threshold': 3.0  # Desviaciones estándar
        },
        
        # Configuración de splits
        'splitting': {
            'train_ratio': 0.8,  # Proporción de entrenamiento
            'val_ratio': 0.2,  # Proporción de validación
            'method': 'temporal',  # 'temporal', 'random'
            'create_walk_forward': False,  # Crear splits walk-forward
            'walk_forward_splits': 5  # Número de splits walk-forward
        },
        
        # Configuración de reportes
        'reporting': {
            'output_dir': 'reports/',
            'generate_json': True,
            'generate_markdown': True,
            'generate_excel': False,
            'include_charts': True
        },
        
        # Configuración de logging
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/pipeline_02.log',
            'console': True
        },
        
        # Configuración de rendimiento
        'performance': {
            'use_parallel': True,  # Usar procesamiento paralelo
            'n_jobs': -1,  # Número de jobs (-1 = todos los cores)
            'chunk_size': 10000,  # Tamaño de chunks para procesamiento
            'memory_limit': '4GB'  # Límite de memoria
        }
    }

def get_market_config() -> Dict[str, Any]:
    """
    Obtener configuración específica de mercado
    
    Returns:
        Dict: Configuración de mercado
    """
    return {
        'm5_tolerance': 0.1,  # ±0.1 minutos para consistencia M5
        'min_consistency': 0.95,  # 95% de consistencia mínima
        'min_real_percentage': 80,  # Mínimo 80% de datos reales
        'timezone': 'US/Eastern',
        'market_hours': {
            'open_hour': 9,
            'open_minute': 30,
            'close_hour': 16,
            'close_minute': 0
        }
    }

def get_feature_selection_config() -> Dict[str, Any]:
    """
    Obtener configuración específica de selección de features
    
    Returns:
        Dict: Configuración de selección de features
    """
    return {
        'variance_threshold': 1e-5,
        'correlation_threshold': 0.95,
        'use_heuristic': True,
        'use_importance': True,
        'recommended_features': [
            'open', 'high', 'low', 'close', 'log_return',
            'ema_21', 'ema_55', 'sma_200', 'atr_14', 'volatility_20',
            'rsi_14', 'macd_histogram', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'vwap', 'volume_ratio', 'doji', 'bullish_engulfing', 'bearish_engulfing',
            'minutes_since_open', 'minutes_to_close', 'session_progress'
        ],
        'mandatory_features': ['time', 'data_flag']
    }

def get_normalization_config() -> Dict[str, Any]:
    """
    Obtener configuración específica de normalización
    
    Returns:
        Dict: Configuración de normalización
    """
    return {
        'method': 'feature_specific',  # 'feature_specific', 'robust', 'standard', 'minmax'
        'price_rolling_window': 20,
        'volume_rolling_window': 20,
        'outlier_threshold': 10,
        'feature_categories': {
            'price': ['open', 'high', 'low', 'close'],
            'volume': ['tick_volume', 'volume_sma_20', 'volume_relative', 'volume_ratio'],
            'technical_bounded': ['rsi_14', 'rsi_28', 'stochastic_k', 'stochastic_d', 'macd_histogram'],
            'returns': ['log_return', 'return_5m', 'volume_change', 'volume_delta'],
            'temporal': ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos'],
            'session': ['session_progress', 'minutes_since_open', 'minutes_to_close'],
            'exclude': ['time', 'symbol', 'source_timeframe', 'quality_flag', 'data_flag']
        }
    }

def get_quality_checklist_config() -> Dict[str, Any]:
    """
    Obtener configuración específica del checklist de calidad
    
    Returns:
        Dict: Configuración del checklist
    """
    return {
        'min_coverage_percentage': 0.7,
        'max_outlier_percentage': 0.01,
        'expected_daily_bars': 78,  # 6.5 horas * 12 registros/hora
        'score_weights': {
            'temporal_coverage': 0.25,
            'market_integrity': 0.25,
            'numerical_coherence': 0.20,
            'data_quality': 0.15,
            'features_analysis': 0.10,
            'splits_analysis': 0.05
        }
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validar configuración
    
    Args:
        config: Configuración a validar
        
    Returns:
        bool: True si la configuración es válida
    """
    required_keys = [
        'market', 'integrity', 'temporal', 'quality_checklist',
        'features', 'normalization', 'cleaning', 'splitting', 'reporting'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Configuración faltante: {key}")
    
    # Validar valores específicos
    if config['splitting']['train_ratio'] + config['splitting']['val_ratio'] != 1.0:
        raise ValueError("Las proporciones de train y val deben sumar 1.0")
    
    if config['market']['min_consistency'] < 0 or config['market']['min_consistency'] > 1:
        raise ValueError("min_consistency debe estar entre 0 y 1")
    
    if config['features']['correlation_threshold'] < 0 or config['features']['correlation_threshold'] > 1:
        raise ValueError("correlation_threshold debe estar entre 0 y 1")
    
    return True

def create_output_directories(config: Dict[str, Any]) -> None:
    """
    Crear directorios de salida necesarios
    
    Args:
        config: Configuración del pipeline
    """
    directories = [
        config['reporting']['output_dir'],
        os.path.dirname(config['logging']['file']),
        'data/validated/',
        'logs/'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Configuración principal para el pipeline
VALIDATION_CONFIG = get_config()