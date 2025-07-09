#!/usr/bin/env python3
"""
Configuración y constantes globales para el pipeline de trading
"""
from collections import OrderedDict
import MetaTrader5 as mt5
import pytz
import os

# Configuración de variables de entorno para optimización
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
os.environ['VECLIB_MAXIMUM_THREADS'] = '16'
os.environ['NUMBA_NUM_THREADS'] = '16'

# ===============================================================================
# CONFIGURACIÓN GLOBAL
# ===============================================================================

SYMBOL_ALIASES = {
    'US500': [
        'US500', 'US500.cash', 'US500_m', 'US.500', 'US500Cash',
        'SP500', 'SP500.cash', 'SP500_m', 'SP.500', 'SP500m',
        'SPX500', 'SPX500.cash', 'SPX', 'SPXUSD', 'SPY',
        'USA500', 'USA500.cash', 'USA.500'
    ],
    'USDCOP': [
        'USDCOP', 'USD/COP', 'USD_COP', 'USDCOP.r', 'USDCOP.forex',
        'COPUSD', 'COP/USD', 'COP_USD', 'COPUSD.r', 'COPUSD.forex',
        'USDCOP.', 'COPUSD.', 'USD.COP', 'COP.USD'
    ]
}

TIMEFRAMES = OrderedDict([
    ('M5',  {'enum': mt5.TIMEFRAME_M5,  'minutes': 5,  'priority': 1, 'quality': 1.0}),
    ('M1',  {'enum': mt5.TIMEFRAME_M1,  'minutes': 1,  'priority': 2, 'quality': 0.95}),
    ('M10', {'enum': mt5.TIMEFRAME_M10, 'minutes': 10, 'priority': 3, 'quality': 0.85}),
    ('M15', {'enum': mt5.TIMEFRAME_M15, 'minutes': 15, 'priority': 4, 'quality': 0.80}),
    ('M20', {'enum': mt5.TIMEFRAME_M20, 'minutes': 20, 'priority': 5, 'quality': 0.75}),
    ('M30', {'enum': mt5.TIMEFRAME_M30, 'minutes': 30, 'priority': 6, 'quality': 0.70}),
    ('H1',  {'enum': mt5.TIMEFRAME_H1,  'minutes': 60, 'priority': 7, 'quality': 0.65})
])

# TODOS los métodos de captura disponibles para máxima flexibilidad
CAPTURE_METHODS = [
    # Métodos de rates (OHLC)
    'rates_range',      # Captura por rango de fechas
    'rates_from',       # Captura desde una fecha específica
    'rates_from_pos',   # Captura desde una posición específica
    
    # Métodos de ticks (datos más granulares)
    'ticks_range',      # Ticks por rango de fechas
    'ticks_from',       # Ticks desde una fecha específica
    
    # Métodos adicionales para máxima cobertura
    'rates_range_pos',  # Rates por rango con posición
    'ticks_range_pos'   # Ticks por rango con posición
]

# Configuración de resampling y agregación
RESAMPLING_CONFIG = {
    'target_timeframe': 'M5',  # Timeframe objetivo final
    'target_minutes': 5,       # Minutos del timeframe objetivo
    
    # Timeframes que se pueden resamplear a M5
    'resampleable_timeframes': ['M1', 'M10', 'M15', 'M20', 'M30'],
    
    # Configuración de agregación OHLC
    'ohlc_aggregation': {
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last',
        'tick_volume': 'sum',
        'real_volume': 'sum',
        'spread': 'mean'
    },
    
    # Configuración específica para ticks
    'tick_processing': {
        'min_ticks_per_bar': 1,      # Mínimo de ticks para formar una barra
        'max_ticks_per_request': 1000000,  # Máximo de ticks por request
        'tick_aggregation_method': 'ohlc',  # Método de agregación de ticks
        'preserve_tick_data': False   # Si preservar datos de ticks originales
    }
}

SEGMENT_SIZES = [1, 7, 30]

# Configuraciones de mercado por tipo
MARKET_CONFIGS = {
    'US500': {
        'open_hour': 9,
        'open_minute': 30,
        'close_hour': 16,
        'close_minute': 0,
        'timezone': pytz.timezone('US/Eastern'),
        'trading_hours_per_day': 6.5,
        'bars_per_day': 78,  # 6.5 hours * 12 bars/hour (M5)
        'market_type': 'stock_market',
        'trading_days': 'weekdays_only',
        'holidays': 'us_market_holidays',
        'capture_preferences': {
            'primary_methods': ['rates_from_pos', 'rates_range'],
            'fallback_methods': ['rates_from', 'ticks_range'],
            'tick_methods': ['ticks_range', 'ticks_from'],
            'max_timeframe_for_ticks': 'M30'  # Usar ticks hasta M30
        }
    },
    'USDCOP': {
        'open_hour': 0,      # Forex abre domingo 5PM ET (convertido a UTC)
        'open_minute': 0,
        'close_hour': 23,    # Forex cierra viernes 5PM ET
        'close_minute': 59,
        'timezone': pytz.UTC,  # Forex opera en UTC
        'trading_hours_per_day': 24,  # 24 horas al día
        'bars_per_day': 288,  # 24 hours * 12 bars/hour (M5)
        'market_type': 'forex',
        'trading_days': 'forex_schedule',
        'holidays': 'none',  # Forex no tiene feriados tradicionales
        'capture_preferences': {
            'primary_methods': ['rates_from_pos', 'ticks_range'],
            'fallback_methods': ['rates_range', 'ticks_from'],
            'tick_methods': ['ticks_range', 'ticks_from'],
            'max_timeframe_for_ticks': 'M30',  # Usar ticks hasta M30
            'prefer_ticks_for_forex': True     # Preferir ticks para Forex
        }
    }
}

# Configuración por defecto (para compatibilidad)
MARKET_PARAMS = MARKET_CONFIGS['US500']

MARKET_EVENTS = {
    'US500': {
        '2020-03': {
            'event': 'COVID-19 Pandemic - Market Crash',
            'impact': 'Alta volatilidad, cierres temporales',
            'data_quality': 'Variable'
        },
        '2020-03-09': {
            'event': 'Black Monday I - Circuit Breaker',
            'impact': 'Caída del 7.6%, trading pausado 15 min',
            'data_quality': 'Gaps intradiarios'
        },
        '2020-03-12': {
            'event': 'Black Thursday', 
            'impact': 'Mayor caída desde 1987 (-9.5%)',
            'data_quality': 'Alta volatilidad'
        },
        '2020-03-16': {
            'event': 'Black Monday II - Circuit Breaker',
            'impact': 'Caída del 12%, trading pausado',
            'data_quality': 'Gaps significativos'
        },
        '2020-03-18': {
            'event': 'Black Wednesday - Circuit Breaker',
            'impact': 'Volatilidad extrema, múltiples pausas',
            'data_quality': 'Datos fragmentados'
        },
        '2021-01-27': {
            'event': 'GameStop Short Squeeze',
            'impact': 'Alta volatilidad en acciones meme',
            'data_quality': 'Normal para índices'
        },
        '2022-03': {
            'event': 'Russia-Ukraine Conflict',
            'impact': 'Volatilidad geopolítica',
            'data_quality': 'Normal'
        },
        '2023-03': {
            'event': 'SVB Banking Crisis',
            'impact': 'Colapso de Silicon Valley Bank',
            'data_quality': 'Volatilidad en sector financiero'
        },
        '2023-05': {
            'event': 'US Debt Ceiling Crisis',
            'impact': 'Incertidumbre política',
            'data_quality': 'Normal'
        }
    },
    'USDCOP': {
        '2020-03': {
            'event': 'COVID-19 Pandemic - Peso Colombiano Devaluación',
            'impact': 'Alta volatilidad, COP alcanza máximos históricos',
            'data_quality': 'Variable'
        },
        '2020-03-20': {
            'event': 'USD/COP alcanza 4,153 - Máximo histórico',
            'impact': 'Devaluación extrema del peso',
            'data_quality': 'Alta volatilidad'
        },
        '2021-05': {
            'event': 'Protestas en Colombia',
            'impact': 'Presión sobre el peso colombiano',
            'data_quality': 'Volatilidad incrementada'
        },
        '2022-06': {
            'event': 'Elecciones presidenciales Colombia',
            'impact': 'Incertidumbre política',
            'data_quality': 'Normal con picos'
        },
        '2022-10': {
            'event': 'USD/COP supera 5,000',
            'impact': 'Nueva barrera psicológica',
            'data_quality': 'Normal'
        },
        '2023-01': {
            'event': 'Intervención Banco de la República',
            'impact': 'Intentos de estabilización',
            'data_quality': 'Normal'
        }
    }
}

OUTPUT_DIR = 'data'
LOG_DIR = 'logs'

# ===============================================================================
# CONSTANTES DE TRAZABILIDAD Y AUDITORÍA DE DATOS
# ===============================================================================

# Catálogo de valores permitidos para data_origin (trazabilidad estándar)
DATA_ORIGINS = {
    # Datos nativos (capturados directamente)
    'M5_NATIVO': {
        'description': 'Datos capturados directamente en M5 (máxima calidad)',
        'quality_score': 1.0,
        'category': 'native'
    },
    'M1_NATIVO': {
        'description': 'Datos capturados directamente en M1',
        'quality_score': 0.95,
        'category': 'native'
    },
    'M10_NATIVO': {
        'description': 'Datos capturados directamente en M10',
        'quality_score': 0.85,
        'category': 'native'
    },
    'M15_NATIVO': {
        'description': 'Datos capturados directamente en M15',
        'quality_score': 0.80,
        'category': 'native'
    },
    'M20_NATIVO': {
        'description': 'Datos capturados directamente en M20',
        'quality_score': 0.75,
        'category': 'native'
    },
    'M30_NATIVO': {
        'description': 'Datos capturados directamente en M30',
        'quality_score': 0.70,
        'category': 'native'
    },
    'H1_NATIVO': {
        'description': 'Datos capturados directamente en H1',
        'quality_score': 0.65,
        'category': 'native'
    },
    'TICKS_NATIVO': {
        'description': 'Datos de ticks capturados directamente',
        'quality_score': 0.98,
        'category': 'native'
    },
    
    # Datos agregados (resampleados)
    'M5_AGREGADO_M1': {
        'description': 'Datos M1 agregados a M5',
        'quality_score': 0.95,
        'category': 'aggregated'
    },
    'M5_AGREGADO_M10': {
        'description': 'Datos M10 agregados a M5',
        'quality_score': 0.85,
        'category': 'aggregated'
    },
    'M5_AGREGADO_M15': {
        'description': 'Datos M15 agregados a M5',
        'quality_score': 0.80,
        'category': 'aggregated'
    },
    'M5_AGREGADO_M20': {
        'description': 'Datos M20 agregados a M5',
        'quality_score': 0.75,
        'category': 'aggregated'
    },
    'M5_AGREGADO_M30': {
        'description': 'Datos M30 agregados a M5',
        'quality_score': 0.70,
        'category': 'aggregated'
    },
    'M5_AGREGADO_H1': {
        'description': 'Datos H1 agregados a M5',
        'quality_score': 0.65,
        'category': 'aggregated'
    },
    'M5_AGREGADO_TICKS': {
        'description': 'Datos de ticks agregados a M5',
        'quality_score': 0.98,
        'category': 'aggregated'
    },
    
    # Datos imputados (sintéticos)
    'M5_IMPUTADO_BROWNIAN': {
        'description': 'Datos imputados usando Brownian Bridge',
        'quality_score': 0.7,
        'category': 'imputed'
    },
    'M5_IMPUTADO_INTERPOLADO': {
        'description': 'Datos imputados usando interpolación lineal',
        'quality_score': 0.6,
        'category': 'imputed'
    },
    'M5_IMPUTADO_SIMPLE': {
        'description': 'Datos imputados usando método simple (forward fill)',
        'quality_score': 0.5,
        'category': 'imputed'
    },
    'M5_IMPUTADO_GAUSSIANO': {
        'description': 'Datos imputados usando distribución gaussiana',
        'quality_score': 0.65,
        'category': 'imputed'
    },
    
    # Datos fuera de horario de mercado
    'M5_FUERA_DE_HORARIO': {
        'description': 'Datos capturados fuera del horario de mercado',
        'quality_score': 0.3,
        'category': 'outside_market'
    },
    
    # Datos de respaldo/fallback
    'M5_FALLBACK': {
        'description': 'Datos capturados usando método de respaldo',
        'quality_score': 0.8,
        'category': 'fallback'
    },
    
    # Datos desconocidos (para compatibilidad)
    'DESCONOCIDO': {
        'description': 'Origen de datos desconocido',
        'quality_score': 0.0,
        'category': 'unknown'
    }
}

# Lista de valores permitidos para validación
DATA_ORIGIN_VALUES = list(DATA_ORIGINS.keys())

# Mapeo de métodos de captura a data_origin
CAPTURE_METHOD_TO_DATA_ORIGIN = {
    # Métodos de rates nativos
    'rates_range': {
        'M5': 'M5_NATIVO',
        'M1': 'M1_NATIVO',
        'M10': 'M10_NATIVO',
        'M15': 'M15_NATIVO',
        'M20': 'M20_NATIVO',
        'M30': 'M30_NATIVO',
        'H1': 'H1_NATIVO'
    },
    'rates_from': {
        'M5': 'M5_NATIVO',
        'M1': 'M1_NATIVO',
        'M10': 'M10_NATIVO',
        'M15': 'M15_NATIVO',
        'M20': 'M20_NATIVO',
        'M30': 'M30_NATIVO',
        'H1': 'H1_NATIVO'
    },
    'rates_from_pos': {
        'M5': 'M5_NATIVO',
        'M1': 'M1_NATIVO',
        'M10': 'M10_NATIVO',
        'M15': 'M15_NATIVO',
        'M20': 'M20_NATIVO',
        'M30': 'M30_NATIVO',
        'H1': 'H1_NATIVO'
    },
    
    # Métodos de ticks
    'ticks_range': {
        'ticks': 'TICKS_NATIVO'
    },
    'ticks_from': {
        'ticks': 'TICKS_NATIVO'
    },
    
    # Métodos de agregación
    'aggregation': {
        'M1': 'M5_AGREGADO_M1',
        'M10': 'M5_AGREGADO_M10',
        'M15': 'M5_AGREGADO_M15',
        'M20': 'M5_AGREGADO_M20',
        'M30': 'M5_AGREGADO_M30',
        'H1': 'M5_AGREGADO_H1',
        'ticks': 'M5_AGREGADO_TICKS'
    },
    
    # Métodos de imputación
    'brownian_bridge': {
        'imputed': 'M5_IMPUTADO_BROWNIAN'
    },
    'interpolation': {
        'imputed': 'M5_IMPUTADO_INTERPOLADO'
    },
    'simple_fill': {
        'imputed': 'M5_IMPUTADO_SIMPLE'
    },
    'gaussian': {
        'imputed': 'M5_IMPUTADO_GAUSSIANO'
    },
    
    # Métodos de resampling
    'resampling': {
        'M1': 'M5_AGREGADO_M1',
        'M10': 'M5_AGREGADO_M10',
        'M15': 'M5_AGREGADO_M15',
        'M20': 'M5_AGREGADO_M20',
        'M30': 'M5_AGREGADO_M30',
        'H1': 'M5_AGREGADO_H1'
    }
}

# Configuración de formato de tiempo estándar
TIME_FORMAT_CONFIG = {
    'display_format': '%Y-%m-%d %H:%M:%S',
    'display_format_with_tz': '%Y-%m-%d %H:%M:%S%z',
    'iso_format': '%Y-%m-%dT%H:%M:%S',
    'iso_format_with_tz': '%Y-%m-%dT%H:%M:%S%z',
    'default_timezone': 'UTC',
    'market_timezone': 'US/Eastern'  # Para US500
}

# Configuración de validación de datos
DATA_VALIDATION_CONFIG = {
    'required_columns': ['time', 'open', 'high', 'low', 'close', 'data_origin'],
    'optional_columns': ['tick_volume', 'spread', 'real_volume', 'quality_score'],
    'min_quality_score': 0.0,
    'max_quality_score': 1.0,
    'min_price': 0.0,
    'max_price_threshold': 1000000.0,  # $1M por acción/contrato
    'min_volume': 0,
    'max_gap_minutes': 30,  # Gaps mayores a 30 min son sospechosos
    'max_imputed_percentage': 0.05  # Máximo 5% de datos imputados
}

# Configuración de reportes de calidad
QUALITY_REPORT_CONFIG = {
    'include_data_origin_breakdown': True,
    'include_capture_methods': True,
    'include_quality_scores': True,
    'include_gap_analysis': True,
    'include_imputation_analysis': True,
    'include_temporal_analysis': True,
    'max_worst_records': 100,
    'completeness_threshold': 0.95  # 95% de completitud mínimo
}

# Configuración de hardware
CPU_CORES = os.cpu_count() or 16
RAY_WORKERS = min(CPU_CORES, 16)
DASK_WORKERS = min(CPU_CORES // 2, 8)
CHUNK_SIZE = 10000  # Tamaño óptimo para procesamiento por chunks

# Configuración de directorios temporales según plataforma
import sys
if sys.platform == "win32":
    RAY_TEMP_DIR = "C:\\tmp\\ray" if os.path.exists("C:\\") else os.path.join(os.environ.get('TEMP', ''), 'ray')
    RAY_SPILL_DIR = "C:\\tmp\\ray_spill" if os.path.exists("C:\\") else os.path.join(os.environ.get('TEMP', ''), 'ray_spill')
else:
    RAY_TEMP_DIR = "/tmp/ray"
    RAY_SPILL_DIR = "/tmp/ray_spill"
