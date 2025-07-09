#!/usr/bin/env python3
"""
Indicadores técnicos y funciones optimizadas con Numba/GPU
"""
import warnings
import logging
import numpy as np
import pandas as pd
from typing import Tuple

logger = logging.getLogger('technical_indicators')

# Detectar bibliotecas disponibles
try:
    from numba import jit, prange
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba no disponible. Instalarlo acelerará cálculos matemáticos: pip install numba")

try:
    import bottleneck as bn
    BOTTLENECK_AVAILABLE = True
except ImportError:
    BOTTLENECK_AVAILABLE = False
    bn = np  # Fallback a numpy

# Detección de GPU
GPU_AVAILABLE = False
GPU_INFO = {}
cudf = None
cp = np  # Fallback a numpy por defecto

# Intentar detectar GPU de múltiples formas
try:
    # Método 1: Verificar CUDA a través de numba
    import numba
    from numba import cuda
    
    if cuda.is_available():
        # Obtener información de la GPU
        gpu = cuda.get_current_device()
        GPU_INFO['name'] = gpu.name.decode('utf-8') if hasattr(gpu.name, 'decode') else str(gpu.name)
        GPU_INFO['compute_capability'] = gpu.compute_capability
        GPU_INFO['memory'] = gpu.total_memory // (1024**2)  # MB
        logger.info(f"GPU detectada via Numba: {GPU_INFO['name']} ({GPU_INFO['memory']}MB)")
        
        # Intentar importar cuDF y CuPy
        try:
            import cudf
            import cupy as cp
            GPU_AVAILABLE = True
            logger.info("cuDF y CuPy importados correctamente")
        except ImportError as e:
            logger.warning(f"GPU detectada pero cuDF/CuPy no disponibles: {e}")
            logger.warning("Instalar con: pip install cudf-cu11 cupy-cuda11x")
    else:
        logger.info("CUDA no disponible a través de Numba")
        
except ImportError:
    logger.warning("Numba no instalado - GPU no disponible")
except Exception as e:
    logger.warning(f"Error detectando GPU: {e}")

# ===============================================================================
# FUNCIONES NUMBA JIT COMPILADAS
# ===============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def calculate_returns_numba(close_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcular retornos y log-retornos con Numba"""
        n = len(close_prices)
        returns = np.empty(n, dtype=np.float32)
        log_returns = np.empty(n, dtype=np.float32)
        
        returns[0] = 0.0
        log_returns[0] = 0.0
        
        for i in prange(1, n):
            if close_prices[i-1] != 0:
                returns[i] = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                if close_prices[i] > 0:
                    log_returns[i] = np.log(close_prices[i] / close_prices[i-1])
                else:
                    log_returns[i] = 0.0
            else:
                returns[i] = 0.0
                log_returns[i] = 0.0
                
        return returns, log_returns
    
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def calculate_rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """RSI optimizado con Numba"""
        n = len(prices)
        rsi = np.full(n, np.nan, dtype=np.float32)
        
        if n < period + 1:
            return rsi
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))
        else:
            rsi[period] = 100
            
        for i in range(period + 1, n):
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100
                
        return rsi
    
    @jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def brownian_bridge_numba(start_val: float, end_val: float, steps: int, 
                             variance_factor: float = 0.001) -> np.ndarray:
        """Brownian Bridge para imputacion con Numba"""
        values = np.empty(steps, dtype=np.float32)
        
        if steps <= 0:
            return values
            
        for i in prange(steps):
            alpha = (i + 1) / (steps + 1)
            base = start_val + alpha * (end_val - start_val)
            variance = alpha * (1 - alpha) * abs(end_val - start_val) * variance_factor
            noise = np.random.normal(0, np.sqrt(variance))
            values[i] = base + noise
            
        return values

else:
    # Fallback a numpy si Numba no está disponible
    def calculate_returns_numba(close_prices):
        returns = np.zeros_like(close_prices)
        log_returns = np.zeros_like(close_prices)
        returns[1:] = np.diff(close_prices) / close_prices[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            log_returns[1:] = np.log(close_prices[1:] / close_prices[:-1])
        log_returns[np.isnan(log_returns)] = 0
        log_returns[np.isinf(log_returns)] = 0
        return returns, log_returns
    
    def calculate_rsi_numba(prices, period):
        delta = np.diff(prices)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gains).rolling(window=period).mean()
        avg_loss = pd.Series(losses).rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    def brownian_bridge_numba(start_val, end_val, steps, variance_factor=0.001):
        alphas = np.linspace(1/(steps+1), steps/(steps+1), steps)
        base = start_val + alphas * (end_val - start_val)
        variance = alphas * (1 - alphas) * abs(end_val - start_val) * variance_factor
        noise = np.random.normal(0, np.sqrt(variance))
        return base + noise

# ===============================================================================
# FUNCIONES GPU MEJORADAS
# ===============================================================================

def generate_features_gpu(df: pd.DataFrame) -> pd.DataFrame:
    """Generar features usando GPU si está disponible"""
    if not GPU_AVAILABLE or len(df) < 10000:
        logger.info("GPU no disponible o dataset pequeño, usando CPU")
        return generate_features_cpu_optimized(df)
    
    try:
        logger.info(f"Generando features con GPU ({GPU_INFO.get('name', 'Unknown')})")
        start_time = time.time()
        
        # Convertir a cuDF
        gdf = cudf.from_pandas(df)
        
        # Cálculos en GPU
        gdf['return'] = gdf['close'].pct_change()
        gdf['log_return'] = cp.log(gdf['close'] / gdf['close'].shift(1))
        gdf['price_change'] = gdf['close'] - gdf['close'].shift(1)
        
        # Rangos
        gdf['high_low_range'] = gdf['high'] - gdf['low']
        gdf['high_low_pct'] = (gdf['high'] - gdf['low']) / gdf['close'] * 100
        gdf['body'] = gdf['close'] - gdf['open']
        gdf['body_pct'] = gdf['body'] / gdf['close'] * 100
        
        # EMAs en GPU (usando cuDF rolling)
        for period in [9, 21, 55, 100, 200]:
            gdf[f'ema_{period}'] = gdf['close'].ewm(span=period, adjust=False).mean()
        
        # RSI en GPU
        for period in [14, 21, 28]:
            delta = gdf['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            gdf[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands en GPU
        for period in [20, 30]:
            sma = gdf['close'].rolling(window=period).mean()
            std = gdf['close'].rolling(window=period).std()
            
            for num_std in [2, 3]:
                upper = sma + (std * num_std)
                lower = sma - (std * num_std)
                
                gdf[f'bb_upper_{period}_{num_std}'] = upper
                gdf[f'bb_lower_{period}_{num_std}'] = lower
                gdf[f'bb_width_{period}_{num_std}'] = upper - lower
                gdf[f'bb_position_{period}_{num_std}'] = (gdf['close'] - lower) / (upper - lower)
        
        # Convertir de vuelta a pandas
        df_result = gdf.to_pandas()
        
        gpu_time = time.time() - start_time
        logger.info(f"Features GPU generados exitosamente en {gpu_time:.2f} segundos")
        
        # Completar con features adicionales en CPU
        df_result = add_remaining_features(df_result)
        
        return df_result
        
    except Exception as e:
        logger.error(f"Error en procesamiento GPU: {e}")
        logger.info("Fallback a procesamiento CPU")
        return generate_features_cpu_optimized(df)

def add_remaining_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agregar features que no se calcularon en GPU"""
    # Features temporales
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['day_of_week'] = df['time'].dt.dayofweek
    df['day_of_month'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['quarter'] = df['time'].dt.quarter
    
    # Codificación cíclica
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    # Features de sesión
    df['minutes_since_open'] = (df['hour'] - 9) * 60 + df['minute'] - 30
    df['minutes_to_close'] = (16 - df['hour']) * 60 - df['minute']
    
    return df

def generate_features_cpu_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """Generar features optimizado para CPU con Numba y vectorización (sin fragmentación)"""
    if df.empty:
        return df
    
    logger.info("Generando features optimizados (CPU, sin fragmentación)")
    df = df.copy().sort_values('time').reset_index(drop=True)
    close_prices = df['close'].values.astype(np.float32)
    
    # Acumular nuevas columnas aquí
    features = {}
    
    # Cálculos con Numba si está disponible
    if NUMBA_AVAILABLE:
        returns, log_returns = calculate_returns_numba(close_prices)
        features['return'] = pd.Series(returns, index=df.index)
        features['log_return'] = pd.Series(log_returns, index=df.index)
    else:
        features['return'] = df['close'].pct_change()
        with np.errstate(divide='ignore', invalid='ignore'):
            features['log_return'] = np.log(df['close'] / df['close'].shift(1))
        features['log_return'] = features['log_return'].fillna(0)
    features['price_change'] = df['close'] - df['close'].shift(1)
    features['high_low_range'] = df['high'] - df['low']
    features['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
    features['body'] = df['close'] - df['open']
    features['body_pct'] = features['body'] / df['close'] * 100
    features['upper_shadow'] = df[['open', 'close']].max(axis=1) - df['high']
    features['lower_shadow'] = df['low'] - df[['open', 'close']].min(axis=1)
    # Medias móviles con Bottleneck si está disponible
    if BOTTLENECK_AVAILABLE:
        for period in [9, 21, 55, 100, 200]:
            features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            features[f'ema_{period}_slope'] = features[f'ema_{period}'].diff()
        for period in [20, 50, 100, 200]:
            sma_array = bn.move_mean(close_prices, window=period, min_count=1)
            features[f'sma_{period}'] = pd.Series(sma_array, index=df.index)
            features[f'sma_{period}_slope'] = features[f'sma_{period}'].diff()
    else:
        for period in [9, 21, 55, 100, 200]:
            features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            features[f'ema_{period}_slope'] = features[f'ema_{period}'].diff()
        for period in [20, 50, 100, 200]:
            features[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            features[f'sma_{period}_slope'] = features[f'sma_{period}'].diff()
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    features['macd'] = exp1 - exp2
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']
    if NUMBA_AVAILABLE:
        for period in [14, 21, 28]:
            rsi_array = calculate_rsi_numba(close_prices, period)
            features[f'rsi_{period}'] = pd.Series(rsi_array, index=df.index)
    else:
        for period in [14, 21, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    for period in [20, 30]:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        for num_std in [2, 3]:
            upper = sma + (std * num_std)
            lower = sma - (std * num_std)
            features[f'bb_upper_{period}_{num_std}'] = upper
            features[f'bb_lower_{period}_{num_std}'] = lower
            features[f'bb_width_{period}_{num_std}'] = upper - lower
            features[f'bb_position_{period}_{num_std}'] = (df['close'] - lower) / (upper - lower)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    for period in [14, 21]:
        features[f'atr_{period}'] = true_range.rolling(window=period).mean()
    for period in [14, 21]:
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        features[f'stoch_k_{period}'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(window=3).mean()
    for period in [10, 20, 30]:
        # Asegurar que log_return sea una Serie de pandas antes de usar .rolling()
        if isinstance(features['log_return'], np.ndarray):
            log_return_series = pd.Series(features['log_return'], index=df.index)
        else:
            log_return_series = features['log_return']
        features[f'volatility_{period}'] = log_return_series.rolling(window=period).std() * np.sqrt(252 * 78)
    # Calcular Parkinson volatility asegurando que sea una Serie de pandas
    high_low_ratio = np.log(df['high'] / df['low']) ** 2
    if isinstance(high_low_ratio, np.ndarray):
        high_low_series = pd.Series(high_low_ratio, index=df.index)
    else:
        high_low_series = high_low_ratio
    features['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * high_low_series.rolling(window=20).mean()
    ) * np.sqrt(252 * 78)
    features['volume_sma_20'] = df['tick_volume'].rolling(window=20).mean()
    features['volume_ratio'] = df['tick_volume'] / features['volume_sma_20']
    features['volume_change'] = df['tick_volume'].pct_change()
    features['vwap'] = (df['close'] * df['tick_volume']).rolling(window=20).sum() / df['tick_volume'].rolling(window=20).sum()
    features['price_to_vwap'] = df['close'] / features['vwap']
    features['hour'] = df['time'].dt.hour
    features['minute'] = df['time'].dt.minute
    features['day_of_week'] = df['time'].dt.dayofweek
    features['day_of_month'] = df['time'].dt.day
    features['month'] = df['time'].dt.month
    features['quarter'] = df['time'].dt.quarter
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['minute_sin'] = np.sin(2 * np.pi * features['minute'] / 60)
    features['minute_cos'] = np.cos(2 * np.pi * features['minute'] / 60)
    features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    features['minutes_since_open'] = (features['hour'] - 9) * 60 + features['minute'] - 30
    features['minutes_to_close'] = (16 - features['hour']) * 60 - features['minute']
    features['session_progress'] = features['minutes_since_open'] / (6.5 * 60)
    features['is_first_hour'] = (features['minutes_since_open'] <= 60).astype(int)
    features['is_last_hour'] = (features['minutes_to_close'] <= 60).astype(int)
    features['is_lunch_time'] = ((features['hour'] >= 12) & (features['hour'] < 14)).astype(int)
    features['is_morning'] = (features['hour'] < 12).astype(int)
    features['is_afternoon'] = (features['hour'] >= 12).astype(int)
    features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    features['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
    features['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
    features['doji'] = (abs(features['body_pct']) < 0.1).astype(int)
    bullish_cond = ((features['body'] > 0) & 
                   (features['body'].shift(1) < 0) & 
                   (df['open'] < df['close'].shift(1)) & 
                   (df['close'] > df['open'].shift(1)))
    features['bullish_engulfing'] = bullish_cond.astype(int)
    bearish_cond = ((features['body'] < 0) & 
                   (features['body'].shift(1) > 0) & 
                   (df['open'] > df['close'].shift(1)) & 
                   (df['close'] < df['open'].shift(1)))
    features['bearish_engulfing'] = bearish_cond.astype(int)
    # Asegurar que 'return' esté en df antes de cálculos dependientes
    df['return'] = features['return']

    # Ahora sí, calcular features que dependen de 'return'
    features['cumulative_return_day'] = df.groupby(df['time'].dt.date)['return'].cumsum()
    features['cumulative_return_week'] = df.groupby([df['time'].dt.year, df['time'].dt.isocalendar().week])['return'].cumsum()

    for window in [20, 50]:
        if BOTTLENECK_AVAILABLE:
            mean_array = bn.move_mean(close_prices, window=window, min_count=1)
            std_array = bn.move_std(close_prices, window=window, min_count=1)
            min_array = bn.move_min(close_prices, window=window, min_count=1)
            max_array = bn.move_max(close_prices, window=window, min_count=1)
            features[f'rolling_mean_{window}'] = pd.Series(mean_array, index=df.index)
            features[f'rolling_std_{window}'] = pd.Series(std_array, index=df.index)
            features[f'rolling_min_{window}'] = pd.Series(min_array, index=df.index)
            features[f'rolling_max_{window}'] = pd.Series(max_array, index=df.index)
        else:
            features[f'rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
            features[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()
            features[f'rolling_min_{window}'] = df['close'].rolling(window=window).min()
            features[f'rolling_max_{window}'] = df['close'].rolling(window=window).max()
        features[f'rolling_range_{window}'] = features[f'rolling_max_{window}'] - features[f'rolling_min_{window}']

    # Concatenar todas las features de una vez para evitar fragmentación
    features_df = pd.DataFrame(features, index=df.index)
    df = pd.concat([df, features_df], axis=1)
    df = df.copy()  # Defragmentar memoria

    # --- FIX: Asegurar que 'return' esté presente ---
    if 'return' not in df.columns:
        logger.warning("La columna 'return' no fue generada. Se crea como ceros para evitar errores en el pipeline/test.")
        df['return'] = 0.0
    # --- END FIX ---

    feature_cols = [col for col in df.columns if col not in 
                   ['time', 'open', 'high', 'low', 'close', 'tick_volume', 
                    'spread', 'real_volume', 'data_flag', 'quality_score',
                    'capture_method', 'source_timeframe', 'worker_id']]
    logger.info(f"Features generados: {len(feature_cols)}")
    return df

def warmup_numba():
    """Pre-compilar funciones Numba para evitar latencia en primera ejecución"""
    if NUMBA_AVAILABLE:
        logger.info("Pre-compilando funciones Numba...")
        
        # Datos dummy para warmup
        dummy_prices = np.random.randn(100).astype(np.float32)
        
        # Ejecutar cada función una vez
        try:
            _ = calculate_returns_numba(dummy_prices)
            _ = calculate_rsi_numba(dummy_prices, 14)
            _ = brownian_bridge_numba(1.0, 2.0, 10)
            logger.info("Funciones Numba pre-compiladas correctamente")
        except Exception as e:
            logger.warning(f"Error pre-compilando algunas funciones Numba: {e}")
            logger.warning("El pipeline continuará pero puede ser más lento en la primera ejecución")
