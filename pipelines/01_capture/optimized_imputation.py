#!/usr/bin/env python3
"""
Imputación optimizada de gaps usando procesamiento por chunks
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
import gc

logger = logging.getLogger(__name__)

def detect_gaps_optimized(df: pd.DataFrame, max_gap_minutes: int = 30) -> pd.DataFrame:
    """
    Detectar gaps de forma optimizada
    """
    if df.empty:
        return pd.DataFrame()
    
    logger.info("Detectando gaps...")
    
    # Ordenar por tiempo
    df_sorted = df.sort_values('time').reset_index(drop=True)
    
    # Calcular diferencias de tiempo
    time_diffs = df_sorted['time'].diff().dt.total_seconds() / 60
    
    # Encontrar gaps mayores al threshold
    gap_mask = time_diffs > max_gap_minutes
    
    if not gap_mask.any():
        logger.info("No se encontraron gaps significativos")
        return pd.DataFrame()
    
    # Crear DataFrame de gaps
    gaps = []
    for i in range(len(df_sorted) - 1):
        if gap_mask.iloc[i + 1]:
            gap_info = {
                'start_time': df_sorted.iloc[i]['time'],
                'end_time': df_sorted.iloc[i + 1]['time'],
                'gap_minutes': time_diffs.iloc[i + 1],
                'start_index': i,
                'end_index': i + 1
            }
            gaps.append(gap_info)
    
    gaps_df = pd.DataFrame(gaps)
    logger.info(f"Encontrados {len(gaps_df)} gaps para imputación")
    
    return gaps_df

def impute_chunk(chunk_data: Tuple[pd.DataFrame, int, int]) -> pd.DataFrame:
    """
    Imputar un chunk específico usando Brownian Bridge
    """
    chunk, start_idx, end_idx = chunk_data
    
    if chunk.empty:
        return chunk
    
    try:
        # Implementar imputación Brownian Bridge simple
        # Para gaps pequeños, usar interpolación lineal
        if len(chunk) <= 10:
            # Interpolación lineal para gaps pequeños
            chunk['close'] = chunk['close'].interpolate(method='linear')
            chunk['open'] = chunk['open'].interpolate(method='linear')
            chunk['high'] = chunk['high'].interpolate(method='linear')
            chunk['low'] = chunk['low'].interpolate(method='linear')
            chunk['tick_volume'] = chunk['tick_volume'].fillna(0)
        else:
            # Para gaps grandes, usar Brownian Bridge
            chunk = impute_brownian_bridge_chunk(chunk)
        
        return chunk
        
    except Exception as e:
        logger.error(f"Error imputando chunk {start_idx}-{end_idx}: {e}")
        return chunk

def impute_brownian_bridge_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Imputación Brownian Bridge para un chunk
    """
    if chunk.empty:
        return chunk
    
    # Obtener valores de inicio y fin
    start_price = chunk['close'].iloc[0]
    end_price = chunk['close'].iloc[-1]
    
    # Generar caminata aleatoria entre start y end
    n_points = len(chunk)
    if n_points <= 2:
        return chunk
    
    # Crear caminata Browniana
    steps = np.random.normal(0, 1, n_points - 2)
    cumulative = np.cumsum(steps)
    
    # Escalar para conectar start_price con end_price
    scaled_steps = cumulative * (end_price - start_price) / cumulative[-1] if cumulative[-1] != 0 else 0
    
    # Generar precios
    prices = [start_price] + [start_price + step for step in scaled_steps] + [end_price]
    
    # Aplicar a las columnas OHLC
    chunk['close'] = prices
    chunk['open'] = prices
    chunk['high'] = [max(p, p * 1.001) for p in prices]  # High ligeramente mayor
    chunk['low'] = [min(p, p * 0.999) for p in prices]   # Low ligeramente menor
    
    # Volumen aleatorio
    chunk['tick_volume'] = np.random.randint(100, 1000, len(chunk))
    
    return chunk

def impute_brownian_bridge_optimized(df: pd.DataFrame, max_gap_minutes: int = 30, 
                                   chunk_size: int = 10000, max_workers: int = None) -> pd.DataFrame:
    """
    Imputación optimizada usando procesamiento por chunks
    """
    if df.empty:
        return df
    
    logger.info("Iniciando imputación optimizada...")
    start_time = pd.Timestamp.now()
    
    # Detectar gaps
    gaps = detect_gaps_optimized(df, max_gap_minutes)
    
    if gaps.empty:
        logger.info("No hay gaps para imputar")
        return df
    
    # Filtrar solo gaps que necesitan imputación
    gaps_to_impute = gaps[gaps['gap_minutes'] <= max_gap_minutes]
    
    if gaps_to_impute.empty:
        logger.info("No hay gaps dentro del rango para imputar")
        return df
    
    logger.info(f"Imputando {len(gaps_to_impute)} gaps...")
    
    # Ordenar DataFrame por tiempo
    df_sorted = df.sort_values('time').reset_index(drop=True)
    
    # Crear chunks para procesamiento
    chunks = []
    for _, gap in gaps_to_impute.iterrows():
        start_idx = gap['start_index']
        end_idx = gap['end_index']
        
        # Extraer chunk con margen
        chunk_start = max(0, start_idx - 5)
        chunk_end = min(len(df_sorted), end_idx + 5)
        
        chunk = df_sorted.iloc[chunk_start:chunk_end].copy()
        chunks.append((chunk, chunk_start, chunk_end))
    
    # Procesar chunks en paralelo
    if max_workers is None:
        max_workers = min(os.cpu_count(), len(chunks))
    
    logger.info(f"Procesando {len(chunks)} chunks con {max_workers} workers...")
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(impute_chunk, chunks))
        
        # Reconstruir DataFrame
        df_imputed = pd.concat(results, ignore_index=True)
        df_imputed = df_imputed.sort_values('time').reset_index(drop=True)
        
        # Eliminar duplicados
        df_imputed = df_imputed.drop_duplicates(subset=['time'], keep='first')
        
        # Liberar memoria
        del chunks, results
        gc.collect()
        
        end_time = pd.Timestamp.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Imputación completada en {duration:.2f} segundos")
        logger.info(f"Registros originales: {len(df)}, Registros finales: {len(df_imputed)}")
        
        return df_imputed
        
    except Exception as e:
        logger.error(f"Error en imputación paralela: {e}")
        logger.info("Fallback a imputación secuencial...")
        
        # Fallback secuencial
        df_imputed = df_sorted.copy()
        for _, gap in gaps_to_impute.iterrows():
            start_idx = gap['start_index']
            end_idx = gap['end_index']
            
            # Imputación simple para el gap
            gap_data = df_imputed.iloc[start_idx:end_idx + 1]
            gap_imputed = impute_chunk((gap_data, start_idx, end_idx))
            df_imputed.iloc[start_idx:end_idx + 1] = gap_imputed
        
        return df_imputed

def validate_imputation_quality(df_original: pd.DataFrame, df_imputed: pd.DataFrame) -> dict:
    """
    Validar calidad de la imputación
    """
    logger.info("Validando calidad de imputación...")
    
    # Estadísticas básicas
    original_stats = {
        'count': len(df_original),
        'price_mean': df_original['close'].mean(),
        'price_std': df_original['close'].std(),
        'volume_mean': df_original['tick_volume'].mean()
    }
    
    imputed_stats = {
        'count': len(df_imputed),
        'price_mean': df_imputed['close'].mean(),
        'price_std': df_imputed['close'].std(),
        'volume_mean': df_imputed['tick_volume'].mean()
    }
    
    # Calcular diferencias
    quality_metrics = {
        'records_added': imputed_stats['count'] - original_stats['count'],
        'price_mean_diff': abs(imputed_stats['price_mean'] - original_stats['price_mean']) / original_stats['price_mean'] * 100,
        'price_std_diff': abs(imputed_stats['price_std'] - original_stats['price_std']) / original_stats['price_std'] * 100,
        'volume_mean_diff': abs(imputed_stats['volume_mean'] - original_stats['volume_mean']) / original_stats['volume_mean'] * 100
    }
    
    logger.info(f"Métricas de calidad: {quality_metrics}")
    
    return quality_metrics

if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Crear datos de prueba
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    test_data = pd.DataFrame({
        'time': dates,
        'open': np.random.randn(1000) + 100,
        'high': np.random.randn(1000) + 101,
        'low': np.random.randn(1000) + 99,
        'close': np.random.randn(1000) + 100,
        'tick_volume': np.random.randint(100, 1000, 1000)
    })
    
    # Crear gaps artificiales
    test_data = test_data.drop([100, 101, 102, 200, 201, 202, 203, 204])
    
    # Imputar gaps
    result = impute_brownian_bridge_optimized(test_data, max_gap_minutes=30)
    
    # Validar calidad
    quality = validate_imputation_quality(test_data, result)
    
    print(f"Imputación completada. Calidad: {quality}") 