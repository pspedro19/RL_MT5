#!/usr/bin/env python3
"""
Motor de captura HPC optimizado con soporte para todos los métodos de captura
"""
import os
import sys
import time
import json
import logging
import itertools
import gc
import warnings
import concurrent.futures as cf
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
import pytz
import numpy as np
import pandas as pd
from tqdm import tqdm
import MetaTrader5 as mt5

from config.constants import (
    TIMEFRAMES, CAPTURE_METHODS, SEGMENT_SIZES, 
    RAY_WORKERS, RAY_TEMP_DIR, RAY_SPILL_DIR, CPU_CORES,
    RESAMPLING_CONFIG, MARKET_CONFIGS, DATA_ORIGINS
)
from utils.data_traceability import DataTraceabilityManager
from utils.exceptions import MT5InitializationError, DataRetrievalError
from data.quality.quality_tracker import DataQualityTracker
from data.processing import (
    process_ticks_to_ohlc, resample_timeframe_to_m5,
    combine_and_prioritize_dataframes
)

# Detectar bibliotecas HPC disponibles
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    warnings.warn("Ray no disponible. Instalarlo mejorará significativamente el rendimiento: pip install ray")

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None
    warnings.warn("Polars no disponible. Instalarlo mejorará el rendimiento: pip install polars")

logger = logging.getLogger('hpc_capture')

# ===============================================================================
# WORKERS RAY PARA CAPTURA PARALELA
# ===============================================================================

if RAY_AVAILABLE:
    @ray.remote
    class RayCaptureWorker:
        """Worker Ray para captura paralela de datos MT5"""
        
        def __init__(self, worker_id: int, symbol: str):
            self.worker_id = worker_id
            self.symbol = symbol
            self.mt5_connected = False
            
        def connect_mt5(self) -> None:
            """Conectar MT5 en el worker"""
            try:
                if not mt5.initialize():
                    raise MT5InitializationError("Error inicializando MT5 en worker")
                self.mt5_connected = True
            except Exception as e:
                logger.error(f"Worker {self.worker_id} - Error conectando MT5: {e}")
                raise
                
        def capture_segment(self, method: str, timeframe: str, tf_config: dict,
                           start: datetime, end: datetime) -> Optional[pd.DataFrame]:
            """Capturar segmento de datos"""
            if not self.mt5_connected:
                self.connect_mt5()
                
            try:
                if method == 'rates_range':
                    rates = mt5.copy_rates_range(
                        self.symbol, tf_config['enum'], start, end
                    )
                elif method == 'rates_from':
                    count = int((end - start).total_seconds() / 60 / tf_config['minutes'])
                    rates = mt5.copy_rates_from(
                        self.symbol, tf_config['enum'], start, count
                    )
                elif method == 'rates_from_pos':
                    current = datetime.now(pytz.UTC)
                    bars_back = int((current - start).total_seconds() / 60 / tf_config['minutes'])
                    count = int((end - start).total_seconds() / 60 / tf_config['minutes'])
                    rates = mt5.copy_rates_from_pos(
                        self.symbol, tf_config['enum'], bars_back, count
                    )
                elif method == 'ticks_range':
                    ticks = mt5.copy_ticks_range(
                        self.symbol, start, end, mt5.COPY_TICKS_ALL
                    )
                    if ticks is not None and len(ticks) > 0:
                        rates = self._ticks_to_ohlc(ticks, tf_config['minutes'])
                    else:
                        rates = None
                elif method == 'ticks_from':
                    count = 100000
                    ticks = mt5.copy_ticks_from(
                        self.symbol, start, count, mt5.COPY_TICKS_ALL
                    )
                    if ticks is not None and len(ticks) > 0:
                        ticks_df = pd.DataFrame(ticks)
                        ticks_df['time'] = pd.to_datetime(ticks_df['time'], unit='s', utc=True)
                        ticks_df = ticks_df[ticks_df['time'] <= end]
                        if len(ticks_df) > 0:
                            rates = self._ticks_to_ohlc(
                                ticks_df.to_records(index=False), tf_config['minutes']
                            )
                        else:
                            rates = None
                    else:
                        rates = None
                else:
                    rates = None
                
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce').dt.tz_localize(None)
                    df['worker_id'] = self.worker_id
                    
                    # Asignar data_origin estandarizado
                    traceability_manager = DataTraceabilityManager()
                    df = traceability_manager.assign_data_origin(
                        df, capture_method=method, source_timeframe=timeframe
                    )
                    
                    return df
                    
            except Exception as e:
                logger.debug(f"Worker {self.worker_id} - Error en {method} {timeframe}: {e}")
                
            return None
        
        def _ticks_to_ohlc(self, ticks, timeframe_minutes: int) -> Optional[np.ndarray]:
            """Convertir ticks a OHLC"""
            try:
                df = pd.DataFrame(ticks)
                df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce').dt.tz_localize(None)
                df = df.set_index('time')
                
                ohlc = df['last'].resample(f'{timeframe_minutes}min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last'
                })
                
                if 'volume' in df.columns:
                    volume = df['volume'].resample(f'{timeframe_minutes}min').sum()
                    ohlc['tick_volume'] = volume
                else:
                    ohlc['tick_volume'] = df['last'].resample(f'{timeframe_minutes}min').count()
                
                if 'spread' in df.columns:
                    ohlc['spread'] = df['spread'].resample(f'{timeframe_minutes}min').mean()
                else:
                    ohlc['spread'] = 0
                
                ohlc['real_volume'] = 0
                ohlc = ohlc.dropna()
                
                if len(ohlc) > 0:
                    ohlc = ohlc.reset_index()
                    ohlc['time'] = ohlc['time'].astype(np.int64) // 10**9
                    return ohlc.to_records(index=False)
                    
            except Exception as e:
                logger.debug(f"Error convirtiendo ticks a OHLC: {e}")
                
            return None
        
        def __del__(self):
            """Cleanup al destruir el worker"""
            if hasattr(self, 'mt5_connected') and self.mt5_connected:
                mt5.shutdown()

# ===============================================================================
# CLASE CAPTURE ENGINE ESTÁNDAR
# ===============================================================================

class CaptureEngine:
    """Motor de captura estándar (sin Ray)"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.capture_stats = defaultdict(int)
        
    def capture_segment(self, method: str, timeframe: str, tf_config: dict,
                       start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Capturar un segmento usando método y timeframe específicos"""
        try:
            if method == 'rates_range':
                rates = mt5.copy_rates_range(
                    self.symbol, tf_config['enum'], start, end
                )
            elif method == 'rates_from':
                count = int((end - start).total_seconds() / 60 / tf_config['minutes'])
                rates = mt5.copy_rates_from(
                    self.symbol, tf_config['enum'], start, count
                )
            elif method == 'rates_from_pos':
                current = datetime.now(pytz.UTC)
                bars_back = int((current - start).total_seconds() / 60 / tf_config['minutes'])
                count = int((end - start).total_seconds() / 60 / tf_config['minutes'])
                rates = mt5.copy_rates_from_pos(
                    self.symbol, tf_config['enum'], bars_back, count
                )
            elif method == 'ticks_range':
                ticks = mt5.copy_ticks_range(
                    self.symbol, start, end, mt5.COPY_TICKS_ALL
                )
                if ticks is not None and len(ticks) > 0:
                    rates = self._ticks_to_ohlc(ticks, tf_config['minutes'])
                else:
                    rates = None
            elif method == 'ticks_from':
                count = 100000
                ticks = mt5.copy_ticks_from(
                    self.symbol, start, count, mt5.COPY_TICKS_ALL
                )
                if ticks is not None and len(ticks) > 0:
                    ticks_df = pd.DataFrame(ticks)
                    ticks_df['time'] = pd.to_datetime(ticks_df['time'], unit='s', utc=True)
                    ticks_df = ticks_df[ticks_df['time'] <= end]
                    if len(ticks_df) > 0:
                        rates = self._ticks_to_ohlc(
                            ticks_df.to_records(index=False), tf_config['minutes']
                        )
                    else:
                        rates = None
                else:
                    rates = None
            else:
                rates = None
            
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce').dt.tz_localize(None)
                
                # Asignar data_origin estandarizado
                traceability_manager = DataTraceabilityManager()
                df = traceability_manager.assign_data_origin(
                    df, capture_method=method, source_timeframe=timeframe
                )
                
                self.capture_stats[f"{method}_{timeframe}"] += len(df)
                return df
                
        except Exception as e:
            logger.debug(f"Error en {method} {timeframe}: {e}")
            
        return None
    
    def _ticks_to_ohlc(self, ticks, timeframe_minutes: int) -> Optional[np.ndarray]:
        """Convertir ticks a OHLC"""
        try:
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce').dt.tz_localize(None)
            df = df.set_index('time')
            
            ohlc = df['last'].resample(f'{timeframe_minutes}min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })
            
            if 'volume' in df.columns:
                volume = df['volume'].resample(f'{timeframe_minutes}min').sum()
                ohlc['tick_volume'] = volume
            else:
                ohlc['tick_volume'] = df['last'].resample(f'{timeframe_minutes}min').count()
            
            if 'spread' in df.columns:
                ohlc['spread'] = df['spread'].resample(f'{timeframe_minutes}min').mean()
            else:
                ohlc['spread'] = 0
            
            ohlc['real_volume'] = 0
            ohlc = ohlc.dropna()
            
            if len(ohlc) > 0:
                ohlc = ohlc.reset_index()
                ohlc['time'] = ohlc['time'].astype(np.int64) // 10**9
                return ohlc.to_records(index=False)
                
        except Exception as e:
            logger.debug(f"Error convirtiendo ticks a OHLC: {e}")
            
        return None

# ===============================================================================
# MOTOR DE CAPTURA HPC CON RAY Y QUALITY TRACKING
# ===============================================================================

class HpcCaptureEngine:
    """Motor de captura HPC con soporte completo para todos los métodos de captura"""
    
    def __init__(self, symbol: str, use_ray: bool = True, instrument: str = 'US500'):
        self.symbol = symbol
        self.instrument = instrument
        self.use_ray = use_ray and RAY_AVAILABLE
        self.quality_tracker = DataQualityTracker(instrument)
        
        # Obtener configuración de mercado
        self.market_config = MARKET_CONFIGS.get(instrument, MARKET_CONFIGS['US500'])
        self.capture_prefs = self.market_config.get('capture_preferences', {})
        
        # Inicializar Ray si está disponible
        if self.use_ray:
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                logger.info("Ray inicializado para captura paralela")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Ray: {e}")
                self.use_ray = False
        
        logger.info(f"Motor HPC inicializado para {symbol} ({instrument})")
        logger.info(f"Ray disponible: {self.use_ray}")
        logger.info(f"Polars disponible: {POLARS_AVAILABLE}")
    
    def capture_with_all_methods(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Capturar datos usando todos los métodos disponibles de forma combinada"""
        logger.info(f"Capturando datos con todos los métodos disponibles")
        logger.info(f"Período: {start_date} - {end_date}")
        
        all_dataframes = []
        source_timeframes = []
        
        # 1. Intentar captura con métodos de rates (OHLC)
        try:
            rates_data = self._capture_with_rates_methods(start_date, end_date)
            if not rates_data.empty:
                all_dataframes.append(rates_data)
                source_timeframes.append('rates_combined')
        except DataRetrievalError as e:
            logger.warning(f"Rates fallidos: {e}")

        # 2. Intentar captura con métodos de ticks
        try:
            ticks_data = self._capture_with_ticks_methods(start_date, end_date)
            if not ticks_data.empty:
                all_dataframes.append(ticks_data)
                source_timeframes.append('ticks_aggregated')
        except DataRetrievalError as e:
            logger.warning(f"Ticks fallidos: {e}")
        
        # 3. Combinar y priorizar todos los datos
        if all_dataframes:
            combined_df = combine_and_prioritize_dataframes(
                all_dataframes, source_timeframes, self.instrument
            )
            
            # 4. Resamplear todo a M5 si es necesario
            final_df = self._ensure_m5_timeframe(combined_df)
            
            logger.info(f"Captura completada: {len(final_df)} registros finales")
            return final_df
        else:
            logger.error("No se pudieron capturar datos con ningún método")
            raise DataRetrievalError("No se pudieron capturar datos con ningún método")
    
    def _capture_with_rates_methods(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Capturar datos usando métodos de rates (OHLC)"""
        logger.info("Capturando datos con métodos de rates...")
        
        rates_dataframes = []
        timeframes_used = []
        
        # Obtener métodos preferidos para rates
        primary_methods = self.capture_prefs.get('primary_methods', ['rates_from_pos'])
        fallback_methods = self.capture_prefs.get('fallback_methods', ['rates_range'])
        
        # Intentar con timeframes en orden de prioridad
        for timeframe, config in TIMEFRAMES.items():
            if timeframe == 'M5':  # Prioridad máxima
                continue  # Lo manejamos al final
            
            logger.info(f"Intentando captura {timeframe}...")
            
            # Intentar métodos primarios
            for method in primary_methods:
                if 'rates' in method:
                    df = self._capture_single_timeframe(
                        timeframe, method, start_date, end_date
                    )
                    if not df.empty:
                        rates_dataframes.append(df)
                        timeframes_used.append(timeframe)
                        logger.info(f"Captura exitosa: {timeframe} con {method}")
                        break
            
            # Si no se pudo con métodos primarios, intentar fallbacks
            if timeframe not in timeframes_used:
                for method in fallback_methods:
                    if 'rates' in method:
                        df = self._capture_single_timeframe(
                            timeframe, method, start_date, end_date
                        )
                        if not df.empty:
                            rates_dataframes.append(df)
                            timeframes_used.append(timeframe)
                            logger.info(f"Captura exitosa (fallback): {timeframe} con {method}")
                            break
        
        # Combinar todos los rates capturados
        if rates_dataframes:
            combined_rates = pd.concat(rates_dataframes, ignore_index=True)
            combined_rates = combined_rates.sort_values('time').reset_index(drop=True)

            # Resamplear a M5
            combined_rates = self._resample_all_to_m5(combined_rates, timeframes_used)

            logger.info(f"Rates combinados: {len(combined_rates)} registros")
            return combined_rates

        raise DataRetrievalError("No se pudieron capturar datos de rates")
    
    def _capture_with_ticks_methods(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Capturar datos usando métodos de ticks"""
        logger.info("Capturando datos con métodos de ticks...")
        
        # Obtener métodos de ticks
        tick_methods = self.capture_prefs.get('tick_methods', ['ticks_range'])
        max_timeframe = self.capture_prefs.get('max_timeframe_for_ticks', 'M30')
        
        # Verificar si debemos usar ticks para este instrumento
        prefer_ticks = self.capture_prefs.get('prefer_ticks_for_forex', False)
        if self.instrument == 'USDCOP' and prefer_ticks:
            logger.info("Usando ticks como método preferido para Forex")
        
        all_ticks_data = []
        
        for method in tick_methods:
            logger.info(f"Intentando captura de ticks con {method}...")
            
            try:
                ticks_df = self._capture_ticks(method, start_date, end_date)
                if not ticks_df.empty:
                    all_ticks_data.append(ticks_df)
                    logger.info(f"Ticks capturados con {method}: {len(ticks_df)} registros")
            except Exception as e:
                logger.warning(f"Error capturando ticks con {method}: {e}")
        
        # Combinar todos los ticks
        if all_ticks_data:
            combined_ticks = pd.concat(all_ticks_data, ignore_index=True)
            combined_ticks = combined_ticks.sort_values('time').reset_index(drop=True)

            # Convertir ticks a OHLC M5
            ohlc_from_ticks = process_ticks_to_ohlc(combined_ticks, 'M5')

            logger.info(f"Ticks procesados a OHLC: {len(ohlc_from_ticks)} barras")
            return ohlc_from_ticks

        raise DataRetrievalError("No se pudieron capturar datos de ticks")
    
    def _capture_single_timeframe(self, timeframe: str, method: str, 
                                 start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Capturar datos de un timeframe específico con un método específico"""
        try:
            tf_config = TIMEFRAMES[timeframe]
            tf_enum = tf_config['enum']
            
            if method == 'rates_range':
                rates = mt5.copy_rates_range(self.symbol, tf_enum, start_date, end_date)
            elif method == 'rates_from':
                rates = mt5.copy_rates_from(self.symbol, tf_enum, start_date, 0)
            elif method == 'rates_from_pos':
                rates = mt5.copy_rates_from_pos(self.symbol, tf_enum, 0, 0)
            else:
                logger.warning(f"Método no soportado: {method}")
                return pd.DataFrame()
            
            if rates is None or len(rates) == 0:
                return pd.DataFrame()
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce').dt.tz_localize(None)
            
            # Asegurar tipos compatibles para comparación
            start_date_pd = pd.to_datetime(start_date).tz_localize(None) if hasattr(pd.to_datetime(start_date), 'tz_localize') else pd.to_datetime(start_date)
            end_date_pd = pd.to_datetime(end_date).tz_localize(None) if hasattr(pd.to_datetime(end_date), 'tz_localize') else pd.to_datetime(end_date)
            df = df[(df['time'] >= start_date_pd) & (df['time'] <= end_date_pd)]
            
            # Añadir metadatos
            df['source_timeframe'] = timeframe
            df['capture_method'] = method
            df['quality_score'] = tf_config['quality']
            df['data_flag'] = f'real_{timeframe}'
            
            return df
            
        except Exception as e:
            logger.warning(f"Error capturando {timeframe} con {method}: {e}")
            return pd.DataFrame()
    
    def _capture_ticks(self, method: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Capturar datos de ticks"""
        try:
            if method == 'ticks_range':
                ticks = mt5.copy_ticks_range(self.symbol, start_date, end_date, mt5.COPY_TICKS_ALL)
            elif method == 'ticks_from':
                ticks = mt5.copy_ticks_from(self.symbol, start_date, 0, mt5.COPY_TICKS_ALL)
            else:
                logger.warning(f"Método de ticks no soportado: {method}")
                return pd.DataFrame()
            
            if ticks is None or len(ticks) == 0:
                return pd.DataFrame()
            
            # Convertir a DataFrame
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce').dt.tz_localize(None)
            
            # Asegurar tipos compatibles para comparación
            start_date_pd = pd.to_datetime(start_date).tz_localize(None) if hasattr(pd.to_datetime(start_date), 'tz_localize') else pd.to_datetime(start_date)
            end_date_pd = pd.to_datetime(end_date).tz_localize(None) if hasattr(pd.to_datetime(end_date), 'tz_localize') else pd.to_datetime(end_date)
            df = df[(df['time'] >= start_date_pd) & (df['time'] <= end_date_pd)]
            
            # Añadir metadatos
            df['source_timeframe'] = 'ticks'
            df['capture_method'] = method
            df['quality_score'] = 0.98  # Alta calidad para ticks
            df['data_flag'] = 'real_ticks'
            
            return df
            
        except Exception as e:
            logger.warning(f"Error capturando ticks con {method}: {e}")
            return pd.DataFrame()
    
    def _resample_all_to_m5(self, df: pd.DataFrame, source_timeframes: List[str]) -> pd.DataFrame:
        """Resamplear todos los datos a M5"""
        if df.empty:
            return df
        
        logger.info("Resampleando todos los datos a M5...")
        
        # Agrupar por timeframe de origen
        resampled_dfs = []
        
        for timeframe in source_timeframes:
            timeframe_data = df[df['source_timeframe'] == timeframe]
            if not timeframe_data.empty:
                resampled = resample_timeframe_to_m5(timeframe_data, timeframe)
                if not resampled.empty:
                    resampled_dfs.append(resampled)
        
        # Combinar todos los datos resampleados
        if resampled_dfs:
            combined = pd.concat(resampled_dfs, ignore_index=True)
            combined = combined.sort_values('time').reset_index(drop=True)
            
            # Eliminar duplicados manteniendo el de mayor calidad
            combined = combined.drop_duplicates(subset=['time'], keep='first')
            
            logger.info(f"Resampling completado: {len(combined)} barras M5")
            return combined
        
        return df
    
    def _ensure_m5_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Asegurar que todos los datos estén en timeframe M5"""
        if df.empty:
            return df
        
        # Verificar si ya está en M5
        if 'source_timeframe' in df.columns:
            m5_data = df[df['source_timeframe'] == 'M5']
            other_data = df[df['source_timeframe'] != 'M5']
            
            if not other_data.empty:
                # Resamplear datos que no son M5
                resampled_other = self._resample_all_to_m5(other_data, other_data['source_timeframe'].unique().tolist())
                
                # Combinar M5 nativo con resampleado
                if not m5_data.empty:
                    final_df = pd.concat([m5_data, resampled_other], ignore_index=True)
                    final_df = final_df.sort_values('time').reset_index(drop=True)
                    final_df = final_df.drop_duplicates(subset=['time'], keep='first')
                    return final_df
                else:
                    return resampled_other
        
        return df
    
    def get_quality_report(self) -> Dict:
        """Obtener reporte de calidad del motor de captura"""
        return self.quality_tracker.generate_quality_report()
    
    def save_quality_report(self, output_path: str):
        """Guardar reporte de calidad en archivo"""
        report = self.get_quality_report()
        with open(output_path, 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Reporte de calidad guardado: {output_path}")
    
    def __del__(self):
        """Cleanup al destruir el objeto"""
        if self.use_ray and ray.is_initialized():
            try:
                ray.shutdown()
            except:
                pass
