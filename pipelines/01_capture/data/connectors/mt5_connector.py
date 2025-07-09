#!/usr/bin/env python3
"""
Conector optimizado para MetaTrader5 con soporte completo para todos los métodos de captura
"""
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import warnings
import time
import gc

from config.constants import (
    TIMEFRAMES, CAPTURE_METHODS, RESAMPLING_CONFIG,
    MARKET_CONFIGS, SYMBOL_ALIASES
)
from data.processing import (
    process_ticks_to_ohlc, resample_timeframe_to_m5,
    combine_and_prioritize_dataframes
)

logger = logging.getLogger(__name__)

# ===============================================================================
# CONECTOR MT5 MEJORADO
# ===============================================================================

class MT5Connector:
    """Conector MT5 con soporte completo para todos los métodos de captura"""
    
    def __init__(self, instrument: str = 'US500', use_aliases: bool = True):
        self.instrument = instrument
        self.use_aliases = use_aliases
        self.symbol = None
        self.connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
        # Obtener configuración de mercado
        self.market_config = MARKET_CONFIGS.get(instrument, MARKET_CONFIGS['US500'])
        self.capture_prefs = self.market_config.get('capture_preferences', {})
        
        # Inicializar MT5
        self._initialize_mt5()
        
        logger.info(f"Conector MT5 inicializado para {instrument}")
    
    def _initialize_mt5(self) -> bool:
        """Inicializar conexión con MT5"""
        try:
            if not mt5.initialize():
                logger.error("Error inicializando MT5")
                return False
            
            # Buscar símbolo
            self.symbol = self._find_symbol()
            if not self.symbol:
                logger.error(f"No se pudo encontrar símbolo para {self.instrument}")
                return False
            
            # Verificar que el símbolo está disponible
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Símbolo {self.symbol} no disponible en MT5")
                return False
            
            self.connected = True
            logger.info(f"MT5 conectado exitosamente. Símbolo: {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando MT5: {e}")
            return False
    
    def _find_symbol(self) -> Optional[str]:
        """Encontrar el símbolo correcto para el instrumento"""
        if not self.use_aliases:
            return self.instrument
        
        # Buscar en aliases
        aliases = SYMBOL_ALIASES.get(self.instrument, [self.instrument])
        
        for alias in aliases:
            try:
                # Verificar si el símbolo existe
                symbol_info = mt5.symbol_info(alias)
                if symbol_info is not None:
                    logger.info(f"Símbolo encontrado: {alias}")
                    return alias
            except Exception as e:
                logger.debug(f"Alias {alias} no disponible: {e}")
                continue
        
        # Si no se encuentra en aliases, intentar con el nombre original
        try:
            symbol_info = mt5.symbol_info(self.instrument)
            if symbol_info is not None:
                logger.info(f"Usando símbolo original: {self.instrument}")
                return self.instrument
        except Exception as e:
            logger.debug(f"Símbolo original {self.instrument} no disponible: {e}")
        
        return None
    
    def capture_data_comprehensive(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Capturar datos usando todos los métodos disponibles de forma comprehensiva"""
        logger.info(f"Captura comprehensiva: {start_date} - {end_date}")
        
        all_dataframes = []
        source_timeframes = []
        
        # 1. Captura con métodos de rates (OHLC)
        rates_data = self._capture_rates_comprehensive(start_date, end_date)
        if not rates_data.empty:
            all_dataframes.append(rates_data)
            source_timeframes.append('rates_combined')
        
        # 2. Captura con métodos de ticks
        ticks_data = self._capture_ticks_comprehensive(start_date, end_date)
        if not ticks_data.empty:
            all_dataframes.append(ticks_data)
            source_timeframes.append('ticks_aggregated')
        
        # 3. Combinar y priorizar
        if all_dataframes:
            combined_df = combine_and_prioritize_dataframes(
                all_dataframes, source_timeframes, self.instrument
            )
            
            # 4. Asegurar timeframe M5
            final_df = self._ensure_m5_timeframe(combined_df)
            
            logger.info(f"Captura comprehensiva completada: {len(final_df)} registros")
            return final_df
        else:
            logger.error("No se pudieron capturar datos con ningún método")
            return pd.DataFrame()
    
    def _capture_rates_comprehensive(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Capturar datos de rates usando todos los métodos disponibles"""
        logger.info("Capturando rates con todos los métodos...")
        
        rates_dataframes = []
        timeframes_used = []
        
        # Obtener métodos preferidos
        primary_methods = self.capture_prefs.get('primary_methods', ['rates_from_pos', 'rates_range'])
        fallback_methods = self.capture_prefs.get('fallback_methods', ['rates_from'])
        
        # Intentar con cada timeframe
        for timeframe, config in TIMEFRAMES.items():
            if timeframe == 'M5':  # Lo manejamos al final
                continue
            
            logger.info(f"Intentando rates {timeframe}...")
            
            # Intentar métodos primarios
            for method in primary_methods:
                if 'rates' in method:
                    df = self._capture_single_rate(method, timeframe, config, start_date, end_date)
                    if not df.empty:
                        rates_dataframes.append(df)
                        timeframes_used.append(timeframe)
                        logger.info(f"Rates exitoso: {timeframe} con {method}")
                        break
            
            # Si no se pudo, intentar fallbacks
            if timeframe not in timeframes_used:
                for method in fallback_methods:
                    if 'rates' in method:
                        df = self._capture_single_rate(method, timeframe, config, start_date, end_date)
                        if not df.empty:
                            rates_dataframes.append(df)
                            timeframes_used.append(timeframe)
                            logger.info(f"Rates exitoso (fallback): {timeframe} con {method}")
                            break
        
        # Combinar todos los rates
        if rates_dataframes:
            combined_rates = pd.concat(rates_dataframes, ignore_index=True)
            combined_rates = combined_rates.sort_values('time').reset_index(drop=True)
            
            # Resamplear a M5
            combined_rates = self._resample_to_m5(combined_rates, timeframes_used)
            
            logger.info(f"Rates combinados: {len(combined_rates)} registros")
            return combined_rates
        
        return pd.DataFrame()
    
    def _capture_ticks_comprehensive(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Capturar datos de ticks usando todos los métodos disponibles"""
        logger.info("Capturando ticks con todos los métodos...")
        
        # Obtener métodos de ticks
        tick_methods = self.capture_prefs.get('tick_methods', ['ticks_range', 'ticks_from'])
        max_timeframe = self.capture_prefs.get('max_timeframe_for_ticks', 'M30')
        
        all_ticks_data = []
        
        for method in tick_methods:
            logger.info(f"Intentando ticks con {method}...")
            
            try:
                ticks_df = self._capture_ticks_single(method, start_date, end_date)
                if not ticks_df.empty:
                    all_ticks_data.append(ticks_df)
                    logger.info(f"Ticks exitoso con {method}: {len(ticks_df)} registros")
            except Exception as e:
                logger.warning(f"Error capturando ticks con {method}: {e}")
        
        # Combinar todos los ticks
        if all_ticks_data:
            combined_ticks = pd.concat(all_ticks_data, ignore_index=True)
            combined_ticks = combined_ticks.sort_values('time').reset_index(drop=True)
            
            # Convertir a OHLC M5
            ohlc_from_ticks = process_ticks_to_ohlc(combined_ticks, 'M5')
            
            logger.info(f"Ticks procesados a OHLC: {len(ohlc_from_ticks)} barras")
            return ohlc_from_ticks
        
        return pd.DataFrame()
    
    def _capture_single_rate(self, method: str, timeframe: str, config: dict,
                            start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Capturar datos de rates con un método específico"""
        try:
            tf_enum = config['enum']
            
            if method == 'rates_range':
                rates = mt5.copy_rates_range(self.symbol, tf_enum, start_date, end_date)
            elif method == 'rates_from':
                # Calcular número de barras necesarias
                total_minutes = int((end_date - start_date).total_seconds() / 60)
                bars_needed = total_minutes // config['minutes']
                rates = mt5.copy_rates_from(self.symbol, tf_enum, start_date, bars_needed)
            elif method == 'rates_from_pos':
                # Calcular posición desde el inicio
                current_time = datetime.now()
                total_minutes = int((current_time - start_date).total_seconds() / 60)
                bars_back = total_minutes // config['minutes']
                bars_needed = int((end_date - start_date).total_seconds() / 60 / config['minutes'])
                rates = mt5.copy_rates_from_pos(self.symbol, tf_enum, bars_back, bars_needed)
            else:
                logger.warning(f"Método de rates no soportado: {method}")
                return pd.DataFrame()
            
            if rates is None or len(rates) == 0:
                return pd.DataFrame()
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Filtrar por rango de fechas
            df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
            
            # Añadir metadatos
            df['source_timeframe'] = timeframe
            df['capture_method'] = method
            df['quality_score'] = config['quality']
            df['data_flag'] = f'real_{timeframe}'
            
            return df
            
        except Exception as e:
            logger.warning(f"Error capturando rates {timeframe} con {method}: {e}")
            return pd.DataFrame()
    
    def _capture_ticks_single(self, method: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Capturar datos de ticks con un método específico"""
        try:
            if method == 'ticks_range':
                ticks = mt5.copy_ticks_range(self.symbol, start_date, end_date, mt5.COPY_TICKS_ALL)
            elif method == 'ticks_from':
                # Para ticks_from, necesitamos estimar cuántos ticks capturar
                # Usar un número grande y luego filtrar
                max_ticks = 1000000  # 1 millón de ticks máximo
                ticks = mt5.copy_ticks_from(self.symbol, start_date, max_ticks, mt5.COPY_TICKS_ALL)
            else:
                logger.warning(f"Método de ticks no soportado: {method}")
                return pd.DataFrame()
            
            if ticks is None or len(ticks) == 0:
                return pd.DataFrame()
            
            # Convertir a DataFrame
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Filtrar por rango de fechas
            df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
            
            # Añadir metadatos
            df['source_timeframe'] = 'ticks'
            df['capture_method'] = method
            df['quality_score'] = 0.98  # Alta calidad para ticks
            df['data_flag'] = 'real_ticks'
            
            return df
            
        except Exception as e:
            logger.warning(f"Error capturando ticks con {method}: {e}")
            return pd.DataFrame()
    
    def _resample_to_m5(self, df: pd.DataFrame, source_timeframes: List[str]) -> pd.DataFrame:
        """Resamplear datos a M5"""
        if df.empty:
            return df
        
        logger.info("Resampleando datos a M5...")
        
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
                source_timeframes = other_data['source_timeframe'].unique().tolist()
                resampled_other = self._resample_to_m5(other_data, source_timeframes)
                
                # Combinar M5 nativo con resampleado
                if not m5_data.empty:
                    final_df = pd.concat([m5_data, resampled_other], ignore_index=True)
                    final_df = final_df.sort_values('time').reset_index(drop=True)
                    final_df = final_df.drop_duplicates(subset=['time'], keep='first')
                    return final_df
                else:
                    return resampled_other
        
        return df
    
    def get_symbol_info(self) -> Optional[Dict]:
        """Obtener información del símbolo"""
        if not self.connected or not self.symbol:
            return None
        
        try:
            info = mt5.symbol_info(self.symbol)
            if info is not None:
                return {
                    'symbol': self.symbol,
                    'name': info.name,
                    'currency_base': info.currency_base,
                    'currency_profit': info.currency_profit,
                    'point': info.point,
                    'digits': info.digits,
                    'spread': info.spread,
                    'spread_float': info.spread_float,
                    'volume_min': info.volume_min,
                    'volume_max': info.volume_max,
                    'volume_step': info.volume_step
                }
        except Exception as e:
            logger.error(f"Error obteniendo información del símbolo: {e}")
        
        return None
    
    def test_connection(self) -> bool:
        """Probar conexión con MT5"""
        if not self.connected:
            return self._initialize_mt5()
        
        try:
            # Intentar obtener información del símbolo
            info = mt5.symbol_info(self.symbol)
            return info is not None
        except Exception as e:
            logger.error(f"Error probando conexión: {e}")
            return False
    
    def __del__(self):
        """Cleanup al destruir el conector"""
        if self.connected:
            try:
                mt5.shutdown()
            except:
                pass
