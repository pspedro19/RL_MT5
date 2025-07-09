"""
Generación de splits temporales para series de tiempo financieras
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Generator
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TemporalSplitter:
    """Generador de splits temporales para datos de trading"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializar splitter
        
        Args:
            config: Configuración del splitter
        """
        self.config = config or {}
        self.min_train_samples = self.config.get('min_train_samples', 1000)
        self.min_val_samples = self.config.get('min_val_samples', 200)
        self.gap_periods = self.config.get('gap_periods', 0)  # Gap entre train y val
        
    def create_train_val_split(self, df: pd.DataFrame, 
                             train_ratio: float = 0.8,
                             time_col: str = 'time') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Split temporal simple train/validation
        
        Args:
            df: DataFrame ordenado por tiempo
            train_ratio: Proporción de datos para entrenamiento
            time_col: Columna de tiempo
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: (train_df, val_df, split_info)
        """
        logger.info(f"Creando split train/val con ratio {train_ratio:.0%}")
        
        # Asegurar orden temporal
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        
        # Calcular índice de split
        n_samples = len(df_sorted)
        split_idx = int(n_samples * train_ratio)
        
        # Aplicar gap si está configurado
        if self.gap_periods > 0:
            split_idx = max(0, split_idx - self.gap_periods)
            gap_end_idx = min(n_samples, split_idx + self.gap_periods)
            train_df = df_sorted.iloc[:split_idx].copy()
            val_df = df_sorted.iloc[gap_end_idx:].copy()
        else:
            train_df = df_sorted.iloc[:split_idx].copy()
            val_df = df_sorted.iloc[split_idx:].copy()
        
        # Validar tamaños mínimos
        if len(train_df) < self.min_train_samples:
            logger.warning(f"Train set muy pequeño: {len(train_df)} < {self.min_train_samples}")
        if len(val_df) < self.min_val_samples:
            logger.warning(f"Validation set muy pequeño: {len(val_df)} < {self.min_val_samples}")
        
        # Información del split
        split_info = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'train_ratio': len(train_df) / n_samples,
            'val_ratio': len(val_df) / n_samples,
            'gap_periods': self.gap_periods,
            'train_date_range': {
                'start': train_df[time_col].min(),
                'end': train_df[time_col].max()
            },
            'val_date_range': {
                'start': val_df[time_col].min(),
                'end': val_df[time_col].max()
            },
            'temporal_overlap': False  # Por diseño no hay overlap
        }
        
        logger.info(f"Split creado: Train={len(train_df):,} ({split_info['train_ratio']:.1%}), "
                   f"Val={len(val_df):,} ({split_info['val_ratio']:.1%})")
        
        return train_df, val_df, split_info
    
    def create_walk_forward_splits(self, df: pd.DataFrame, 
                                 n_splits: int = 5,
                                 train_size: Optional[int] = None,
                                 val_size: Optional[int] = None,
                                 time_col: str = 'time') -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict], None, None]:
        """
        Crear splits para walk-forward analysis
        
        Args:
            df: DataFrame ordenado por tiempo
            n_splits: Número de splits
            train_size: Tamaño fijo de entrenamiento (opcional)
            val_size: Tamaño fijo de validación (opcional)
            time_col: Columna de tiempo
            
        Yields:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: (train_df, val_df, split_info)
        """
        logger.info(f"Creando {n_splits} splits walk-forward")
        
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        n_samples = len(df_sorted)
        
        if train_size is None:
            # Calcular tamaño basado en splits
            total_val_size = n_samples // (n_splits + 1)
            train_size = n_samples - n_splits * total_val_size
            
        if val_size is None:
            val_size = (n_samples - train_size) // n_splits
        
        for i in range(n_splits):
            # Índices para este split
            train_start = 0
            train_end = train_size + i * val_size
            val_start = train_end + self.gap_periods
            val_end = val_start + val_size
            
            # Verificar límites
            if val_end > n_samples:
                val_end = n_samples
                if val_start >= val_end:
                    logger.warning(f"Split {i+1} excede los límites del dataset")
                    continue
            
            # Crear splits
            train_df = df_sorted.iloc[train_start:train_end].copy()
            val_df = df_sorted.iloc[val_start:val_end].copy()
            
            # Info del split
            split_info = {
                'split_number': i + 1,
                'train_size': len(train_df),
                'val_size': len(val_df),
                'train_date_range': {
                    'start': train_df[time_col].min(),
                    'end': train_df[time_col].max()
                },
                'val_date_range': {
                    'start': val_df[time_col].min(),
                    'end': val_df[time_col].max()
                }
            }
            
            logger.info(f"Split {i+1}/{n_splits}: Train={len(train_df):,}, Val={len(val_df):,}")
            
            yield train_df, val_df, split_info
    
    def create_expanding_window_splits(self, df: pd.DataFrame,
                                     initial_train_size: int,
                                     val_size: int,
                                     step_size: Optional[int] = None,
                                     max_splits: Optional[int] = None,
                                     time_col: str = 'time') -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict], None, None]:
        """
        Crear splits con ventana expansiva
        
        Args:
            df: DataFrame ordenado por tiempo
            initial_train_size: Tamaño inicial de entrenamiento
            val_size: Tamaño de validación
            step_size: Tamaño del paso (default: val_size)
            max_splits: Máximo número de splits
            time_col: Columna de tiempo
            
        Yields:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: (train_df, val_df, split_info)
        """
        logger.info("Creando splits con ventana expansiva")
        
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        n_samples = len(df_sorted)
        
        if step_size is None:
            step_size = val_size
        
        split_count = 0
        current_train_end = initial_train_size
        
        while current_train_end + self.gap_periods + val_size <= n_samples:
            # Crear split
            train_df = df_sorted.iloc[:current_train_end].copy()
            val_start = current_train_end + self.gap_periods
            val_end = val_start + val_size
            val_df = df_sorted.iloc[val_start:val_end].copy()
            
            split_count += 1
            
            # Info del split
            split_info = {
                'split_number': split_count,
                'train_size': len(train_df),
                'val_size': len(val_df),
                'train_date_range': {
                    'start': train_df[time_col].min(),
                    'end': train_df[time_col].max()
                },
                'val_date_range': {
                    'start': val_df[time_col].min(),
                    'end': val_df[time_col].max()
                },
                'window_type': 'expanding'
            }
            
            logger.info(f"Split {split_count}: Train={len(train_df):,} (expanding), Val={len(val_df):,}")
            
            yield train_df, val_df, split_info
            
            # Siguiente ventana
            current_train_end += step_size
            
            # Verificar máximo de splits
            if max_splits and split_count >= max_splits:
                break
    
    def create_time_based_splits(self, df: pd.DataFrame,
                               train_months: int = 12,
                               val_months: int = 3,
                               step_months: int = 1,
                               time_col: str = 'time') -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict], None, None]:
        """
        Crear splits basados en períodos de tiempo fijos
        
        Args:
            df: DataFrame ordenado por tiempo
            train_months: Meses de entrenamiento
            val_months: Meses de validación
            step_months: Meses de avance entre splits
            time_col: Columna de tiempo
            
        Yields:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: (train_df, val_df, split_info)
        """
        logger.info(f"Creando splits basados en tiempo: {train_months}m train, {val_months}m val")
        
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        
        # Convertir a datetime si es necesario
        if not pd.api.types.is_datetime64_any_dtype(df_sorted[time_col]):
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        
        # Fechas límite
        min_date = df_sorted[time_col].min()
        max_date = df_sorted[time_col].max()
        
        # Generar splits
        current_date = min_date
        split_count = 0
        
        while current_date + pd.DateOffset(months=train_months+val_months) <= max_date:
            # Definir períodos
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=train_months)
            val_start = train_end
            val_end = val_start + pd.DateOffset(months=val_months)
            
            # Filtrar datos
            train_mask = (df_sorted[time_col] >= train_start) & (df_sorted[time_col] < train_end)
            val_mask = (df_sorted[time_col] >= val_start) & (df_sorted[time_col] < val_end)
            
            train_df = df_sorted[train_mask].copy()
            val_df = df_sorted[val_mask].copy()
            
            # Verificar tamaños mínimos
            if len(train_df) >= self.min_train_samples and len(val_df) >= self.min_val_samples:
                split_count += 1
                
                split_info = {
                    'split_number': split_count,
                    'train_size': len(train_df),
                    'val_size': len(val_df),
                    'train_period': f"{train_start.date()} to {train_end.date()}",
                    'val_period': f"{val_start.date()} to {val_end.date()}",
                    'train_months': train_months,
                    'val_months': val_months
                }
                
                logger.info(f"Split {split_count}: {split_info['train_period']} → {split_info['val_period']}")
                
                yield train_df, val_df, split_info
            
            # Avanzar al siguiente período
            current_date += pd.DateOffset(months=step_months)
    
    def create_custom_splits(self, df: pd.DataFrame,
                           split_dates: List[Dict[str, str]],
                           time_col: str = 'time') -> Generator[Tuple[pd.DataFrame, pd.DataFrame, Dict], None, None]:
        """
        Crear splits personalizados basados en fechas específicas
        
        Args:
            df: DataFrame ordenado por tiempo
            split_dates: Lista de diccionarios con fechas de split
                       [{'train_start': '2023-01-01', 'train_end': '2023-12-31',
                         'val_start': '2024-01-01', 'val_end': '2024-03-31'}, ...]
            time_col: Columna de tiempo
            
        Yields:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: (train_df, val_df, split_info)
        """
        logger.info(f"Creando {len(split_dates)} splits personalizados")
        
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        
        # Convertir a datetime si es necesario
        if not pd.api.types.is_datetime64_any_dtype(df_sorted[time_col]):
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        
        for i, dates in enumerate(split_dates):
            # Parsear fechas
            train_start = pd.to_datetime(dates['train_start'])
            train_end = pd.to_datetime(dates['train_end'])
            val_start = pd.to_datetime(dates['val_start'])
            val_end = pd.to_datetime(dates['val_end'])
            
            # Filtrar datos
            train_mask = (df_sorted[time_col] >= train_start) & (df_sorted[time_col] <= train_end)
            val_mask = (df_sorted[time_col] >= val_start) & (df_sorted[time_col] <= val_end)
            
            train_df = df_sorted[train_mask].copy()
            val_df = df_sorted[val_mask].copy()
            
            split_info = {
                'split_number': i + 1,
                'train_size': len(train_df),
                'val_size': len(val_df),
                'train_date_range': {
                    'start': train_start,
                    'end': train_end
                },
                'val_date_range': {
                    'start': val_start,
                    'end': val_end
                },
                'custom_split': True
            }
            
            logger.info(f"Split {i+1}: Train={len(train_df):,} "
                       f"({train_start.date()} to {train_end.date()}), "
                       f"Val={len(val_df):,} ({val_start.date()} to {val_end.date()})")
            
            yield train_df, val_df, split_info
    
    def analyze_splits(self, splits: List[Tuple[pd.DataFrame, pd.DataFrame, Dict]]) -> Dict:
        """
        Analizar características de los splits generados
        
        Args:
            splits: Lista de splits generados
            
        Returns:
            Dict: Análisis de los splits
        """
        logger.info("Analizando splits generados...")
        
        analysis = {
            'total_splits': len(splits),
            'train_sizes': [],
            'val_sizes': [],
            'train_val_ratios': [],
            'temporal_coverage': []
        }
        
        for train_df, val_df, info in splits:
            analysis['train_sizes'].append(len(train_df))
            analysis['val_sizes'].append(len(val_df))
            analysis['train_val_ratios'].append(len(train_df) / len(val_df) if len(val_df) > 0 else 0)
            
            if 'train_date_range' in info and 'val_date_range' in info:
                train_days = (info['train_date_range']['end'] - 
                            info['train_date_range']['start']).days
                val_days = (info['val_date_range']['end'] - 
                          info['val_date_range']['start']).days
                analysis['temporal_coverage'].append({
                    'train_days': train_days,
                    'val_days': val_days
                })
        
        # Estadísticas agregadas
        analysis['avg_train_size'] = np.mean(analysis['train_sizes'])
        analysis['avg_val_size'] = np.mean(analysis['val_sizes'])
        analysis['min_train_size'] = np.min(analysis['train_sizes'])
        analysis['max_train_size'] = np.max(analysis['train_sizes'])
        analysis['min_val_size'] = np.min(analysis['val_sizes'])
        analysis['max_val_size'] = np.max(analysis['val_sizes'])
        
        return analysis