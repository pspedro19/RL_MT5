#!/usr/bin/env python3
"""
Script específico para migrar el archivo US500 existente a la nueva estructura de trazabilidad
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import DATA_ORIGINS, DATA_ORIGIN_VALUES
from utils.data_traceability import DataTraceabilityManager, format_time_standard

logger = logging.getLogger(__name__)


def migrate_us500_file(input_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Migrar archivo US500 específico a nueva estructura de trazabilidad
    
    Args:
        input_path: Ruta al archivo de entrada
        output_path: Ruta de salida (opcional)
    
    Returns:
        Dict con resultados de migración
    """
    logger.info(f"Migrando archivo US500: {input_path}")
    
    try:
        # Cargar datos
        df = pd.read_parquet(input_path)
        logger.info(f"Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
        
        # Verificar si ya tiene data_origin
        if 'data_origin' in df.columns:
            logger.info("Archivo ya tiene data_origin, no necesita migración")
            return {
                'file': input_path,
                'migrated': False,
                'message': 'Archivo ya tiene data_origin'
            }
        
        # Analizar el archivo para determinar el origen más probable
        # Basándonos en el nombre del archivo y las características
        file_name = os.path.basename(input_path)
        
        # Determinar origen basado en el nombre del archivo
        if 'hpc' in file_name.lower():
            # Archivo procesado con HPC, probablemente datos nativos M5
            most_likely_origin = 'M5_NATIVO'
            capture_method = 'rates_range'
            source_timeframe = 'M5'
        elif 'm5' in file_name.lower():
            # Archivo M5 específico
            most_likely_origin = 'M5_NATIVO'
            capture_method = 'rates_range'
            source_timeframe = 'M5'
        else:
            # Fallback
            most_likely_origin = 'M5_NATIVO'
            capture_method = 'rates_range'
            source_timeframe = 'M5'
        
        logger.info(f"Origen determinado: {most_likely_origin}")
        
        # Crear copia del DataFrame
        df_migrated = df.copy()
        
        # Asignar data_origin
        df_migrated['data_origin'] = most_likely_origin
        
        # Asignar quality_score basado en data_origin
        df_migrated['quality_score'] = DATA_ORIGINS[most_likely_origin]['quality_score']
        
        # Formatear tiempo si es necesario
        if 'time' in df_migrated.columns:
            df_migrated['time'] = format_time_standard(df_migrated['time'])
        
        # Validar migración
        traceability_manager = DataTraceabilityManager()
        is_valid, errors = traceability_manager.validate_data_origin(df_migrated)
        
        if not is_valid:
            logger.error(f"Migración falló: {errors}")
            return {
                'file': input_path,
                'migrated': False,
                'error': f'Migración falló: {errors}'
            }
        
        # Guardar archivo migrado
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_migrated.parquet"
        
        df_migrated.to_parquet(output_path, index=False)
        
        # Generar estadísticas
        stats = {
            'total_records': len(df_migrated),
            'data_origin_distribution': df_migrated['data_origin'].value_counts().to_dict(),
            'quality_score_stats': {
                'mean': float(df_migrated['quality_score'].mean()),
                'min': float(df_migrated['quality_score'].min()),
                'max': float(df_migrated['quality_score'].max()),
                'std': float(df_migrated['quality_score'].std())
            },
            'time_range': {
                'start': df_migrated['time'].min().isoformat(),
                'end': df_migrated['time'].max().isoformat()
            }
        }
        
        logger.info(f"Migración exitosa: {output_path}")
        logger.info(f"Estadísticas: {stats}")
        
        return {
            'file': input_path,
            'migrated': True,
            'output_path': output_path,
            'stats': stats,
            'data_origin_used': most_likely_origin
        }
        
    except Exception as e:
        error_msg = f"Error migrando {input_path}: {str(e)}"
        logger.error(error_msg)
        
        return {
            'file': input_path,
            'migrated': False,
            'error': error_msg
        }


def analyze_us500_file(file_path: str) -> Dict[str, Any]:
    """
    Analizar archivo US500 para determinar características
    
    Args:
        file_path: Ruta al archivo
    
    Returns:
        Dict con análisis del archivo
    """
    logger.info(f"Analizando archivo: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        
        analysis = {
            'file_name': os.path.basename(file_path),
            'total_records': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'time_range': {
                'start': df['time'].min().isoformat() if 'time' in df.columns else None,
                'end': df['time'].max().isoformat() if 'time' in df.columns else None
            },
            'price_stats': {
                'open_mean': float(df['open'].mean()) if 'open' in df.columns else None,
                'high_mean': float(df['high'].mean()) if 'high' in df.columns else None,
                'low_mean': float(df['low'].mean()) if 'low' in df.columns else None,
                'close_mean': float(df['close'].mean()) if 'close' in df.columns else None
            },
            'quality_score_stats': {
                'mean': float(df['quality_score'].mean()) if 'quality_score' in df.columns else None,
                'min': float(df['quality_score'].min()) if 'quality_score' in df.columns else None,
                'max': float(df['quality_score'].max()) if 'quality_score' in df.columns else None
            },
            'has_traceability': {
                'data_origin': 'data_origin' in df.columns,
                'data_flag': 'data_flag' in df.columns,
                'source_timeframe': 'source_timeframe' in df.columns,
                'capture_method': 'capture_method' in df.columns
            }
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analizando {file_path}: {str(e)}")
        return {'error': str(e)}


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrar archivo US500 a nueva trazabilidad')
    parser.add_argument('--input', required=True, help='Archivo de entrada')
    parser.add_argument('--output', help='Archivo de salida (opcional)')
    parser.add_argument('--action', choices=['analyze', 'migrate', 'both'], 
                       default='both', help='Acción a realizar')
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.action in ['analyze', 'both']:
        print("=== ANÁLISIS DEL ARCHIVO ===")
        analysis = analyze_us500_file(args.input)
        print(f"Análisis: {analysis}")
        print()
    
    if args.action in ['migrate', 'both']:
        print("=== MIGRACIÓN DEL ARCHIVO ===")
        result = migrate_us500_file(args.input, args.output)
        print(f"Resultado: {result}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 