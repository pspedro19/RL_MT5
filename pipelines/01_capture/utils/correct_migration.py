#!/usr/bin/env python3
"""
Script de migración corregido con filtro de horarios y trazabilidad granular
Aplica el filtro de horarios de mercado PRIMERO, luego análisis granular de trazabilidad
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import json

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import DATA_ORIGINS, DATA_ORIGIN_VALUES
from utils.market_hours_filter import apply_market_hours_filter, validate_market_hours_compliance
from utils.granular_traceability import apply_granular_traceability, validate_granular_traceability
from utils.data_traceability import format_time_standard

logger = logging.getLogger(__name__)


def correct_migration_pipeline(input_path: str, output_path: str = None, 
                             instrument: str = 'US500') -> Dict[str, Any]:
    """
    Pipeline completo de migración corregido
    
    Args:
        input_path: Ruta al archivo de entrada
        output_path: Ruta de salida (opcional)
        instrument: Instrumento ('US500', 'USDCOP', etc.)
    
    Returns:
        Dict con resultados completos de migración
    """
    logger.info(f"Iniciando migración corregida para {instrument}")
    logger.info(f"Archivo de entrada: {input_path}")
    
    try:
        # PASO 1: Cargar datos
        logger.info("PASO 1: Cargando datos...")
        df = pd.read_parquet(input_path)
        logger.info(f"Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
        
        # Verificar si ya tiene data_origin
        if 'data_origin' in df.columns:
            logger.info("Archivo ya tiene data_origin, validando estructura...")
            validation = validate_granular_traceability(df)
            if validation['valid']:
                return {
                    'file': input_path,
                    'migrated': False,
                    'message': 'Archivo ya tiene data_origin válido',
                    'validation': validation
                }
            else:
                logger.warning("Archivo tiene data_origin pero no es válido, re-migrando...")
        
        # PASO 2: Validar horarios de mercado ANTES de procesar
        logger.info("PASO 2: Validando horarios de mercado...")
        market_validation = validate_market_hours_compliance(df, instrument)
        logger.info(f"Validación de horarios: {market_validation}")
        
        # PASO 3: Aplicar filtro de horarios de mercado (PRIMERO)
        logger.info("PASO 3: Aplicando filtro de horarios de mercado...")
        df_filtered, market_analysis = apply_market_hours_filter(df, instrument)
        logger.info(f"Filtrado completado: {len(df_filtered)} registros después del filtro")
        
        # PASO 4: Formatear tiempo estándar
        logger.info("PASO 4: Formateando tiempo estándar...")
        if 'time' in df_filtered.columns:
            df_filtered['time'] = format_time_standard(df_filtered['time'])
        
        # PASO 5: Aplicar trazabilidad granular
        logger.info("PASO 5: Aplicando trazabilidad granular...")
        df_traced, traceability_analysis = apply_granular_traceability(df_filtered, instrument)
        logger.info(f"Trazabilidad aplicada: {len(df_traced)} registros")
        
        # PASO 6: Validar resultado final
        logger.info("PASO 6: Validando resultado final...")
        final_validation = validate_granular_traceability(df_traced)
        
        if not final_validation['valid']:
            logger.error(f"Validación final falló: {final_validation}")
            return {
                'file': input_path,
                'migrated': False,
                'error': f'Validación final falló: {final_validation}'
            }
        
        # PASO 7: Guardar archivo migrado
        logger.info("PASO 7: Guardando archivo migrado...")
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_corrected_migrated.parquet"
        
        df_traced.to_parquet(output_path, index=False)
        
        # PASO 8: Generar reporte completo
        logger.info("PASO 8: Generando reporte completo...")
        
        # Estadísticas finales
        final_stats = {
            'total_records_original': len(df),
            'total_records_filtered': len(df_filtered),
            'total_records_final': len(df_traced),
            'records_removed_by_market_hours': len(df) - len(df_filtered),
            'removal_percentage': ((len(df) - len(df_filtered)) / len(df) * 100) if len(df) > 0 else 0,
            'time_range': {
                'start': df_traced['time'].min().isoformat(),
                'end': df_traced['time'].max().isoformat()
            },
            'data_origin_distribution': df_traced['data_origin'].value_counts().to_dict(),
            'quality_score_stats': {
                'mean': float(df_traced['quality_score'].mean()),
                'min': float(df_traced['quality_score'].min()),
                'max': float(df_traced['quality_score'].max()),
                'std': float(df_traced['quality_score'].std())
            }
        }
        
        # Reporte completo
        complete_report = {
            'migration_info': {
                'file': input_path,
                'output_file': output_path,
                'instrument': instrument,
                'migration_timestamp': datetime.now().isoformat(),
                'migrated': True
            },
            'market_hours_analysis': market_analysis,
            'traceability_analysis': traceability_analysis,
            'final_validation': final_validation,
            'final_stats': final_stats
        }
        
        logger.info("Migración corregida completada exitosamente")
        logger.info(f"Archivo guardado: {output_path}")
        
        return complete_report
        
    except Exception as e:
        error_msg = f"Error en migración corregida: {str(e)}"
        logger.error(error_msg)
        
        return {
            'file': input_path,
            'migrated': False,
            'error': error_msg
        }


def analyze_migration_results(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analizar resultados de migración
    
    Args:
        report: Reporte de migración
    
    Returns:
        Dict con análisis de resultados
    """
    if not report.get('migrated', False):
        return {'error': 'Migración no exitosa'}
    
    # Extraer datos del reporte
    market_analysis = report.get('market_hours_analysis', {})
    traceability_analysis = report.get('traceability_analysis', {})
    final_stats = report.get('final_stats', {})
    
    # Análisis de calidad
    quality_analysis = {
        'market_hours_compliance': {
            'records_removed': final_stats.get('records_removed_by_market_hours', 0),
            'removal_percentage': final_stats.get('removal_percentage', 0),
            'compliance_rate': 100 - final_stats.get('removal_percentage', 0)
        },
        'data_origin_quality': {
            'native_data_percentage': 0,
            'aggregated_data_percentage': 0,
            'imputed_data_percentage': 0,
            'outside_market_percentage': 0
        },
        'overall_quality_score': final_stats.get('quality_score_stats', {}).get('mean', 0)
    }
    
    # Calcular distribución por categoría
    origin_dist = final_stats.get('data_origin_distribution', {})
    total_records = final_stats.get('total_records_final', 0)
    
    if total_records > 0:
        for origin, count in origin_dist.items():
            percentage = (count / total_records) * 100
            category = DATA_ORIGINS.get(origin, {}).get('category', 'unknown')
            
            if category == 'native':
                quality_analysis['data_origin_quality']['native_data_percentage'] += percentage
            elif category == 'aggregated':
                quality_analysis['data_origin_quality']['aggregated_data_percentage'] += percentage
            elif category == 'imputed':
                quality_analysis['data_origin_quality']['imputed_data_percentage'] += percentage
            elif category == 'outside_market':
                quality_analysis['data_origin_quality']['outside_market_percentage'] += percentage
    
    # Evaluación de calidad
    quality_score = quality_analysis['overall_quality_score']
    if quality_score >= 0.9:
        quality_grade = 'A'
    elif quality_score >= 0.8:
        quality_grade = 'B'
    elif quality_score >= 0.7:
        quality_grade = 'C'
    elif quality_score >= 0.6:
        quality_grade = 'D'
    else:
        quality_grade = 'F'
    
    quality_analysis['quality_grade'] = quality_grade
    
    return {
        'quality_analysis': quality_analysis,
        'summary': {
            'total_records_processed': final_stats.get('total_records_original', 0),
            'total_records_final': final_stats.get('total_records_final', 0),
            'data_quality_grade': quality_grade,
            'market_hours_compliance': f"{quality_analysis['market_hours_compliance']['compliance_rate']:.1f}%",
            'native_data_percentage': f"{quality_analysis['data_origin_quality']['native_data_percentage']:.1f}%"
        }
    }


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migración corregida con filtro de horarios y trazabilidad granular')
    parser.add_argument('--input', required=True, help='Archivo de entrada')
    parser.add_argument('--output', help='Archivo de salida (opcional)')
    parser.add_argument('--instrument', default='US500', choices=['US500', 'USDCOP'], 
                       help='Instrumento')
    parser.add_argument('--report', help='Ruta para guardar reporte detallado')
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=== MIGRACIÓN CORREGIDA CON FILTRO DE HORARIOS Y TRAZABILIDAD GRANULAR ===")
    print(f"Instrumento: {args.instrument}")
    print(f"Archivo de entrada: {args.input}")
    print()
    
    # Ejecutar migración
    result = correct_migration_pipeline(args.input, args.output, args.instrument)
    
    if result.get('migrated', False):
        print("✅ MIGRACIÓN EXITOSA")
        
        # Analizar resultados
        analysis = analyze_migration_results(result)
        
        print("\n=== RESUMEN DE RESULTADOS ===")
        summary = analysis.get('summary', {})
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n=== ANÁLISIS DE CALIDAD ===")
        quality = analysis.get('quality_analysis', {})
        print(f"  Grado de calidad: {quality.get('quality_grade', 'N/A')}")
        print(f"  Puntuación general: {quality.get('overall_quality_score', 0):.3f}")
        print(f"  Datos nativos: {quality.get('data_origin_quality', {}).get('native_data_percentage', 0):.1f}%")
        print(f"  Cumplimiento horarios: {quality.get('market_hours_compliance', {}).get('compliance_rate', 0):.1f}%")
        
        # Guardar reporte detallado
        if args.report:
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nReporte detallado guardado en: {args.report}")
        
    else:
        print("❌ MIGRACIÓN FALLÓ")
        print(f"Error: {result.get('error', 'Error desconocido')}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 