#!/usr/bin/env python3
"""
Script de validación y migración de trazabilidad de datos
Migra datos legacy a la nueva estructura estandarizada
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json

# Agregar el directorio del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import DATA_ORIGINS, DATA_ORIGIN_VALUES, DATA_VALIDATION_CONFIG
from utils.data_traceability import (
    DataTraceabilityManager, validate_dataframe_traceability,
    format_time_standard, get_data_origin_info, list_data_origins_by_category
)

logger = logging.getLogger(__name__)


class TraceabilityValidator:
    """Validador y migrador de trazabilidad de datos"""
    
    def __init__(self):
        """Inicializar validador"""
        self.traceability_manager = DataTraceabilityManager()
        self.validation_results = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'migrated_files': 0,
            'errors': [],
            'warnings': []
        }
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validar un archivo de datos
        
        Args:
            file_path: Ruta al archivo a validar
        
        Returns:
            Dict con resultados de validación
        """
        logger.info(f"Validando archivo: {file_path}")
        
        try:
            # Cargar datos
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, parse_dates=['time'])
            elif file_path.endswith('.feather'):
                df = pd.read_feather(file_path)
            else:
                return {
                    'file': file_path,
                    'valid': False,
                    'error': 'Formato de archivo no soportado'
                }
            
            # Validar trazabilidad
            traceability_report = validate_dataframe_traceability(df)
            
            # Verificar columnas requeridas
            missing_columns = []
            for col in DATA_VALIDATION_CONFIG['required_columns']:
                if col not in df.columns:
                    missing_columns.append(col)
            
            # Verificar formato de tiempo
            time_format_valid = True
            if 'time' in df.columns:
                try:
                    # Verificar que sea datetime
                    if not pd.api.types.is_datetime64_any_dtype(df['time']):
                        time_format_valid = False
                except Exception:
                    time_format_valid = False
            
            # Resultados de validación
            is_valid = (
                traceability_report['validation']['is_valid'] and
                len(missing_columns) == 0 and
                time_format_valid
            )
            
            result = {
                'file': file_path,
                'valid': is_valid,
                'total_records': len(df),
                'traceability_report': traceability_report,
                'missing_columns': missing_columns,
                'time_format_valid': time_format_valid,
                'data_origin_present': 'data_origin' in df.columns,
                'legacy_columns_present': any(col in df.columns for col in ['data_flag', 'source_timeframe', 'capture_method'])
            }
            
            if not is_valid:
                self.validation_results['invalid_files'] += 1
                self.validation_results['errors'].append({
                    'file': file_path,
                    'errors': traceability_report['validation']['errors'],
                    'missing_columns': missing_columns
                })
            else:
                self.validation_results['valid_files'] += 1
            
            self.validation_results['total_files'] += 1
            
            return result
            
        except Exception as e:
            error_msg = f"Error validando {file_path}: {str(e)}"
            logger.error(error_msg)
            
            self.validation_results['invalid_files'] += 1
            self.validation_results['errors'].append({
                'file': file_path,
                'error': error_msg
            })
            
            return {
                'file': file_path,
                'valid': False,
                'error': error_msg
            }
    
    def migrate_file(self, file_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Migrar un archivo legacy a la nueva estructura de trazabilidad
        
        Args:
            file_path: Ruta al archivo a migrar
            output_path: Ruta de salida (opcional)
        
        Returns:
            Dict con resultados de migración
        """
        logger.info(f"Migrando archivo: {file_path}")
        
        try:
            # Cargar datos
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, parse_dates=['time'])
            elif file_path.endswith('.feather'):
                df = pd.read_feather(file_path)
            else:
                return {
                    'file': file_path,
                    'migrated': False,
                    'error': 'Formato de archivo no soportado'
                }
            
            # Verificar si necesita migración
            if 'data_origin' in df.columns:
                return {
                    'file': file_path,
                    'migrated': False,
                    'message': 'Archivo ya tiene data_origin, no necesita migración'
                }
            
            # Migrar datos legacy
            df_migrated = self.traceability_manager.convert_legacy_data_flags(df)
            
            # Formatear tiempo
            if 'time' in df_migrated.columns:
                df_migrated['time'] = format_time_standard(df_migrated['time'])
            
            # Validar migración
            is_valid, errors = self.traceability_manager.validate_data_origin(df_migrated)
            
            if not is_valid:
                return {
                    'file': file_path,
                    'migrated': False,
                    'error': f'Migración falló: {errors}'
                }
            
            # Guardar archivo migrado
            if output_path is None:
                base_name = os.path.splitext(file_path)[0]
                output_path = f"{base_name}_migrated.parquet"
            
            df_migrated.to_parquet(output_path, index=False)
            
            # Generar reporte de migración
            traceability_report = self.traceability_manager.generate_traceability_report()
            
            self.validation_results['migrated_files'] += 1
            
            return {
                'file': file_path,
                'migrated': True,
                'output_path': output_path,
                'total_records': len(df_migrated),
                'traceability_report': traceability_report,
                'data_origin_distribution': df_migrated['data_origin'].value_counts().to_dict()
            }
            
        except Exception as e:
            error_msg = f"Error migrando {file_path}: {str(e)}"
            logger.error(error_msg)
            
            return {
                'file': file_path,
                'migrated': False,
                'error': error_msg
            }
    
    def validate_directory(self, directory_path: str, file_pattern: str = "*.parquet") -> Dict[str, Any]:
        """
        Validar todos los archivos en un directorio
        
        Args:
            directory_path: Ruta al directorio
            file_pattern: Patrón de archivos a validar
        
        Returns:
            Dict con resultados de validación
        """
        import glob
        
        logger.info(f"Validando directorio: {directory_path}")
        
        # Encontrar archivos
        pattern = os.path.join(directory_path, file_pattern)
        files = glob.glob(pattern)
        
        if not files:
            return {
                'directory': directory_path,
                'error': f'No se encontraron archivos con patrón: {file_pattern}'
            }
        
        # Validar cada archivo
        results = []
        for file_path in files:
            result = self.validate_file(file_path)
            results.append(result)
        
        return {
            'directory': directory_path,
            'total_files': len(files),
            'file_results': results,
            'summary': self.validation_results
        }
    
    def migrate_directory(self, directory_path: str, output_directory: str = None,
                         file_pattern: str = "*.parquet") -> Dict[str, Any]:
        """
        Migrar todos los archivos en un directorio
        
        Args:
            directory_path: Ruta al directorio de entrada
            output_directory: Ruta al directorio de salida (opcional)
            file_pattern: Patrón de archivos a migrar
        
        Returns:
            Dict con resultados de migración
        """
        import glob
        
        logger.info(f"Migrando directorio: {directory_path}")
        
        # Crear directorio de salida si no existe
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        # Encontrar archivos
        pattern = os.path.join(directory_path, file_pattern)
        files = glob.glob(pattern)
        
        if not files:
            return {
                'directory': directory_path,
                'error': f'No se encontraron archivos con patrón: {file_pattern}'
            }
        
        # Migrar cada archivo
        results = []
        for file_path in files:
            if output_directory:
                filename = os.path.basename(file_path)
                output_path = os.path.join(output_directory, filename)
            else:
                output_path = None
            
            result = self.migrate_file(file_path, output_path)
            results.append(result)
        
        return {
            'directory': directory_path,
            'output_directory': output_directory,
            'total_files': len(files),
            'file_results': results,
            'summary': self.validation_results
        }
    
    def generate_validation_report(self, output_path: str = None) -> Dict[str, Any]:
        """
        Generar reporte completo de validación
        
        Args:
            output_path: Ruta para guardar el reporte (opcional)
        
        Returns:
            Dict con reporte completo
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': self.validation_results,
            'data_origins_catalog': DATA_ORIGINS,
            'validation_config': DATA_VALIDATION_CONFIG,
            'recommendations': self._generate_recommendations()
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Reporte guardado en: {output_path}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generar recomendaciones basadas en los resultados de validación"""
        recommendations = []
        
        if self.validation_results['invalid_files'] > 0:
            recommendations.append(
                f"Se encontraron {self.validation_results['invalid_files']} archivos inválidos. "
                "Revisar y corregir los errores antes de usar los datos."
            )
        
        if self.validation_results['migrated_files'] > 0:
            recommendations.append(
                f"Se migraron {self.validation_results['migrated_files']} archivos legacy. "
                "Verificar que la migración fue correcta."
            )
        
        if self.validation_results['total_files'] == 0:
            recommendations.append("No se encontraron archivos para validar.")
        
        if not recommendations:
            recommendations.append("Todos los archivos son válidos y están actualizados.")
        
        return recommendations


def main():
    """Función principal para ejecutar validación y migración"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validar y migrar trazabilidad de datos')
    parser.add_argument('--input', required=True, help='Archivo o directorio de entrada')
    parser.add_argument('--output', help='Archivo o directorio de salida (para migración)')
    parser.add_argument('--action', choices=['validate', 'migrate', 'both'], 
                       default='validate', help='Acción a realizar')
    parser.add_argument('--pattern', default='*.parquet', 
                       help='Patrón de archivos (para directorios)')
    parser.add_argument('--report', help='Ruta para guardar reporte de validación')
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    validator = TraceabilityValidator()
    
    if os.path.isfile(args.input):
        # Procesar archivo individual
        if args.action in ['validate', 'both']:
            result = validator.validate_file(args.input)
            print(f"Validación: {result}")
        
        if args.action in ['migrate', 'both']:
            result = validator.migrate_file(args.input, args.output)
            print(f"Migración: {result}")
    
    elif os.path.isdir(args.input):
        # Procesar directorio
        if args.action in ['validate', 'both']:
            result = validator.validate_directory(args.input, args.pattern)
            print(f"Validación de directorio: {result}")
        
        if args.action in ['migrate', 'both']:
            result = validator.migrate_directory(args.input, args.output, args.pattern)
            print(f"Migración de directorio: {result}")
    
    else:
        print(f"Error: {args.input} no es un archivo o directorio válido")
        return 1
    
    # Generar reporte
    if args.report:
        report = validator.generate_validation_report(args.report)
        print(f"Reporte generado: {args.report}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 