"""
Pipeline principal de validación y calidad de dataset
"""
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import json

# Importar todos los componentes
from validation.market_validator import MarketValidator
from validation.data_integrity import DataIntegrityValidator
from validation.temporal_validator import TemporalValidator
from validation.quality_checklist import QualityChecklist
from features.feature_selector import FeatureSelector
from preprocessing.data_normalizer import DataNormalizer
from preprocessing.data_cleaner import DataCleaner
from data.splitters import TemporalSplitter
from reports.quality_reporter import QualityReporter
from reports.gap_reporter import GapReporter

# Importar nuevos componentes de calidad avanzada
from quality_tracker import DataQualityTracker
from analysis.data_analyzer import analyze_data_by_period, analyze_data_parallel_dask, generate_analysis_summary
from reports.advanced_reports import (
    generate_json_report_enhanced,
    generate_quality_markdown_report,
    generate_markdown_report_enhanced
)

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """Pipeline completo de validación y preparación de datos"""
    
    def __init__(self, config: Dict):
        """
        Inicializar pipeline con configuración
        
        Args:
            config: Diccionario de configuración
        """
        self.config = config
        
        # Inicializar componentes
        self.market_validator = MarketValidator(config.get('market', {}))
        self.integrity_validator = DataIntegrityValidator(config.get('integrity', {}))
        self.temporal_validator = TemporalValidator(config.get('temporal', {}))
        self.quality_checklist = QualityChecklist(config.get('quality_checklist', {}))
        self.feature_selector = FeatureSelector(config.get('features', {}))
        self.normalizer = DataNormalizer(
            method=config.get('normalization', {}).get('method', 'feature_specific'),
            config=config.get('normalization', {})
        )
        self.cleaner = DataCleaner(config.get('cleaning', {}))
        self.splitter = TemporalSplitter(config.get('splitting', {}))
        self.reporter = QualityReporter(config.get('reporting', {}))
        
        # Inicializar sistema de tracking de calidad avanzado
        self.quality_tracker = DataQualityTracker()
        
        # Estado del pipeline
        self.validation_results = {}
        self.processed_data = None
        
    def run(self, input_file: str, output_dir: str = 'data/validated/') -> Dict:
        """
        Ejecutar pipeline completo de validación
        
        Args:
            input_file: Archivo CSV de entrada
            output_dir: Directorio de salida
            
        Returns:
            Dict: Resultados del pipeline
        """
        logger.info("="*80)
        logger.info("INICIANDO PIPELINE DE VALIDACIÓN Y CALIDAD")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        try:
            # 1. Cargar datos
            df = self._load_data(input_file)
            self.validation_results['dataset_info'] = {
                'total_records': len(df),
                'total_features': len(df.columns),
                'date_range': {
                    'start': str(df['time'].min()),
                    'end': str(df['time'].max())
                }
            }

            # 2. Validaciones de mercado avanzadas (incluye M5, flags, horarios)
            logger.info("Ejecutando validaciones de mercado avanzadas...")
            try:
                market_validations = self.market_validator.validate_all(df)
                self.validation_results['market_validations'] = market_validations
            except Exception as e:
                logger.warning(f"Error en validaciones de mercado: {e}")
                market_validations = {'overall_valid': False, 'error': str(e)}
                self.validation_results['market_validations'] = market_validations
            
            # 3. Validaciones de integridad
            try:
                self._run_integrity_validations(df)
            except Exception as e:
                logger.warning(f"Error en validaciones de integridad: {e}")
                self.validation_results['integrity_validations'] = {'error': str(e)}
            
            # 4. Validaciones temporales
            try:
                self._run_temporal_validations(df)
            except Exception as e:
                logger.warning(f"Error en validaciones temporales: {e}")
                self.validation_results['temporal_validations'] = {'error': str(e)}
            
            # 5. Limpieza de datos
            try:
                df_clean = self._clean_data(df)
            except Exception as e:
                logger.warning(f"Error en limpieza de datos: {e}")
                df_clean = df.copy()  # Usar datos originales si falla la limpieza
            
            # 6. Selección automática de features con heurística
            logger.info("Ejecutando selección automática de features...")
            try:
                df_selected, selected_features, feature_importances = self.feature_selector.select_features(
                    df_clean, target_col='log_return' if 'log_return' in df_clean.columns else None
                )
                self.validation_results['feature_selection'] = {
                    'selected_features': selected_features,
                    'total_selected': len(selected_features),
                    'feature_importances': feature_importances
                }
            except Exception as e:
                logger.warning(f"Error en selección de features: {e}")
                df_selected = df_clean
                selected_features = list(df_clean.columns)
                self.validation_results['feature_selection'] = {
                    'selected_features': selected_features,
                    'total_selected': len(selected_features),
                    'feature_importances': {},
                    'error': str(e)
                }
            
            # 7. Normalización específica por tipo de feature
            logger.info("Aplicando normalización específica por tipo de feature...")
            try:
                df_normalized = self.normalizer.normalize(df_selected)
                self.validation_results['normalization'] = self.normalizer.get_normalization_info()
            except Exception as e:
                logger.warning(f"Error en normalización: {e}")
                df_normalized = df_selected
                self.validation_results['normalization'] = {'error': str(e)}
            
            # 8. Aplicar trazabilidad estandarizada
            logger.info("Aplicando trazabilidad estandarizada...")
            try:
                # Importar gestor de trazabilidad
                import sys
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_capture'))
                from utils.data_traceability import DataTraceabilityManager
                
                traceability_manager = DataTraceabilityManager()
                
                # Si no tiene data_origin, aplicar trazabilidad
                if 'data_origin' not in df_normalized.columns:
                    logger.info("Aplicando trazabilidad estandarizada a todos los registros...")
                    
                    # Determinar el origen más probable basado en el procesamiento
                    # Como es un archivo validado, la mayoría debería ser M5_NATIVO
                    df_normalized['data_origin'] = 'M5_NATIVO'
                    df_normalized['quality_score'] = 1.0
                    
                    # Para registros imputados, cambiar el origen
                    if 'imputed' in df_normalized.columns:
                        imputed_mask = df_normalized['imputed'] == True
                        df_normalized.loc[imputed_mask, 'data_origin'] = 'M5_IMPUTADO_BROWNIAN'
                        df_normalized.loc[imputed_mask, 'quality_score'] = 0.8
                    
                    logger.info(f"Trazabilidad aplicada: {df_normalized['data_origin'].value_counts().to_dict()}")
                else:
                    logger.info("DataFrame ya tiene trazabilidad aplicada")
                    
            except Exception as e:
                logger.warning(f"Error aplicando trazabilidad: {e}")
                logger.info("Continuando sin trazabilidad...")
            
            # 9. Aplicar tracking de calidad a todos los records
            logger.info("Aplicando tracking de calidad avanzado...")
            try:
                df_normalized.apply(self.quality_tracker.track_record, axis=1)
            except Exception as e:
                logger.warning(f"Error en tracking de calidad: {e}")
            
            # 9. Análisis avanzado de datos
            logger.info("Realizando análisis avanzado de datos...")
            try:
                if len(df_normalized) > 100000:
                    analysis = analyze_data_parallel_dask(df_normalized)
                else:
                    analysis = analyze_data_by_period(df_normalized)
            except Exception as e:
                logger.warning(f"Error en análisis de datos: {e}")
                analysis = {
                    'summary': {
                        'total_records': len(df_normalized),
                        'date_range': {
                            'start': str(df_normalized['time'].min()),
                            'end': str(df_normalized['time'].max())
                        },
                        'error': str(e)
                    }
                }
            
            # 10. Generar reporte de calidad avanzado
            logger.info("Generando reporte de calidad avanzado...")
            try:
                quality_report = self.quality_tracker.generate_quality_report()
            except Exception as e:
                logger.warning(f"Error generando reporte de calidad: {e}")
                quality_report = {
                    'summary': {
                        'total_records': len(df_normalized),
                        'error': str(e)
                    }
                }
            
            # 11. Crear splits temporales
            logger.info("Creando splits temporales...")
            try:
                splits = self._create_splits(df_normalized)
            except Exception as e:
                logger.warning(f"Error creando splits: {e}")
                splits = []
            
            # 12. Generar checklist de calidad avanzado
            logger.info("Generando checklist de calidad avanzado...")
            if len(splits) >= 1:
                try:
                    train_df, val_df, _ = splits[0]
                    checklist = self.quality_checklist.generate_checklist(df_normalized, train_df, val_df)
                    self.validation_results['quality_checklist'] = checklist
                except Exception as e:
                    logger.error(f"Error generando checklist de calidad: {e}")
                    checklist = {'overall_score': 0, 'error': str(e)}
            else:
                logger.warning("No se pudieron crear splits para el checklist")
                checklist = {'overall_score': 0, 'error': 'No splits available'}
            
            # 13. Guardar resultados
            try:
                self._save_results(df_normalized, splits, output_dir)
            except Exception as e:
                logger.warning(f"Error guardando resultados: {e}")
            
            # 14. Guardar información de selección de features
            try:
                self.feature_selector.save_selection_info(output_dir)
            except Exception as e:
                logger.warning(f"Error guardando información de features: {e}")
            
            # 15. Guardar checklist de calidad
            if checklist:
                try:
                    self.quality_checklist.save_checklist_report(checklist, output_dir)
                except Exception as e:
                    logger.warning(f"Error guardando checklist: {e}")
            
            # 16. Generar reportes avanzados
            try:
                self._generate_advanced_reports(df_normalized, analysis, quality_report, output_dir)
            except Exception as e:
                logger.warning(f"Error generando reportes avanzados: {e}")
            
            # 17. Generar reporte básico
            try:
                report = self._generate_report(df_normalized)
            except Exception as e:
                logger.warning(f"Error generando reporte básico: {e}")
                report = {}
            
            # Tiempo total
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # Resumen final
            pipeline_results = {
                'status': 'SUCCESS',
                'elapsed_time_seconds': elapsed_time,
                'validation_results': self.validation_results,
                'output_directory': output_dir,
                'splits_created': len(splits),
                'quality_score': checklist.get('overall_score', 0) if checklist else 0,
                'market_validations_passed': market_validations.get('overall_valid', False),
                'features_selected': len(selected_features),
                'normalization_method': self.normalizer.method if hasattr(self.normalizer, 'method') else 'unknown'
            }
            
            logger.info("="*80)
            logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
            logger.info(f"Tiempo total: {elapsed_time:.2f} segundos")
            logger.info(f"Score de calidad: {pipeline_results['quality_score']:.1f}/100")
            logger.info(f"Validaciones de mercado: {'PASÓ' if pipeline_results['market_validations_passed'] else 'FALLÓ'}")
            logger.info(f"Features seleccionados: {pipeline_results['features_selected']}")
            logger.info("="*80)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Error en pipeline: {str(e)}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'elapsed_time_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    def _load_data(self, filepath: str) -> pd.DataFrame:
        """Cargar datos del archivo (parquet, feather, csv)"""
        logger.info(f"Cargando datos de {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.parquet':
            df = pd.read_parquet(filepath)
        elif ext == '.feather':
            df = pd.read_feather(filepath)
        elif ext == '.csv':
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Formato de archivo no soportado: {ext}")
        
        # Convertir columna de tiempo
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        logger.info(f"Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
        return df
    
    def _run_integrity_validations(self, df: pd.DataFrame):
        """Ejecutar validaciones de integridad"""
        logger.info("Ejecutando validaciones de integridad...")
        
        integrity_results = self.integrity_validator.validate_all(df)
        self.validation_results['integrity_validations'] = integrity_results
        
        # Verificar si pasaron todas las validaciones
        all_passed = all(result.get('passed', False) for result in integrity_results.values())
        if not all_passed:
            logger.warning("Algunas validaciones de integridad fallaron")
    
    def _run_market_validations(self, df: pd.DataFrame):
        """Ejecutar validaciones de mercado (ya se hace en el método principal)"""
        pass  # Ya se ejecuta en el método principal
    
    def _run_temporal_validations(self, df: pd.DataFrame):
        """Ejecutar validaciones temporales"""
        logger.info("Ejecutando validaciones temporales...")
        
        temporal_results = self.temporal_validator.validate_all(df)
        self.validation_results['temporal_validations'] = temporal_results
        
        # Verificar si pasaron todas las validaciones
        all_passed = all(result.get('passed', False) for result in temporal_results.values())
        if not all_passed:
            logger.warning("Algunas validaciones temporales fallaron")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpiar datos"""
        logger.info("Limpiando datos...")
        
        df_clean = self.cleaner.clean(df)
        logger.info(f"Datos limpios: {len(df_clean)} registros")
        
        return df_clean
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Seleccionar features (ya se hace en el método principal)"""
        pass  # Ya se ejecuta en el método principal
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizar datos (ya se hace en el método principal)"""
        pass  # Ya se ejecuta en el método principal
    
    def _create_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]]:
        """Crear splits de entrenamiento/validación y prueba con rangos fijos"""
        logger.info("Creando splits fijos por año...")

        train_df, val_df, test_df, split_info = self.splitter.create_fixed_year_splits(df)

        self.validation_results['data_splits'] = {
            'train_range': split_info['train_date_range'],
            'val_range': split_info['val_date_range'],
            'test_range': split_info['test_date_range'],
        }

        splits = [(train_df, val_df, test_df, split_info)]

        logger.info(f"Splits creados: {len(splits)}")
        return splits
    
    def _save_results(self, df: pd.DataFrame, splits: List, output_dir: str):
        """Guardar resultados del pipeline"""
        logger.info(f"Guardando resultados en {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar dataset completo
        df.to_csv(os.path.join(output_dir, 'validated_dataset.csv'), index=False)
        
        # Guardar splits
        for i, (train_df, val_df, test_df, split_info) in enumerate(splits):
            split_dir = os.path.join(output_dir, f'split_{i+1}')
            os.makedirs(split_dir, exist_ok=True)

            train_df.to_csv(os.path.join(split_dir, 'train.csv'), index=False)
            val_df.to_csv(os.path.join(split_dir, 'validation.csv'), index=False)
            test_df.to_csv(os.path.join(split_dir, 'test.csv'), index=False)
            
            # Guardar información del split
            with open(os.path.join(split_dir, 'split_info.json'), 'w') as f:
                json.dump(split_info, f, indent=2)
        
        # Guardar información general
        with open(os.path.join(output_dir, 'pipeline_info.json'), 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info("Resultados guardados exitosamente")
    
    def _generate_report(self, df: pd.DataFrame) -> Dict:
        """Generar reporte básico"""
        logger.info("Generando reporte básico...")
        
        report = self.reporter.generate_quality_report(self.validation_results)
        return report
    
    def _generate_advanced_reports(self, df: pd.DataFrame, analysis: Dict, quality_report: Dict, output_dir: str):
        """Generar reportes avanzados"""
        logger.info("Generando reportes avanzados...")
        gap_reporter = GapReporter(log_dir=output_dir)

        # Guardar métricas y registros adicionales
        try:
            # Cobertura diaria
            coverage_metrics = gap_reporter.daily_coverage_metrics(df)
            gap_reporter.save_daily_coverage(coverage_metrics)

            # Gaps y duplicados
            gaps_info = gap_reporter.detect_gaps(df)
            gap_reporter.save_gap_log(gaps_info['gaps'])
            gap_reporter.save_gaps(gaps_info['gaps'])

            duplicates_df = df[df.duplicated(subset=['time'], keep=False)]
            gap_reporter.save_duplicates(duplicates_df)

            # Outliers sencillos por IQR
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_records = []
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (df[col] < lower) | (df[col] > upper)
                if mask.any():
                    tmp = df.loc[mask, ['time', col]].copy()
                    tmp['feature'] = col
                    tmp.rename(columns={col: 'value'}, inplace=True)
                    outlier_records.append(tmp)
            outliers_df = pd.concat(outlier_records, ignore_index=True) if outlier_records else pd.DataFrame()
            gap_reporter.save_outliers(outliers_df)

            nan_summary = df.isna().sum().to_dict()
            nan_summary['total_nan'] = int(sum(nan_summary.values()))
            gap_reporter.save_nan_summary(nan_summary)
        except Exception as e:
            logger.warning(f"Error generando archivos de gap reporter: {e}")

        try:
            # Importar funciones de reportes
            from reports.advanced_reports import (
                generate_json_report_enhanced,
                generate_markdown_report_enhanced,
                generate_quality_markdown_report
            )
            
            # Generar reporte JSON mejorado
            json_path = os.path.join(output_dir, 'enhanced_report.json')
            generate_json_report_enhanced(df, analysis, {}, quality_report, 'US500', json_path)
            
            # Generar reporte Markdown mejorado
            md_path = os.path.join(output_dir, 'enhanced_report.md')
            generate_markdown_report_enhanced(df, analysis, {}, quality_report, 'US500', md_path)
            
            # Generar reporte de calidad Markdown
            quality_md_path = os.path.join(output_dir, 'quality_report.md')
            generate_quality_markdown_report(quality_report, quality_md_path)
            
            logger.info("Reportes avanzados generados exitosamente")
            
        except ImportError as e:
            logger.warning(f"No se pudieron importar funciones de reportes avanzados: {e}")
            logger.info("Generando reportes básicos...")
            
            # Fallback a reportes básicos
            try:
                # Generar reporte JSON básico
                json_path = os.path.join(output_dir, 'basic_report.json')
                basic_report = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': 'US500',
                    'total_records': len(df),
                    'columns': list(df.columns),
                    'date_range': {
                        'start': str(df['time'].min()),
                        'end': str(df['time'].max())
                    },
                    'quality_summary': quality_report.get('summary', {}),
                    'analysis_summary': analysis.get('summary', {})
                }
                
                with open(json_path, 'w') as f:
                    json.dump(basic_report, f, indent=2)
                
                logger.info(f"Reporte básico guardado: {json_path}")
                
            except Exception as e2:
                logger.error(f"Error generando reporte básico: {e2}")
                
        except Exception as e:
            logger.error(f"Error generando reportes avanzados: {e}")
            logger.info("Continuando sin reportes avanzados...")
    
    def validate_only(self, input_file: str) -> Dict:
        """
        Ejecutar solo validaciones sin procesamiento
        
        Args:
            input_file: Archivo CSV de entrada
            
        Returns:
            Dict: Resultados de validación
        """
        logger.info("Ejecutando solo validaciones...")
        
        try:
            df = self._load_data(input_file)
            
            # Ejecutar todas las validaciones
            market_validations = self.market_validator.validate_all(df)
            integrity_validations = self.integrity_validator.validate_all(df)
            temporal_validations = self.temporal_validator.validate_all(df)
            
            validation_results = {
                'market_validations': market_validations,
                'integrity_validations': integrity_validations,
                'temporal_validations': temporal_validations,
                'overall_passed': (
                    market_validations.get('overall_valid', False) and
                    all(result.get('passed', False) for result in integrity_validations.values()) and
                    all(result.get('passed', False) for result in temporal_validations.values())
                )
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error en validaciones: {str(e)}")
            return {'error': str(e)}