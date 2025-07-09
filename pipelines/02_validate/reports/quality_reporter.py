"""
Generación de reportes de calidad para datasets de trading
"""
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

logger = logging.getLogger(__name__)


class QualityReporter:
    """Generador de reportes de calidad para validación de datasets"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializar reporter
        
        Args:
            config: Configuración del reporter
        """
        self.config = config or {}
        self.report_timestamp = datetime.now()
        self.validation_results = {}
        
    def generate_quality_report(self, validation_results: Dict) -> Dict:
        """
        Generar reporte completo de calidad
        
        Args:
            validation_results: Resultados de todas las validaciones
            
        Returns:
            Dict: Reporte completo
        """
        logger.info("Generando reporte de calidad...")
        
        report = {
            'report_metadata': {
                'timestamp': self.report_timestamp.isoformat(),
                'version': '1.0',
                'generator': 'QualityReporter'
            },
            'dataset_summary': self._generate_dataset_summary(validation_results),
            'validation_summary': self._generate_validation_summary(validation_results),
            'detailed_results': validation_results,
            'quality_score': self._calculate_quality_score(validation_results),
            'recommendations': self._generate_recommendations(validation_results),
            'checklist': self._generate_checklist(validation_results)
        }
        
        logger.info(f"Reporte generado. Score de calidad: {report['quality_score']['overall_score']:.1f}/100")
        
        return report
    
    def _generate_dataset_summary(self, results: Dict) -> Dict:
        """Generar resumen del dataset"""
        
        summary = {
            'total_records': results.get('dataset_info', {}).get('total_records', 0),
            'total_features': results.get('dataset_info', {}).get('total_features', 0),
            'date_range': results.get('dataset_info', {}).get('date_range', {}),
            'data_composition': {
                'real_data_percentage': results.get('data_integrity', {})
                    .get('data_flags', {}).get('real_percentage', 0),
                'imputed_data_percentage': results.get('data_integrity', {})
                    .get('data_flags', {}).get('imputed_percentage', 0)
            }
        }
        
        # Agregar información temporal
        if 'temporal' in results:
            temporal_coverage = results['temporal'].get('temporal_coverage', {})
            summary['temporal_info'] = {
                'trading_days': temporal_coverage.get('unique_trading_days', 0),
                'coverage_quality': temporal_coverage.get('daily_coverage', {})
            }
        
        # Agregar información de gaps
        if 'gap_report' in results:
            summary['gap_report'] = {
                'total_slots': results['gap_report'].get('total_slots', 0),
                'missing_slots': results['gap_report'].get('missing_slots', 0),
                'coverage_pct': results['gap_report'].get('coverage_pct', 0),
                'gaps_log': results['gap_report'].get('gaps_log', ''),
                'daily_coverage_log': results['gap_report'].get('daily_coverage_log', '')
            }
        
        return summary
    
    def _generate_validation_summary(self, results: Dict) -> Dict:
        """Generar resumen de validaciones"""
        
        summary = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'warnings': 0,
            'validation_details': {}
        }
        
        # Categorías de validación
        validation_categories = ['market_hours', 'data_integrity', 'temporal', 'features']
        
        for category in validation_categories:
            if category not in results:
                continue
                
            category_results = results[category]
            passed = 0
            failed = 0
            
            # Buscar resultados de validación en la estructura
            for key, value in category_results.items():
                if isinstance(value, dict) and 'validation_passed' in value:
                    summary['total_validations'] += 1
                    if value['validation_passed']:
                        passed += 1
                        summary['passed_validations'] += 1
                    else:
                        failed += 1
                        summary['failed_validations'] += 1
            
            summary['validation_details'][category] = {
                'passed': passed,
                'failed': failed,
                'status': 'PASSED' if failed == 0 else 'FAILED'
            }
        
        summary['overall_status'] = 'PASSED' if summary['failed_validations'] == 0 else 'FAILED'
        
        return summary
    
    def _calculate_quality_score(self, results: Dict) -> Dict:
        """
        Calcular score de calidad del dataset
        
        Args:
            results: Resultados de validaciones
            
        Returns:
            Dict: Scores de calidad
        """
        scores = {}
        weights = {
            'data_completeness': 0.25,
            'temporal_consistency': 0.25,
            'data_integrity': 0.25,
            'market_compliance': 0.25
        }
        
        # Data completeness
        nan_percentage = results.get('cleaning_summary', {}).get('quality_metrics', {}).get('nan_percentage', 0)
        scores['data_completeness'] = max(0, 100 - nan_percentage * 10)
        
        # Temporal consistency
        m5_consistency = results.get('temporal', {}).get('m5_consistency', {}).get('consistency_percentage', 0)
        scores['temporal_consistency'] = m5_consistency
        
        # Data integrity
        real_data_pct = results.get('data_integrity', {}).get('data_flags', {}).get('real_percentage', 0)
        scores['data_integrity'] = real_data_pct
        
        # Market compliance
        market_hours_pct = results.get('market_hours', {}).get('market_hours', {}).get('market_hours_percentage', 0)
        scores['market_compliance'] = market_hours_pct
        
        # Overall score
        overall_score = sum(scores[k] * weights[k] for k in scores)
        
        return {
            'overall_score': overall_score,
            'component_scores': scores,
            'weights': weights,
            'grade': self._get_quality_grade(overall_score)
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Obtener grado de calidad basado en score"""
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 85:
            return 'B+'
        elif score >= 80:
            return 'B'
        elif score >= 75:
            return 'C+'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, results: Dict) -> List[Dict]:
        """Generar recomendaciones basadas en los resultados"""
        
        recommendations = []
        
        # Verificar datos imputados
        imputed_pct = results.get('data_integrity', {}).get('data_flags', {}).get('imputed_percentage', 0)
        if imputed_pct > 20:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'DATA_QUALITY',
                'issue': f'Alto porcentaje de datos imputados ({imputed_pct:.1f}%)',
                'recommendation': 'Considerar obtener más datos reales o revisar el proceso de imputación'
            })
        
        # Verificar consistencia M5
        m5_consistency = results.get('temporal', {}).get('m5_consistency', {}).get('consistency_percentage', 0)
        if m5_consistency < 95:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'TEMPORAL',
                'issue': f'Consistencia M5 baja ({m5_consistency:.1f}%)',
                'recommendation': 'Revisar gaps temporales y considerar re-muestreo uniforme'
            })
        
        # Verificar horarios de mercado
        outside_market = results.get('market_hours', {}).get('market_hours', {}).get('outside_market_hours', 0)
        if outside_market > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'MARKET_HOURS',
                'issue': f'{outside_market} registros fuera de horarios de mercado',
                'recommendation': 'Filtrar datos para incluir solo horarios oficiales de trading'
            })
        
        # Verificar features con alta correlación
        high_corr = results.get('feature_selection', {}).get('high_correlation_pairs', [])
        if len(high_corr) > 5:
            recommendations.append({
                'priority': 'LOW',
                'category': 'FEATURES',
                'issue': f'{len(high_corr)} pares de features altamente correlacionados',
                'recommendation': 'Considerar eliminar features redundantes'
            })
        
        return recommendations
    
    def generate_checklist(self, df: pd.DataFrame) -> Dict:
        """
        Generar checklist de calidad detallado
        
        Args:
            df: DataFrame validado
            
        Returns:
            Dict: Checklist completo
        """
        logger.info("Generando checklist de calidad...")
        
        checklist = {
            'basic_integrity': self._check_basic_integrity(df),
            'temporal_quality': self._check_temporal_quality(df),
            'feature_quality': self._check_feature_quality(df),
            'statistical_quality': self._check_statistical_quality(df),
            'trading_specific': self._check_trading_specific(df)
        }
        
        # Calcular resumen
        total_checks = sum(len(v) for v in checklist.values())
        passed_checks = sum(
            sum(1 for check in v.values() if check.get('status') == 'PASS')
            for v in checklist.values()
        )
        
        checklist['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'success_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0
        }
        
        return checklist
    
    def _check_basic_integrity(self, df: pd.DataFrame) -> Dict:
        """Verificaciones básicas de integridad"""
        
        checks = {}
        
        # Duplicados
        duplicates = df.duplicated().sum()
        checks['no_duplicates'] = {
            'status': 'PASS' if duplicates == 0 else 'FAIL',
            'value': duplicates,
            'message': f'{duplicates} filas duplicadas'
        }
        
        # NaN
        total_nans = df.isna().sum().sum()
        checks['no_missing_values'] = {
            'status': 'PASS' if total_nans == 0 else 'WARN',
            'value': total_nans,
            'message': f'{total_nans} valores faltantes'
        }
        
        # Tipos de datos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        checks['numeric_columns'] = {
            'status': 'PASS' if len(numeric_cols) > 10 else 'WARN',
            'value': len(numeric_cols),
            'message': f'{len(numeric_cols)} columnas numéricas'
        }
        
        return checks
    
    def _check_temporal_quality(self, df: pd.DataFrame) -> Dict:
        """Verificaciones de calidad temporal"""
        
        checks = {}
        
        if 'time' in df.columns:
            # Orden temporal
            is_sorted = df['time'].is_monotonic_increasing
            checks['temporal_order'] = {
                'status': 'PASS' if is_sorted else 'FAIL',
                'value': is_sorted,
                'message': 'Datos ordenados temporalmente' if is_sorted else 'Datos desordenados'
            }
            
            # Cobertura temporal
            date_range = (df['time'].max() - df['time'].min()).days
            checks['temporal_coverage'] = {
                'status': 'PASS' if date_range > 30 else 'WARN',
                'value': date_range,
                'message': f'{date_range} días de datos'
            }
        
        return checks
    
    def _check_feature_quality(self, df: pd.DataFrame) -> Dict:
        """Verificaciones de calidad de features"""
        
        checks = {}
        
        # Varianza cero
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zero_variance = [col for col in numeric_cols if df[col].var() < 1e-10]
        
        checks['no_zero_variance'] = {
            'status': 'PASS' if len(zero_variance) == 0 else 'WARN',
            'value': len(zero_variance),
            'message': f'{len(zero_variance)} features con varianza cero'
        }
        
        # Features esenciales
        essential_features = ['open', 'high', 'low', 'close', 'volume']
        missing_essential = [f for f in essential_features if f not in df.columns]
        
        checks['essential_features'] = {
            'status': 'PASS' if len(missing_essential) == 0 else 'FAIL',
            'value': len(missing_essential),
            'message': f'{len(missing_essential)} features esenciales faltantes'
        }
        
        return checks
    
    def _check_statistical_quality(self, df: pd.DataFrame) -> Dict:
        """Verificaciones estadísticas"""
        
        checks = {}
        
        # Outliers extremos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        extreme_outliers = 0
        
        for col in numeric_cols:
            if df[col].std() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                extreme_outliers += (z_scores > 5).sum()
        
        checks['extreme_outliers'] = {
            'status': 'PASS' if extreme_outliers < len(df) * 0.01 else 'WARN',
            'value': extreme_outliers,
            'message': f'{extreme_outliers} valores extremos (>5 std)'
        }
        
        return checks
    
    def _check_trading_specific(self, df: pd.DataFrame) -> Dict:
        """Verificaciones específicas de trading"""
        
        checks = {}
        
        # OHLCV coherencia
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_valid = ((df['high'] >= df[['open', 'close']].max(axis=1)) & 
                         (df['low'] <= df[['open', 'close']].min(axis=1)) & 
                         (df['low'] <= df['high'])).all()
            
            checks['ohlcv_integrity'] = {
                'status': 'PASS' if ohlc_valid else 'FAIL',
                'value': ohlc_valid,
                'message': 'OHLCV coherente' if ohlc_valid else 'OHLCV incoherente'
            }
        
        # Volumen no negativo
        if 'volume' in df.columns or 'tick_volume' in df.columns:
            vol_col = 'volume' if 'volume' in df.columns else 'tick_volume'
            negative_volume = (df[vol_col] < 0).sum()
            
            checks['non_negative_volume'] = {
                'status': 'PASS' if negative_volume == 0 else 'FAIL',
                'value': negative_volume,
                'message': f'{negative_volume} registros con volumen negativo'
            }
        
        return checks
    
    def export_markdown_report(self, report: Dict, filepath: str):
        """
        Exportar reporte en formato Markdown
        
        Args:
            report: Reporte generado
            filepath: Ruta del archivo
        """
        logger.info(f"Exportando reporte Markdown a {filepath}")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            # Header
            f.write("# Reporte de Calidad de Dataset\n\n")
            f.write(f"**Generado:** {report['report_metadata']['timestamp']}\n\n")
            
            # Quality Score
            quality_score = report['quality_score']
            f.write(f"## Score de Calidad: {quality_score['overall_score']:.1f}/100 ({quality_score['grade']})\n\n")
            
            # Component Scores
            f.write("### Scores por Componente\n\n")
            for component, score in quality_score['component_scores'].items():
                f.write(f"- **{component.replace('_', ' ').title()}:** {score:.1f}/100\n")
            f.write("\n")
            
            # Dataset Summary
            f.write("## Resumen del Dataset\n\n")
            summary = report['dataset_summary']
            f.write(f"- **Total de registros:** {summary['total_records']:,}\n")
            f.write(f"- **Total de features:** {summary['total_features']}\n")
            f.write(f"- **Rango temporal:** {summary['date_range'].get('start', 'N/A')} a {summary['date_range'].get('end', 'N/A')}\n")
            f.write(f"- **Datos reales:** {summary['data_composition']['real_data_percentage']:.1f}%\n")
            f.write(f"- **Datos imputados:** {summary['data_composition']['imputed_data_percentage']:.1f}%\n\n")
            
            # Validation Summary
            f.write("## Resumen de Validaciones\n\n")
            val_summary = report['validation_summary']
            f.write(f"- **Total:** {val_summary['total_validations']}\n")
            f.write(f"- **Aprobadas:** {val_summary['passed_validations']} ✓\n")
            f.write(f"- **Fallidas:** {val_summary['failed_validations']} ✗\n")
            f.write(f"- **Estado general:** {val_summary['overall_status']}\n\n")
            
            # Recommendations
            if report.get('recommendations'):
                f.write("## Recomendaciones\n\n")
                for rec in report['recommendations']:
                    f.write(f"### {rec['priority']} - {rec['category']}\n")
                    f.write(f"**Problema:** {rec['issue']}\n\n")
                    f.write(f"**Recomendación:** {rec['recommendation']}\n\n")
            
            # Checklist Summary
            if 'checklist' in report:
                f.write("## Checklist de Calidad\n\n")
                checklist_summary = report['checklist'].get('summary', {})
                f.write(f"- **Verificaciones totales:** {checklist_summary.get('total_checks', 0)}\n")
                f.write(f"- **Aprobadas:** {checklist_summary.get('passed_checks', 0)}\n")
                f.write(f"- **Tasa de éxito:** {checklist_summary.get('success_rate', 0):.1f}%\n")
        
        logger.info("Reporte Markdown exportado exitosamente")
    
    def export_json_report(self, report: Dict, filepath: str):
        """
        Exportar reporte en formato JSON
        
        Args:
            report: Reporte generado
            filepath: Ruta del archivo
        """
        logger.info(f"Exportando reporte JSON a {filepath}")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convertir datetime a string si es necesario
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=serialize)
        
        logger.info("Reporte JSON exportado exitosamente")