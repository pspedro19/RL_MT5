"""
Sistema de Tracking de Calidad Avanzado para Pipeline 02
Basado en el código de referencia data_extraction_02_simple.py
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Any
import json
import os

logger = logging.getLogger(__name__)


class DataQualityTracker:
    """Sistema avanzado de tracking de calidad de datos"""
    
    def __init__(self):
        """Inicializar tracker de calidad"""
        # Tracking de completitud diaria
        self.daily_completeness = defaultdict(lambda: {
            'expected': 0, 'captured': 0, 'sources': defaultdict(int)
        })
        
        # Tracking de completitud horaria
        self.hourly_completeness = defaultdict(lambda: defaultdict(lambda: {
            'expected': 0, 'captured': 0, 'sources': defaultdict(int)
        }))
        
        # Tracking de gaps
        self.gap_tracking = defaultdict(list)
        
        # Tracking de imputaciones
        self.imputation_tracking = defaultdict(list)
        
        # Records de peor calidad
        self.worst_quality_records = []
        
        # Tiempos de ejecución
        self.execution_times = {}
        
        # Información del broker
        self.broker_info = {}
        
        # Estadísticas de fuentes
        self.source_stats = defaultdict(lambda: {
            'total_bars': 0, 'total_days': 0, 'avg_quality': 0.0
        })
        
    def track_record(self, record: pd.Series):
        """Trackear un record individual"""
        try:
            # Extraer información temporal
            timestamp = pd.to_datetime(record['time'])
            date = timestamp.date()
            hour = timestamp.replace(minute=0, second=0, microsecond=0)
            
            # Determinar fuente de datos
            source = self._determine_data_source(record)
            
            # Actualizar completitud diaria
            self.daily_completeness[date]['captured'] += 1
            self.daily_completeness[date]['sources'][source] += 1
            
            # Actualizar completitud horaria
            self.hourly_completeness[date][hour]['captured'] += 1
            self.hourly_completeness[date][hour]['sources'][source] += 1
            
            # Trackear calidad del record
            quality_score = record.get('quality_score', 1.0)
            if quality_score < 0.8:  # Records de baja calidad
                self.worst_quality_records.append({
                    'timestamp': timestamp.isoformat(),
                    'quality_score': quality_score,
                    'source': source,
                    'data_flag': record.get('data_flag', 'unknown')
                })
                # Mantener solo los 100 peores
                self.worst_quality_records = sorted(
                    self.worst_quality_records, 
                    key=lambda x: x['quality_score']
                )[:100]
            
            # Actualizar estadísticas de fuente
            self.source_stats[source]['total_bars'] += 1
            self.source_stats[source]['avg_quality'] = (
                (self.source_stats[source]['avg_quality'] * 
                 (self.source_stats[source]['total_bars'] - 1) + quality_score) /
                self.source_stats[source]['total_bars']
            )
            
        except Exception as e:
            logger.warning(f"Error trackeando record: {e}")
    
    def track_gap(self, gap_start: datetime, gap_end: datetime, 
                  gap_minutes: int, filled_by: str = None, variance: float = None):
        """Trackear un gap detectado"""
        date = gap_start.date()
        
        gap_info = {
            'start': gap_start,
            'end': gap_end,
            'minutes': gap_minutes,
            'bars_missing': gap_minutes // 5,  # Asumiendo M5
            'filled_by': filled_by,
            'variance': variance
        }
        
        self.gap_tracking[date].append(gap_info)
    
    def track_imputation(self, timestamp: datetime, method: str, source_data: str):
        """Trackear una imputación realizada"""
        date = timestamp.date()
        
        imputation_info = {
            'timestamp': timestamp,
            'method': method,
            'source_data': source_data
        }
        
        self.imputation_tracking[date].append(imputation_info)
    
    def calculate_expected_bars(self, start_date: datetime, end_date: datetime) -> Dict[datetime, int]:
        """Calcular barras esperadas por día"""
        expected_bars = {}
        
        current_date = start_date.date()
        end_date = end_date.date()
        
        while current_date <= end_date:
            # Verificar si es día de trading
            if self._is_trading_day(current_date):
                # 6.5 horas de trading = 390 minutos
                # M5 = 78 barras por día
                expected_bars[current_date] = 78
            else:
                expected_bars[current_date] = 0
            
            current_date += timedelta(days=1)
        
        return expected_bars
    
    def _is_trading_day(self, date) -> bool:
        """Verificar si es día de trading"""
        # Lunes = 0, Domingo = 6
        if date.weekday() >= 5:  # Fin de semana
            return False
        
        # Aquí se podrían agregar festivos específicos
        # Por ahora, asumimos que todos los días hábiles son de trading
        return True
    
    def _determine_data_source(self, record: pd.Series) -> str:
        """Determinar la fuente de datos del record"""
        # Usar data_origin si está disponible
        if 'data_origin' in record:
            data_origin = record['data_origin']
            
            # Mapear data_origin a fuente legible
            if data_origin == 'M5_NATIVO':
                return 'M5'
            elif data_origin == 'M1_NATIVO':
                return 'M1'
            elif data_origin == 'M5_AGREGADO_M1':
                return 'M1->M5'
            elif data_origin == 'M5_AGREGADO_M10':
                return 'M10->M5'
            elif data_origin == 'M5_AGREGADO_M15':
                return 'M15->M5'
            elif data_origin == 'M5_AGREGADO_M20':
                return 'M20->M5'
            elif data_origin == 'M5_AGREGADO_M30':
                return 'M30->M5'
            elif data_origin == 'M5_AGREGADO_H1':
                return 'H1->M5'
            elif 'IMPUTADO' in data_origin:
                return 'imputed'
            elif data_origin == 'TICKS_NATIVO':
                return 'ticks'
            else:
                return data_origin
        
        # Fallback a data_flag legacy
        data_flag = record.get('data_flag', 'unknown')
        
        if data_flag == 'real_m5':
            return 'M5'
        elif data_flag == 'aggregated_m1':
            return 'M1->M5'
        elif data_flag == 'aggregated_m10':
            return 'M10->M5'
        elif data_flag == 'aggregated_m15':
            return 'M15->M5'
        elif data_flag == 'aggregated_m20':
            return 'M20->M5'
        elif data_flag == 'aggregated_m30':
            return 'M30->M5'
        elif data_flag == 'aggregated_h1':
            return 'H1->M5'
        elif data_flag == 'imputed':
            return 'imputed'
        else:
            return 'unknown'
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generar reporte completo de calidad"""
        report = {
            'summary': self._generate_summary(),
            'daily_analysis': self._analyze_daily_completeness(),
            'hourly_analysis': self._analyze_hourly_completeness(),
            'gap_analysis': self._analyze_gaps(),
            'source_contribution': self._analyze_source_contributions(),
            'imputation_analysis': self._analyze_imputations(),
            'quality_scores': self._analyze_quality_scores(),
            'yearly_analysis': self._analyze_yearly_data(),
            'monthly_analysis': self._analyze_monthly_data(),
            'data_origin_analysis': self._analyze_data_origin(),
            'worst_quality_records': self.worst_quality_records[:50],
            'execution_times': self.execution_times,
            'broker_info': self.broker_info
        }
        
        return report
    
    def _generate_summary(self) -> Dict:
        """Generar resumen ejecutivo"""
        total_expected = sum(day['expected'] for day in self.daily_completeness.values())
        total_captured = sum(day['captured'] for day in self.daily_completeness.values())
        
        # Contar contribuciones por timeframe
        tf_totals = defaultdict(int)
        for day_data in self.daily_completeness.values():
            for tf, count in day_data['sources'].items():
                tf_totals[tf] += count
                
        return {
            'total_expected_bars': total_expected,
            'total_captured_bars': total_captured,
            'overall_completeness': (total_captured / total_expected * 100) if total_expected > 0 else 0,
            'trading_days_analyzed': len(self.daily_completeness),
            'primary_source_breakdown': {
                tf: {
                    'bars': count,
                    'percentage': (count / total_captured * 100) if total_captured > 0 else 0
                }
                for tf, count in sorted(tf_totals.items(), key=lambda x: x[1], reverse=True)
            }
        }
    
    def _analyze_daily_completeness(self) -> Dict:
        """Analizar completitud diaria"""
        daily_stats = []
        
        for date, data in sorted(self.daily_completeness.items()):
            completeness = (data['captured'] / data['expected'] * 100) if data['expected'] > 0 else 0
            
            # Intentar identificar causa si completitud es baja
            cause = "Normal"
            if completeness < 50:
                cause = "Posible festivo o cierre"
            elif completeness < 90:
                if date.year == 2020 and date.month == 3:
                    cause = "COVID-19 volatilidad"
                elif max(data['sources'].items(), key=lambda x: x[1])[0] != 'M5' if data['sources'] else False:
                    cause = "M5 no disponible"
                else:
                    cause = "Gap en datos"
            
            daily_stats.append({
                'date': date.isoformat(),
                'expected_bars': data['expected'],
                'captured_bars': data['captured'],
                'completeness_pct': completeness,
                'primary_source': max(data['sources'].items(), key=lambda x: x[1])[0] if data['sources'] else 'none',
                'sources_used': dict(data['sources']),
                'likely_cause': cause
            })
            
        # Estadísticas agregadas
        completeness_values = [d['completeness_pct'] for d in daily_stats]
        
        return {
            'daily_details': daily_stats,
            'statistics': {
                'mean_completeness': np.mean(completeness_values) if completeness_values else 0,
                'min_completeness': np.min(completeness_values) if completeness_values else 0,
                'max_completeness': np.max(completeness_values) if completeness_values else 0,
                'std_completeness': np.std(completeness_values) if completeness_values else 0,
                'days_100_complete': sum(1 for v in completeness_values if v >= 100),
                'days_95_complete': sum(1 for v in completeness_values if v >= 95),
                'days_below_90': sum(1 for v in completeness_values if v < 90)
            }
        }
    
    def _analyze_hourly_completeness(self) -> Dict:
        """Analizar completitud por hora del día"""
        hourly_aggregate = defaultdict(lambda: {
            'total_expected': 0, 'total_captured': 0, 'sources': defaultdict(int)
        })
        
        for date, hours in self.hourly_completeness.items():
            for hour, data in hours.items():
                hour_of_day = hour.hour
                hourly_aggregate[hour_of_day]['total_expected'] += data['expected']
                hourly_aggregate[hour_of_day]['total_captured'] += data['captured']
                for source, count in data['sources'].items():
                    hourly_aggregate[hour_of_day]['sources'][source] += count
                    
        hourly_stats = []
        for hour in sorted(hourly_aggregate.keys()):
            data = hourly_aggregate[hour]
            completeness = (data['total_captured'] / data['total_expected'] * 100) if data['total_expected'] > 0 else 0
            
            hourly_stats.append({
                'hour': f"{hour:02d}:00",
                'total_expected': data['total_expected'],
                'total_captured': data['total_captured'],
                'completeness_pct': completeness,
                'primary_source': max(data['sources'].items(), key=lambda x: x[1])[0] if data['sources'] else 'none',
                'source_breakdown': dict(data['sources'])
            })
            
        return {
            'hourly_patterns': hourly_stats,
            'observations': self._identify_hourly_patterns(hourly_stats)
        }
    
    def _identify_hourly_patterns(self, hourly_stats: List[Dict]) -> Dict:
        """Identificar patrones en la completitud horaria"""
        patterns = {
            'lowest_completeness_hours': [],
            'highest_completeness_hours': [],
            'opening_hour_issues': False,
            'closing_hour_issues': False,
            'lunch_hour_pattern': False
        }
        
        # Ordenar por completitud
        sorted_hours = sorted(hourly_stats, key=lambda x: x['completeness_pct'])
        
        patterns['lowest_completeness_hours'] = [h['hour'] for h in sorted_hours[:3]]
        patterns['highest_completeness_hours'] = [h['hour'] for h in sorted_hours[-3:]]
        
        # Verificar horas específicas
        for stat in hourly_stats:
            if stat['hour'] == '09:00' and stat['completeness_pct'] < 90:
                patterns['opening_hour_issues'] = True
            if stat['hour'] == '15:00' and stat['completeness_pct'] < 90:
                patterns['closing_hour_issues'] = True
            if stat['hour'] in ['12:00', '13:00'] and stat['completeness_pct'] < 95:
                patterns['lunch_hour_pattern'] = True
                
        return patterns
    
    def _analyze_gaps(self) -> Dict:
        """Analizar gaps detectados"""
        all_gaps = []
        gap_summary = {
            'total_gaps': 0,
            'total_gap_minutes': 0,
            'gaps_by_size': defaultdict(int),
            'gaps_filled': 0,
            'gaps_unfilled': 0,
            'filling_methods': defaultdict(int),
            'total_variance': 0.0,
            'avg_variance_per_gap': 0.0
        }
        
        for date, gaps in self.gap_tracking.items():
            for gap in gaps:
                all_gaps.append({
                    'date': date.isoformat(),
                    'start': gap['start'].isoformat(),
                    'end': gap['end'].isoformat(),
                    'minutes': gap['minutes'],
                    'bars_missing': gap['bars_missing'],
                    'filled': gap['filled_by'] is not None,
                    'filled_by': gap['filled_by'],
                    'variance': gap.get('variance', 0.0)
                })
                
                gap_summary['total_gaps'] += 1
                gap_summary['total_gap_minutes'] += gap['minutes']
                
                if gap.get('variance'):
                    gap_summary['total_variance'] += gap['variance']
                
                # Categorizar por tamaño
                if gap['minutes'] <= 10:
                    gap_summary['gaps_by_size']['small (<=10 min)'] += 1
                elif gap['minutes'] <= 30:
                    gap_summary['gaps_by_size']['medium (11-30 min)'] += 1
                elif gap['minutes'] <= 60:
                    gap_summary['gaps_by_size']['large (31-60 min)'] += 1
                else:
                    gap_summary['gaps_by_size']['very large (>60 min)'] += 1
                    
                if gap['filled_by']:
                    gap_summary['gaps_filled'] += 1
                    gap_summary['filling_methods'][gap['filled_by']] += 1
                else:
                    gap_summary['gaps_unfilled'] += 1
        
        if gap_summary['gaps_filled'] > 0:
            gap_summary['avg_variance_per_gap'] = gap_summary['total_variance'] / gap_summary['gaps_filled']
        
        gap_summary['gap_fill_rate'] = (
            gap_summary['gaps_filled'] / gap_summary['total_gaps'] * 100
        ) if gap_summary['total_gaps'] > 0 else 0
        
        return {
            'summary': gap_summary,
            'detailed_gaps': all_gaps[:100]  # Limitar a 100 gaps para el reporte
        }
    
    def _analyze_source_contributions(self) -> Dict:
        """Analizar contribuciones por fuente de datos"""
        source_analysis = {}
        
        for source, stats in self.source_stats.items():
            source_analysis[source] = {
                'total_bars': stats['total_bars'],
                'avg_quality': stats['avg_quality'],
                'percentage_of_total': 0  # Se calculará después
            }
        
        # Calcular porcentajes
        total_bars = sum(stats['total_bars'] for stats in self.source_stats.values())
        if total_bars > 0:
            for source in source_analysis:
                source_analysis[source]['percentage_of_total'] = (
                    source_analysis[source]['total_bars'] / total_bars * 100
                )
        
        return source_analysis
    
    def _analyze_imputations(self) -> Dict:
        """Analizar imputaciones realizadas"""
        imputation_summary = {
            'total_imputations': 0,
            'imputations_by_method': defaultdict(int),
            'imputations_by_source': defaultdict(int),
            'imputations_by_date': defaultdict(int)
        }
        
        for date, imputations in self.imputation_tracking.items():
            for imp in imputations:
                imputation_summary['total_imputations'] += 1
                imputation_summary['imputations_by_method'][imp['method']] += 1
                imputation_summary['imputations_by_source'][imp['source_data']] += 1
                imputation_summary['imputations_by_date'][date.isoformat()] += 1
        
        return imputation_summary
    
    def _analyze_quality_scores(self) -> Dict:
        """Analizar distribución de scores de calidad"""
        if not self.worst_quality_records:
            return {'no_quality_data': True}
        
        scores = [r['quality_score'] for r in self.worst_quality_records]
        
        return {
            'min_score': min(scores),
            'max_score': max(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'score_distribution': self._categorize_quality_scores(scores)
        }
    
    def _categorize_quality_scores(self, scores: List[float]) -> Dict:
        """Categorizar scores de calidad"""
        categories = {
            'excellent (0.95-1.0)': 0,
            'good (0.90-0.95)': 0,
            'acceptable (0.80-0.90)': 0,
            'poor (0.70-0.80)': 0,
            'very_poor (<0.70)': 0
        }
        
        for score in scores:
            if score >= 0.95:
                categories['excellent (0.95-1.0)'] += 1
            elif score >= 0.90:
                categories['good (0.90-0.95)'] += 1
            elif score >= 0.80:
                categories['acceptable (0.80-0.90)'] += 1
            elif score >= 0.70:
                categories['poor (0.70-0.80)'] += 1
            else:
                categories['very_poor (<0.70)'] += 1
        
        return categories
    
    def _analyze_yearly_data(self) -> Dict:
        """Analizar datos por año"""
        yearly_stats = defaultdict(lambda: {
            'total_records': 0,
            'data_sources': defaultdict(lambda: {'bars': 0, 'percentage': 0}),
            'data_origin': {'real': 0, 'imputed': 0},
            'quality_stats': {'min': 1.0, 'max': 0.0, 'mean': 0.0}
        })
        
        # Procesar datos diarios por año
        for date, data in self.daily_completeness.items():
            year = date.year
            yearly_stats[year]['total_records'] += data['captured']
            
            # Analizar fuentes
            for source, count in data['sources'].items():
                yearly_stats[year]['data_sources'][source]['bars'] += count
            
            # Analizar origen (real vs imputado)
            real_count = sum(count for source, count in data['sources'].items() 
                           if source != 'imputed')
            imputed_count = data['sources'].get('imputed', 0)
            
            yearly_stats[year]['data_origin']['real'] += real_count
            yearly_stats[year]['data_origin']['imputed'] += imputed_count
        
        # Calcular porcentajes
        for year in yearly_stats:
            total = yearly_stats[year]['total_records']
            if total > 0:
                for source in yearly_stats[year]['data_sources']:
                    bars = yearly_stats[year]['data_sources'][source]['bars']
                    yearly_stats[year]['data_sources'][source]['percentage'] = bars / total * 100
                
                real = yearly_stats[year]['data_origin']['real']
                imputed = yearly_stats[year]['data_origin']['imputed']
                yearly_stats[year]['data_origin']['real_percentage'] = real / total * 100
                yearly_stats[year]['data_origin']['imputed_percentage'] = imputed / total * 100
        
        return dict(yearly_stats)
    
    def _analyze_monthly_data(self) -> Dict:
        """Analizar datos por mes"""
        monthly_stats = defaultdict(lambda: {
            'total_records': 0,
            'trading_days': 0,
            'avg_records_per_day': 0,
            'completeness_pct': 0
        })
        
        for date, data in self.daily_completeness.items():
            month_key = f"{date.year}-{date.month:02d}"
            monthly_stats[month_key]['total_records'] += data['captured']
            monthly_stats[month_key]['trading_days'] += 1
        
        # Calcular promedios
        for month in monthly_stats:
            trading_days = monthly_stats[month]['trading_days']
            if trading_days > 0:
                monthly_stats[month]['avg_records_per_day'] = (
                    monthly_stats[month]['total_records'] / trading_days
                )
        
        return dict(monthly_stats)
    
    def _analyze_data_origin(self) -> Dict:
        """Analizar origen de los datos (real vs imputado)"""
        total_real = 0
        total_imputed = 0
        
        for date, data in self.daily_completeness.items():
            for source, count in data['sources'].items():
                if source == 'imputed':
                    total_imputed += count
                else:
                    total_real += count
        
        total = total_real + total_imputed
        
        return {
            'real_data': {
                'bars': total_real,
                'percentage': (total_real / total * 100) if total > 0 else 0
            },
            'imputed_data': {
                'bars': total_imputed,
                'percentage': (total_imputed / total * 100) if total > 0 else 0
            }
        }
    
    def save_quality_report(self, output_path: str):
        """Guardar reporte de calidad en JSON"""
        report = self.generate_quality_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Reporte de calidad guardado en: {output_path}")
    
    def generate_completeness_heatmap(self, output_path: str):
        """Generar heatmap de completitud (placeholder)"""
        # Esta funcionalidad requeriría matplotlib/seaborn
        # Por ahora, solo guardamos los datos para generar el heatmap después
        heatmap_data = {}
        
        for date, hours in self.hourly_completeness.items():
            date_str = date.isoformat()
            heatmap_data[date_str] = {}
            
            for hour, data in hours.items():
                hour_str = f"{hour.hour:02d}:00"
                completeness = (data['captured'] / data['expected'] * 100) if data['expected'] > 0 else 0
                heatmap_data[date_str][hour_str] = completeness
        
        # Guardar datos del heatmap
        heatmap_file = output_path.replace('.png', '_data.json')
        with open(heatmap_file, 'w', encoding='utf-8') as f:
            json.dump(heatmap_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Datos de heatmap guardados en: {heatmap_file}")
        logger.info("Para generar el heatmap visual, instala matplotlib y seaborn") 