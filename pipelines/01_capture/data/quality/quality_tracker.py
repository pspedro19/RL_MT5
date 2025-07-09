from __future__ import annotations
#!/usr/bin/env python3
"""
Sistema de tracking de calidad de datos
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Any
import warnings
import pytz

from config.constants import MARKET_CONFIGS, DATA_ORIGINS
from utils.market_calendar import (
    get_expected_trading_days,
    get_expected_trading_days_forex,
    get_early_close_days,
)

logger = logging.getLogger(__name__)

# ===============================================================================
# CLASE PARA TRACKING DE CALIDAD DE DATOS
# ===============================================================================

class DataQualityTracker:
    """Sistema avanzado de tracking de calidad y completitud de datos"""

    def __init__(self, instrument: str = 'US500', large_gap_threshold: int = 60):
        self.instrument = instrument
        self.market_config = MARKET_CONFIGS.get(instrument, MARKET_CONFIGS['US500'])

        # Umbral para considerar un gap como "grande"
        self.large_gap_threshold = large_gap_threshold

        # Contadores de re-captura de gaps grandes
        self.large_gap_retried = 0
        self.large_gap_unresolved = 0
        
        self.source_tracking = defaultdict(lambda: defaultdict(int))
        self.gap_tracking = defaultdict(list)
        self.daily_completeness = defaultdict(lambda: {
            'expected': 0,
            'captured': 0,
            'sources': defaultdict(int)
        })
        self.daily_coverage_path: Optional[str] = None
        self.hourly_completeness = defaultdict(lambda: defaultdict(lambda: {'expected': 0, 'captured': 0, 'sources': defaultdict(int)}))
        self.timeframe_contributions = defaultdict(lambda: defaultdict(int))
        self.imputation_tracking = defaultdict(list)
        self.quality_metrics = defaultdict(list)
        
        # Nuevos trackers para análisis por año/mes
        self.yearly_tracking = defaultdict(lambda: {
            'total_records': 0,
            'native_m5': 0,
            'aggregated_m1': 0,
            'other_timeframes': defaultdict(int),
            'imputed_records': 0,
            'real_captured': 0,
            'sources': defaultdict(int),
            'methods': defaultdict(int),
            'origins': defaultdict(int)
        })
        
        self.monthly_tracking = defaultdict(lambda: defaultdict(lambda: {
            'total_records': 0,
            'native_m5': 0,
            'aggregated_m1': 0,
            'other_timeframes': defaultdict(int),
            'imputed_records': 0,
            'real_captured': 0,
            'sources': defaultdict(int),
            'methods': defaultdict(int),
            'origins': defaultdict(int),
            'expected_bars': 0,
            'completeness': 0
        }))
        
        # Tracking de tiempos de ejecución
        self.execution_times = {
            'capture': 0,
            'filtering': 0,
            'imputation': 0,
            'features': 0,
            'cleaning': 0,
            'io': 0,
            'total': 0
        }
        
        # Tracking de métricas adicionales
        self.worst_quality_records = []  # Top N peores registros por calidad
        self.monthly_trends = {}  # Tendencias mensuales
        self.broker_info = {}  # Información del broker
        self.variance_per_gap = defaultdict(float)  # Varianza estimada por gap

        # Historial de data_origin por timestamp para análisis posterior
        self.data_origin_history = []
        self.synthetic_streaks = pd.DataFrame(columns=["start", "end", "length"])
        
    def track_record(self, record: pd.Series):
        """Trackear un registro individual"""
        timestamp = record['time']
        source = record.get('source_timeframe', 'unknown')
        method = record.get('capture_method', 'unknown')
        quality = record.get('quality_score', 1.0)
        data_origin = record.get('data_origin')

        # Guardar historial para análisis de secuencias sintéticas
        self.data_origin_history.append((timestamp, data_origin))
        
        # Actualizar origenes si están disponibles
        if data_origin:
            self.yearly_tracking[timestamp.year]['origins'][data_origin] += 1
            self.monthly_tracking[timestamp.year][timestamp.month]['origins'][data_origin] += 1

            if data_origin == 'M5_NATIVO':
                source = 'M5'
            elif data_origin == 'M5_AGREGADO_M1':
                source = 'M1'
            elif data_origin.startswith('M5_AGREGADO_'):
                source = data_origin.replace('M5_AGREGADO_', '')
            elif 'IMPUTADO' in data_origin:
                method = 'imputation'
                source = 'M5'

        # Tracking por fuente
        self.source_tracking[source]['total'] += 1
        self.source_tracking[source]['quality_sum'] += quality
        
        # Tracking por método
        self.timeframe_contributions[method]['total'] += 1
        
        # Tracking temporal
        year = timestamp.year
        month = timestamp.month
        hour = timestamp.hour
        
        # Yearly tracking
        self.yearly_tracking[year]['total_records'] += 1
        self.yearly_tracking[year]['sources'][source] += 1
        self.yearly_tracking[year]['methods'][method] += 1
        
        if data_origin:
            if data_origin == 'M5_NATIVO':
                self.yearly_tracking[year]['native_m5'] += 1
            elif data_origin == 'M5_AGREGADO_M1':
                self.yearly_tracking[year]['aggregated_m1'] += 1
            elif data_origin.startswith('M5_AGREGADO_'):
                tf = data_origin.replace('M5_AGREGADO_', '')
                self.yearly_tracking[year]['other_timeframes'][tf] += 1
            elif 'IMPUTADO' in data_origin:
                self.yearly_tracking[year]['imputed_records'] += 1
            else:
                self.yearly_tracking[year]['other_timeframes'][data_origin] += 1

            if 'IMPUTADO' not in data_origin:
                self.yearly_tracking[year]['real_captured'] += 1
        else:
            if source == 'M5':
                self.yearly_tracking[year]['native_m5'] += 1
            elif source == 'M1':
                self.yearly_tracking[year]['aggregated_m1'] += 1
            else:
                self.yearly_tracking[year]['other_timeframes'][source] += 1

            if method == 'imputation':
                self.yearly_tracking[year]['imputed_records'] += 1
            else:
                self.yearly_tracking[year]['real_captured'] += 1
        
        # Monthly tracking
        self.monthly_tracking[year][month]['total_records'] += 1
        self.monthly_tracking[year][month]['sources'][source] += 1
        self.monthly_tracking[year][month]['methods'][method] += 1

        if data_origin:
            if data_origin == 'M5_NATIVO':
                self.monthly_tracking[year][month]['native_m5'] += 1
            elif data_origin == 'M5_AGREGADO_M1':
                self.monthly_tracking[year][month]['aggregated_m1'] += 1
            elif data_origin.startswith('M5_AGREGADO_'):
                tf = data_origin.replace('M5_AGREGADO_', '')
                self.monthly_tracking[year][month]['other_timeframes'][tf] += 1
            elif 'IMPUTADO' in data_origin:
                self.monthly_tracking[year][month]['imputed_records'] += 1
            else:
                self.monthly_tracking[year][month]['other_timeframes'][data_origin] += 1

            if 'IMPUTADO' not in data_origin:
                self.monthly_tracking[year][month]['real_captured'] += 1
        else:
            if source == 'M5':
                self.monthly_tracking[year][month]['native_m5'] += 1
            elif source == 'M1':
                self.monthly_tracking[year][month]['aggregated_m1'] += 1
            else:
                self.monthly_tracking[year][month]['other_timeframes'][source] += 1

            if method == 'imputation':
                self.monthly_tracking[year][month]['imputed_records'] += 1
            else:
                self.monthly_tracking[year][month]['real_captured'] += 1
        
        # Hourly tracking
        self.hourly_completeness[year][hour]['captured'] += 1
        self.hourly_completeness[year][hour]['sources'][source] += 1
        
        # Quality metrics
        self.quality_metrics[year].append(quality)
        
        # Track worst quality records
        if len(self.worst_quality_records) < 100:
            self.worst_quality_records.append({
                'timestamp': timestamp,
                'quality': quality,
                'source': source,
                'method': method
            })
        elif quality < min(r['quality'] for r in self.worst_quality_records):
            # Replace worst record
            self.worst_quality_records = sorted(
                self.worst_quality_records + [{
                    'timestamp': timestamp,
                    'quality': quality,
                    'source': source,
                    'method': method
                }],
                key=lambda x: x['quality']
            )[:100]
    
    def track_gap(self, gap_start: datetime, gap_end: datetime, gap_minutes: int,
                  filled_by: str = None, variance: float = None,
                  reason: str = None):
        """Trackear un gap detectado con información detallada"""
        gap_info = {
            'start': gap_start,
            'end': gap_end,
            'minutes': gap_minutes,
            'bars_missing': int(gap_minutes / 5),
            'filled_by': filled_by,
            'variance': variance,
            'reason': reason,
            'market_hours': self._is_market_hours(gap_start, gap_end)
        }
        
        self.gap_tracking[gap_start.date()].append(gap_info)
        
        if variance is not None:
            self.variance_per_gap[gap_minutes] = variance
    
    def _is_market_hours(self, start_time: datetime, end_time: datetime) -> bool:
        """Determinar si un período está dentro del horario de mercado (robusto a naive/aware)"""
        import pytz
        utc = pytz.UTC
        eastern = pytz.timezone('US/Eastern')

        def ensure_aware(dt):
            if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                return utc.localize(dt)
            return dt

        start_time = ensure_aware(start_time)
        end_time = ensure_aware(end_time)
        start_et = start_time.astimezone(eastern)
        end_et = end_time.astimezone(eastern)

        if self.instrument == 'USDCOP':
            # Forex opera 24/5, pero consideramos horario principal
            if start_et.weekday() >= 5 or end_et.weekday() >= 5:
                return False
            start_hour = start_et.hour
            end_hour = end_et.hour
            return 8 <= start_hour <= 17 and 8 <= end_hour <= 17
        else:
            # Mercado USA: 9:30 AM - 4:00 PM ET, Lunes a Viernes
            if start_et.weekday() >= 5 or end_et.weekday() >= 5:
                return False
            start_hour = start_et.hour + start_et.minute / 60
            end_hour = end_et.hour + end_et.minute / 60
            return 9.5 <= start_hour <= 16 and 9.5 <= end_hour <= 16
    
    def track_imputation(self, timestamp: datetime, method: str, source_data: str):
        """Trackear una imputación realizada"""
        self.imputation_tracking[timestamp.date()].append({
            'timestamp': timestamp,
            'method': method,
            'source_data': source_data
        })
    
    def calculate_expected_bars(self, start_date: datetime, end_date: datetime) -> Dict[datetime, int]:
        """Calcular barras esperadas por día según el instrumento"""
        expected = {}
        
        if self.instrument == 'USDCOP':
            # Lógica específica para Forex (24/5)
            trading_days = get_expected_trading_days_forex(start_date, end_date)
            
            for day in trading_days:
                current = datetime.combine(day, datetime.min.time())
                # Forex opera de domingo 5PM ET a viernes 5PM ET
                if current.weekday() < 5:  # Lunes a Viernes
                    expected[current] = self.market_config['bars_per_day']  # 288 bars
                elif current.weekday() == 6:  # Domingo
                    # Solo contar desde las 22:00 UTC
                    expected[current] = self.market_config['bars_per_day'] // 24 * 2  # 2 horas
        else:
            # Lógica para mercado USA
            trading_days = get_expected_trading_days(start_date, end_date, self.instrument)

            # Determinar días con cierre temprano (13:00)
            early_close_dates = set()
            for year in range(start_date.year, end_date.year + 1):
                early_close_days = get_early_close_days(year)
                early_close_dates.update({dt.date() for dt in early_close_days.keys()})

            for day in trading_days:
                current = datetime.combine(day, datetime.min.time())
                if day in early_close_dates:
                    # Mercado cierra a la 1 PM -> 53 barras de 5 minutos
                    expected[current] = 53
                else:
                    expected[current] = self.market_config['bars_per_day']  # 78 bars para US500
        
        return expected
    
    def update_daily_completeness(self, date: datetime, captured: int, expected: int):
        """Actualizar completitud diaria"""
        self.daily_completeness[date]['captured'] = captured
        self.daily_completeness[date]['expected'] = expected

        year = date.year
        month = date.month
        self.monthly_tracking[year][month]['expected_bars'] += expected
        total = self.monthly_tracking[year][month]['total_records']
        expected_total = self.monthly_tracking[year][month]['expected_bars']
        if expected_total > 0:
            self.monthly_tracking[year][month]['completeness'] = total / expected_total
        else:
            self.monthly_tracking[year][month]['completeness'] = 0
    
    def update_hourly_completeness(self, year: int, hour: int, captured: int, expected: int):
        """Actualizar completitud horaria"""
        self.hourly_completeness[year][hour]['captured'] = captured
        self.hourly_completeness[year][hour]['expected'] = expected

    def save_daily_coverage_csv(self, path: str = 'daily_coverage.csv') -> None:
        """Guardar matriz de cobertura diaria en CSV."""
        records = []
        for date, stats in sorted(self.daily_completeness.items()):
            expected = stats['expected']
            captured = stats['captured']
            pct = (captured / expected * 100) if expected > 0 else 0
            grade = self._grade_completeness(pct)
            records.append({
                'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                'expected_bars': expected,
                'captured_bars': captured,
                'coverage_pct': pct,
                'grade': grade
            })

        if records:
            df = pd.DataFrame(records)
            df.to_csv(path, index=False)
            self.daily_coverage_path = path
        else:
            self.daily_coverage_path = None
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generar reporte completo de calidad"""
        report = {
            'summary': self._generate_summary(),
            'daily_completeness': self._analyze_daily_completeness(),
            'hourly_completeness': self._analyze_hourly_completeness(),
            'gaps': self._analyze_gaps(),
            'source_contributions': self._analyze_source_contributions(),
            'method_effectiveness': self._analyze_method_effectiveness(),
            'imputations': self._analyze_imputations(),
            'quality_scores': self._analyze_quality_scores(),
            'yearly_data': self._analyze_yearly_data(),
            'monthly_data': self._analyze_monthly_data(),
            'data_origin': self._analyze_data_origin(),
            'instrument': self.instrument,
            'market_config': self.market_config
        }

        # Guardar matriz diaria de cobertura
        try:
            self.save_daily_coverage_csv()
        except Exception as e:
            logger.warning(f"No se pudo guardar daily_coverage.csv: {e}")

        if self.daily_coverage_path:
            report['daily_coverage_file'] = self.daily_coverage_path

        # Detectar tramos sintéticos y exportar a parquet
        streaks_df = self.detect_synthetic_streaks()
        if not streaks_df.empty:
            try:
                streaks_df.to_parquet("synthetic_streaks.parquet", index=False)
            except Exception as e:
                logger.warning(f"No se pudo guardar synthetic_streaks.parquet: {e}")

        return report

    def _grade_completeness(self, pct: float) -> str:
        """Clasificar porcentaje de completitud en una letra."""
        if pct >= 95:
            return 'A'
        elif pct >= 90:
            return 'B'
        elif pct >= 80:
            return 'C'
        else:
            return 'D'
    
    def _generate_summary(self) -> Dict:
        """Generar resumen general"""
        total_records = sum(tracking['total_records'] for tracking in self.yearly_tracking.values())
        total_imputed = sum(tracking['imputed_records'] for tracking in self.yearly_tracking.values())

        total_expected = sum(day['expected'] for day in self.daily_completeness.values())
        total_captured = sum(day['captured'] for day in self.daily_completeness.values())

        return {
            'total_records': total_records,
            'total_imputed': total_imputed,
            'imputation_rate': total_imputed / total_records if total_records > 0 else 0,
            'total_expected_bars': total_expected,
            'total_captured_bars': total_captured,
            'overall_completeness': (total_captured / total_expected * 100) if total_expected > 0 else 0,
            'trading_days_analyzed': len(self.daily_completeness),
            'years_covered': list(self.yearly_tracking.keys()),
            'data_sources': list(self.source_tracking.keys()),
            'capture_methods': list(self.timeframe_contributions.keys())
        }
    
    def _analyze_daily_completeness(self) -> Dict:
        """Analizar completitud diaria"""
        daily_stats = []
        
        for date, stats in self.daily_completeness.items():
            if stats['expected'] > 0:
                completeness = stats['captured'] / stats['expected']
                pct = completeness * 100
                daily_stats.append({
                    'date': date,
                    'captured': stats['captured'],
                    'expected': stats['expected'],
                    'completeness': completeness,
                    'coverage_pct': pct,
                    'grade': self._grade_completeness(pct),
                    'sources': dict(stats['sources'])
                })
        
        if not daily_stats:
            return {'average_completeness': 0, 'daily_stats': []}
        
        avg_completeness = sum(s['completeness'] for s in daily_stats) / len(daily_stats)
        
        return {
            'average_completeness': avg_completeness,
            'daily_stats': daily_stats
        }
    
    def _analyze_hourly_completeness(self) -> Dict:
        """Analizar completitud horaria"""
        hourly_stats = []
        
        for year in self.hourly_completeness:
            for hour in range(24):
                if hour in self.hourly_completeness[year]:
                    stats = self.hourly_completeness[year][hour]
                    if stats['expected'] > 0:
                        completeness = stats['captured'] / stats['expected']
                        hourly_stats.append({
                            'year': year,
                            'hour': hour,
                            'captured': stats['captured'],
                            'expected': stats['expected'],
                            'completeness': completeness,
                            'sources': dict(stats['sources'])
                        })
        
        if not hourly_stats:
            return {'average_completeness': 0, 'hourly_stats': []}
        
        avg_completeness = sum(s['completeness'] for s in hourly_stats) / len(hourly_stats)
        
        return {
            'average_completeness': avg_completeness,
            'hourly_stats': hourly_stats
        }
    
    def _analyze_gaps(self) -> Dict:
        """Analizar gaps detectados con clasificación detallada"""
        gap_analysis = {
            'total_gaps': 0,
            'total_gap_minutes': 0,
            'average_gap_minutes': 0,
            'max_gap_minutes': 0,
            'gap_distribution': {},
            'gaps_by_day': {},
            'summary': {},
            'gap_classification': {
                'market_hours': {
                    'total': 0,
                    'imputed': 0,
                    'not_imputed': 0,
                    'total_minutes': 0,
                    'average_minutes': 0
                },
                'outside_market_hours': {
                    'total': 0,
                    'ignored': 0,
                    'total_minutes': 0,
                    'average_minutes': 0
                }
            },
            'imputation_summary': {
                'imputed': 0,
                'not_imputed': 0,
                'ignored': 0,
                'reasons': {}
            }
        }
        
        all_gaps = []
        for day_gaps in self.gap_tracking.values():
            all_gaps.extend(day_gaps)

        if not all_gaps:
            gap_analysis['summary'] = {
                'total_gaps': 0,
                'gaps_filled': 0,
                'gap_fill_rate': 0,
                'large_gap_threshold': self.large_gap_threshold,
                'retried_large_gaps': 0,
                'unresolved_large_gaps': 0,
                'large_gaps': 0
            }
            return gap_analysis
        
        gap_analysis['total_gaps'] = len(all_gaps)
        gap_analysis['total_gap_minutes'] = sum(g['minutes'] for g in all_gaps)
        gap_analysis['average_gap_minutes'] = gap_analysis['total_gap_minutes'] / len(all_gaps)
        gap_analysis['max_gap_minutes'] = max(g['minutes'] for g in all_gaps)
        
        # Clasificar gaps
        market_hours_gaps = [g for g in all_gaps if g.get('market_hours', False)]
        outside_market_gaps = [g for g in all_gaps if not g.get('market_hours', False)]
        
        # Análisis de gaps en horario de mercado
        gap_analysis['gap_classification']['market_hours']['total'] = len(market_hours_gaps)
        gap_analysis['gap_classification']['market_hours']['total_minutes'] = sum(g['minutes'] for g in market_hours_gaps)
        if market_hours_gaps:
            gap_analysis['gap_classification']['market_hours']['average_minutes'] = (
                gap_analysis['gap_classification']['market_hours']['total_minutes'] / len(market_hours_gaps)
            )
        
        imputed_gaps = [g for g in market_hours_gaps if g.get('filled_by') == 'brownian_bridge']
        not_imputed_gaps = [g for g in market_hours_gaps if g.get('filled_by') == 'not_imputed']
        
        gap_analysis['gap_classification']['market_hours']['imputed'] = len(imputed_gaps)
        gap_analysis['gap_classification']['market_hours']['not_imputed'] = len(not_imputed_gaps)
        
        # Análisis de gaps fuera de horario
        gap_analysis['gap_classification']['outside_market_hours']['total'] = len(outside_market_gaps)
        gap_analysis['gap_classification']['outside_market_hours']['total_minutes'] = sum(g['minutes'] for g in outside_market_gaps)
        gap_analysis['gap_classification']['outside_market_hours']['ignored'] = len(outside_market_gaps)
        if outside_market_gaps:
            gap_analysis['gap_classification']['outside_market_hours']['average_minutes'] = (
                gap_analysis['gap_classification']['outside_market_hours']['total_minutes'] / len(outside_market_gaps)
            )
        
        # Resumen de imputación
        gap_analysis['imputation_summary']['imputed'] = len(imputed_gaps)
        gap_analysis['imputation_summary']['not_imputed'] = len(not_imputed_gaps)
        gap_analysis['imputation_summary']['ignored'] = len(outside_market_gaps)

        # Razones de no imputación
        reasons = {}
        for gap in all_gaps:
            reason = gap.get('reason', 'Unknown')
            reasons[reason] = reasons.get(reason, 0) + 1
        gap_analysis['imputation_summary']['reasons'] = reasons
        
        # Distribución de gaps por tamaño
        gap_sizes = [g['minutes'] for g in all_gaps]
        gap_analysis['gap_distribution'] = {
            'small_5_15': len([g for g in gap_sizes if 5 <= g <= 15]),
            'medium_15_30': len([g for g in gap_sizes if 15 < g <= 30]),
            'large_30_60': len([g for g in gap_sizes if 30 < g <= 60]),
            'very_large_60+': len([g for g in gap_sizes if g > 60])
        }

        # Resumen general y métricas de gaps grandes
        large_gaps = [g for g in all_gaps if g['minutes'] > self.large_gap_threshold]
        retried = len([g for g in large_gaps if g.get('filled_by') == 'recaptured'])
        unresolved = len(large_gaps) - retried
        filled_count = len([g for g in all_gaps if g.get('filled_by')])
        gap_analysis['summary'] = {
            'total_gaps': len(all_gaps),
            'gaps_filled': filled_count,
            'gap_fill_rate': filled_count / len(all_gaps) * 100 if all_gaps else 0,
            'large_gap_threshold': self.large_gap_threshold,
            'retried_large_gaps': retried,
            'unresolved_large_gaps': unresolved,
            'large_gaps': len(large_gaps)
        }
        
        # Gaps por día
        for date, gaps in self.gap_tracking.items():
            market_hours_count = len([g for g in gaps if g.get('market_hours', False)])
            outside_market_count = len([g for g in gaps if not g.get('market_hours', False)])
            
            gap_analysis['gaps_by_day'][str(date)] = {
                'count': len(gaps),
                'market_hours': market_hours_count,
                'outside_market_hours': outside_market_count,
                'total_minutes': sum(g['minutes'] for g in gaps),
                'average_minutes': sum(g['minutes'] for g in gaps) / len(gaps) if gaps else 0
            }
        
        return gap_analysis
    
    def _analyze_source_contributions(self) -> Dict:
        """Analizar contribuciones por fuente"""
        source_stats = {}
        
        for source, stats in self.source_tracking.items():
            if stats['total'] > 0:
                avg_quality = stats['quality_sum'] / stats['total']
                source_stats[source] = {
                    'total_records': stats['total'],
                    'average_quality': avg_quality
                }
        
        return source_stats
    
    def _analyze_method_effectiveness(self) -> Dict:
        """Analizar efectividad de métodos de captura"""
        method_stats = {}
        
        for method, stats in self.timeframe_contributions.items():
            method_stats[method] = {
                'total_records': stats['total']
            }
        
        return method_stats
    
    def _analyze_imputations(self) -> Dict:
        """Analizar imputaciones realizadas"""
        total_imputations = sum(len(imputations) for imputations in self.imputation_tracking.values())
        
        method_counts = defaultdict(int)
        for date_imputations in self.imputation_tracking.values():
            for imp in date_imputations:
                method_counts[imp['method']] += 1
        
        return {
            'total_imputations': total_imputations,
            'method_counts': dict(method_counts)
        }
    
    def _analyze_quality_scores(self) -> Dict:
        """Analizar scores de calidad"""
        all_scores = []
        for year_scores in self.quality_metrics.values():
            all_scores.extend(year_scores)
        
        if not all_scores:
            return {'average_quality': 0, 'quality_distribution': {}}
        
        avg_quality = sum(all_scores) / len(all_scores)
        
        # Categorizar scores
        quality_categories = {
            'excellent': len([s for s in all_scores if s >= 0.9]),
            'good': len([s for s in all_scores if 0.7 <= s < 0.9]),
            'fair': len([s for s in all_scores if 0.5 <= s < 0.7]),
            'poor': len([s for s in all_scores if s < 0.5])
        }
        
        return {
            'average_quality': avg_quality,
            'quality_distribution': quality_categories
        }
    
    def _analyze_yearly_data(self) -> Dict:
        """Analizar datos por año"""
        yearly_stats = {}
        
        for year, tracking in self.yearly_tracking.items():
            if tracking['total_records'] > 0:
                imputation_rate = tracking['imputed_records'] / tracking['total_records']
                native_m5_rate = tracking['native_m5'] / tracking['total_records']
                
                yearly_stats[year] = {
                    'total_records': tracking['total_records'],
                    'imputation_rate': imputation_rate,
                    'native_m5_rate': native_m5_rate,
                    'sources': dict(tracking['sources']),
                    'methods': dict(tracking['methods'])
                }
        
        return yearly_stats
    
    def _analyze_monthly_data(self) -> Dict:
        """Analizar datos por mes"""
        monthly_stats = {}
        
        for year in self.monthly_tracking:
            monthly_stats[year] = {}
            for month, tracking in self.monthly_tracking[year].items():
                if tracking['total_records'] > 0:
                    imputation_rate = tracking['imputed_records'] / tracking['total_records']
                    native_m5_rate = tracking['native_m5'] / tracking['total_records']
                    
                    monthly_stats[year][month] = {
                        'total_records': tracking['total_records'],
                        'imputation_rate': imputation_rate,
                        'native_m5_rate': native_m5_rate,
                        'expected_bars': tracking['expected_bars'],
                        'completeness': tracking['completeness']
                    }
        
        return monthly_stats
    
    def _analyze_data_origin(self) -> Dict:
        """Analizar origen de los datos con desglose detallado"""
        total_native_m5 = 0
        total_aggregated_m1 = 0
        total_imputed = 0
        total_records = 0
        other_timeframes = {}
        for tracking in self.yearly_tracking.values():
            for origin, count in tracking.get('origins', {}).items():
                total_records += count
                if origin == 'M5_NATIVO':
                    total_native_m5 += count
                elif origin == 'M5_AGREGADO_M1':
                    total_aggregated_m1 += count
                elif origin.startswith('M5_AGREGADO_'):
                    tf = origin.replace('M5_AGREGADO_', '')
                    other_timeframes[tf] = other_timeframes.get(tf, 0) + count
                elif 'IMPUTADO' in origin:
                    total_imputed += count
                else:
                    other_timeframes[origin] = other_timeframes.get(origin, 0) + count
        
        if total_records == 0:
            return {
                'data_origin_breakdown': {},
                'source_details': {},
                'yearly_breakdown': {},
                'monthly_breakdown': {}
            }
        
        # Desglose general
        origin_breakdown = {
            'native_m5': total_native_m5 / total_records,
            'aggregated_m1': total_aggregated_m1 / total_records,
            'imputed': total_imputed / total_records,
            'other_timeframes': sum(other_timeframes.values()) / total_records
        }
        
        # Detalles por fuente
        source_details = {
            'native_m5': {
                'count': total_native_m5,
                'percentage': total_native_m5 / total_records * 100,
                'description': 'Datos capturados directamente en M5 (mejor calidad)'
            },
            'aggregated_m1': {
                'count': total_aggregated_m1,
                'percentage': total_aggregated_m1 / total_records * 100,
                'description': 'Datos de M1 agregados a M5 (buena calidad)'
            },
            'imputed': {
                'count': total_imputed,
                'percentage': total_imputed / total_records * 100,
                'description': 'Datos sintéticos para llenar gaps (usar con precaución)'
            }
        }
        
        # Añadir otros timeframes
        for timeframe, count in other_timeframes.items():
            source_details[timeframe] = {
                'count': count,
                'percentage': count / total_records * 100,
                'description': f'Datos de {timeframe} agregados a M5'
            }
        
        # Desglose por año
        yearly_breakdown = {}
        for year, tracking in self.yearly_tracking.items():
            if tracking['total_records'] > 0:
                yearly_breakdown[year] = {
                    'total_records': tracking['total_records'],
                    'native_m5_rate': tracking['native_m5'] / tracking['total_records'],
                    'aggregated_m1_rate': tracking['aggregated_m1'] / tracking['total_records'],
                    'imputed_rate': tracking['imputed_records'] / tracking['total_records'],
                    'other_timeframes': {tf: count / tracking['total_records'] for tf, count in tracking['other_timeframes'].items()}
                }
        
        # Desglose por mes
        monthly_breakdown = {}
        for year in self.monthly_tracking:
            monthly_breakdown[year] = {}
            for month, tracking in self.monthly_tracking[year].items():
                if tracking['total_records'] > 0:
                    monthly_breakdown[year][month] = {
                        'total_records': tracking['total_records'],
                        'native_m5_rate': tracking['native_m5'] / tracking['total_records'],
                        'aggregated_m1_rate': tracking['aggregated_m1'] / tracking['total_records'],
                        'imputed_rate': tracking['imputed_records'] / tracking['total_records'],
                        'other_timeframes': {tf: count / tracking['total_records'] for tf, count in tracking['other_timeframes'].items()},
                        'completeness': tracking['completeness']
                    }
        
        return {
            'data_origin_breakdown': origin_breakdown,
            'source_details': source_details,
            'yearly_breakdown': yearly_breakdown,
            'monthly_breakdown': monthly_breakdown
        }

    def detect_synthetic_streaks(self, min_length: int = 3) -> pd.DataFrame:
        """Detectar tramos consecutivos marcados como imputados.

        Parameters
        ----------
        min_length : int
            Longitud mínima de la secuencia para ser reportada.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas ['start', 'end', 'length']
        """

        if not self.data_origin_history:
            self.synthetic_streaks = pd.DataFrame(columns=["start", "end", "length"])
            return self.synthetic_streaks

        df = pd.DataFrame(self.data_origin_history, columns=["timestamp", "data_origin"])\
            .sort_values("timestamp")
        df["is_imputed"] = df["data_origin"].astype(str).str.contains("IMPUTADO")

        if df.empty:
            self.synthetic_streaks = pd.DataFrame(columns=["start", "end", "length"])
            return self.synthetic_streaks

        df["streak_id"] = (df["is_imputed"] != df["is_imputed"].shift()).cumsum()
        streaks = (
            df[df["is_imputed"]]
            .groupby("streak_id")
            .agg(start=("timestamp", "first"), end=("timestamp", "last"), length=("timestamp", "count"))
            .reset_index(drop=True)
        )

        streaks = streaks[streaks["length"] >= min_length]
        self.synthetic_streaks = streaks
        return streaks
    
    def generate_completeness_heatmap(self, output_path: str):
        """Generar CSV con datos de completitud en lugar de visualización"""
        # Cambiar la extensión a CSV
        csv_path = output_path.replace('.png', '_completeness.csv')
        
        logger = logging.getLogger('pipeline_hpc')
        logger.info("Generando CSV de completitud...")
        
        # Preparar datos para CSV
        dates = sorted(self.daily_completeness.keys())
        hours = list(range(24))  # Mercado opera 24 horas
        
        # Crear lista de registros para CSV
        completeness_data = []
        
        for date in dates:
            for hour in hours:
                hour_dt = pytz.UTC.localize(
                    datetime.combine(date, datetime.min.time()).replace(hour=hour)
                )
                
                record = {
                    'date': date.isoformat(),
                    'hour': hour,
                    'expected': 0,
                    'captured': 0,
                    'completeness_pct': 0,
                    'primary_source': 'N/A'
                }
                
                if date in self.hourly_completeness and hour_dt in self.hourly_completeness[date]:
                    data = self.hourly_completeness[date][hour_dt]
                    record['expected'] = data['expected']
                    record['captured'] = data['captured']
                    
                    if data['expected'] > 0:
                        record['completeness_pct'] = (data['captured'] / data['expected']) * 100
                    
                    # Fuente principal
                    if data['sources']:
                        primary = max(data['sources'].items(), key=lambda x: x[1])
                        record['primary_source'] = primary[0]
                
                completeness_data.append(record)
        
        # Guardar CSV de completitud horaria
        df_completeness = pd.DataFrame(completeness_data)
        df_completeness.to_csv(csv_path, index=False)
        logger.info(f"CSV de completitud guardado: {csv_path}")
        
        # Generar CSVs adicionales
        
        # 1. CSV de resumen anual
        yearly_csv_path = output_path.replace('.png', '_yearly_summary.csv')
        yearly_data = []
        
        for year in sorted(self.yearly_tracking.keys()):
            data = self.yearly_tracking[year]
            total = data['total_records']
            
            if total > 0:
                yearly_data.append({
                    'year': year,
                    'total_records': total,
                    'native_m5': data['native_m5'],
                    'native_m5_pct': (data['native_m5'] / total * 100),
                    'aggregated_m1': data['aggregated_m1'],
                    'aggregated_m1_pct': (data['aggregated_m1'] / total * 100),
                    'imputed': data['imputed_records'],
                    'imputed_pct': (data['imputed_records'] / total * 100),
                    'real_captured': data['real_captured'],
                    'real_captured_pct': (data['real_captured'] / total * 100),
                    **{f'{tf}_count': count for tf, count in data['sources'].items()},
                    **{f'{tf}_pct': (count / total * 100) for tf, count in data['sources'].items()}
                })
        
        if yearly_data:
            df_yearly = pd.DataFrame(yearly_data)
            df_yearly.to_csv(yearly_csv_path, index=False)
            logger.info(f"CSV de resumen anual guardado: {yearly_csv_path}")
        
        # 2. CSV de resumen mensual
        monthly_csv_path = output_path.replace('.png', '_monthly_summary.csv')
        monthly_data = []
        
        for year in sorted(self.monthly_tracking.keys()):
            for month in sorted(self.monthly_tracking[year].keys()):
                data = self.monthly_tracking[year][month]
                total = data['total_records']
                
                if total > 0:
                    monthly_data.append({
                        'year': year,
                        'month': month,
                        'month_name': datetime(year, month, 1).strftime('%B'),
                        'total_records': total,
                        'expected_bars': data['expected_bars'],
                        'completeness_pct': (total / data['expected_bars'] * 100) if data['expected_bars'] > 0 else 0,
                        'native_m5': data['native_m5'],
                        'native_m5_pct': (data['native_m5'] / total * 100),
                        'aggregated_m1': data['aggregated_m1'],
                        'aggregated_m1_pct': (data['aggregated_m1'] / total * 100),
                        'imputed': data['imputed_records'],
                        'imputed_pct': (data['imputed_records'] / total * 100),
                        **{f'{tf}_count': count for tf, count in data['sources'].items()},
                        **{f'{tf}_pct': (count / total * 100) for tf, count in data['sources'].items()}
                    })
        
        if monthly_data:
            df_monthly = pd.DataFrame(monthly_data)
            df_monthly.to_csv(monthly_csv_path, index=False)
            logger.info(f"CSV de resumen mensual guardado: {monthly_csv_path}")
        
        # 3. CSV de gaps detectados
        gaps_csv_path = output_path.replace('.png', '_gaps_detected.csv')
        gaps_data = []
        
        for date, gaps in self.gap_tracking.items():
            for gap in gaps:
                gaps_data.append({
                    'date': date.isoformat(),
                    'gap_start': gap['start'].isoformat(),
                    'gap_end': gap['end'].isoformat(),
                    'gap_minutes': gap['minutes'],
                    'bars_missing': gap['bars_missing'],
                    'filled': gap['filled_by'] is not None,
                    'filled_by': gap['filled_by'] or 'N/A',
                    'variance': gap.get('variance', 0.0)
                })
        
        if gaps_data:
            df_gaps = pd.DataFrame(gaps_data)
            df_gaps.to_csv(gaps_csv_path, index=False)
            logger.info(f"CSV de gaps guardado: {gaps_csv_path}")
