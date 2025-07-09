#!/usr/bin/env python3
"""
Script de prueba para las mejoras de calidad implementadas
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Añadir el directorio del pipeline al path
sys.path.append('pipelines/01_capture')

from data.processing import detect_gaps_optimized, is_within_market_hours
from data.quality.quality_tracker import DataQualityTracker
from utils.report_generator import generate_quality_markdown_report, generate_json_report_enhanced

REQUIRED_FILES = [
    'pipelines/01_capture/data/processing.py',
    'pipelines/01_capture/data/quality/quality_tracker.py',
]

for path in REQUIRED_FILES:
    if not os.path.exists(path):
        print(f"Faltan archivos requeridos: {path}")
        sys.exit(1)

def create_test_data():
    """Crear datos de prueba con gaps conocidos"""
    print("Creando datos de prueba...")
    
    # Crear fechas de prueba
    start_date = datetime(2024, 1, 1, 9, 30, tzinfo=pytz.timezone('US/Eastern'))
    end_date = datetime(2024, 1, 3, 16, 0, tzinfo=pytz.timezone('US/Eastern'))
    
    # Generar timestamps cada 5 minutos
    timestamps = []
    current = start_date
    while current <= end_date:
        # Solo incluir horario de mercado (9:30 AM - 4:00 PM ET, Lunes a Viernes)
        if (current.weekday() < 5 and 
            9.5 <= (current.hour + current.minute / 60) <= 16):
            timestamps.append(current)
        current += timedelta(minutes=5)
    
    # Crear datos de prueba
    data = []
    for i, ts in enumerate(timestamps):
        # Simular algunos gaps
        if i % 100 == 50:  # Gap cada 100 registros
            continue
        
        # Simular diferentes fuentes de datos
        if i % 3 == 0:
            source = 'M5'
            method = 'native'
            quality = 1.0
        elif i % 3 == 1:
            source = 'M1'
            method = 'aggregated'
            quality = 0.95
        else:
            source = 'M10'
            method = 'aggregated'
            quality = 0.9
        
        data.append({
            'time': ts,
            'open': 100 + np.random.normal(0, 1),
            'high': 101 + np.random.normal(0, 1),
            'low': 99 + np.random.normal(0, 1),
            'close': 100 + np.random.normal(0, 1),
            'tick_volume': int(np.random.exponential(1000)),
            'spread': np.random.uniform(0.1, 0.5),
            'real_volume': int(np.random.exponential(1000)),
            'data_flag': 'real',
            'quality_score': quality,
            'source_timeframe': source,
            'capture_method': method
        })
    
    df = pd.DataFrame(data)
    print(f"Datos de prueba creados: {len(df)} registros")
    return df

def test_gap_detection():
    """Probar la detección mejorada de gaps"""
    print("\n=== Probando detección de gaps ===")
    
    df = create_test_data()

    # Detectar gaps
    gaps_df = detect_gaps_optimized(df, 'US500')

    assert isinstance(gaps_df, pd.DataFrame), "detect_gaps_optimized debe devolver un DataFrame"
    assert not gaps_df.empty, "No se detectaron gaps"
    
    print(f"Gaps detectados: {len(gaps_df)}")
    if not gaps_df.empty:
        print("\nPrimeros 5 gaps:")
        print(gaps_df.head())
        
        # Estadísticas
        market_hours_gaps = gaps_df[gaps_df['market_hours'] == True]
        outside_market_gaps = gaps_df[gaps_df['market_hours'] == False]
        
        print(f"\nGaps dentro de horario de mercado: {len(market_hours_gaps)}")
        print(f"Gaps fuera de horario de mercado: {len(outside_market_gaps)}")
        
        if not market_hours_gaps.empty:
            imputable = market_hours_gaps[market_hours_gaps['imputation_status'] == 'imputable']
            not_imputable = market_hours_gaps[market_hours_gaps['imputation_status'] == 'not_imputable']
            print(f"  - Imputables: {len(imputable)}")
            print(f"  - No imputables: {len(not_imputable)}")

def test_quality_tracker():
    """Probar el quality tracker mejorado"""
    print("\n=== Probando quality tracker ===")
    
    df = create_test_data()
    tracker = DataQualityTracker('US500')
    
    # Trackear registros
    for _, record in df.iterrows():
        tracker.track_record(record)
    
    # Calcular barras esperadas y capturadas por día
    df['date'] = df['time'].dt.date
    days = df['date'].unique()
    expected_per_day = 78  # 9:30 a 16:00 son 6.5h = 78 barras M5
    for day in days:
        captured = (df['date'] == day).sum()
        tracker.update_daily_completeness(day, captured, expected_per_day)
    
    # Simular algunos gaps
    tracker.track_gap(
        datetime(2024, 1, 1, 10, 0, tzinfo=pytz.timezone('US/Eastern')),
        datetime(2024, 1, 1, 10, 15, tzinfo=pytz.timezone('US/Eastern')),
        15, 'brownian_bridge', 0.5, 'Gap pequeño dentro de horario de mercado'
    )
    tracker.track_gap(
        datetime(2024, 1, 1, 10, 30, tzinfo=pytz.timezone('US/Eastern')),
        datetime(2024, 1, 1, 11, 0, tzinfo=pytz.timezone('US/Eastern')),
        30, 'not_imputed', None, 'Gap muy grande para imputación segura'
    )
    tracker.track_gap(
        datetime(2024, 1, 1, 18, 0, tzinfo=pytz.timezone('US/Eastern')),
        datetime(2024, 1, 1, 19, 0, tzinfo=pytz.timezone('US/Eastern')),
        60, 'ignored', None, 'Fuera del horario de mercado (normal)'
    )
    
    quality_report = tracker.generate_quality_report()
    assert 'summary' in quality_report, "El reporte de calidad no tiene seccion summary"

    print("Reporte de calidad generado:")
    print(f"- Total de registros: {quality_report['summary']['total_records']}")
    print(f"- Completitud: {quality_report['summary']['overall_completeness']:.2f}%")
    
    # Análisis de gaps
    gap_analysis = quality_report['gap_analysis']
    print(f"\nAnálisis de gaps:")
    print(f"- Total de gaps: {gap_analysis['total_gaps']}")
    print(f"- Gaps en horario de mercado: {gap_analysis['gap_classification']['market_hours']['total']}")
    print(f"- Gaps fuera de horario: {gap_analysis['gap_classification']['outside_market_hours']['total']}")
    
    # Análisis de origen de datos
    origin_analysis = quality_report['data_origin_analysis']
    print(f"\nAnálisis de origen de datos:")
    for source, details in origin_analysis['source_details'].items():
        print(f"- {source}: {details['percentage']:.1f}% ({details['count']} registros)")

def test_early_close_expected_bars():
    """Probar cálculo de barras en días de cierre temprano"""
    print("\n=== Probando cálculo de barras esperadas para cierre temprano ===")

    tracker = DataQualityTracker('US500')
    tz = pytz.timezone('US/Eastern')
    early_close_day = datetime(2024, 11, 27, 12, 0, tzinfo=tz)

    result = tracker.calculate_expected_bars(early_close_day, early_close_day)
    key = datetime.combine(early_close_day.date(), datetime.min.time())

    expected = result.get(key)
    print(f"Barras esperadas {early_close_day.date()}: {expected}")
    assert expected == 53, "Las barras esperadas deben ser 53 en días de cierre temprano"

def test_report_generation():
    """Probar la generación de reportes mejorados"""
    print("\n=== Probando generación de reportes ===")
    
    df = create_test_data()
    tracker = DataQualityTracker('US500')
    
    # Trackear registros
    for _, record in df.iterrows():
        tracker.track_record(record)
    
    # Simular gaps
    tracker.track_gap(
        datetime(2024, 1, 1, 10, 0, tzinfo=pytz.timezone('US/Eastern')),
        datetime(2024, 1, 1, 10, 15, tzinfo=pytz.timezone('US/Eastern')),
        15, 'brownian_bridge', 0.5, 'Gap pequeño dentro de horario de mercado'
    )
    
    # Generar reportes
    quality_report = tracker.generate_quality_report()
    
    # Metadata de prueba
    metadata = {
        'system_info': {
            'hostname': 'test-machine',
            'platform': 'Windows',
            'processor': 'Intel i7',
            'cpu_count': 8,
            'memory_gb': 16.0
        },
        'git_info': {
            'commit': 'abc123',
            'branch': 'main',
            'tag': 'v2.4',
            'dirty': False,
            'remote': 'origin'
        },
        'file_checksums': {
            'test_data.csv': 'sha256:abc123...'
        }
    }
    
    # Generar reporte Markdown
    generate_quality_markdown_report(
        quality_report,
        'test_quality_report.md',
        metadata
    )
    print("Reporte Markdown generado: test_quality_report.md")
    assert os.path.exists('test_quality_report.md'), "No se generó test_quality_report.md"
    
    # Generar reporte JSON
    analysis = {'summary': {'total_records': len(df)}}
    capture_stats = {'total_captured': len(df)}
    
    generate_json_report_enhanced(
        df, analysis, capture_stats, quality_report,
        'US500', 'test_quality_report.json', metadata
    )
    print("Reporte JSON generado: test_quality_report.json")
    assert os.path.exists('test_quality_report.json'), "No se generó test_quality_report.json"

def main():
    """Función principal de prueba"""
    print("=== PRUEBA DE MEJORAS DE CALIDAD ===")
    print("Pipeline 02 MASTER HPC v2.4 - Enhanced Quality Tracking")
    print("=" * 50)
    
    try:
        test_gap_detection()
        test_quality_tracker()
        test_early_close_expected_bars()
        test_report_generation()
        
        print("\n=== TODAS LAS PRUEBAS COMPLETADAS ===")
        print("✅ Detección de gaps mejorada")
        print("✅ Quality tracker con clasificación detallada")
        print("✅ Reportes con análisis de procedencia")
        print("✅ Análisis de gaps por horario de mercado")
            except Exception as e:
        print(f"\n❌ Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
