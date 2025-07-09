#!/usr/bin/env python3
"""
Generación de reportes mejorados con métricas de calidad
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger('report_generator')

def create_ascii_sparkline(values: List[float], width: int = 20) -> str:
    """Crear sparkline ASCII de valores"""
    if not values:
        return ""
    
    chars = "▁▂▃▄▅▆▇█"
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return chars[4] * min(len(values), width)
    
    # Normalizar valores
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    
    # Seleccionar muestra si hay más valores que ancho
    if len(normalized) > width:
        step = len(normalized) / width
        sampled = [normalized[int(i * step)] for i in range(width)]
    else:
        sampled = normalized
    
    # Convertir a caracteres
    result = ""
    for val in sampled:
        idx = int(val * (len(chars) - 1))
        result += chars[idx]
    
    return result

def generate_json_report_enhanced(df: pd.DataFrame, analysis: Dict, capture_stats: Dict, 
                                 quality_report: Dict, symbol: str, output_path: str,
                                 metadata: Dict = None) -> None:
    """Generar reporte JSON completo con métricas de calidad avanzadas"""
    from features.technical.indicators import (
        NUMBA_AVAILABLE, GPU_AVAILABLE, GPU_INFO,
        BOTTLENECK_AVAILABLE
    )
    from config.constants import CPU_CORES, DASK_WORKERS
    
    # Detectar otras bibliotecas
    try:
        import ray
        RAY_AVAILABLE = True
    except ImportError:
        RAY_AVAILABLE = False
    
    try:
        import polars as pl
        POLARS_AVAILABLE = True
    except ImportError:
        POLARS_AVAILABLE = False
    
    try:
        import dask
        DASK_AVAILABLE = True
    except ImportError:
        DASK_AVAILABLE = False
    
    RAY_WORKERS = 4  # Valor por defecto
    
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '2.4_HPC_OPTIMIZED_ENHANCED_QUALITY',
            'symbol': symbol,
            'system_info': metadata.get('system_info', {}) if metadata else {},
            'git_info': metadata.get('git_info', {}) if metadata else {},
            'broker_info': quality_report.get('broker_info', {}),
            'hpc_libraries': {
                'ray': RAY_AVAILABLE,
                'numba': NUMBA_AVAILABLE,
                'gpu': GPU_AVAILABLE,
                'gpu_info': GPU_INFO if GPU_AVAILABLE else None,
                'polars': POLARS_AVAILABLE,
                'dask': DASK_AVAILABLE,
                'bottleneck': BOTTLENECK_AVAILABLE
            },
            'hardware': {
                'cpu_cores': CPU_CORES,
                'ray_workers': RAY_WORKERS if RAY_AVAILABLE else 0,
                'dask_workers': DASK_WORKERS if DASK_AVAILABLE else 0,
                'memory_gb': metadata.get('system_info', {}).get('memory_gb', 0) if metadata else 0
            },
            'file_checksums': metadata.get('file_checksums', {}) if metadata else {}
        },
        'capture_statistics': capture_stats,
        'data_analysis': analysis,
        'quality_metrics': quality_report,
        'feature_list': [col for col in df.columns if col not in 
                        ['time', 'open', 'high', 'low', 'close', 'tick_volume', 
                         'spread', 'real_volume', 'data_flag', 'quality_score']],
        'data_quality_summary': {
            'null_counts': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'source_distribution': df['source_timeframe'].value_counts().to_dict() if 'source_timeframe' in df.columns else {},
            'data_flag_distribution': df['data_flag'].value_counts().to_dict() if 'data_flag' in df.columns else {}
        },
        'execution_times': quality_report.get('execution_times', {}),
        'worst_quality_records': quality_report.get('worst_quality_records', []),
        'enhanced_gap_analysis': {
            'gap_classification': quality_report.get('gap_analysis', {}).get('gap_classification', {}),
            'imputation_summary': quality_report.get('gap_analysis', {}).get('imputation_summary', {}),
            'gap_distribution': quality_report.get('gap_analysis', {}).get('gap_distribution', {}),
            'gaps_by_day': quality_report.get('gap_analysis', {}).get('gaps_by_day', {})
        },
        'enhanced_data_origin': {
            'source_details': quality_report.get('data_origin_analysis', {}).get('source_details', {}),
            'yearly_breakdown': quality_report.get('data_origin_analysis', {}).get('yearly_breakdown', {}),
            'monthly_breakdown': quality_report.get('data_origin_analysis', {}).get('monthly_breakdown', {}),
            'data_origin_breakdown': quality_report.get('data_origin_analysis', {}).get('data_origin_breakdown', {})
        }
    }
    
    # Función para convertir tuplas a strings en el diccionario
    def convert_tuples_to_strings(obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: convert_tuples_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tuples_to_strings(item) for item in obj]
        else:
            return obj
    
    # Convertir tuplas a strings antes de serializar
    report = convert_tuples_to_strings(report)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Reporte JSON mejorado guardado: {output_path}")

def generate_quality_markdown_report(quality_report: Dict, output_path: str, 
                                   metadata: Dict = None, execution_times: Dict = None) -> None:
    """Generar reporte detallado de calidad en Markdown con todas las métricas avanzadas"""
    
    # Extraer información del reporte
    summary = quality_report.get('summary', {})
    daily = quality_report.get('daily_analysis', {})
    hourly = quality_report.get('hourly_analysis', {})
    gaps = quality_report.get('gap_analysis', {})
    sources = quality_report.get('source_contribution', {})
    imputations = quality_report.get('imputation_analysis', {})
    quality_scores = quality_report.get('quality_scores', {})
    yearly = quality_report.get('yearly_analysis', {})
    monthly = quality_report.get('monthly_analysis', {})
    origin = quality_report.get('data_origin_analysis', {})
    broker_info = quality_report.get('broker_info', {})
    worst_records = quality_report.get('worst_quality_records', [])
    
    # Extraer nueva información de gaps y procedencia
    gap_classification = gaps.get('gap_classification', {})
    imputation_summary = gaps.get('imputation_summary', {})
    source_details = origin.get('source_details', {})
    yearly_breakdown = origin.get('yearly_breakdown', {})
    monthly_breakdown = origin.get('monthly_breakdown', {})
    
    # Obtener información del sistema y git
    git_info = metadata.get('git_info', {}) if metadata else {}
    system_info = metadata.get('system_info', {}) if metadata else {}
    
    report = f"""# REPORTE DE CALIDAD DE DATOS - PIPELINE 02 MASTER HPC v2.4

**Generado:** {datetime.now().strftime('%d de %B del %Y a las %H:%M:%S')}  
**Zona Horaria:** {datetime.now().astimezone().strftime('%Z')}

---

## TABLA DE CONTENIDOS

1. [Guía de Lectura](#guía-de-lectura)
2. [Metadatos y Proveniencia](#metadatos-y-proveniencia)
3. [Resumen Ejecutivo](#resumen-ejecutivo)
4. [Cobertura y Completitud](#cobertura-y-completitud)
5. [Análisis de Origen de Datos](#análisis-de-origen-de-datos)
6. [Análisis de Gaps](#análisis-de-gaps)
7. [Análisis Temporal](#análisis-temporal)
8. [Métricas de Calidad](#métricas-de-calidad)
9. [Rendimiento y Recursos](#rendimiento-y-recursos)
10. [Conclusiones y Recomendaciones](#conclusiones-y-recomendaciones)

---

## GUÍA DE LECTURA

### Qué es este reporte?

Este documento analiza la calidad y completitud de los datos históricos del S&P 500 capturados desde MetaTrader 5. 
El pipeline intentó obtener datos cada 5 minutos (M5) durante el horario de mercado (9:30 AM - 4:00 PM ET).

### Cómo interpretar las métricas:

- **Completitud**: Porcentaje de datos capturados vs esperados. Objetivo: >=95%
- **M5 Nativo**: Datos obtenidos directamente en intervalos de 5 minutos (mejor calidad)
- **M1->M5**: Datos de 1 minuto agregados a 5 minutos (buena calidad)
- **Imputados**: Datos estimados para llenar huecos (usar con precaución)
- **Sparklines**: sparklines ASCII muestran tendencias visuales en texto

### Símbolos de estado:
- Excelente (>=95%)
- Aceptable (90-94%)
- Requiere atención (<90%)

---

## METADATOS Y PROVENIENCIA

### Información del Sistema
- **Hostname:** {system_info.get('hostname', 'unknown')}
- **Plataforma:** {system_info.get('platform', 'unknown')}
- **Procesador:** {system_info.get('processor', 'unknown')[:50]}...
- **Cores CPU:** {system_info.get('cpu_count', 0)}
- **Memoria:** {system_info.get('memory_gb', 0):.1f} GB

### Control de Versiones
- **Git Commit:** {git_info.get('commit', 'unknown')}
- **Rama:** {git_info.get('branch', 'unknown')}
- **Tag:** {git_info.get('tag', 'unknown')}
- **Estado:** {'Modificado localmente' if git_info.get('dirty', False) else 'Limpio'}
- **Remote:** {git_info.get('remote', 'unknown')}
- **Pipeline Version:** 2.4_HPC_ENHANCED_QUALITY

### Información del Broker
- **Conectado:** {'Si' if broker_info.get('connected', False) else 'No'}
- **Broker:** {broker_info.get('broker', 'unknown')}
- **Server:** {broker_info.get('server', 'unknown')}
- **Build:** {broker_info.get('build', 'unknown')}
- **Ping:** {broker_info.get('ping_ms', -1)} ms

### Checksums de Archivos
"""
    
    # Añadir checksums si metadata está disponible
    if metadata and 'file_checksums' in metadata:
        for filename, checksum in metadata['file_checksums'].items():
            report += f"- **{filename}:** `{checksum[:16]}...`\n"
    else:
        report += "- *Checksums se calcularán al finalizar*\n"
    
    report += f"""
---

## RESUMEN EJECUTIVO

### Métricas Clave de Calidad

| Métrica | Valor | Estado | Target |
|---------|-------|--------|--------|
| Completitud Global | {summary.get('overall_completeness', 0):.2f}% | {'Excelente' if summary.get('overall_completeness', 0) >= 95 else 'Aceptable' if summary.get('overall_completeness', 0) >= 90 else 'Requiere atención'} | >=95% |
| Barras Esperadas | {summary.get('total_expected_bars', 0):,} | - | - |
| Barras Capturadas | {summary.get('total_captured_bars', 0):,} | - | - |
| Días de Trading | {summary.get('trading_days_analyzed', 0)} | - | - |

### Origen de Datos Global
"""
    
    # Origen de datos global con sparkline
    monthly_completeness = []
    sparkline = ""
    
    if 'summary' in origin:
        origin_summary = origin['summary']
        real_vs_imputed = origin_summary.get('real_vs_imputed', {})
        
        # Crear sparkline de tendencia mensual si hay datos
        if monthly:
            for year in sorted(monthly.keys()):
                for month in sorted(monthly[year].keys()):
                    if monthly[year][month]['expected_bars'] > 0:
                        completeness = monthly[year][month]['completeness']
                        monthly_completeness.append(completeness)
        
        sparkline = create_ascii_sparkline(monthly_completeness[-12:]) if monthly_completeness else ""
        
        report += f"""
| Tipo de Datos | Registros | Porcentaje | Tendencia |
|---------------|-----------|------------|-----------|
| Datos Reales | {real_vs_imputed.get('real_data', {}).get('count', 0):,} | {real_vs_imputed.get('real_data', {}).get('percentage', 0):.1f}% | - |
| Datos Imputados | {real_vs_imputed.get('imputed_data', {}).get('count', 0):,} | {real_vs_imputed.get('imputed_data', {}).get('percentage', 0):.1f}% | - |

### Distribución por Timeframe Principal
| Fuente | Registros | Porcentaje | Calidad |
|--------|-----------|------------|---------|
"""
        source_breakdown = origin_summary.get('source_breakdown', {})
        report += f"| M5 Nativo | {source_breakdown.get('native_m5', {}).get('count', 0):,} | {source_breakdown.get('native_m5', {}).get('percentage', 0):.1f}% | Óptima |\n"
        report += f"| M1->M5 Agregado | {source_breakdown.get('aggregated_from_m1', {}).get('count', 0):,} | {source_breakdown.get('aggregated_from_m1', {}).get('percentage', 0):.1f}% | Buena |\n"
        
        # Otros timeframes
        for tf, data in source_breakdown.get('other_timeframes', {}).items():
            if data['percentage'] > 0.1:
                report += f"| {tf} | {data['count']:,} | {data['percentage']:.1f}% | Aceptable |\n"
    
    # Tendencia de completitud
    if monthly_completeness:
        report += f"\n**Tendencia Completitud (12 meses):** {sparkline}\n"
    
    # Añadir nueva sección de análisis de gaps
    report += f"""

---

## ANÁLISIS DETALLADO DE GAPS

### Clasificación de Gaps por Horario de Mercado

| Categoría | Cantidad | Minutos Totales | Promedio | Estado |
|-----------|----------|-----------------|----------|--------|
| Dentro de Horario de Mercado | {gap_classification.get('market_hours', {}).get('total', 0)} | {gap_classification.get('market_hours', {}).get('total_minutes', 0)} | {gap_classification.get('market_hours', {}).get('average_minutes', 0):.1f} min | {'⚠️ Requiere atención' if gap_classification.get('market_hours', {}).get('total', 0) > 0 else '✅ Sin problemas'} |
| Fuera de Horario de Mercado | {gap_classification.get('outside_market_hours', {}).get('total', 0)} | {gap_classification.get('outside_market_hours', {}).get('total_minutes', 0)} | {gap_classification.get('outside_market_hours', {}).get('average_minutes', 0):.1f} min | ✅ Normal |

### Estado de Imputación de Gaps

| Estado | Cantidad | Porcentaje | Descripción |
|--------|----------|------------|-------------|
| Imputados | {imputation_summary.get('imputed', 0)} | {imputation_summary.get('imputed', 0) / max(gaps.get('total_gaps', 1), 1) * 100:.1f}% | Gaps llenados con datos sintéticos |
| No Imputados | {imputation_summary.get('not_imputed', 0)} | {imputation_summary.get('not_imputed', 0) / max(gaps.get('total_gaps', 1), 1) * 100:.1f}% | Gaps muy grandes o problemáticos |
| Ignorados | {imputation_summary.get('ignored', 0)} | {imputation_summary.get('ignored', 0) / max(gaps.get('total_gaps', 1), 1) * 100:.1f}% | Fuera de horario de mercado |

### Razones de No Imputación

"""
    
    # Añadir razones de no imputación
    reasons = imputation_summary.get('reasons', {})
    if reasons:
        for reason, count in reasons.items():
            percentage = count / max(gaps.get('total_gaps', 1), 1) * 100
            report += f"- **{reason}:** {count} gaps ({percentage:.1f}%)\n"
    else:
        report += "- *No hay gaps sin imputar*\n"
    
    # Añadir nueva sección de procedencia de datos
    report += f"""

---

## ANÁLISIS DETALLADO DE PROCEDENCIA DE DATOS

### Desglose por Fuente y Calidad

| Fuente | Registros | Porcentaje | Calidad | Descripción |
|--------|-----------|------------|---------|-------------|
"""
    
    # Añadir fuentes de datos
    for source, details in source_details.items():
        quality_emoji = "🟢" if source == 'native_m5' else "🟡" if source == 'aggregated_m1' else "🔴"
        report += f"| {source.upper()} | {details['count']:,} | {details['percentage']:.1f}% | {quality_emoji} {details['description']} |\n"
    
    # Añadir análisis por año
    report += f"""

### Análisis de Procedencia por Año

| Año | Total | M5 Nativo | M1 Agregado | Imputados | Completitud |
|-----|-------|-----------|-------------|-----------|-------------|
"""
    
    for year in sorted(yearly_breakdown.keys()):
        data = yearly_breakdown[year]
        native_pct = data['native_m5_rate'] * 100
        m1_pct = data['aggregated_m1_rate'] * 100
        imputed_pct = data['imputed_rate'] * 100
        
        report += f"| {year} | {data['total_records']:,} | {native_pct:.1f}% | {m1_pct:.1f}% | {imputed_pct:.1f}% | - |\n"
    
    # Añadir análisis por mes (últimos 12 meses)
    report += f"""

### Análisis de Procedencia por Mes (Últimos 12 meses)

| Año-Mes | Total | M5 Nativo | M1 Agregado | Imputados | Completitud |
|---------|-------|-----------|-------------|-----------|-------------|
"""
    
    # Obtener últimos 12 meses
    all_months = []
    for year in monthly_breakdown:
        for month in monthly_breakdown[year]:
            all_months.append((year, month, monthly_breakdown[year][month]))
    
    # Ordenar por fecha y tomar últimos 12
    all_months.sort(reverse=True)
    recent_months = all_months[:12]
    
    for year, month, data in recent_months:
        native_pct = data['native_m5_rate'] * 100
        m1_pct = data['aggregated_m1_rate'] * 100
        imputed_pct = data['imputed_rate'] * 100
        completeness = data['completeness'] * 100
        
        report += f"| {year}-{month:02d} | {data['total_records']:,} | {native_pct:.1f}% | {m1_pct:.1f}% | {imputed_pct:.1f}% | {completeness:.1f}% |\n"
    
    # Resto del reporte (continuación con las secciones restantes)
    report += """
---

## COBERTURA Y COMPLETITUD

### Target vs Achieved

| Métrica | Target | Achieved | Delta | Estado |
|---------|--------|----------|-------|--------|
"""
    
    completeness = summary.get('overall_completeness', 0)
    target = 95
    delta = completeness - target
    
    report += f"| Completitud | >={target}% | {completeness:.2f}% | {delta:+.2f}% | {'Cumplido' if delta >= 0 else 'No cumplido'} |\n"
    
    # ... (Resto del reporte continúa igual que en el código original)
    
    # Añadir información de tiempos de ejecución si está disponible
    if execution_times:
        from main import format_duration
        report += f"""
---

## RENDIMIENTO Y RECURSOS

### Tiempos de Ejecución por Fase

| Fase | Tiempo | % del Total | Throughput |
|------|--------|-------------|------------|
"""
        total_time = execution_times.get('total', 0)
        for phase, time_spent in execution_times.items():
            if phase != 'total' and time_spent > 0:
                pct = (time_spent / total_time * 100) if total_time > 0 else 0
                
                # Calcular throughput aproximado
                if phase == 'capture' and summary.get('total_captured_bars', 0) > 0:
                    throughput = f"{summary['total_captured_bars'] / time_spent:.0f} bars/s"
                elif phase == 'features' and summary.get('total_captured_bars', 0) > 0:
                    throughput = f"{summary['total_captured_bars'] / time_spent:.0f} records/s"
                else:
                    throughput = "-"
                    
                report += f"| {phase.capitalize()} | {format_duration(time_spent)} | {pct:.1f}% | {throughput} |\n"
        
        report += f"\n**Tiempo Total:** {format_duration(total_time)}\n"
    
    report += """
---

*Reporte de Calidad v2.4 - Pipeline 02 MASTER HPC*  
*Generado automáticamente - No editar manualmente*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Reporte de calidad Markdown guardado: {output_path}")

def generate_markdown_report_enhanced(df: pd.DataFrame, analysis: Dict, capture_stats: Dict,
                                    quality_report: Dict, symbol: str, output_path: str) -> None:
    """Generar reporte ejecutivo mejorado en Markdown con métricas de calidad"""
    from features.technical.indicators import GPU_AVAILABLE, GPU_INFO, RAY_AVAILABLE, NUMBA_AVAILABLE
    from features.technical.indicators import POLARS_AVAILABLE, DASK_AVAILABLE, BOTTLENECK_AVAILABLE
    from config.constants import RAY_WORKERS, DASK_WORKERS
    
    gpu_status = "Activo" if GPU_AVAILABLE else "No disponible"
    if GPU_AVAILABLE and GPU_INFO:
        gpu_status += f" ({GPU_INFO.get('name', 'Unknown')} - {GPU_INFO.get('memory', 'Unknown')}MB)"
    
    # Extraer métricas clave de calidad
    quality_summary = quality_report.get('summary', {})
    completeness = quality_summary.get('overall_completeness', 0)
    gaps_summary = quality_report.get('gap_analysis', {}).get('summary', {})
    origin_data = quality_report.get('data_origin_analysis', {}).get('summary', {})
    
    report = f"""# REPORTE EJECUTIVO - PIPELINE 02 MASTER HPC v2.4

**Generado:** {datetime.now().strftime('%d de %B del %Y a las %H:%M')}  
**Símbolo:** {symbol}  
**Version:** Pipeline 02 MASTER - HPC Optimizado v2.4 (Enhanced Quality Tracking)

---

## RESUMEN EJECUTIVO

### Datos Generales
- **Total de registros capturados:** {analysis['summary']['total_records']:,}
- **Periodo:** {analysis['summary']['date_range']['start']} - {analysis['summary']['date_range']['end']}
- **Días de trading:** {analysis['summary']['trading_days']:,}
- **Completitud global:** {completeness:.2f}%

### Calidad de Datos - Métricas Clave
- **Barras esperadas:** {quality_summary.get('total_expected_bars', 0):,}
- **Barras capturadas:** {quality_summary.get('total_captured_bars', 0):,}
- **Datos reales vs imputados:** {origin_data.get('real_vs_imputed', {}).get('real_data', {}).get('percentage', 0):.1f}% reales / {origin_data.get('real_vs_imputed', {}).get('imputed_data', {}).get('percentage', 0):.1f}% imputados
- **Gaps detectados:** {gaps_summary.get('total_gaps', 0)}
- **Gaps llenados:** {gaps_summary.get('gaps_filled', 0)} ({quality_report.get('gap_analysis', {}).get('gap_fill_rate', 0):.1f}%)
- **Score promedio de calidad:** {df['quality_score'].mean():.3f}

### Distribución de Fuentes de Datos
"""
    
    # Agregar tabla de distribución de fuentes
    if 'primary_source_breakdown' in quality_summary:
        report += "\n| Fuente | Barras | Porcentaje |\n"
        report += "|--------|---------|------------|\n"
        for tf, data in quality_summary['primary_source_breakdown'].items():
            report += f"| {tf} | {data['bars']:,} | {data['percentage']:.1f}% |\n"
    
    # Resumen por año
    if 'yearly_analysis' in quality_report:
        report += "\n### Resumen por Año\n"
        report += "\n| Año | Total | M5 Nativo | M1->M5 | Imputado |\n"
        report += "|-----|-------|-----------|--------|----------|\n"
        
        for year in sorted(quality_report['yearly_analysis'].keys()):
            year_data = quality_report['yearly_analysis'][year]
            report += f"| {year} | {year_data['total_records']:,} | "
            report += f"{year_data['data_sources']['native_m5']['percentage']:.1f}% | "
            report += f"{year_data['data_sources']['aggregated_m1']['percentage']:.1f}% | "
            report += f"{year_data['data_origin']['imputed']['percentage']:.1f}% |\n"
    
    report += f"""

### Optimizaciones HPC Aplicadas
- **Ray:** {"Activo" if RAY_AVAILABLE else "No disponible"} ({RAY_WORKERS if RAY_AVAILABLE else 0} workers)
- **Numba JIT:** {"Activo" if NUMBA_AVAILABLE else "No disponible"}
- **GPU/CUDA:** {gpu_status}
- **Polars:** {"Activo" if POLARS_AVAILABLE else "No disponible"}
- **Dask:** {"Activo" if DASK_AVAILABLE else "No disponible"} ({DASK_WORKERS if DASK_AVAILABLE else 0} workers)
- **Bottleneck:** {"Activo" if BOTTLENECK_AVAILABLE else "No disponible"}

### Features Generados
- **Total de features:** {len([col for col in df.columns if col not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'data_flag', 'quality_score']])}
- **Categorías:** Indicadores técnicos, temporales, estructurales, volumen, volatilidad

---

*Pipeline 02 MASTER HPC v2.4 - Transparencia Total en el Origen de Cada Dato*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Reporte Markdown ejecutivo guardado: {output_path}")
