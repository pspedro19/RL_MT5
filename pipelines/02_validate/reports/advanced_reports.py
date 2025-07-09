"""
Generación de Reportes Avanzados para Pipeline 02
Basado en el código de referencia data_extraction_02_simple.py
"""
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import os
import hashlib
import platform
import socket
import subprocess

logger = logging.getLogger(__name__)


def generate_json_report_enhanced(df: pd.DataFrame, analysis: Dict, capture_stats: Dict, 
                                 quality_report: Dict, symbol: str, output_path: str,
                                 metadata: Dict = None) -> None:
    """Generar reporte JSON completo con métricas de calidad avanzadas"""
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '2.0_ENHANCED_QUALITY',
            'symbol': symbol,
            'system_info': get_system_info(),
            'git_info': get_git_info(),
            'broker_info': quality_report.get('broker_info', {}),
            'file_checksums': metadata.get('file_checksums', {}) if metadata else {}
        },
        'capture_statistics': capture_stats,
        'data_analysis': analysis,
        'quality_metrics': quality_report,
        'feature_list': [col for col in df.columns if col not in
                        ['time', 'open', 'high', 'low', 'close', 'tick_volume',
                         'spread', 'real_volume', 'data_flag', 'quality_score']],
        'contextual_info': {
            'adj_factor_present': 'adj_factor' in df.columns,
            'cb_level_present': 'cb_level' in df.columns,
            'halt_flag_present': 'halt_flag' in df.columns
        },
        'data_quality_summary': {
            'null_counts': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_records': df.duplicated().sum(),
            'unique_dates': df['time'].dt.date.nunique()
        },
        'statistical_summary': {
            'price_stats': {
                'open': {'mean': df['open'].mean(), 'std': df['open'].std(), 'min': df['open'].min(), 'max': df['open'].max()},
                'high': {'mean': df['high'].mean(), 'std': df['high'].std(), 'min': df['high'].min(), 'max': df['high'].max()},
                'low': {'mean': df['low'].mean(), 'std': df['low'].std(), 'min': df['low'].min(), 'max': df['low'].max()},
                'close': {'mean': df['close'].mean(), 'std': df['close'].std(), 'min': df['close'].min(), 'max': df['close'].max()}
            },
            'volume_stats': {
                'mean': df['tick_volume'].mean(),
                'std': df['tick_volume'].std(),
                'min': df['tick_volume'].min(),
                'max': df['tick_volume'].max(),
                'total': df['tick_volume'].sum()
            }
        }
    }
    
    # Guardar reporte
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Reporte JSON avanzado guardado en: {output_path}")


def generate_quality_markdown_report(quality_report: Dict, output_path: str, 
                                   metadata: Dict = None, execution_times: Dict = None) -> None:
    """Generar reporte Markdown detallado de calidad"""
    
    md_content = []
    
    # Header
    md_content.append("# REPORTE DE CALIDAD AVANZADO")
    md_content.append(f"**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append(f"**Pipeline:** 2.0 Enhanced Quality")
    md_content.append("")
    
    # Resumen ejecutivo
    md_content.append("## 📊 RESUMEN EJECUTIVO")
    summary = quality_report.get('summary', {})
    md_content.append(f"- **Total de barras esperadas:** {summary.get('total_expected_bars', 0):,}")
    md_content.append(f"- **Total de barras capturadas:** {summary.get('total_captured_bars', 0):,}")
    md_content.append(f"- **Completitud global:** {summary.get('overall_completeness', 0):.2f}%")
    md_content.append(f"- **Días de trading analizados:** {summary.get('trading_days_analyzed', 0)}")
    md_content.append("")
    
    # Distribución de fuentes
    md_content.append("### 📈 Distribución por Fuente de Datos")
    source_breakdown = summary.get('primary_source_breakdown', {})
    for source, data in source_breakdown.items():
        md_content.append(f"- **{source}:** {data['bars']:,} barras ({data['percentage']:.1f}%)")
    md_content.append("")
    
    # Análisis diario
    md_content.append("## 📅 ANÁLISIS DIARIO")
    daily_analysis = quality_report.get('daily_analysis', {})
    daily_stats = daily_analysis.get('statistics', {})
    
    md_content.append(f"- **Completitud promedio:** {daily_stats.get('mean_completeness', 0):.2f}%")
    md_content.append(f"- **Completitud mínima:** {daily_stats.get('min_completeness', 0):.2f}%")
    md_content.append(f"- **Completitud máxima:** {daily_stats.get('max_completeness', 0):.2f}%")
    md_content.append(f"- **Días con 100% completitud:** {daily_stats.get('days_100_complete', 0)}")
    md_content.append(f"- **Días con 95%+ completitud:** {daily_stats.get('days_95_complete', 0)}")
    md_content.append(f"- **Días con <90% completitud:** {daily_stats.get('days_below_90', 0)}")
    md_content.append("")
    
    # Análisis de gaps
    md_content.append("## 🔍 ANÁLISIS DE GAPS")
    gap_analysis = quality_report.get('gap_analysis', {})
    gap_summary = gap_analysis.get('summary', {})
    
    md_content.append(f"- **Total de gaps:** {gap_summary.get('total_gaps', 0)}")
    md_content.append(f"- **Gaps llenados:** {gap_summary.get('gaps_filled', 0)}")
    md_content.append(f"- **Tasa de llenado:** {gap_summary.get('gap_fill_rate', 0):.1f}%")
    md_content.append(f"- **Minutos totales de gaps:** {gap_summary.get('total_gap_minutes', 0)}")
    md_content.append("")
    
    # Distribución de gaps por tamaño
    gaps_by_size = gap_summary.get('gaps_by_size', {})
    if gaps_by_size:
        md_content.append("### 📏 Distribución de Gaps por Tamaño")
        for size, count in gaps_by_size.items():
            md_content.append(f"- **{size}:** {count} gaps")
        md_content.append("")
    
    # Análisis horario
    md_content.append("## 🕐 ANÁLISIS HORARIO")
    hourly_analysis = quality_report.get('hourly_analysis', {})
    hourly_patterns = hourly_analysis.get('hourly_patterns', [])
    
    if hourly_patterns:
        md_content.append("### 📊 Patrones de Completitud por Hora")
        md_content.append("| Hora | Completitud | Fuente Principal |")
        md_content.append("|------|-------------|-------------------|")
        
        for pattern in hourly_patterns[:10]:  # Mostrar solo las primeras 10 horas
            hour = pattern['hour']
            completeness = pattern['completeness_pct']
            primary_source = pattern['primary_source']
            md_content.append(f"| {hour} | {completeness:.1f}% | {primary_source} |")
        md_content.append("")
    
    # Análisis por año
    md_content.append("## 📈 ANÁLISIS POR AÑO")
    yearly_analysis = quality_report.get('yearly_analysis', {})
    
    if yearly_analysis:
        md_content.append("| Año | Registros | M5 Nativo | Imputado |")
        md_content.append("|-----|-----------|-----------|----------|")
        
        for year in sorted(yearly_analysis.keys()):
            year_data = yearly_analysis[year]
            total = year_data.get('total_records', 0)
            
            # Obtener porcentajes
            m5_pct = 0
            imputed_pct = 0
            
            if 'data_sources' in year_data:
                m5_data = year_data['data_sources'].get('M5', {})
                m5_pct = m5_data.get('percentage', 0)
            
            if 'data_origin' in year_data:
                imputed_data = year_data['data_origin'].get('imputed', {})
                imputed_pct = imputed_data.get('percentage', 0)
            
            md_content.append(f"| {year} | {total:,} | {m5_pct:.1f}% | {imputed_pct:.1f}% |")
        md_content.append("")
    
    # Records de peor calidad
    md_content.append("## ⚠️ RECORDS DE PEOR CALIDAD")
    worst_records = quality_report.get('worst_quality_records', [])
    
    if worst_records:
        md_content.append("| Timestamp | Score | Fuente | Flag |")
        md_content.append("|-----------|-------|--------|------|")
        
        for record in worst_records[:10]:  # Mostrar solo los 10 peores
            timestamp = record.get('timestamp', 'N/A')
            score = record.get('quality_score', 0)
            source = record.get('source', 'N/A')
            flag = record.get('data_flag', 'N/A')
            md_content.append(f"| {timestamp} | {score:.3f} | {source} | {flag} |")
        md_content.append("")

    # Sección contextual
    contextual_cols = ['adj_factor', 'cb_level', 'halt_flag']
    available = [c for c in contextual_cols if c in df.columns]
    if available:
        md_content.append("## 📝 CONTEXTO ADICIONAL")
        for col in available:
            if col == 'halt_flag':
                md_content.append(f"- **{col}:** {df[col].sum()} eventos de halt")
            else:
                md_content.append(f"- **{col}:** rango {df[col].min()} - {df[col].max()}")
        md_content.append("")
    
    # Tiempos de ejecución
    if execution_times:
        md_content.append("## ⏱️ TIEMPOS DE EJECUCIÓN")
        for phase, duration in execution_times.items():
            if duration > 0:
                md_content.append(f"- **{phase.capitalize()}:** {format_duration(duration)}")
        md_content.append("")
    
    # Información del sistema
    md_content.append("## 💻 INFORMACIÓN DEL SISTEMA")
    system_info = get_system_info()
    md_content.append(f"- **Hostname:** {system_info.get('hostname', 'N/A')}")
    md_content.append(f"- **Plataforma:** {system_info.get('platform', 'N/A')}")
    md_content.append(f"- **CPU:** {system_info.get('cpu_count', 'N/A')} cores")
    md_content.append(f"- **Memoria:** {system_info.get('memory_gb', 0):.1f} GB")
    md_content.append("")
    
    # Checksums de archivos
    if metadata and 'file_checksums' in metadata:
        md_content.append("## 🔒 CHECKSUMS DE ARCHIVOS")
        for filename, checksum in metadata['file_checksums'].items():
            md_content.append(f"- **{filename}:** `{checksum[:16]}...`")
        md_content.append("")
    
    # Footer
    md_content.append("---")
    md_content.append("*Reporte generado automáticamente por Pipeline 02 Enhanced Quality*")
    
    # Guardar archivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    logger.info(f"Reporte Markdown de calidad guardado en: {output_path}")


def generate_markdown_report_enhanced(df: pd.DataFrame, analysis: Dict, capture_stats: Dict,
                                    quality_report: Dict, symbol: str, output_path: str) -> None:
    """Generar resumen ejecutivo Markdown"""
    
    md_content = []
    
    # Header
    md_content.append("# RESUMEN EJECUTIVO - PIPELINE 02")
    md_content.append(f"**Símbolo:** {symbol}")
    md_content.append(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append("")
    
    # Resumen de datos
    md_content.append("## 📊 RESUMEN DE DATOS")
    md_content.append(f"- **Total de registros:** {len(df):,}")
    md_content.append(f"- **Período:** {df['time'].min().strftime('%Y-%m-%d')} - {df['time'].max().strftime('%Y-%m-%d')}")
    md_content.append(f"- **Días de trading:** {df['time'].dt.date.nunique()}")
    md_content.append(f"- **Features generados:** {len([c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'data_flag', 'quality_score']])}")
    md_content.append(f"- **Memoria utilizada:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    md_content.append("")
    
    # Métricas de calidad
    md_content.append("## 🎯 MÉTRICAS DE CALIDAD")
    summary = quality_report.get('summary', {})
    md_content.append(f"- **Completitud global:** {summary.get('overall_completeness', 0):.2f}%")
    
    gap_analysis = quality_report.get('gap_analysis', {})
    gap_summary = gap_analysis.get('summary', {})
    md_content.append(f"- **Gaps detectados:** {gap_summary.get('total_gaps', 0)}")
    md_content.append(f"- **Gaps llenados:** {gap_summary.get('gaps_filled', 0)} ({gap_summary.get('gap_fill_rate', 0):.1f}%)")
    
    daily_analysis = quality_report.get('daily_analysis', {})
    daily_stats = daily_analysis.get('statistics', {})
    md_content.append(f"- **Días con 100% completitud:** {daily_stats.get('days_100_complete', 0)}")
    md_content.append("")
    
    # Distribución de fuentes
    md_content.append("## 📈 DISTRIBUCIÓN DE FUENTES")
    source_breakdown = summary.get('primary_source_breakdown', {})
    for source, data in source_breakdown.items():
        md_content.append(f"- **{source}:** {data['bars']:,} barras ({data['percentage']:.1f}%)")
    md_content.append("")
    
    # Análisis estadístico
    md_content.append("## 📊 ANÁLISIS ESTADÍSTICO")
    price_stats = analysis.get('statistical_summary', {}).get('price_stats', {})
    if price_stats:
        close_stats = price_stats.get('close', {})
        md_content.append(f"- **Precio promedio:** ${close_stats.get('mean', 0):.2f}")
        md_content.append(f"- **Volatilidad:** {close_stats.get('std', 0):.2f}")
        md_content.append(f"- **Rango de precios:** ${close_stats.get('min', 0):.2f} - ${close_stats.get('max', 0):.2f}")
    
    volume_stats = analysis.get('statistical_summary', {}).get('volume_stats', {})
    if volume_stats:
        md_content.append(f"- **Volumen total:** {volume_stats.get('total', 0):,}")
        md_content.append(f"- **Volumen promedio:** {volume_stats.get('mean', 0):.1f}")
    md_content.append("")
    
    # Análisis por año
    md_content.append("## 📅 ANÁLISIS POR AÑO")
    yearly_analysis = quality_report.get('yearly_analysis', {})
    
    if yearly_analysis:
        for year in sorted(yearly_analysis.keys()):
            year_data = yearly_analysis[year]
            total = year_data.get('total_records', 0)
            m5_pct = 0
            imputed_pct = 0
            
            if 'data_sources' in year_data:
                m5_data = year_data['data_sources'].get('M5', {})
                m5_pct = m5_data.get('percentage', 0)
            
            if 'data_origin' in year_data:
                imputed_data = year_data['data_origin'].get('imputed', {})
                imputed_pct = imputed_data.get('percentage', 0)
            
            md_content.append(f"- **{year}:** {total:,} registros | M5 nativo: {m5_pct:.1f}% | Imputado: {imputed_pct:.1f}%")
    md_content.append("")
    
    # Recomendaciones
    md_content.append("## 💡 RECOMENDACIONES")
    
    overall_completeness = summary.get('overall_completeness', 0)
    if overall_completeness >= 95:
        md_content.append("- ✅ **Excelente calidad de datos** - El dataset está listo para ML")
    elif overall_completeness >= 90:
        md_content.append("- ⚠️ **Buena calidad** - Considerar revisar gaps específicos")
    elif overall_completeness >= 80:
        md_content.append("- ⚠️ **Calidad aceptable** - Revisar fuentes de datos y gaps")
    else:
        md_content.append("- ❌ **Calidad insuficiente** - Revisar captura de datos")
    
    gap_fill_rate = gap_summary.get('gap_fill_rate', 0)
    if gap_fill_rate >= 90:
        md_content.append("- ✅ **Excelente llenado de gaps**")
    elif gap_fill_rate >= 70:
        md_content.append("- ⚠️ **Llenado de gaps aceptable**")
    else:
        md_content.append("- ❌ **Llenado de gaps insuficiente**")
    
    md_content.append("")
    
    # Footer
    md_content.append("---")
    md_content.append("*Resumen generado por Pipeline 02 Enhanced Quality*")
    
    # Guardar archivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    logger.info(f"Resumen ejecutivo guardado en: {output_path}")


def get_system_info() -> Dict[str, Any]:
    """Obtener información del sistema"""
    return {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'memory_gb': get_available_memory_gb(),
        'os_version': platform.version(),
        'architecture': platform.machine()
    }


def get_git_info() -> Dict[str, str]:
    """Obtener información de git"""
    git_info = {
        'commit': 'unknown',
        'branch': 'unknown',
        'tag': 'unknown',
        'dirty': False,
        'remote': 'unknown'
    }
    
    try:
        # Commit actual
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            git_info['commit'] = result.stdout.strip()[:8]
        
        # Rama actual
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()
        
        # Tag más cercano
        result = subprocess.run(['git', 'describe', '--tags', '--abbrev=0'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            git_info['tag'] = result.stdout.strip()
        
        # Estado del repositorio
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            git_info['dirty'] = bool(result.stdout.strip())
            
        # Remote origin
        result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            git_info['remote'] = result.stdout.strip()
            
    except Exception as e:
        logger.debug(f"Error obteniendo información de git: {e}")
    
    return git_info


def get_available_memory_gb() -> float:
    """Obtener memoria disponible en GB"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return 0.0


def format_duration(seconds: float) -> str:
    """Formatear duración en formato legible"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h" 