#!/usr/bin/env python3
"""
Script mejorado para ejecutar el pipeline completo: Pipeline 01 + Pipeline 02
Versión 3.0 con limpieza automática, validaciones robustas, mejor manejo de errores
y soporte completo para Windows con UTF-8
"""
import sys
import os
import subprocess
import shutil
import glob
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# ===============================================================================
# CONFIGURACIÓN DE ENCODING Y LOGGING PARA WINDOWS
# ===============================================================================
if sys.platform == "win32":
    import locale
    try:
        # Configurar encoding UTF-8 para Windows
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        # Configurar locale para Windows
        locale.setlocale(locale.LC_ALL, '')
    except AttributeError:
        # Fallback para Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except Exception:
        pass

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_execution.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def safe_print(message):
    """Imprimir mensaje de forma segura en Windows con caracteres Unicode"""
    try:
        print(message)
        logger.info(message)
    except UnicodeEncodeError:
        # Reemplazar caracteres problemáticos
        safe_message = message.replace('✓', '[OK]').replace('✗', '[ERROR]').replace('🚀', '[START]').replace('❌', '[ERROR]').replace('✅', '[SUCCESS]').replace('🎉', '[SUCCESS]').replace('💡', '[INFO]').replace('📁', '[DIR]').replace('🧹', '[CLEAN]').replace('🗑️', '[DELETE]').replace('⚠️', '[WARN]').replace('📦', '[BACKUP]').replace('🔍', '[SEARCH]')
        print(safe_message)
        logger.info(safe_message)

def clean_data_directory():
    """Limpiar la carpeta data al inicio de cada ejecución, preservando solo la carpeta validated/"""
    data_dir = Path("data")
    if not data_dir.exists():
        safe_print("📁 Creando directorio data/...")
        data_dir.mkdir(exist_ok=True)
        return True
    
    safe_print("🧹 Limpiando carpeta data/ (preservando validated/)...")
    
    # Contar archivos antes de limpiar
    files_before = len(list(data_dir.glob("*")))
    
    # Preservar la carpeta validated/ si existe
    validated_dir = data_dir / "validated"
    validated_backup = None
    
    if validated_dir.exists():
        safe_print("📦 Preservando carpeta validated/...")
        # Crear backup temporal
        validated_backup = data_dir / "validated_backup"
        if validated_backup.exists():
            shutil.rmtree(validated_backup)
        shutil.copytree(validated_dir, validated_backup)
    
    # Eliminar todos los archivos y carpetas excepto validated/
    deleted_count = 0
    for item in data_dir.iterdir():
        if item.name != "validated" and item.name != "validated_backup":
            try:
                if item.is_file():
                    item.unlink()
                    safe_print(f"🗑️  Eliminado archivo: {item.name}")
                    deleted_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    safe_print(f"🗑️  Eliminada carpeta: {item.name}")
                    deleted_count += 1
            except Exception as e:
                safe_print(f"⚠️  No se pudo eliminar {item.name}: {e}")
    
    # Restaurar validated/ si existía
    if validated_backup and validated_backup.exists():
        if validated_dir.exists():
            shutil.rmtree(validated_dir)
        shutil.move(validated_backup, validated_dir)
        safe_print("📦 Carpeta validated/ restaurada")
    
    # Verificar limpieza
    files_after = len(list(data_dir.glob("*")))
    safe_print(f"✅ Limpieza completada: {files_before} → {files_after} elementos ({deleted_count} eliminados)")
    return True

def validate_python_environment():
    """Validar que el entorno Python tenga las dependencias necesarias"""
    safe_print("🔍 Validando entorno Python...")
    
    required_packages = [
        'pandas', 'numpy', 'MetaTrader5', 'pytz', 'numba'
    ]
    
    optional_packages = [
        'ray', 'cudf', 'polars', 'dask', 'bottleneck'
    ]
    
    missing_packages = []
    available_optimizations = []
    
    for package in required_packages:
        try:
            __import__(package)
            safe_print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            safe_print(f"  ✗ {package} (faltante)")
    
    for package in optional_packages:
        try:
            __import__(package)
            available_optimizations.append(package)
            safe_print(f"  ✓ {package} (optimización)")
        except ImportError:
            safe_print(f"  - {package} (opcional)")
    
    if missing_packages:
        safe_print(f"⚠️  Paquetes faltantes: {', '.join(missing_packages)}")
        safe_print("💡 Instalar con: pip install " + " ".join(missing_packages))
        return False
    
    if available_optimizations:
        safe_print(f"🚀 Optimizaciones disponibles: {', '.join(available_optimizations)}")
    
    safe_print("✅ Entorno Python validado")
    return True

def validate_pipeline_files():
    """Validar que existan los archivos necesarios de los pipelines"""
    safe_print("🔍 Validando archivos de pipeline...")
    
    required_files = [
        "pipelines/01_capture/main.py",
        "pipelines/02_validate/run.py",
        "pipelines/02_validate/config.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            safe_print(f"  ✓ {file_path}")
        else:
            missing_files.append(file_path)
            safe_print(f"  ✗ {file_path} (faltante)")
    
    if missing_files:
        safe_print(f"❌ Archivos faltantes: {', '.join(missing_files)}")
        return False
    
    safe_print("✅ Archivos de pipeline validados")
    return True

def run_command(command, description, timeout=7200):  # Aumentado a 2 horas
    """Ejecutar comando con mejor manejo de errores y timeout.

    Returns
    -------
    Tuple[bool, str]
        success flag and captured output (stderr y stdout combinados).
    """
    safe_print(f"\n{'='*60}")
    safe_print(f"🚀 {description}")
    safe_print(f"{'='*60}")
    safe_print(f"Comando: {command}")
    safe_print("-" * 60)
    
    start_time = time.time()
    captured_output = []
    
    try:
        # Usar subprocess.Popen para mejor control
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        
        # Mostrar output en tiempo real y capturar
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                safe_print(line)
                captured_output.append(line)
        
        # Esperar a que termine
        return_code = process.poll()
        
        joined_output = "\n".join(captured_output)

        if return_code == 0:
            elapsed_time = time.time() - start_time
            safe_print(f"✅ {description} completado exitosamente en {elapsed_time:.1f}s")
            return True, joined_output
        else:
            safe_print(f"❌ {description} falló con código {return_code}")
            return False, joined_output
            
    except subprocess.TimeoutExpired:
        safe_print(f"❌ {description} excedió el tiempo límite ({timeout}s)")
        process.kill()
        return False, ""
    except Exception as e:
        safe_print(f"❌ Error ejecutando {description}: {e}")
        return False, str(e)

def find_latest_data_file(symbol, start_year, end_year):
    """Buscar el archivo de datos más reciente con mejor lógica"""
    safe_print(f"🔍 Buscando archivo de datos para {symbol} ({start_year}-{end_year})...")
    
    # Patrones de búsqueda (ordenados por prioridad)
    search_patterns = [
        f"data/{symbol.lower()}_m5_hpc_{start_year}_{end_year}.parquet",
        f"data/{symbol.lower()}_m5_hpc_{start_year}_{end_year}.csv",
        f"data/{symbol.lower()}_m5_hpc_{start_year}_{end_year}.feather"
    ]
    
    # Buscar archivo exacto
    for pattern in search_patterns:
        if os.path.exists(pattern):
            safe_print(f"✅ Archivo encontrado: {pattern}")
            return pattern
    
    # Si no se encuentra, buscar archivos similares
    safe_print("💡 Archivo exacto no encontrado, buscando alternativas...")
    
    # Buscar archivos con patrones similares
    similar_patterns = [
        f"data/*{symbol.lower()}*{start_year}*{end_year}*.parquet",
        f"data/*{symbol.lower()}*{start_year}*{end_year}*.csv",
        f"data/*{symbol.lower()}*{start_year}*{end_year}*.feather"
    ]
    
    for pattern in similar_patterns:
        files = glob.glob(pattern)
        if files:
            # Ordenar por fecha de modificación (más reciente primero)
            files.sort(key=os.path.getmtime, reverse=True)
            latest_file = files[0]
            safe_print(f"✅ Archivo alternativo encontrado: {latest_file}")
            return latest_file
    
    # Si aún no se encuentra, buscar cualquier archivo del símbolo
    fallback_patterns = [
        f"data/*{symbol.lower()}*.parquet",
        f"data/*{symbol.lower()}*.csv",
        f"data/*{symbol.lower()}*.feather"
    ]
    
    for pattern in fallback_patterns:
        files = glob.glob(pattern)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            latest_file = files[0]
            safe_print(f"⚠️  Usando archivo disponible: {latest_file}")
            return latest_file
    
    safe_print(f"❌ No se encontró ningún archivo de datos para {symbol}")
    return None

def validate_data_file(file_path):
    """Validar que el archivo de datos existe y es accesible"""
    if not file_path or not os.path.exists(file_path):
        return False, "Archivo no existe"
    
    try:
        # Verificar que se puede leer
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Archivo vacío"
        
        # Intentar leer las primeras líneas para verificar formato
        if file_path.endswith('.csv'):
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line or ',' not in first_line:
                    return False, "Formato CSV inválido"
        
        safe_print(f"✅ Archivo validado: {file_path} ({file_size / (1024*1024):.1f} MB)")
        return True, "OK"
        
    except Exception as e:
        return False, f"Error validando archivo: {e}"

def create_execution_report(success, pipeline_results, start_time, end_time):
    """Crear reporte de ejecución"""
    report = {
        'execution_date': datetime.now().isoformat(),
        'success': success,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': (end_time - start_time).total_seconds(),
        'pipeline_results': pipeline_results,
        'system_info': {
            'platform': sys.platform,
            'python_version': sys.version,
            'encoding': sys.stdout.encoding if hasattr(sys.stdout, 'encoding') else 'unknown'
        }
    }
    
    report_path = f"execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        safe_print(f"📊 Reporte de ejecución guardado: {report_path}")
    except Exception as e:
        safe_print(f"⚠️  No se pudo guardar reporte: {e}")

def aggregate_quality_reports(symbol: str, start_year: str, end_year: str) -> None:
    """Unir reportes de calidad de ambos pipelines y generar resumen"""
    try:
        p1_path = f"data/quality_report_{symbol.lower()}_{start_year}_{end_year}.json"
        if not os.path.exists(p1_path):
            candidates = glob.glob(f"data/quality_report_{symbol.lower()}*.json")
            p1_path = sorted(candidates, key=os.path.getmtime, reverse=True)[0] if candidates else None
    except Exception:
        p1_path = None

    def load_json(path):
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                safe_print(f"⚠️  No se pudo cargar {path}: {e}")
        return {}

    report1 = load_json(p1_path)
    validated_dir = Path("data/validated")
    report2 = load_json(validated_dir / "pipeline_info.json")
    checklist = load_json(validated_dir / "quality_checklist_report.json")

    summary = {
        'completeness_pct': report1.get('summary', {}).get('overall_completeness'),
        'integrity_passed': report2.get('market_validations', {}).get('market_hours', {}).get('is_valid'),
        'quality_score': checklist.get('overall_score'),
        'total_records': report1.get('summary', {}).get('total_records')
    }

    final_report = {
        'generated_at': datetime.now().isoformat(),
        'pipeline_01_report': report1,
        'pipeline_02_info': report2,
        'pipeline_02_checklist': checklist,
        'summary': summary
    }

    output_json = validated_dir / "final_quality_report.json"
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        safe_print(f"📊 Reporte final de calidad guardado: {output_json}")
    except Exception as e:
        safe_print(f"⚠️  No se pudo guardar reporte final: {e}")

    # Generar versión Markdown sencilla
    output_md = validated_dir / "final_quality_report.md"
    try:
        md_lines = [
            "# Resumen Final de Calidad",
            f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"- **Completitud:** {summary.get('completeness_pct', 'N/A')}%",
            f"- **Integridad de mercado:** {summary.get('integrity_passed')}",
            f"- **Score checklist:** {summary.get('quality_score')}",
            f"- **Registros totales:** {summary.get('total_records')}"
        ]
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        safe_print(f"📊 Reporte final Markdown guardado: {output_md}")
    except Exception as e:
        safe_print(f"⚠️  No se pudo guardar reporte Markdown: {e}")

def main():
    """Función principal mejorada"""
    parser = argparse.ArgumentParser(description='Pipeline completo con limpieza automática y validaciones robustas')
    parser.add_argument('--symbol', default='US500', help='Símbolo a procesar (default: US500)')
    parser.add_argument('--start', default='2020', help='Año de inicio (default: 2020)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y'), help='Año de fin (default: año actual)')
    parser.add_argument('--no-clean', action='store_true', help='No limpiar carpeta data/')
    parser.add_argument('--skip-pipeline1', action='store_true', help='Saltar Pipeline 01 (captura)')
    parser.add_argument('--skip-pipeline2', action='store_true', help='Saltar Pipeline 02 (validación)')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout en segundos (default: 7200)')
    
    args = parser.parse_args()
    
    # Timer de ejecución
    execution_start = datetime.now()
    
    safe_print("="*80)
    safe_print("🚀 INICIANDO PIPELINE COMPLETO - VERSIÓN 3.0")
    safe_print("="*80)
    safe_print(f"Configuración:")
    safe_print(f"  - Símbolo: {args.symbol}")
    safe_print(f"  - Período: {args.start} - {args.end}")
    safe_print(f"  - Limpieza automática: {'No' if args.no_clean else 'Sí'}")
    safe_print(f"  - Pipeline 01: {'Saltado' if args.skip_pipeline1 else 'Ejecutar'}")
    safe_print(f"  - Pipeline 02: {'Saltado' if args.skip_pipeline2 else 'Ejecutar'}")
    safe_print(f"  - Timeout: {args.timeout}s")
    
    # Validaciones iniciales
    if not validate_python_environment():
        safe_print("❌ Validación del entorno falló")
        return 1
    
    if not validate_pipeline_files():
        safe_print("❌ Validación de archivos falló")
        return 1
    
    # Limpieza de carpeta data (si no se desactiva)
    if not args.no_clean:
        if not clean_data_directory():
            safe_print("❌ Limpieza de carpeta data falló")
            return 1
    
    # Resultados de pipelines
    pipeline_results = {
        'pipeline_01': {'success': False, 'error': None},
        'pipeline_02': {'success': False, 'error': None}
    }
    
    # Pipeline 01: Captura de datos
    if not args.skip_pipeline1:
        safe_print("\n" + "="*60)
        safe_print("PIPELINE 01: CAPTURA DE DATOS")
        safe_print("="*60)
        
        pipeline01_cmd = (
            f"python pipelines/01_capture/main.py "
            f"--symbol {args.symbol} "
            f"--start {args.start} "
            f"--end {args.end} "
            f"--output data"
        )
        
        success, output = run_command(pipeline01_cmd, "Pipeline 01 - Captura de datos", args.timeout)
        if success:
            pipeline_results['pipeline_01']['success'] = True
            safe_print("✅ Pipeline 01 completado exitosamente")
        else:
            last_line = output.strip().split('\n')[-1] if output else 'Pipeline 01 falló'
            pipeline_results['pipeline_01']['error'] = last_line
            safe_print(f"❌ Pipeline 01 falló: {last_line}")
            # Continuar con Pipeline 02 si hay datos disponibles
    
    # Buscar archivo de datos para Pipeline 02
    data_file = find_latest_data_file(args.symbol, args.start, args.end)
    
    if not data_file:
        safe_print("❌ No se encontró archivo de datos para Pipeline 02")
        pipeline_results['pipeline_02']['error'] = "No se encontró archivo de datos"
    else:
        # Validar archivo de datos
        is_valid, validation_msg = validate_data_file(data_file)
        if not is_valid:
            safe_print(f"❌ Archivo de datos inválido: {validation_msg}")
            pipeline_results['pipeline_02']['error'] = f"Archivo inválido: {validation_msg}"
        else:
            # Pipeline 02: Validación y procesamiento
            if not args.skip_pipeline2:
                safe_print("\n" + "="*60)
                safe_print("PIPELINE 02: VALIDACIÓN Y PROCESAMIENTO")
                safe_print("="*60)
                
                pipeline02_cmd = (
                    f"python pipelines/02_validate/run.py "
                    f"--input {data_file} "
                    f"--config pipelines/02_validate/config.py"
                )
                
                success, output = run_command(pipeline02_cmd, "Pipeline 02 - Validación y procesamiento", args.timeout)
                if success:
                    pipeline_results['pipeline_02']['success'] = True
                    safe_print("✅ Pipeline 02 completado exitosamente")
                else:
                    last_line = output.strip().split('\n')[-1] if output else 'Pipeline 02 falló'
                    pipeline_results['pipeline_02']['error'] = last_line
                    safe_print(f"❌ Pipeline 02 falló: {last_line}")
    
    # Resumen final
    execution_end = datetime.now()
    execution_duration = execution_end - execution_start
    
    safe_print("\n" + "="*80)
    safe_print("📊 RESUMEN DE EJECUCIÓN")
    safe_print("="*80)
    
    success_count = sum(1 for result in pipeline_results.values() if result['success'])
    total_pipelines = len([k for k in pipeline_results.keys() if not (k == 'pipeline_01' and args.skip_pipeline1) or (k == 'pipeline_02' and args.skip_pipeline2)])
    
    safe_print(f"Tiempo total de ejecución: {execution_duration}")
    safe_print(f"Pipelines exitosos: {success_count}/{total_pipelines}")
    
    for pipeline_name, result in pipeline_results.items():
        if result['success']:
            safe_print(f"  ✅ {pipeline_name}: Exitoso")
        else:
            error_msg = result.get('error', 'Falló')
            safe_print(f"  ❌ {pipeline_name}: {error_msg}")
    
    # Crear reporte de ejecución
    overall_success = success_count == total_pipelines
    create_execution_report(overall_success, pipeline_results, execution_start, execution_end)

    # Generar reporte de calidad combinado
    aggregate_quality_reports(args.symbol, args.start, args.end)
    
    if overall_success:
        safe_print("\n🎉 ¡PIPELINE COMPLETADO EXITOSAMENTE!")
        return 0
    else:
        safe_print("\n⚠️  Pipeline completado con errores")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        safe_print("\n⚠️  Ejecución interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        safe_print(f"\n❌ Error crítico: {e}")
        logger.error(f"Error crítico en main: {e}", exc_info=True)        sys.exit(1) 