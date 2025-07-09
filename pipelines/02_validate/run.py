#!/usr/bin/env python3
"""
Script mejorado para ejecutar el Pipeline 02 - Validación y Preparación de Datos
Versión 2.0 con mejor manejo de rutas, compatibilidad Windows y logging robusto
"""
import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Configurar encoding UTF-8 para Windows
if sys.platform == "win32":
    import locale
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        locale.setlocale(locale.LC_ALL, '')
    except AttributeError:
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
        logging.FileHandler('pipeline_02.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def safe_print(message):
    """Imprimir mensaje de forma segura en Windows"""
    try:
        print(message)
        logger.info(message)
    except UnicodeEncodeError:
        safe_message = message.replace('✓', '[OK]').replace('✗', '[ERROR]').replace('🚀', '[START]').replace('❌', '[ERROR]').replace('✅', '[SUCCESS]').replace('🎉', '[SUCCESS]').replace('💡', '[INFO]').replace('📁', '[DIR]').replace('🔍', '[SEARCH]')
        print(safe_message)
        logger.info(safe_message)

def validate_input_file(input_path):
    """Validar que el archivo de entrada existe y es accesible"""
    if not input_path or not os.path.exists(input_path):
        return False, f"Archivo no existe: {input_path}"
    
    try:
        file_size = os.path.getsize(input_path)
        if file_size == 0:
            return False, "Archivo está vacío"
        
        # Verificar extensión
        valid_extensions = ['.parquet', '.csv', '.feather']
        file_ext = Path(input_path).suffix.lower()
        if file_ext not in valid_extensions:
            return False, f"Formato de archivo no soportado: {file_ext}"
        
        safe_print(f"✅ Archivo de entrada validado: {input_path} ({file_size / (1024*1024):.1f} MB)")
        return True, "OK"
        
    except Exception as e:
        return False, f"Error validando archivo: {e}"

def find_config_file(config_path):
    """Buscar archivo de configuración con fallbacks"""
    # Si se proporciona una ruta absoluta, usarla
    if os.path.isabs(config_path):
        if os.path.exists(config_path):
            return config_path
    
    # Buscar en el directorio actual del script
    script_dir = Path(__file__).parent
    possible_paths = [
        config_path,  # Ruta relativa proporcionada
        script_dir / config_path,  # En el directorio del script
        script_dir / "config.py",  # config.py en el directorio del script
        Path("config.py"),  # En el directorio de trabajo actual
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            safe_print(f"✅ Archivo de configuración encontrado: {path}")
            return str(path)
    
    safe_print(f"❌ No se encontró archivo de configuración en: {possible_paths}")
    return None

def load_config(config_path):
    """Cargar configuración con manejo de errores robusto"""
    try:
        # Importar configuración
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        if hasattr(config_module, 'get_config'):
            config = config_module.get_config()
            safe_print("✅ Configuración cargada exitosamente")
            return config
        else:
            safe_print("❌ Función get_config() no encontrada en el archivo de configuración")
            return None
            
    except Exception as e:
        safe_print(f"❌ Error cargando configuración: {e}")
        logger.error(f"Error cargando configuración: {e}", exc_info=True)
        return None

def main():
    """Ejecutar Pipeline 02 con parámetros mejorados"""
    
    parser = argparse.ArgumentParser(description='Pipeline 02 - Validación y Preparación de Datos')
    parser.add_argument('--input', type=str, 
                       help='Archivo de entrada del Pipeline 01 (buscará automáticamente si no se especifica)')
    parser.add_argument('--output-dir', type=str, default='data/validated',
                       help='Directorio de salida')
    parser.add_argument('--config', type=str, default='config.py',
                       help='Archivo de configuración')
    parser.add_argument('--symbol', type=str, default='US500',
                       help='Símbolo para buscar archivo automáticamente')
    parser.add_argument('--start-year', type=str, default='2020',
                       help='Año de inicio para buscar archivo')
    parser.add_argument('--end-year', type=str, default='2025',
                       help='Año de fin para buscar archivo')
    
    args = parser.parse_args()
    
    safe_print("🚀 Pipeline 02 - Validación y Preparación de Datos")
    safe_print("=" * 60)
    
    # Buscar archivo de entrada si no se especifica
    input_file = args.input
    if not input_file:
        safe_print("🔍 Buscando archivo de entrada automáticamente...")
        
        # Patrones de búsqueda
        search_patterns = [
            f"data/{args.symbol.lower()}_m5_hpc_{args.start_year}_{args.end_year}.parquet",
            f"data/{args.symbol.lower()}_m5_hpc_{args.start_year}_{args.end_year}.csv",
            f"data/{args.symbol.lower()}_m5_hpc_{args.start_year}_{args.end_year}.feather",
            f"data/*{args.symbol.lower()}*.parquet",
            f"data/*{args.symbol.lower()}*.csv",
            f"data/*{args.symbol.lower()}*.feather"
        ]
        
        for pattern in search_patterns:
            import glob
            files = glob.glob(pattern)
            if files:
                # Ordenar por fecha de modificación (más reciente primero)
                files.sort(key=os.path.getmtime, reverse=True)
                input_file = files[0]
                safe_print(f"✅ Archivo encontrado: {input_file}")
                break
        
        if not input_file:
            safe_print("❌ No se encontró archivo de entrada")
            safe_print("💡 Usa --input para especificar manualmente el archivo")
            return 1
    
    safe_print(f"📁 Archivo de entrada: {input_file}")
    safe_print(f"📁 Directorio de salida: {args.output_dir}")
    
    # Validar archivo de entrada
    is_valid, validation_msg = validate_input_file(input_file)
    if not is_valid:
        safe_print(f"❌ Archivo de entrada inválido: {validation_msg}")
        return 1
    
    # Buscar y cargar configuración
    config_path = find_config_file(args.config)
    if not config_path:
        safe_print("❌ No se pudo encontrar archivo de configuración")
        return 1
    
    config = load_config(config_path)
    if not config:
        safe_print("❌ No se pudo cargar configuración")
        return 1
    
    safe_print("=" * 60)
    
    try:
        # Crear directorio de salida
        os.makedirs(args.output_dir, exist_ok=True)
        safe_print(f"📁 Directorio de salida creado: {args.output_dir}")
        
        # Importar y ejecutar el pipeline
        try:
            from main import ValidationPipeline
        except ImportError as e:
            safe_print(f"❌ Error importando ValidationPipeline: {e}")
            safe_print("💡 Asegúrate de que el archivo main.py existe en el directorio")
            return 1
        
        # Inicializar y ejecutar pipeline
        safe_print("🚀 Iniciando pipeline de validación...")
        pipeline = ValidationPipeline(config)
        results = pipeline.run(input_file, args.output_dir)
        
        if results['status'] == 'SUCCESS':
            safe_print("✅ Pipeline 02 completado exitosamente!")
            safe_print(f"⏱️  Tiempo total: {results['elapsed_time_seconds']:.2f} segundos")
            if 'quality_score' in results:
                safe_print(f"📊 Score de calidad: {results['quality_score']:.1f}/100")
            safe_print(f"📁 Archivos generados en: {args.output_dir}")
            
            # Mostrar archivos generados
            try:
                output_files = list(Path(args.output_dir).glob("*"))
                if output_files:
                    safe_print("📋 Archivos generados:")
                    for file in output_files:
                        if file.is_file():
                            size_mb = file.stat().st_size / (1024 * 1024)
                            safe_print(f"  - {file.name} ({size_mb:.1f} MB)")
                        else:
                            safe_print(f"  - {file.name}/ (carpeta)")
            except Exception as e:
                safe_print(f"⚠️  No se pudieron listar archivos generados: {e}")
                
        else:
            error_msg = results.get('error', 'Error desconocido')
            safe_print(f"❌ Error en Pipeline 02: {error_msg}")
            return 1
            
    except ImportError as e:
        safe_print(f"❌ Error importando módulos: {e}")
        safe_print("💡 Asegúrate de tener todas las dependencias instaladas:")
        safe_print("   pip install -r requirements.txt")
        logger.error(f"Error importando módulos: {e}", exc_info=True)
        return 1
    except Exception as e:
        safe_print(f"❌ Error ejecutando Pipeline 02: {e}")
        logger.error(f"Error ejecutando Pipeline 02: {e}", exc_info=True)
        return 1
    
    safe_print("🎉 Pipeline 02 completado exitosamente")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        safe_print("\n⚠️  Ejecución interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        safe_print(f"\n❌ Error crítico: {e}")
        logger.error(f"Error crítico en main: {e}", exc_info=True)
        sys.exit(1) 