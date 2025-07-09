#!/usr/bin/env python3
"""
Script de validación del entorno para RL_SP500
Versión mejorada con validaciones completas de dependencias, configuración y estructura
"""
import sys
import os
import platform
import subprocess
import importlib
from pathlib import Path
import logging

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
        logging.FileHandler('environment_validation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def safe_print(message):
    """Imprimir mensaje de forma segura en Windows"""
    try:
        print(message)
        logger.info(message)
    except UnicodeEncodeError:
        safe_message = message.replace('✓', '[OK]').replace('✗', '[ERROR]').replace('🚀', '[START]').replace('❌', '[ERROR]').replace('✅', '[SUCCESS]').replace('🎉', '[SUCCESS]').replace('💡', '[INFO]').replace('📁', '[DIR]').replace('🔍', '[SEARCH]').replace('⚠️', '[WARN]').replace('📊', '[INFO]')
        print(safe_message)
        logger.info(safe_message)

def check_python_version():
    """Verificar versión de Python"""
    safe_print("🔍 Verificando versión de Python...")
    
    version = sys.version_info
    min_version = (3, 8)
    
    if version >= min_version:
        safe_print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        safe_print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requiere Python 3.8+")
        return False

def check_required_packages():
    """Verificar paquetes requeridos"""
    safe_print("\n🔍 Verificando paquetes requeridos...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'MetaTrader5': 'MetaTrader5',
        'pytz': 'pytz',
        'numba': 'numba'
    }
    
    optional_packages = {
        'ray': 'ray',
        'cudf': 'cudf',
        'polars': 'polars',
        'dask': 'dask',
        'bottleneck': 'bottleneck',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_required = []
    available_optional = []
    missing_optional = []
    
    # Verificar paquetes requeridos
    for package_name, import_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            safe_print(f"  ✅ {package_name}")
        except ImportError:
            missing_required.append(package_name)
            safe_print(f"  ❌ {package_name} (faltante)")
    
    # Verificar paquetes opcionales
    for package_name, import_name in optional_packages.items():
        try:
            importlib.import_module(import_name)
            available_optional.append(package_name)
            safe_print(f"  ✓ {package_name} (optimización)")
        except ImportError:
            missing_optional.append(package_name)
            safe_print(f"  - {package_name} (opcional)")
    
    if missing_required:
        safe_print(f"\n⚠️  Paquetes requeridos faltantes: {', '.join(missing_required)}")
        safe_print("💡 Instalar con: pip install " + " ".join(missing_required))
        return False
    
    if available_optional:
        safe_print(f"\n🚀 Optimizaciones disponibles: {', '.join(available_optional)}")
    
    if missing_optional:
        safe_print(f"\n💡 Paquetes opcionales no instalados: {', '.join(missing_optional)}")
        safe_print("💡 Para mejor rendimiento, instalar: pip install " + " ".join(missing_optional))
    
    return True

def check_mt5_connection():
    """Verificar conexión a MetaTrader5"""
    safe_print("\n🔍 Verificando conexión a MetaTrader5...")
    
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            safe_print("❌ No se pudo inicializar MetaTrader5")
            safe_print("💡 Asegúrate de que MetaTrader5 esté instalado y ejecutándose")
            return False
        
        # Verificar conexión
        account_info = mt5.account_info()
        if account_info is None:
            safe_print("❌ No se pudo obtener información de la cuenta")
            safe_print("💡 Verifica que estés conectado a MetaTrader5")
            return False
        
        safe_print(f"✅ MetaTrader5 conectado - Cuenta: {account_info.login}")
        safe_print(f"📊 Servidor: {account_info.server}")
        safe_print(f"💰 Balance: {account_info.balance}")
        
        # Verificar símbolos disponibles
        symbols = mt5.symbols_get()
        if symbols:
            safe_print(f"📈 Símbolos disponibles: {len(symbols)}")
            # Buscar US500
            us500_symbols = [s for s in symbols if 'US500' in s.name or 'SP500' in s.name]
            if us500_symbols:
                safe_print(f"✅ Símbolos US500/SP500 encontrados: {len(us500_symbols)}")
                for symbol in us500_symbols[:3]:  # Mostrar primeros 3
                    safe_print(f"  - {symbol.name}")
            else:
                safe_print("⚠️  No se encontraron símbolos US500/SP500")
        
        mt5.shutdown()
        return True
        
    except ImportError:
        safe_print("❌ MetaTrader5 no está instalado")
        safe_print("💡 Instalar con: pip install MetaTrader5")
        return False
    except Exception as e:
        safe_print(f"❌ Error verificando MetaTrader5: {e}")
        return False

def check_project_structure():
    """Verificar estructura del proyecto"""
    safe_print("\n🔍 Verificando estructura del proyecto...")
    
    required_dirs = [
        'pipelines',
        'pipelines/01_capture',
        'pipelines/02_validate',
        'data',
        'logs'
    ]
    
    required_files = [
        'run.py',
        'pipelines/01_capture/main.py',
        'pipelines/02_validate/run.py',
        'pipelines/02_validate/config.py',
        'pipelines/02_validate/main.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Verificar directorios
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            safe_print(f"  ✅ {dir_path}/")
        else:
            missing_dirs.append(dir_path)
            safe_print(f"  ❌ {dir_path}/ (faltante)")
    
    # Verificar archivos
    for file_path in required_files:
        if os.path.exists(file_path):
            safe_print(f"  ✅ {file_path}")
        else:
            missing_files.append(file_path)
            safe_print(f"  ❌ {file_path} (faltante)")
    
    if missing_dirs or missing_files:
        safe_print(f"\n⚠️  Elementos faltantes:")
        if missing_dirs:
            safe_print(f"  Directorios: {', '.join(missing_dirs)}")
        if missing_files:
            safe_print(f"  Archivos: {', '.join(missing_files)}")
        return False
    
    return True

def check_system_info():
    """Verificar información del sistema"""
    safe_print("\n🔍 Información del sistema...")
    
    safe_print(f"📊 Sistema operativo: {platform.system()} {platform.release()}")
    safe_print(f"📊 Arquitectura: {platform.machine()}")
    safe_print(f"📊 Procesador: {platform.processor()}")
    safe_print(f"📊 Python: {sys.version}")
    
    # Verificar memoria disponible
    try:
        import psutil
        memory = psutil.virtual_memory()
        safe_print(f"📊 Memoria total: {memory.total / (1024**3):.1f} GB")
        safe_print(f"📊 Memoria disponible: {memory.available / (1024**3):.1f} GB")
        safe_print(f"📊 Uso de memoria: {memory.percent}%")
        
        # Verificar CPU
        cpu_count = psutil.cpu_count()
        safe_print(f"📊 CPUs: {cpu_count}")
        
    except ImportError:
        safe_print("📊 psutil no disponible - información de memoria limitada")
    
    # Verificar GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')
            for i, gpu in enumerate(gpu_info):
                if gpu:
                    name, memory = gpu.split(', ')
                    safe_print(f"📊 GPU {i+1}: {name} ({memory} MB)")
        else:
            safe_print("📊 GPU: No detectada o nvidia-smi no disponible")
    except Exception:
        safe_print("📊 GPU: No detectada")
    
    return True

def check_encoding_support():
    """Verificar soporte de encoding"""
    safe_print("\n🔍 Verificando soporte de encoding...")
    
    try:
        # Probar caracteres Unicode
        test_chars = "✓✗🚀❌✅🎉💡📁🔍⚠️📊"
        safe_print(f"✅ Caracteres Unicode soportados: {test_chars}")
        
        # Verificar encoding del sistema
        if hasattr(sys.stdout, 'encoding'):
            safe_print(f"✅ Encoding stdout: {sys.stdout.encoding}")
        else:
            safe_print("⚠️  Encoding stdout: No disponible")
        
        # Verificar locale
        try:
            import locale
            current_locale = locale.getlocale()
            safe_print(f"✅ Locale: {current_locale}")
        except Exception as e:
            safe_print(f"⚠️  Locale: Error - {e}")
        
        return True
        
    except Exception as e:
        safe_print(f"❌ Error verificando encoding: {e}")
        return False

def generate_requirements_file():
    """Generar archivo requirements.txt"""
    safe_print("\n🔍 Generando requirements.txt...")
    
    try:
        import pkg_resources
        installed_packages = [d.project_name for d in pkg_resources.working_set]
        
        # Paquetes relevantes para el proyecto
        relevant_packages = [
            'pandas', 'numpy', 'MetaTrader5', 'pytz', 'numba',
            'ray', 'cudf', 'polars', 'dask', 'bottleneck',
            'scikit-learn', 'matplotlib', 'seaborn', 'psutil'
        ]
        
        requirements = []
        for package in relevant_packages:
            if package in installed_packages:
                try:
                    version = pkg_resources.get_distribution(package).version
                    requirements.append(f"{package}=={version}")
                except Exception:
                    requirements.append(package)
        
        with open('requirements.txt', 'w', encoding='utf-8') as f:
            f.write("# Requirements generados automáticamente\n")
            f.write("# Para instalar: pip install -r requirements.txt\n\n")
            for req in sorted(requirements):
                f.write(f"{req}\n")
        
        safe_print(f"✅ requirements.txt generado con {len(requirements)} paquetes")
        return True
        
    except Exception as e:
        safe_print(f"❌ Error generando requirements.txt: {e}")
        return False

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validar entorno para RL_SP500')
    parser.add_argument('--generate-requirements', action='store_true', 
                       help='Generar archivo requirements.txt')
    parser.add_argument('--skip-mt5', action='store_true',
                       help='Saltar verificación de MetaTrader5')
    
    args = parser.parse_args()
    
    safe_print("="*80)
    safe_print("🔍 VALIDACIÓN DEL ENTORNO - RL_SP500")
    safe_print("="*80)
    
    # Contador de verificaciones
    total_checks = 0
    passed_checks = 0
    
    # 1. Verificar Python
    total_checks += 1
    if check_python_version():
        passed_checks += 1
    
    # 2. Verificar paquetes
    total_checks += 1
    if check_required_packages():
        passed_checks += 1
    
    # 3. Verificar estructura del proyecto
    total_checks += 1
    if check_project_structure():
        passed_checks += 1
    
    # 4. Verificar MetaTrader5 (opcional)
    if not args.skip_mt5:
        total_checks += 1
        if check_mt5_connection():
            passed_checks += 1
    
    # 5. Verificar información del sistema
    total_checks += 1
    if check_system_info():
        passed_checks += 1
    
    # 6. Verificar encoding
    total_checks += 1
    if check_encoding_support():
        passed_checks += 1
    
    # 7. Generar requirements.txt si se solicita
    if args.generate_requirements:
        total_checks += 1
        if generate_requirements_file():
            passed_checks += 1
    
    # Resumen final
    safe_print("\n" + "="*80)
    safe_print("📊 RESUMEN DE VALIDACIÓN")
    safe_print("="*80)
    safe_print(f"Verificaciones pasadas: {passed_checks}/{total_checks}")
    
    if passed_checks == total_checks:
        safe_print("🎉 ¡ENTORNO VALIDADO EXITOSAMENTE!")
        safe_print("✅ El proyecto está listo para ejecutarse")
        return 0
    else:
        failed_checks = total_checks - passed_checks
        safe_print(f"⚠️  {failed_checks} verificaciones fallaron")
        safe_print("💡 Revisa los errores anteriores y corrige los problemas")
        safe_print("💡 Ejecuta este script nuevamente después de las correcciones")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        safe_print("\n⚠️  Validación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        safe_print(f"\n❌ Error crítico: {e}")
        logger.error(f"Error crítico en main: {e}", exc_info=True)
        sys.exit(1) 