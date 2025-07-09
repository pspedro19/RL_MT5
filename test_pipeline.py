#!/usr/bin/env python3
"""
Script de prueba rápida para verificar que todas las mejoras funcionan correctamente
Versión 1.0 - Pruebas de funcionalidad básica
"""
import sys
import os
import subprocess
import time
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

def safe_print(message):
    """Imprimir mensaje de forma segura en Windows"""
    try:
        print(message)
    except UnicodeEncodeError:
        safe_message = message.replace('✓', '[OK]').replace('✗', '[ERROR]').replace('🚀', '[START]').replace('❌', '[ERROR]').replace('✅', '[SUCCESS]').replace('🎉', '[SUCCESS]').replace('💡', '[INFO]').replace('📁', '[DIR]').replace('🔍', '[SEARCH]').replace('⚠️', '[WARN]').replace('📊', '[INFO]')
        print(safe_message)

def test_environment_validation():
    """Probar validación del entorno"""
    safe_print("🔍 Probando validación del entorno...")
    
    try:
        result = subprocess.run([sys.executable, 'validate_environment.py', '--skip-mt5'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            safe_print("✅ Validación del entorno: EXITOSA")
            return True
        else:
            safe_print(f"❌ Validación del entorno: FALLÓ")
            safe_print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        safe_print("❌ Validación del entorno: TIMEOUT")
        return False
    except Exception as e:
        safe_print(f"❌ Validación del entorno: ERROR - {e}")
        return False

def test_clean_data_dry_run():
    """Probar limpieza de datos en modo simulación"""
    safe_print("🔍 Probando limpieza de datos (dry-run)...")
    
    try:
        result = subprocess.run([sys.executable, 'clean_data.py', '--dry-run'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            safe_print("✅ Limpieza de datos (dry-run): EXITOSA")
            return True
        else:
            safe_print(f"❌ Limpieza de datos (dry-run): FALLÓ")
            safe_print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        safe_print("❌ Limpieza de datos (dry-run): TIMEOUT")
        return False
    except Exception as e:
        safe_print(f"❌ Limpieza de datos (dry-run): ERROR - {e}")
        return False

def test_pipeline_02_config():
    """Probar configuración del Pipeline 02"""
    safe_print("🔍 Probando configuración del Pipeline 02...")
    
    try:
        # Verificar que existe el archivo de configuración
        config_path = Path("pipelines/02_validate/config.py")
        if not config_path.exists():
            safe_print("❌ Archivo de configuración no encontrado")
            return False
        
        # Intentar importar la configuración
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        if hasattr(config_module, 'get_config'):
            config = config_module.get_config()
            safe_print("✅ Configuración del Pipeline 02: EXITOSA")
            return True
        else:
            safe_print("❌ Función get_config() no encontrada")
            return False
            
    except Exception as e:
        safe_print(f"❌ Configuración del Pipeline 02: ERROR - {e}")
        return False

def test_script_help():
    """Probar que los scripts muestran ayuda correctamente"""
    safe_print("🔍 Probando ayuda de scripts...")
    
    scripts_to_test = [
        ('run.py', ['--help']),
        ('clean_data.py', ['--help']),
        ('validate_environment.py', ['--help'])
    ]
    
    success_count = 0
    for script, args in scripts_to_test:
        try:
            result = subprocess.run([sys.executable, script] + args, 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and 'usage:' in result.stdout.lower():
                safe_print(f"✅ Ayuda de {script}: EXITOSA")
                success_count += 1
            else:
                safe_print(f"❌ Ayuda de {script}: FALLÓ")
                
        except Exception as e:
            safe_print(f"❌ Ayuda de {script}: ERROR - {e}")
    
    return success_count == len(scripts_to_test)

def test_unicode_support():
    """Probar soporte de caracteres Unicode"""
    safe_print("🔍 Probando soporte Unicode...")
    
    test_messages = [
        "✓ Test de caracteres Unicode",
        "🚀 Pipeline funcionando",
        "✅ Prueba exitosa",
        "💡 Información importante",
        "📁 Directorio de datos",
        "🔍 Búsqueda en progreso",
        "⚠️ Advertencia del sistema",
        "📊 Métricas de rendimiento"
    ]
    
    try:
        for message in test_messages:
            safe_print(message)
        
        safe_print("✅ Soporte Unicode: EXITOSO")
        return True
        
    except Exception as e:
        safe_print(f"❌ Soporte Unicode: ERROR - {e}")
        return False

def test_file_structure():
    """Probar estructura de archivos"""
    safe_print("🔍 Probando estructura de archivos...")
    
    required_files = [
        'run.py',
        'clean_data.py',
        'validate_environment.py',
        'pipelines/02_validate/config.py',
        'pipelines/02_validate/run.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if not missing_files:
        safe_print("✅ Estructura de archivos: EXITOSA")
        return True
    else:
        safe_print(f"❌ Estructura de archivos: FALTAN ARCHIVOS")
        for file_path in missing_files:
            safe_print(f"  - {file_path}")
        return False

def test_imports():
    """Probar importaciones básicas"""
    safe_print("🔍 Probando importaciones...")
    
    try:
        import pandas as pd
        import numpy as np
        import pytz
        safe_print("✅ Importaciones básicas: EXITOSAS")
        
        # Probar importaciones opcionales
        optional_imports = []
        try:
            import numba
            optional_imports.append("numba")
        except ImportError:
            pass
        
        try:
            import ray
            optional_imports.append("ray")
        except ImportError:
            pass
        
        if optional_imports:
            safe_print(f"✅ Optimizaciones disponibles: {', '.join(optional_imports)}")
        
        return True
        
    except ImportError as e:
        safe_print(f"❌ Importaciones básicas: ERROR - {e}")
        return False

def main():
    """Función principal de pruebas"""
    safe_print("="*80)
    safe_print("🧪 PRUEBAS RÁPIDAS - RL_SP500 v3.0")
    safe_print("="*80)
    
    tests = [
        ("Estructura de archivos", test_file_structure),
        ("Importaciones básicas", test_imports),
        ("Soporte Unicode", test_unicode_support),
        ("Ayuda de scripts", test_script_help),
        ("Configuración Pipeline 02", test_pipeline_02_config),
        ("Limpieza de datos (dry-run)", test_clean_data_dry_run),
        ("Validación del entorno", test_environment_validation),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        safe_print(f"\n{'='*60}")
        safe_print(f"🚀 Ejecutando: {test_name}")
        safe_print(f"{'='*60}")
        
        try:
            if test_func():
                passed_tests += 1
                safe_print(f"✅ {test_name}: PASÓ")
            else:
                safe_print(f"❌ {test_name}: FALLÓ")
        except Exception as e:
            safe_print(f"❌ {test_name}: ERROR - {e}")
    
    # Resumen final
    safe_print("\n" + "="*80)
    safe_print("📊 RESUMEN DE PRUEBAS")
    safe_print("="*80)
    safe_print(f"Pruebas pasadas: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        safe_print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        safe_print("✅ El sistema está listo para usar")
        safe_print("\n💡 Próximos pasos:")
        safe_print("   1. Ejecutar: python run.py")
        safe_print("   2. O probar: python run.py --symbol US500 --start 2024 --end 2024")
        return 0
    else:
        failed_tests = total_tests - passed_tests
        safe_print(f"⚠️  {failed_tests} pruebas fallaron")
        safe_print("💡 Revisa los errores anteriores y corrige los problemas")
        safe_print("💡 Ejecuta: python validate_environment.py para diagnóstico completo")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        safe_print("\n⚠️  Pruebas interrumpidas por el usuario")
        sys.exit(1)
    except Exception as e:
        safe_print(f"\n❌ Error crítico en pruebas: {e}")
        sys.exit(1) 