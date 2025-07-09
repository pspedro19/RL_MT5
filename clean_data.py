#!/usr/bin/env python3
"""
Script para limpiar la carpeta data/ preservando solo la carpeta validated/
Versión mejorada con mejor manejo de errores y logging
"""
import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime

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
        logging.FileHandler('clean_data.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def safe_print(message):
    """Imprimir mensaje de forma segura en Windows"""
    try:
        print(message)
        logger.info(message)
    except UnicodeEncodeError:
        safe_message = message.replace('✓', '[OK]').replace('✗', '[ERROR]').replace('🚀', '[START]').replace('❌', '[ERROR]').replace('✅', '[SUCCESS]').replace('🎉', '[SUCCESS]').replace('💡', '[INFO]').replace('📁', '[DIR]').replace('🧹', '[CLEAN]').replace('🗑️', '[DELETE]').replace('⚠️', '[WARN]').replace('📦', '[BACKUP]').replace('🔍', '[SEARCH]')
        print(safe_message)
        logger.info(safe_message)

def clean_data_directory(dry_run=False):
    """Limpiar la carpeta data al inicio de cada ejecución, preservando solo la carpeta validated/"""
    data_dir = Path("data")
    
    if not data_dir.exists():
        safe_print("📁 Creando directorio data/...")
        if not dry_run:
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
        if not dry_run:
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
                if dry_run:
                    if item.is_file():
                        safe_print(f"🗑️  [DRY RUN] Eliminaría archivo: {item.name}")
                    elif item.is_dir():
                        safe_print(f"🗑️  [DRY RUN] Eliminaría carpeta: {item.name}")
                    deleted_count += 1
                else:
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
    if validated_backup and validated_backup.exists() and not dry_run:
        if validated_dir.exists():
            shutil.rmtree(validated_dir)
        shutil.move(validated_backup, validated_dir)
        safe_print("📦 Carpeta validated/ restaurada")
    
    # Verificar limpieza
    files_after = len(list(data_dir.glob("*")))
    if dry_run:
        safe_print(f"✅ [DRY RUN] Simulación completada: {files_before} → {files_after} elementos ({deleted_count} se eliminarían)")
    else:
        safe_print(f"✅ Limpieza completada: {files_before} → {files_after} elementos ({deleted_count} eliminados)")
    
    return True

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Limpiar carpeta data/ preservando validated/')
    parser.add_argument('--dry-run', action='store_true', help='Simular limpieza sin eliminar archivos')
    parser.add_argument('--force', action='store_true', help='Forzar limpieza sin confirmación')
    
    args = parser.parse_args()
    
    safe_print("="*60)
    safe_print("🧹 LIMPIADOR DE CARPETA DATA/")
    safe_print("="*60)
    
    if args.dry_run:
        safe_print("🔍 MODO SIMULACIÓN - No se eliminarán archivos")
    else:
        safe_print("⚠️  ATENCIÓN: Se eliminarán todos los archivos de data/ excepto validated/")
    
    # Verificar si hay archivos para eliminar
    data_dir = Path("data")
    if data_dir.exists():
        files_to_delete = [item for item in data_dir.iterdir() 
                          if item.name != "validated" and item.name != "validated_backup"]
        
        if not files_to_delete:
            safe_print("✅ No hay archivos para eliminar")
            return 0
        
        safe_print(f"📁 Archivos/carpetas a eliminar: {len(files_to_delete)}")
        for item in files_to_delete:
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                safe_print(f"  - {item.name} ({size_mb:.1f} MB)")
            else:
                safe_print(f"  - {item.name}/ (carpeta)")
    
    # Confirmación (si no es dry-run y no se fuerza)
    if not args.dry_run and not args.force:
        response = input("\n¿Continuar con la limpieza? (y/N): ").strip().lower()
        if response not in ['y', 'yes', 'sí', 'si']:
            safe_print("❌ Limpieza cancelada por el usuario")
            return 1
    
    # Ejecutar limpieza
    try:
        clean_data_directory(dry_run=args.dry_run)
        safe_print("✅ Limpieza completada exitosamente")
        return 0
    except Exception as e:
        safe_print(f"❌ Error durante la limpieza: {e}")
        logger.error(f"Error durante la limpieza: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        safe_print("\n⚠️  Limpieza interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        safe_print(f"\n❌ Error crítico: {e}")
        logger.error(f"Error crítico en main: {e}", exc_info=True)
        sys.exit(1) 