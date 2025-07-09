#!/usr/bin/env python3
"""
Script simple para ejecutar el pipeline de trading para cualquier símbolo
"""
import sys
import os
import argparse
from datetime import datetime

# Asegurar que el directorio actual esté en el path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Ejecutar pipeline desde 2020-01-01 hasta la fecha más reciente para el símbolo dado"""
    parser = argparse.ArgumentParser(description="Ejecutar pipeline de captura para cualquier símbolo")
    parser.add_argument('--symbol', type=str, default='US500', help='Símbolo a capturar (ej: US500, USDCOP)')
    parser.add_argument('--instrument', type=str, default='sp500', choices=['sp500', 'usdcop'], help='Instrumento a descargar')
    args = parser.parse_args()

    symbol = args.symbol
    instrument = args.instrument
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    
    print("🚀 RL_SP500 Pipeline - Ejecución Simple")
    print("=" * 50)
    print(f"Símbolo: {symbol}")
    print(f"Instrumento: {instrument}")
    print(f"Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
    print("=" * 50)
    
    # Importar y ejecutar el pipeline principal
    try:
        from main import main as pipeline_main
        
        # Simular argumentos de línea de comandos correctamente
        sys.argv = [
            'main.py',
            '--symbol', symbol,
            '--instrument', instrument,
            '--start', start_date.strftime('%Y-%m-%d'),
            '--end', end_date.strftime('%Y-%m-%d'),
            '--formats', 'parquet', 'csv', 'feather'
        ]
        
        result = pipeline_main()
        
        if result == 0:
            print("✅ Pipeline completado exitosamente!")
        else:
            print("❌ Pipeline falló con errores")
            return result
        
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        print("💡 Asegúrate de tener todas las dependencias instaladas:")
        print("   pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"❌ Error ejecutando pipeline: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Ejecutar solo el símbolo especificado en los argumentos
    exit(main()) 