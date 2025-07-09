#!/usr/bin/env python3
"""
Script de debug para verificar las columnas generadas por generate_features_cpu_optimized
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_indicators():
    print("\n=== INICIO DEBUG INDICATORS ===", flush=True)
    try:
        from features.technical.indicators import generate_features_cpu_optimized
        
        # Crear datos de prueba
        dates = pd.date_range('2024-01-01', periods=300, freq='5min')
        test_data = pd.DataFrame({
            'time': dates,
            'open': np.random.randn(300) + 100,
            'high': np.random.randn(300) + 101,
            'low': np.random.randn(300) + 99,
            'close': np.random.randn(300) + 100,
            'tick_volume': np.random.randint(100, 1000, 300),
            'spread': np.random.randn(300) * 0.1,
            'real_volume': np.random.randint(100, 1000, 300),
            'data_flag': 'test',
            'quality_score': 1.0
        })
        print(f"📊 Datos de entrada: {len(test_data)} registros", flush=True)
        print(f"📋 Columnas de entrada: {list(test_data.columns)}", flush=True)
        
        # Generar features
        result = generate_features_cpu_optimized(test_data)
        
        print(f"\n📊 Datos de salida: {len(result)} registros", flush=True)
        print(f"📋 Columnas de salida ({len(result.columns)}):", flush=True)
        for i, col in enumerate(result.columns):
            print(f"  {i+1:2d}. {col}", flush=True)
        
        return_cols = [col for col in result.columns if 'return' in col.lower()]
        print(f"\n🔍 Columnas con 'return': {return_cols}", flush=True)
        
        required_cols = ['return', 'log_return', 'price_change', 'macd', 'rsi_14', 'volatility_10']
        print(f"\n✅ Verificación de columnas requeridas:", flush=True)
        for col in required_cols:
            status = "✅ PRESENTE" if col in result.columns else "❌ FALTANTE"
            print(f"  {col}: {status}", flush=True)
        
        print(f"\n📈 Muestra de datos (primeras 5 filas):", flush=True)
        for col in ['return', 'log_return', 'price_change']:
            if col in result.columns:
                print(f"{col}: {result[col].head().values}", flush=True)
        print("\n=== FIN DEBUG INDICATORS ===", flush=True)
        return True
    except Exception as e:
        print(f"❌ Error en debug: {e}", flush=True)
        import traceback
        traceback.print_exc()
        print("\n=== FIN DEBUG INDICATORS (con error) ===", flush=True)
        return False

if __name__ == "__main__":
    debug_indicators() 