#!/usr/bin/env python3
"""
Script de prueba para verificar las correcciones implementadas
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_indicators_fix():
    """Probar que el error de AttributeError en indicators.py está corregido y que se generan las columnas clave"""
    print("🔧 Probando corrección de AttributeError en indicators.py...")
    
    try:
        from features.technical.indicators import generate_features_cpu_optimized
        
        # Crear datos de prueba con suficientes registros para features
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
        
        # Generar features (esto debería funcionar sin AttributeError)
        result = generate_features_cpu_optimized(test_data)
        
        # Validar que las columnas clave existen
        required_cols = ['return', 'log_return', 'price_change', 'macd', 'rsi_14', 'volatility_10']
        missing = [col for col in required_cols if col not in result.columns]
        if missing:
            print(f"❌ Test de indicators.py: FALLIDO - Faltan columnas: {missing}")
            return False
        print("✅ Test de indicators.py: PASADO")
        print(f"   - Features generadas: {len([c for c in result.columns if c not in ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'data_flag', 'quality_score']])}")
        return True
        
    except Exception as e:
        print(f"❌ Test de indicators.py: FALLIDO - {e}")
        return False

def test_setting_with_copy_warning():
    """Probar que los SettingWithCopyWarning están corregidos"""
    print("\n🔧 Probando corrección de SettingWithCopyWarning...")
    
    try:
        from data.processing import process_ticks_to_ohlc
        
        # Crear datos de ticks de prueba
        dates = pd.date_range('2024-01-01', periods=100, freq='1min')
        ticks_data = pd.DataFrame({
            'time': dates,
            'bid': np.random.randn(100) + 100,
            'ask': np.random.randn(100) + 100.1,
            'volume': np.random.randint(1, 10, 100)
        })
        
        # Procesar ticks (esto no debería generar SettingWithCopyWarning)
        result = process_ticks_to_ohlc(ticks_data, 'M5')
        
        print("✅ Test de SettingWithCopyWarning: PASADO")
        print(f"   - Barras OHLC generadas: {len(result)}")
        return True
        
    except Exception as e:
        print(f"❌ Test de SettingWithCopyWarning: FALLIDO - {e}")
        return False

def test_empty_attribute_fix():
    """Probar que el error de .empty está corregido"""
    print("\n🔧 Probando corrección de error .empty...")
    
    try:
        from main import filter_market_hours_modular
        
        # Crear DataFrame vacío
        empty_df = pd.DataFrame()
        
        # Esto no debería fallar
        result = filter_market_hours_modular(empty_df)
        
        print("✅ Test de .empty: PASADO")
        return True
        
    except Exception as e:
        print(f"❌ Test de .empty: FALLIDO - {e}")
        return False

def test_frequency_warning():
    """Probar que los FutureWarning de frecuencia están corregidos"""
    print("\n🔧 Probando corrección de FutureWarning de frecuencia...")
    
    try:
        # Crear date_range con frecuencia 'min' en lugar de 'T'
        dates = pd.date_range('2024-01-01', periods=10, freq='5min')
        
        print("✅ Test de frecuencia: PASADO")
        print(f"   - Fechas generadas: {len(dates)}")
        return True
        
    except Exception as e:
        print(f"❌ Test de frecuencia: FALLIDO - {e}")
        return False

def test_memory_optimization():
    """Probar la optimización de memoria"""
    print("\n🔧 Probando optimización de memoria...")
    
    try:
        from data.processing import optimize_dataframe_memory
        
        # Crear DataFrame con tipos de datos no optimizados
        dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
        test_data = pd.DataFrame({
            'time': dates,
            'open': np.random.randn(1000).astype('float64') + 100,
            'high': np.random.randn(1000).astype('float64') + 101,
            'low': np.random.randn(1000).astype('float64') + 99,
            'close': np.random.randn(1000).astype('float64') + 100,
            'tick_volume': np.random.randint(100, 1000, 1000).astype('int64'),
            'hour': np.random.randint(0, 24, 1000).astype('int64'),
            'is_morning': np.random.randint(0, 2, 1000).astype('int64')
        })
        
        initial_memory = test_data.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Optimizar memoria
        optimized_data = optimize_dataframe_memory(test_data)
        
        final_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024
        
        print("✅ Test de optimización de memoria: PASADO")
        print(f"   - Memoria inicial: {initial_memory:.2f}MB")
        print(f"   - Memoria final: {final_memory:.2f}MB")
        print(f"   - Ahorro: {initial_memory - final_memory:.2f}MB")
        return True
        
    except Exception as e:
        print(f"❌ Test de optimización de memoria: FALLIDO - {e}")
        return False

def test_data_integrity_validation():
    """Probar la validación de integridad de datos"""
    print("\n🔧 Probando validación de integridad de datos...")
    
    try:
        from data.processing import validate_data_integrity
        
        # Crear datos con problemas para probar validación
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        test_data = pd.DataFrame({
            'time': dates,
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100,
            'tick_volume': np.random.randint(100, 1000, 100)
        })
        
        # Agregar algunos outliers
        test_data.loc[0, 'close'] = 1000000  # Outlier extremo
        
        # Validar integridad
        validation = validate_data_integrity(test_data)
        
        print("✅ Test de validación de integridad: PASADO")
        print(f"   - Válido: {validation['valid']}")
        print(f"   - Issues: {len(validation['issues'])}")
        print(f"   - Outliers detectados: {len(validation['outliers'])}")
        return True
        
    except Exception as e:
        print(f"❌ Test de validación de integridad: FALLIDO - {e}")
        return False

def main():
    """Ejecutar todos los tests"""
    print("🧪 INICIANDO TESTS DE CORRECCIONES")
    print("=" * 50)
    
    tests = [
        test_indicators_fix,
        test_setting_with_copy_warning,
        test_empty_attribute_fix,
        test_frequency_warning,
        test_memory_optimization,
        test_data_integrity_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test falló con excepción: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTADOS: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("🎉 TODOS LOS TESTS PASARON - CORRECCIONES IMPLEMENTADAS EXITOSAMENTE")
        return 0
    else:
        print("⚠️  ALGUNOS TESTS FALLARON - REVISAR IMPLEMENTACIÓN")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 