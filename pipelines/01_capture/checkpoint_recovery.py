#!/usr/bin/env python3
"""
Script de recuperación para el pipeline de captura
Permite continuar desde donde falló el proceso
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class PipelineCheckpoint:
    """Gestor de checkpoints para el pipeline"""
    
    def __init__(self, checkpoint_file: str = 'pipeline_checkpoint.json'):
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data = {}
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Cargar checkpoint si existe"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self.checkpoint_data = json.load(f)
                logger.info(f"Checkpoint cargado: {self.checkpoint_data}")
                return self.checkpoint_data
            except Exception as e:
                logger.error(f"Error cargando checkpoint: {e}")
        return None
    
    def save_checkpoint(self, year: int, status: str = 'completed', error: str = None):
        """Guardar checkpoint"""
        self.checkpoint_data = {
            'last_completed_year': year,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'error': error
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=2)
            logger.info(f"Checkpoint guardado: año {year}, estado: {status}")
        except Exception as e:
            logger.error(f"Error guardando checkpoint: {e}")
    
    def get_start_year(self, default_start: int = 2020) -> int:
        """Obtener año de inicio basado en checkpoint"""
        if self.checkpoint_data:
            last_year = self.checkpoint_data.get('last_completed_year', default_start - 1)
            return last_year + 1
        return default_start
    
    def clear_checkpoint(self):
        """Limpiar checkpoint"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            logger.info("Checkpoint eliminado")
    
    def is_recovery_mode(self) -> bool:
        """Verificar si estamos en modo recuperación"""
        return os.path.exists(self.checkpoint_file)

def run_with_recovery(start_year: int = 2020, end_year: int = 2025, symbol: str = 'US500'):
    """Ejecutar pipeline con recuperación automática"""
    checkpoint = PipelineCheckpoint()
    
    # Cargar checkpoint si existe
    checkpoint.load_checkpoint()
    
    # Determinar año de inicio
    actual_start_year = checkpoint.get_start_year(start_year)
    
    if checkpoint.is_recovery_mode():
        logger.info(f"Modo recuperación: continuando desde año {actual_start_year}")
    else:
        logger.info(f"Iniciando pipeline desde año {actual_start_year}")
    
    for year in range(actual_start_year, end_year + 1):
        try:
            logger.info(f"Procesando año {year}...")
            
            # Aquí iría la lógica de procesamiento del año
            # Por ahora es un placeholder
            process_year(year, symbol)
            
            # Guardar checkpoint exitoso
            checkpoint.save_checkpoint(year, 'completed')
            
            logger.info(f"Año {year} procesado exitosamente")
            
        except Exception as e:
            logger.error(f"Error procesando año {year}: {e}")
            
            # Guardar checkpoint con error
            checkpoint.save_checkpoint(year, 'error', str(e))
            
            # Re-lanzar excepción para que el usuario decida qué hacer
            raise
    
    # Limpiar checkpoint al completar todo
    checkpoint.clear_checkpoint()
    logger.info("Pipeline completado exitosamente")

def process_year(year: int, symbol: str):
    """Procesar un año específico (placeholder)"""
    # Esta función sería reemplazada con la lógica real del pipeline
    logger.info(f"Procesando {symbol} para año {year}")
    # Simular procesamiento
    import time
    time.sleep(1)

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Ejecutar con recuperación
    run_with_recovery(2020, 2025, 'US500') 