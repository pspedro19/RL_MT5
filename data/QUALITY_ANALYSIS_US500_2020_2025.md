# REPORTE DE CALIDAD DE DATOS - PIPELINE 02 MASTER HPC v2.4

**Generado:** 08 de July del 2025 a las 19:00:28  
**Zona Horaria:** Hora est. Pacífico, Sudamérica

---

## TABLA DE CONTENIDOS

1. [Guía de Lectura](#guía-de-lectura)
2. [Metadatos y Proveniencia](#metadatos-y-proveniencia)
3. [Resumen Ejecutivo](#resumen-ejecutivo)
4. [Cobertura y Completitud](#cobertura-y-completitud)
5. [Análisis de Origen de Datos](#análisis-de-origen-de-datos)
6. [Análisis de Gaps](#análisis-de-gaps)
7. [Análisis Temporal](#análisis-temporal)
8. [Métricas de Calidad](#métricas-de-calidad)
9. [Rendimiento y Recursos](#rendimiento-y-recursos)
10. [Conclusiones y Recomendaciones](#conclusiones-y-recomendaciones)

---

## GUÍA DE LECTURA

### Qué es este reporte?

Este documento analiza la calidad y completitud de los datos históricos del S&P 500 capturados desde MetaTrader 5. 
El pipeline intentó obtener datos cada 5 minutos (M5) durante el horario de mercado (9:30 AM - 4:00 PM ET).

### Cómo interpretar las métricas:

- **Completitud**: Porcentaje de datos capturados vs esperados. Objetivo: >=95%
- **M5 Nativo**: Datos obtenidos directamente en intervalos de 5 minutos (mejor calidad)
- **M1->M5**: Datos de 1 minuto agregados a 5 minutos (buena calidad)
- **Imputados**: Datos estimados para llenar huecos (usar con precaución)
- **Sparklines**: sparklines ASCII muestran tendencias visuales en texto

### Símbolos de estado:
- Excelente (>=95%)
- Aceptable (90-94%)
- Requiere atención (<90%)

---

## METADATOS Y PROVENIENCIA

### Información del Sistema
- **Hostname:** unknown
- **Plataforma:** unknown
- **Procesador:** unknown...
- **Cores CPU:** 0
- **Memoria:** 0.0 GB

### Control de Versiones
- **Git Commit:** unknown
- **Rama:** unknown
- **Tag:** unknown
- **Estado:** Limpio
- **Remote:** unknown
- **Pipeline Version:** 2.4_HPC_ENHANCED_QUALITY

### Información del Broker
- **Conectado:** No
- **Broker:** unknown
- **Server:** unknown
- **Build:** unknown
- **Ping:** -1 ms

### Checksums de Archivos
- **us500_m5_hpc_2020_2025.parquet:** `f4e09fd998aa63ed...`
- **us500_m5_hpc_2020_2025.csv:** `dd49004330e46079...`

---

## RESUMEN EJECUTIVO

### Métricas Clave de Calidad

| Métrica | Valor | Estado | Target |
|---------|-------|--------|--------|
| Completitud Global | 0.00% | Requiere atención | >=95% |
| Barras Esperadas | 0 | - | - |
| Barras Capturadas | 0 | - | - |
| Días de Trading | 0 | - | - |

### Origen de Datos Global


---

## ANÁLISIS DETALLADO DE GAPS

### Clasificación de Gaps por Horario de Mercado

| Categoría | Cantidad | Minutos Totales | Promedio | Estado |
|-----------|----------|-----------------|----------|--------|
| Dentro de Horario de Mercado | 0 | 0 | 0.0 min | ✅ Sin problemas |
| Fuera de Horario de Mercado | 0 | 0 | 0.0 min | ✅ Normal |

### Estado de Imputación de Gaps

| Estado | Cantidad | Porcentaje | Descripción |
|--------|----------|------------|-------------|
| Imputados | 0 | 0.0% | Gaps llenados con datos sintéticos |
| No Imputados | 0 | 0.0% | Gaps muy grandes o problemáticos |
| Ignorados | 0 | 0.0% | Fuera de horario de mercado |

### Razones de No Imputación

- *No hay gaps sin imputar*


---

## ANÁLISIS DETALLADO DE PROCEDENCIA DE DATOS

### Desglose por Fuente y Calidad

| Fuente | Registros | Porcentaje | Calidad | Descripción |
|--------|-----------|------------|---------|-------------|


### Análisis de Procedencia por Año

| Año | Total | M5 Nativo | M1 Agregado | Imputados | Completitud |
|-----|-------|-----------|-------------|-----------|-------------|


### Análisis de Procedencia por Mes (Últimos 12 meses)

| Año-Mes | Total | M5 Nativo | M1 Agregado | Imputados | Completitud |
|---------|-------|-----------|-------------|-----------|-------------|

---

## COBERTURA Y COMPLETITUD

### Target vs Achieved

| Métrica | Target | Achieved | Delta | Estado |
|---------|--------|----------|-------|--------|
| Completitud | >=95% | 0.00% | -95.00% | No cumplido |

---

## RENDIMIENTO Y RECURSOS

### Tiempos de Ejecución por Fase

| Fase | Tiempo | % del Total | Throughput |
|------|--------|-------------|------------|
| Filtering | 0.4s | 0.1% | - |
| Imputation | 1.1m | 7.8% | - |
| Features | 1.9s | 0.2% | - |
| Cleaning | 0.9s | 0.1% | - |
| Io | 2.5m | 18.2% | - |
| Validation | 0.2s | 0.0% | - |
| Memory_optimization | 0.7s | 0.1% | - |

**Tiempo Total:** 13.6m

---

*Reporte de Calidad v2.4 - Pipeline 02 MASTER HPC*  
*Generado automáticamente - No editar manualmente*
