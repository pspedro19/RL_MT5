#!/usr/bin/env python3
"""
Funciones de calendario y festivos del mercado USA y Forex
"""
import calendar
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Set
import pandas as pd

from config.constants import MARKET_CONFIGS

# ===============================================================================
# FUNCIONES DE CALENDARIO OPTIMIZADAS CON CACHE
# ===============================================================================

@lru_cache(maxsize=10)
def get_us_market_holidays(year: int) -> Set[datetime]:
    """Obtener festivos del mercado USA para un año (con cache)"""
    holidays = set()
    tz = MARKET_CONFIGS['US500']['timezone']
    
    specific_holidays = {
        2020: [
            (1, 1), (1, 20), (2, 17), (4, 10), (5, 25),
            (7, 3), (9, 7), (11, 26), (12, 25)
        ],
        2021: [
            (1, 1), (1, 18), (2, 15), (4, 2), (5, 31),
            (7, 5), (9, 6), (11, 25), (12, 24)
        ],
        2022: [
            (1, 3), (1, 17), (2, 21), (4, 15), (5, 30),
            (6, 20), (7, 4), (9, 5), (11, 24), (12, 26)
        ],
        2023: [
            (1, 2), (1, 16), (2, 20), (4, 7), (5, 29),
            (6, 19), (7, 4), (9, 4), (11, 23), (12, 25)
        ],
        2024: [
            (1, 1), (1, 15), (2, 19), (3, 29), (5, 27),
            (6, 19), (7, 4), (9, 2), (11, 28), (12, 25)
        ],
        2025: [
            (1, 1), (1, 20), (2, 17), (4, 18), (5, 26),
            (6, 19), (7, 4), (9, 1), (11, 27), (12, 25)
        ]
    }
    
    if year in specific_holidays:
        for month, day in specific_holidays[year]:
            holidays.add(tz.localize(datetime(year, month, day)))
    else:
        # Cálculo genérico para años no especificados
        holidays.add(tz.localize(datetime(year, 1, 1)))
        
        cal = calendar.monthcalendar(year, 1)
        mlk_day = [week[0] for week in cal if week[0] != 0][2]
        holidays.add(tz.localize(datetime(year, 1, mlk_day)))
        
        cal = calendar.monthcalendar(year, 2)
        presidents_day = [week[0] for week in cal if week[0] != 0][2]
        holidays.add(tz.localize(datetime(year, 2, presidents_day)))
        
        easter = calculate_easter(year)
        good_friday = easter - timedelta(days=2)
        holidays.add(tz.localize(good_friday))
        
        cal = calendar.monthcalendar(year, 5)
        memorial_day = cal[-1][0] if cal[-1][0] != 0 else cal[-2][0]
        holidays.add(tz.localize(datetime(year, 5, memorial_day)))
        
        if year >= 2022:
            holidays.add(tz.localize(datetime(year, 6, 19)))
        
        holidays.add(tz.localize(datetime(year, 7, 4)))
        
        cal = calendar.monthcalendar(year, 9)
        labor_day = cal[0][0] if cal[0][0] != 0 else cal[1][0]
        holidays.add(tz.localize(datetime(year, 9, labor_day)))
        
        cal = calendar.monthcalendar(year, 11)
        thanksgiving = [week[3] for week in cal if week[3] != 0][3]
        holidays.add(tz.localize(datetime(year, 11, thanksgiving)))
        
        holidays.add(tz.localize(datetime(year, 12, 25)))
        
        holidays = apply_holiday_rules(holidays, year, tz)
    
    return holidays

@lru_cache(maxsize=10)
def get_early_close_days(year: int) -> Dict[datetime, int]:
    """Obtener días con cierre temprano del mercado (con cache)"""
    early_closes = {}
    tz = MARKET_CONFIGS['US500']['timezone']
    
    cal = calendar.monthcalendar(year, 11)
    thanksgiving = [week[3] for week in cal if week[3] != 0][3]
    day_before_thanksgiving = tz.localize(datetime(year, 11, thanksgiving - 1))
    early_closes[day_before_thanksgiving] = 13
    
    christmas_eve = datetime(year, 12, 24)
    if christmas_eve.weekday() < 5:
        early_closes[tz.localize(christmas_eve)] = 13
    
    july_4 = datetime(year, 7, 4)
    if july_4.weekday() == 6:
        july_3 = tz.localize(datetime(year, 7, 3))
        if july_3.weekday() < 5:
            early_closes[july_3] = 13
    elif july_4.weekday() == 5:
        july_3 = tz.localize(datetime(year, 7, 3))
        if july_3.weekday() < 5:
            early_closes[july_3] = 13
    
    return early_closes

def apply_holiday_rules(holidays: Set[datetime], year: int, tz) -> Set[datetime]:
    """Aplicar reglas de observación de festivos (función auxiliar)"""
    adjusted_holidays = set()
    
    for holiday in holidays:
        if holiday.weekday() == 5:  # Sábado
            # Observar el viernes anterior
            adjusted_holidays.add(holiday - timedelta(days=1))
        elif holiday.weekday() == 6:  # Domingo
            # Observar el lunes siguiente
            adjusted_holidays.add(holiday + timedelta(days=1))
        else:
            adjusted_holidays.add(holiday)
    
    return adjusted_holidays

def calculate_easter(year: int) -> datetime:
    """Calcular fecha de Pascua usando algoritmo de Gauss"""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day)

def is_market_open(dt: pd.Timestamp, instrument: str = 'US500', holidays: Optional[Set[datetime]] = None) -> bool:
    """Verificar si el mercado está abierto en un momento dado"""
    if instrument == 'USDCOP':
        return is_forex_market_open(dt)
    
    # Lógica para mercado USA
    if dt.tz is None:
        dt = dt.tz_localize('UTC')
    
    market_config = MARKET_CONFIGS[instrument]
    dt_market = dt.astimezone(market_config['timezone'])
    
    if dt_market.weekday() >= 5:
        return False
    
    if holidays:
        date_only = dt_market.replace(hour=0, minute=0, second=0, microsecond=0)
        if date_only in holidays:
            return False
    
    market_open = dt_market.replace(
        hour=market_config['open_hour'],
        minute=market_config['open_minute'],
        second=0, microsecond=0
    )
    market_close = dt_market.replace(
        hour=market_config['close_hour'],
        minute=market_config['close_minute'],
        second=0, microsecond=0
    )
    
    return market_open <= dt_market <= market_close

def is_forex_market_open(dt: pd.Timestamp) -> bool:
    """Verificar si el mercado Forex está abierto en un momento dado"""
    if dt.tz is None:
        dt = dt.tz_localize('UTC')
    
    # Forex cierra desde viernes 22:00 UTC hasta domingo 22:00 UTC
    weekday = dt.weekday()
    hour = dt.hour
    
    # Viernes después de las 22:00 UTC hasta domingo antes de las 22:00 UTC
    if weekday == 4 and hour >= 22:  # Viernes 22:00+
        return False
    elif weekday == 5:  # Sábado completo
        return False
    elif weekday == 6 and hour < 22:  # Domingo antes de 22:00
        return False
    
    return True

def get_expected_trading_days(start_date: datetime, end_date: datetime, instrument: str = 'US500') -> List[datetime]:
    """Obtener lista de días de trading esperados en un período"""
    if instrument == 'USDCOP':
        return get_expected_trading_days_forex(start_date, end_date)
    
    # Lógica para mercado USA
    trading_days = []
    current = start_date.date()
    end = end_date.date()
    
    years = range(start_date.year, end_date.year + 1)
    all_holidays = set()
    for year in years:
        all_holidays.update(get_us_market_holidays(year))
    
    market_config = MARKET_CONFIGS[instrument]
    
    while current <= end:
        if current.weekday() < 5:
            dt_check = market_config['timezone'].localize(
                datetime.combine(current, datetime.min.time())
            )
            if dt_check not in all_holidays:
                trading_days.append(current)
        
        current += timedelta(days=1)
    
    return trading_days

def get_expected_trading_days_forex(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Obtener lista de días de trading esperados para Forex"""
    trading_days = []
    current = start_date.date()
    end = end_date.date()
    
    # Forex opera todos los días excepto sábados y parte del domingo
    while current <= end:
        if current.weekday() < 5:  # Lunes a Viernes
            trading_days.append(current)
        elif current.weekday() == 6:  # Domingo
            # Solo incluir si es después de las 22:00 UTC
            # Para simplificar, incluimos todo el domingo
            trading_days.append(current)
        
        current += timedelta(days=1)
    
    return trading_days

def validate_no_holiday_data(df: pd.DataFrame, instrument: str = 'US500') -> Dict:
    """Validar que no hay datos en días festivos"""
    if instrument == 'USDCOP':
        return {'valid': True, 'message': 'Forex no tiene feriados tradicionales'}
    
    # Lógica para mercado USA
    if df.empty:
        return {'valid': True, 'message': 'DataFrame vacío'}
    
    # Obtener festivos para el rango de fechas
    start_year = df['time'].min().year
    end_year = df['time'].max().year
    
    all_holidays = set()
    for year in range(start_year, end_year + 1):
        all_holidays.update(get_us_market_holidays(year))
    
    # Verificar si hay datos en festivos
    holiday_data = []
    for holiday in all_holidays:
        holiday_date = holiday.date()
        holiday_mask = df['time'].dt.date == holiday_date
        if holiday_mask.any():
            holiday_data.append({
                'date': holiday_date,
                'records': holiday_mask.sum()
            })
    
    if holiday_data:
        return {
            'valid': False,
            'message': f'Se encontraron datos en {len(holiday_data)} días festivos',
            'holiday_data': holiday_data
        }
    
    return {'valid': True, 'message': 'No se encontraron datos en días festivos'}

def get_market_config(instrument: str):
    """Obtener configuración de mercado para un instrumento específico"""
    return MARKET_CONFIGS.get(instrument, MARKET_CONFIGS['US500'])
