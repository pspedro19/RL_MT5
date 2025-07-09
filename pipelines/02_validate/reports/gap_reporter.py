import pandas as pd
import pandas_market_calendars as mcal
import os
import json
from typing import List, Dict, Optional, Union

class GapReporter:
    """
    Detecta y reporta gaps legítimos en datos de trading (excluyendo festivos y fuera de mercado).
    Genera logs detallados y métricas de cobertura.
    """
    def __init__(self, symbol: str = 'US500', freq: str = '5min', log_dir: str = 'logs/'):
        self.symbol = symbol
        self.freq = freq
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def generate_trading_calendar(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        market_times = mcal.date_range(schedule, frequency=self.freq)
        return market_times

    def detect_gaps(self, df: pd.DataFrame, time_col: str = 'time', start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        df[time_col] = pd.to_datetime(df[time_col])
        if not start_date:
            start_date = str(df[time_col].min().date())
        if not end_date:
            end_date = str(df[time_col].max().date())
        market_times = self.generate_trading_calendar(start_date, end_date)
        df_times = set(df[time_col])
        gaps = []
        for ts in market_times:
            if ts not in df_times:
                gaps.append({'timestamp': ts, 'reason': 'missing, not holiday, not early close'})
        return {
            'gaps': gaps,
            'total_slots': len(market_times),
            'missing_slots': len(gaps),
            'coverage_pct': 100 * (len(market_times) - len(gaps)) / len(market_times) if len(market_times) > 0 else 0
        }

    def save_gap_log(self, gaps: List[Dict], filename: str = 'gaps_detailed_log.csv'):
        if not gaps:
            return
        df_gaps = pd.DataFrame(gaps)
        df_gaps.to_csv(os.path.join(self.log_dir, filename), index=False)

    def daily_coverage_metrics(self, df: pd.DataFrame, time_col: str = 'time', start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        df[time_col] = pd.to_datetime(df[time_col])
        if not start_date:
            start_date = str(df[time_col].min().date())
        if not end_date:
            end_date = str(df[time_col].max().date())
        market_times = self.generate_trading_calendar(start_date, end_date)
        market_df = pd.DataFrame({'timestamp': market_times})
        market_df['date'] = market_df['timestamp'].dt.date
        df['date'] = df[time_col].dt.date
        daily_expected = market_df.groupby('date').size()
        daily_actual = df.groupby('date').size()
        daily_coverage = (daily_actual / daily_expected * 100).fillna(0)        metrics = pd.DataFrame({
            'expected_slots': daily_expected,
            'actual_slots': daily_actual,
            'coverage_pct': daily_coverage
        })
        return metrics

    def save_daily_coverage(self, metrics: pd.DataFrame, filename: str = 'daily_coverage.csv'):
        """Guardar métricas de cobertura diaria en CSV"""
        if metrics is None or metrics.empty:
            return
        metrics.to_csv(os.path.join(self.log_dir, filename))

    def save_gaps(self, gaps: Union[List[Dict], pd.DataFrame], filename: str = 'gaps.parquet'):
        """Guardar lista de gaps detectados en formato Parquet"""
        if isinstance(gaps, list):
            if not gaps:
                return
            df_gaps = pd.DataFrame(gaps)
        else:
            df_gaps = gaps
        if df_gaps is None or df_gaps.empty:
            return
        df_gaps.to_parquet(os.path.join(self.log_dir, filename), index=False)

    def save_duplicates(self, duplicates: pd.DataFrame, filename: str = 'duplicates.parquet'):
        """Guardar registros duplicados en Parquet"""
        if duplicates is None or duplicates.empty:
            return
        duplicates.to_parquet(os.path.join(self.log_dir, filename), index=False)

    def save_outliers(self, outliers: pd.DataFrame, filename: str = 'outliers.parquet'):
        """Guardar registros considerados outliers"""
        if outliers is None or outliers.empty:
            return
        outliers.to_parquet(os.path.join(self.log_dir, filename), index=False)

    def save_nan_summary(self, nan_summary: Dict, filename: str = 'nan_summary.json'):
        """Guardar resumen de valores NaN en JSON"""
        if not nan_summary:
            return
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nan_summary, f, indent=2, default=str)
