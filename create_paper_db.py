import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_paper_db(db_path: str = "paper.db", days: int = 1500, start_capital: float = 100_000.0):
    """
    Generates a realistic SQLite database ('paper.db') containing 'equity_log' 
    and 'trades' tables, perfectly formatted for the Apex Epoch Quant Engine.
    """
    logger = logging.getLogger(__name__)
    db_file = Path(db_path)
    
    if db_file.exists():
        logger.info(f"Removing existing {db_path}...")
        db_file.unlink()

    # 1. Generate Realistic Dates (Business days only)
    logger.info(f"Generating {days} days of simulated market data...")
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)

    # 2. Simulate Equity Curve (Geometric Brownian Motion)
    # Target: ~15% Annualized Return, ~12% Annualized Volatility
    np.random.seed(42)
    daily_drift = 0.15 / 252
    daily_vol = 0.12 / np.sqrt(252)
    
    # Inject a realistic market crash (e.g., a 15% drop over 20 days somewhere in the middle)
    returns = np.random.normal(daily_drift, daily_vol, days)
    crash_start = days // 2
    returns[crash_start:crash_start+20] -= 0.015 
    
    equity_curve = start_capital * np.exp(np.cumsum(returns))

    df_equity = pd.DataFrame({
        'date': dates,
        'total_equity': equity_curve
    })

    # 3. Simulate Trade Log
    # Assume the system makes a trade roughly 30% of the time
    trade_mask = np.random.rand(days) < 0.30
    trade_dates = dates[trade_mask]
    n_trades = len(trade_dates)
    
    # Win rate ~55%, Risk/Reward ~1.2
    is_win = np.random.rand(n_trades) < 0.55
    
    realized_pnl = np.empty(n_trades)
    realized_pnl[is_win] = np.random.normal(500, 150, sum(is_win))      # Winning trades
    realized_pnl[~is_win] = np.random.normal(-400, 100, sum(~is_win))   # Losing trades
    
    # Trade Value (Gross position size per trade)
    trade_value = np.abs(realized_pnl) * np.random.uniform(10, 50, n_trades)
    
    # Gross Exposure (System leverage at the time of trade, between 50% and 150%)
    gross_exposure = np.random.uniform(0.5, 1.5, n_trades)

    df_trades = pd.DataFrame({
        'date': trade_dates,
        'realized_pnl': realized_pnl,
        'trade_value': trade_value,
        'gross_exposure': gross_exposure
    })

    # 4. Write to SQLite
    logger.info(f"Writing to {db_path}...")
    with sqlite3.connect(db_file) as conn:
        df_equity.to_sql('equity_log', conn, index=False, if_exists='replace')
        df_trades.to_sql('trades', conn, index=False, if_exists='replace')

    logger.info(f"Successfully created {db_path}!")
    logger.info(f" - Equity rows: {len(df_equity)}")
    logger.info(f" - Trade rows:  {len(df_trades)}")

if __name__ == "__main__":
    generate_paper_db("paper.db")
