#!/usr/bin/env python3
import sqlite3
import pandas as pd
import logging
from datetime import datetime
import os
import config
from backtest import BacktestEngine
from generate_report import QuantEngine, QuantReporter, DEFAULT_MIN_PERIODS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestOrchestrator")

def setup_backtest_db(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE trades (date TEXT, ticker TEXT, action TEXT, price REAL, qty REAL, commission REAL, pnl REAL, reason TEXT, equity REAL)")
        conn.execute("CREATE TABLE equity_log (date TEXT, total_equity REAL, cash REAL, positions_val REAL, est_method TEXT)")
        conn.commit()

def run_portfolio_backtest():
    db_path = "backtest.db"
    setup_backtest_db(db_path)
    
    engine = BacktestEngine(initial_capital=100000.0)
    tickers = list(set(getattr(config, "TICKERS", ["RELIANCE.NS", "TCS.NS"])))
    
    logger.info(f"🚀 Initiating Portfolio Backtest across {len(tickers)} assets...")
    
    for ticker in tickers:
        engine.run(ticker, start_date="2020-01-01") # Adjust date as needed
        
    if not engine.equity_curve:
        logger.error("❌ Backtest yielded no data. Check ticker symbols and network.")
        return

    # Convert results to DataFrame
    df_equity = pd.DataFrame(engine.equity_curve).drop_duplicates(subset=['date'], keep='last')
    df_trades = pd.DataFrame(engine.trade_log)

    # Write to backtest.db
    logger.info("💾 Writing backtest results to local database...")
    with sqlite3.connect(db_path) as conn:
        # Standardize columns for QuantEngine compatibility
        df_equity = df_equity.rename(columns={"equity": "total_equity"})
        df_trades = df_trades.rename(columns={"pnl": "realized_pnl"})
        
        df_equity.to_sql('equity_log', conn, index=False, if_exists='append')
        df_trades.to_sql('trades', conn, index=False, if_exists='append')

    logger.info("📊 Triggering Institutional Audit on Backtest Results...")
    
    # Pass the simulated DB directly into your Risk Engine
    audit_engine = QuantEngine(
        db_path=db_path,
        annualized_risk_free_rate=0.05,
        mc_sims=2000,
        ruin_threshold=-0.30,
        is_out_of_sample=False # Set to True if testing unseen data
    )
    
    try:
        audit_engine.load_data()
        report = audit_engine.run_analytics()
        if report:
            QuantReporter("₹").print_console(report)
    except Exception as e:
        logger.error(f"Audit generation failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_portfolio_backtest()
