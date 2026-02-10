import logging
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import config
from backtest import BacktestEngine

# 1. Setup Logging
if not os.path.exists("logs"): os.makedirs("logs")
if not os.path.exists("results"): os.makedirs("results")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_system.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

# Random Seed Lock for Reproducibility
np.random.seed(42)

# Strategy Metadata (Schema v1.2)
STRATEGY_TAG = "Donchian_Breakout_v1"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M")
SCHEMA_VERSION = "v1.2"

def calculate_cagr(start_val, end_val, start_date, end_date):
    """
    Calculates Compound Annual Growth Rate (Geometric Mean).
    Anchors to the full requested duration to prevent inflation on short histories.
    """
    if start_val <= 0 or end_val <= 0: return 0.0
    
    # Calculate duration in years based on REQUESTED window
    duration = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    years = duration.days / 365.25
    
    if years <= 0: return 0.0
    
    cagr = (end_val / start_val) ** (1 / years) - 1
    return cagr * 100

def calculate_fiduciary_drawdown(equity_curve):
    """
    Calculates Drawdown using Worst-Case Intraday Equity if available.
    This captures tail risk (gaps, slippage, crashes) that Close-to-Close misses.
    """
    if 'worst_equity' in equity_curve.columns:
        series = equity_curve['worst_equity']
    else:
        series = equity_curve['equity']
        
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    return max_dd

def run_simulation(tickers, start_date, end_date, capital):
    """
    Runs the Platinum-Sovereign Backtest on a portfolio of tickers.
    """
    logger.info(f"Initializing Run: {RUN_ID} | Strategy: {STRATEGY_TAG}")
    logger.info(f"Simulation Window: {start_date} to {end_date}")
    
    portfolio_results = {}
    
    # Enhanced Dashboard with MAR Ratio and Round Trips
    print("-" * 118)
    print(f"{'TICKER':<10} | {'EXECS':<6} | {'R.TRIP':<6} | {'FINAL EQ':<12} | {'TOT RET%':<9} | {'CAGR%':<8} | {'MDD%':<8} | {'MAR':<6} | {'TIME':<6}")
    print("-" * 118)

    for ticker in tickers:
        loop_start = time.time()
        
        # Isolation: Fresh Engine per ticker
        engine = BacktestEngine(initial_capital=capital)
        
        try:
            # [FIX #1] Explicit Crypto Detection
            # Using set membership avoids operator precedence bugs
            if ticker in {"BTC-USD", "ETH-USD"}:
                 logger.warning(f"Crypto detected ({ticker}). Volume constraint in backtest.py may need 'max_pct_volume' adjustment.")

            equity_curve = engine.run(ticker, start_date=start_date, end_date=end_date)
            elapsed = time.time() - loop_start
            
            if equity_curve is not None and not equity_curve.empty:
                final_eq = equity_curve['equity'].iloc[-1]
                start_cap = engine.initial_capital
                
                # 1. Total Return
                total_ret = ((final_eq - start_cap) / start_cap) * 100
                
                # 2. True CAGR (Anchored to Simulation Window)
                cagr = calculate_cagr(start_cap, final_eq, start_date, end_date)
                
                # 3. Fiduciary Max Drawdown (Intraday Worst-Case)
                max_dd = calculate_fiduciary_drawdown(equity_curve)
                
                # 4. MAR Ratio (Risk-Adjusted Return)
                mar = abs(cagr / max_dd) if max_dd < 0 else (999.0 if cagr > 0 else 0.0)
                
                # 5. Round Trips
                execution_count = len(engine.trade_log)
                round_trips = execution_count // 2
                
                # Print Summary
                print(f"{ticker:<10} | {execution_count:<6} | {round_trips:<6} | ${final_eq:<11,.0f} | {total_ret:>8.2f}% | {cagr:>7.1f}% | {max_dd:>7.1f}% | {mar:>6.2f} | {elapsed:>5.2f}s")
                
                # 6. Schema Versioning & Metadata
                df_trades = pd.DataFrame(engine.trade_log).copy()
                if not df_trades.empty:
                    df_trades['Ticker'] = ticker
                    df_trades['Strategy'] = STRATEGY_TAG
                    df_trades['Run_ID'] = RUN_ID
                    df_trades['Schema_Ver'] = SCHEMA_VERSION
                    df_trades['InitCapital'] = start_cap
                    df_trades['Mode'] = 'Isolated_Reset'
                    df_trades['Metric_CAGR'] = cagr
                    df_trades['Metric_MDD'] = max_dd
                    df_trades['Metric_MAR'] = mar
                
                portfolio_results[ticker] = {
                    "equity": equity_curve.copy(),
                    "trades": df_trades
                }
                
            else:
                print(f"{ticker:<10} | 0      | 0      | ${capital:<11,.0f} |    0.00%  |    0.0%  |    0.0%  |   0.00 | {elapsed:>5.2f}s")

        except Exception as e:
            logger.error(f"Failed to backtest {ticker}: {e}", exc_info=True)
            continue

    print("-" * 118)
    return portfolio_results

if __name__ == "__main__":
    # --- CONFIGURATION ---
    TICKERS = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD"]
    START = "2020-01-01"
    END = datetime.now().strftime("%Y-%m-%d")
    CAPITAL = 10000.0
    
    # --- EXECUTION ---
    results = run_simulation(TICKERS, START, END, CAPITAL)
    
    # Export Aggregate Trade Logs
    if results:
        all_trades = []
        for t, data in results.items():
            if not data['trades'].empty:
                all_trades.append(data['trades'])
        
        if all_trades:
            final_log = pd.concat(all_trades)
            if 'date' in final_log.columns:
                final_log.sort_values('date', inplace=True)
            
            # Save to results folder with ID
            filename = f"results/trades_{STRATEGY_TAG}_{RUN_ID}.csv"
            final_log.to_csv(filename, index=False)
            logger.info(f"Aggregated trade logs exported to {filename}")
