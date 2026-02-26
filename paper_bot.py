import time
from datetime import datetime
import pandas as pd
import sys
import logging
import os
import sqlite3

# Import The Stack
from mock_broker import MockBroker
from logic import LogicEngine
import config
from data import DataGuard

# --- 1. SETUP LOGGING ---
if not os.path.exists("logs"): os.makedirs("logs")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler(f"logs/paper_bot_{datetime.now().strftime('%Y%m%d')}.log"), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger()

# --- 2. GLOBAL RISK CONFIG ---
MAX_OPEN_POSITIONS = 5          
MIN_ATR_PERCENT = 0.015         
MAX_ATR_PERCENT = 0.08          
MAX_PORTFOLIO_HEAT = 0.06       
DAILY_DD_LIMIT = 0.05           
INTRADAY_DD_LIMIT = 0.03        
RISK_PER_TRADE_PCT = 0.01
MIN_OPERATING_EQUITY = 10000.0  

TICKERS = list(set(getattr(config, "TICKERS", ["RELIANCE.NS", "TCS.NS"])))
broker = MockBroker(db_path="paper.db", initial_capital=100000.0)

# --- 3. PERSISTENT UTILITIES ---

def init_session_db(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS daily_locks (date TEXT NOT NULL, ticker TEXT NOT NULL, PRIMARY KEY (date, ticker))")
            conn.execute("DELETE FROM daily_locks WHERE date < DATE('now', 'utc', '-7 day')")
            conn.execute("CREATE TABLE IF NOT EXISTS system_status (key TEXT PRIMARY KEY, value TEXT, timestamp TEXT)")
            conn.commit()
    except Exception as e:
        logger.error(f"🚨 DB Init Error: {e}")

# --- CIRCUIT BREAKER & HWM ---

def get_circuit_breaker(db_path):
    """FAIL-SAFE CIRCUIT BREAKER"""
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT value FROM system_status WHERE key='halted'").fetchone()
            if row and row[0] == 'TRUE': return True
        return False
    except Exception as e:
        # 🚨 THE FIX: Fail CLOSED. If DB is unreachable, HALT the system.
        logger.critical(f"🚨 CRITICAL: DB Error checking circuit breaker. FAILING CLOSED. {e}")
        return True 

def trip_circuit_breaker(db_path, reason, residual=None):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO system_status (key, value, timestamp) VALUES ('halted', 'TRUE', ?)", (datetime.utcnow().isoformat(),))
            conn.execute("INSERT OR REPLACE INTO system_status (key, value, timestamp) VALUES ('halt_reason', ?, ?)", (str(reason), datetime.utcnow().isoformat()))
            if residual:
                conn.execute("INSERT OR REPLACE INTO system_status (key, value, timestamp) VALUES ('residual_risk', ?, ?)", (str(residual), datetime.utcnow().isoformat()))
            conn.commit()
    except: pass



def get_session_high(db_path, current_equity):
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT value, timestamp FROM system_status WHERE key='session_high'").fetchone()
            if row:
                stored_val = float(row[0])
                if row[1][:10] == datetime.utcnow().isoformat()[:10]:
                    return max(stored_val, current_equity)
    except: pass
    return current_equity

def update_session_high(db_path, equity):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO system_status (key, value, timestamp) VALUES ('session_high', ?, ?)", (str(equity), datetime.utcnow().isoformat()))
            conn.commit()
    except: pass

def is_ticker_locked(db_path, ticker):
    try:
        with sqlite3.connect(db_path) as conn:
            return conn.execute("SELECT 1 FROM daily_locks WHERE date = DATE('now', 'utc') AND ticker = ?", (ticker,)).fetchone() is not None
    except: return True 

def lock_ticker(db_path, ticker):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT OR IGNORE INTO daily_locks (date, ticker) VALUES (DATE('now', 'utc'), ?)", (ticker,))
            conn.commit()
    except: pass

def get_baseline_equity(db_path, current_equity):
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT total_equity FROM equity_log WHERE DATE(date) < DATE('now', 'utc') ORDER BY date DESC, id DESC LIMIT 1").fetchone()
            if row: return row[0]
    except: pass
    return current_equity

# --- 4. ENGINE CONTROLS ---

def refresh_state():
    """Fetches State, Valuation, and Heat."""
    cash = broker.get_cash()
    positions = broker.get_positions()
    equity = cash
    heat = 0.0
    for ticker, pos in positions.items():
        price = broker.get_live_price(ticker) or pos['avg_price']
        equity += (pos['qty'] * price)
        
        stop = broker.get_stop(ticker)
        if not stop or stop <= 0:
            stop = pos['avg_price'] * 0.95
            broker.set_stop(ticker, stop)

        # 🚨 THE FIX: True Liquidation Risk
        # Heat is now explicitly calculated as max(0, Current Price - Stop).
        # We no longer artificially shrink heat when a position goes red.
        risk_per_share = max(0, price - stop)
        heat += (risk_per_share * pos['qty'])
            
    return cash, equity, positions, heat

def shutdown_system(reason):
    """Verified Liquidation with Forensic Pre/Post Logging."""
    logger.critical(f"☢️ EMERGENCY SHUTDOWN: {reason}")
    trip_circuit_breaker("paper.db", reason) 
    
    # 🚨 THE FIX: Forensic Equity Logging
    _, equity_before, positions, _ = refresh_state()
    logger.info(f"📊 Pre-Liquidation Equity: ₹{equity_before:,.2f} | Open Positions: {len(positions)}")

    for attempt in range(3):
        broker.liquidate_portfolio()
        _, equity_after, positions, _ = refresh_state()
        
        if not positions:
            logger.info(f"☢️ Liquidation Verified. Post-Liquidation Equity: ₹{equity_after:,.2f} (Impact: ₹{equity_after - equity_before:,.2f})")
            sys.exit(1)
        
        logger.warning(f"⚠️ Liquidation attempt {attempt+1} failed. Residual: {list(positions.keys())}")
        time.sleep(2)
    
    logger.critical(f"🚨 FATAL: Liquidation Failed. Residual: {list(positions.keys())}")
    trip_circuit_breaker("paper.db", f"Liquidation Failed: {reason}", residual=str(list(positions.keys())))
    sys.exit(2)

def check_risk_metrics(equity, baseline, session_high):
    if equity > session_high:
        update_session_high("paper.db", equity)
        session_high = equity

    if baseline > 0 and (baseline - equity) / baseline > DAILY_DD_LIMIT:
        shutdown_system(f"Daily Drawdown {((baseline - equity) / baseline)*100:.2f}% Breach")

    if session_high > 0 and (session_high - equity) / session_high > INTRADAY_DD_LIMIT:
        shutdown_system(f"Intraday HWM Drawdown {((session_high - equity) / session_high)*100:.2f}% Breach")

    return session_high

def run_cycle():
    logger.info("🛡️ --- STARTING PROP FIRM CYCLE ---")
    init_session_db("paper.db")
    
    if get_circuit_breaker("paper.db"):
        logger.critical("⛔ System is HALTED. Manual intervention required.")
        sys.exit(1)
    
    if datetime.utcnow().weekday() >= 5:
        logger.warning("⛔ Weekend. Exiting.")
        return

    cash, equity, positions, heat = refresh_state()
    baseline = get_baseline_equity("paper.db", equity)
    session_high = get_session_high("paper.db", equity)
    
    session_high = check_risk_metrics(equity, baseline, session_high)

    if equity < MIN_OPERATING_EQUITY:
        shutdown_system(f"Equity ₹{equity:.2f} below floor ₹{MIN_OPERATING_EQUITY:.2f}")

    # 🚨 THE FIX: Data Caching to prevent multi-fetch inside the same run loop
    data_cache = {}

    for ticker in TICKERS:
        if is_ticker_locked("paper.db", ticker): continue

        try:
            # Check Cache before hitting DataGuard
            if ticker not in data_cache:
                time.sleep(1.0) # Respect API limits on initial fetch
                data_cache[ticker] = DataGuard.fetch_data(ticker, "3mo", "1d")
                
            history = data_cache[ticker]
            if history.empty or len(history) < 30: continue
            
            last_close = history['Close'].iloc[-1]
            todays_low = history['Low'].iloc[-1]
            todays_open = history['Open'].iloc[-1]
            
            # --- PHASE A: EXITS ---
            if ticker in positions:
                qty = positions[ticker]['qty']
                avg_price = positions[ticker]['avg_price']
                stop = broker.get_stop(ticker)
                
                if not stop or stop <= 0:
                    stop = avg_price * 0.95
                    broker.set_stop(ticker, stop)

                if todays_low <= stop:
                    fill_price = todays_open if todays_open < stop else stop
                    if broker.execute_order(ticker, qty, "SELL", "Hard Stop", price_override=fill_price):
                        lock_ticker("paper.db", ticker)
                        cash, equity, positions, heat = refresh_state()
                        session_high = check_risk_metrics(equity, baseline, session_high)
                    continue

                pos_state = {'side': 'long', 'entry': avg_price, 'current_sl': stop}
                decision = LogicEngine.check_exit_conditions(history, pos_state)
                if decision and decision['action'] == "EXIT":
                    if broker.execute_order(ticker, qty, "SELL", decision['reason']):
                        lock_ticker("paper.db", ticker)
                        cash, equity, positions, heat = refresh_state()
                        session_high = check_risk_metrics(equity, baseline, session_high)

            # --- PHASE B: ENTRIES ---
            else:
                if len(positions) >= MAX_OPEN_POSITIONS: continue
                if heat > (equity * MAX_PORTFOLIO_HEAT): continue

                rolling_high = history['High'].shift(1).rolling(20).max()
                if (last_close > rolling_high.iloc[-1]) and (history['Close'].iloc[-2] <= rolling_high.iloc[-2]):
                    
                    atr = LogicEngine.get_atr(history)
                    if not atr or pd.isna(atr): continue
                    
                    vol_ratio = atr / last_close
                    if vol_ratio < MIN_ATR_PERCENT or vol_ratio > MAX_ATR_PERCENT: continue

                    # 🚨 THE FIX: Stable Sizing Logic
                    risk_amt = baseline * RISK_PER_TRADE_PCT
                    est_entry = last_close * (1 + getattr(config, "SLIPPAGE_PCT", 0.0005))
                    stop_loss = last_close - (atr * 2.0)
                    risk_per_share = est_entry - stop_loss
                    
                    if risk_per_share <= 0: continue
                    qty = int(risk_amt / risk_per_share)
                    
                    if qty <= 0 or (qty * est_entry) < 5000: continue
                    if (qty * est_entry) > cash: qty = int(cash / est_entry)
                    if qty <= 0: continue

                    added_risk = qty * (est_entry - stop_loss)
                    if (heat + added_risk) > (equity * MAX_PORTFOLIO_HEAT): continue

                    logger.info(f"⚔️ BUYING {qty} {ticker}")
                    if broker.execute_order(ticker, qty, "BUY", "Breakout", stop_loss=stop_loss):
                        lock_ticker("paper.db", ticker)
                        cash, equity, positions, heat = refresh_state()
                        session_high = check_risk_metrics(equity, baseline, session_high)
                    
        except Exception as e:
            logger.error(f"💥 Error processing {ticker}: {e}", exc_info=True)

    logger.info("🏁 --- CYCLE COMPLETE ---")

if __name__ == "__main__":
    run_cycle()
