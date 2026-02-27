import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
import time
import uuid
import config 
import logging

logger = logging.getLogger("QuantBroker")

# 🚨 FIX: Custom Exception for strict transaction routing
class FiduciaryHaltException(Exception):
    """Raised when the broker must persistently halt due to severe limit breaches."""
    pass

class MockBroker:
    """
    The Refined Standard.
    A forensic-grade, ACID-compliant SQLite broker with Adversarial Fiduciary Guardrails.
    """
    def __init__(self, db_path="paper.db", initial_capital=100000.0):
        self.db_path = db_path
        self.price_cache = {} 
        self.cache_ttl = getattr(config, "PRICE_CACHE_SEC", 10)
        
        self.slippage = float(getattr(config, "SLIPPAGE_PCT", 0.0005))
        self.comm_pct = float(getattr(config, "COMMISSION_PCT", 0.001))
        
        self.max_gross_exposure = float(getattr(config, "BROKER_MAX_GROSS_EXPOSURE", 0.80))
        self.broker_dd_limit = float(getattr(config, "BROKER_DAILY_DD_LIMIT", 0.05))
        self.max_trades_per_min = int(getattr(config, "MAX_TRADES_PER_MIN", 10))
        
        if self.slippage < 0 or self.slippage > 0.10: raise ValueError("Unrealistic Slippage")
        if self.comm_pct < 0 or self.comm_pct > 0.05: raise ValueError("Unrealistic Commission")

        self._init_db(initial_capital)

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30) 
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL") 
        conn.execute("PRAGMA synchronous = NORMAL") 
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self, initial_capital):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL CHECK (cash >= 0),
                    realized_pnl REAL DEFAULT 0.0
                )
            """)
            conn.execute("INSERT OR IGNORE INTO account (id, cash) VALUES (1, ?)", (initial_capital,))

            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    ticker TEXT PRIMARY KEY,
                    qty REAL NOT NULL CHECK (qty > 0),
                    avg_price REAL NOT NULL CHECK (avg_price >= 0),
                    stop_loss REAL DEFAULT 0.0,
                    initial_sl REAL DEFAULT 0.0
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL CHECK (action IN ('BUY', 'SELL')),
                    qty REAL NOT NULL CHECK (qty > 0),
                    exec_price REAL NOT NULL CHECK (exec_price > 0),
                    gross_val REAL NOT NULL CHECK (gross_val > 0),
                    comm REAL NOT NULL CHECK (comm >= 0),
                    net_val REAL NOT NULL CHECK (net_val > 0),
                    cash_flow REAL NOT NULL,
                    realized_pnl REAL,
                    reason TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS equity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_val REAL NOT NULL,
                    est_method TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_equity_date ON equity_log(date)")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS broker_status (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp TEXT
                )
            """)
            conn.commit()

    # --- BROKER STATE MANAGEMENT ---

    def is_halted(self) -> bool:
        """Checks if the Broker layer has been persistently locked."""
        try:
            with self._get_conn() as conn:
                row = conn.execute("SELECT value FROM broker_status WHERE key='halted'").fetchone()
                if row and row['value'] == 'TRUE': return True
        except Exception: pass
        return False

    def halt_broker(self, reason: str, active_conn=None):
        """
        Locks the Broker across reboots.
        🚨 RESTORED: Uses its own isolated connection to prevent rollback erasure.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            if active_conn:
                active_conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('halted', 'TRUE', ?)", (timestamp,))
                active_conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('halt_reason', ?, ?)", (str(reason), timestamp))
            else:
                with self._get_conn() as conn:
                    conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('halted', 'TRUE', ?)", (timestamp,))
                    conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('halt_reason', ?, ?)", (str(reason), timestamp))
                    conn.commit()
            logger.critical(f"🔒 BROKER PERSISTENTLY HALTED: {reason}")
        except Exception as e:
            logger.critical(f"FATAL: Failed to persist Broker Halt State: {e}")

    def get_daily_baseline(self, current_equity: float) -> float:
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        with self._get_conn() as conn:
            row = conn.execute("SELECT value, timestamp FROM broker_status WHERE key='daily_baseline'").fetchone()
            
            if row and row['timestamp'][:10] == today:
                return float(row['value'])
                
            if self.is_halted():
                return float(row['value']) if row else current_equity

            conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('daily_baseline', ?, ?)", 
                         (str(current_equity), datetime.now(timezone.utc).isoformat()))
            conn.commit()
            return current_equity

    # --- CORE DATA ---

    def get_live_price(self, ticker, force_refresh=False):
        ticker = ticker.strip().upper()
        now = time.time()
        
        if not force_refresh and ticker in self.price_cache:
            price, ts = self.price_cache[ticker]
            if now - ts < self.cache_ttl: return price

        try:
            dat = yf.Ticker(ticker)
            price = dat.fast_info.get('last_price', None)
            if price is None:
                 data = yf.download(ticker, period="1d", interval="1m", progress=False)
                 if not data.empty:
                     price = float(data['Close'].iloc[-1])
            if price:
                self.price_cache[ticker] = (price, now)
                return price
        except Exception as e:
            logger.error(f"Data feed exception for {ticker}: {e}")
            
        raise ConnectionError(f"CRITICAL: Primary and fallback data feeds failed for {ticker}. Halting to prevent blind execution.")

    def get_cash(self):
        with self._get_conn() as conn:
            row = conn.execute("SELECT cash FROM account WHERE id=1").fetchone()
            if row:
                return row['cash']
            else:
                conn.execute("INSERT INTO account (id, cash) VALUES (1, 0.0)")
                conn.commit()
                return 0.0

    def get_positions(self):
        with self._get_conn() as conn:
            rows = conn.execute("SELECT ticker, qty, avg_price, stop_loss, initial_sl FROM positions").fetchall()
            return {r['ticker']: {'qty': r['qty'], 'avg_price': r['avg_price'], 'stop_loss': r['stop_loss'], 'initial_sl': r['initial_sl']} for r in rows}

    def get_stop(self, ticker):
        with self._get_conn() as conn:
            row = conn.execute("SELECT stop_loss FROM positions WHERE ticker=?", (ticker,)).fetchone()
            return row['stop_loss'] if row else 0.0

    def set_stop(self, ticker, stop_loss):
        with self._get_conn() as conn:
            conn.execute("UPDATE positions SET stop_loss=? WHERE ticker=?", (stop_loss, ticker))
            conn.commit()

    def liquidate_portfolio(self):
        positions = self.get_positions()
        for ticker, pos in positions.items():
            try:
                self.execute_order(ticker, pos['qty'], "SELL", "EMERGENCY LIQUIDATION")
            except Exception as e:
                logger.error(f"Failed to liquidate {ticker}: {e}")

    # --- EXECUTION ---

    def execute_order(self, ticker, qty, side, reason, price_override=None, stop_loss=0.0):
        side = str(side).strip().upper()
        
        if self.is_halted() and side == "BUY":
            logger.critical("Execution rejected. Broker is HALTED (Reduce-Only Mode active).")
            return False

        ticker = str(ticker).strip().upper()
        is_crypto = ticker in getattr(config, "CRYPTO_TICKERS", [])
        
        price_prec = 8 if is_crypto else 2
        qty_prec = 8 if is_crypto else 0
        
        if qty <= 0: return False
        if not is_crypto and not isinstance(qty, int): return False
        if side not in {"BUY", "SELL"}: return False

        fetch_start = time.time()
        
        try:
            raw_price = price_override if price_override else self.get_live_price(ticker)
            if not raw_price or raw_price <= 0: return False

            trade_id = str(uuid.uuid4())[:8]
            
            with self._get_conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                
                one_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
                trade_count_row = conn.execute("SELECT COUNT(*) as cnt FROM trades WHERE date >= ?", (one_min_ago,)).fetchone()
                
                if trade_count_row and trade_count_row['cnt'] >= self.max_trades_per_min:
                    raise FiduciaryHaltException(f"Execution Throttle Breached: {trade_count_row['cnt']} trades in 60s. Malicious loop suspected.")
                
                if not price_override and time.time() - fetch_start > 5:
                    raw_price = self.get_live_price(ticker, force_refresh=True)
                    if not raw_price: raise ValueError("Could not refresh price")

                acct = conn.execute("SELECT cash FROM account WHERE id=1").fetchone()
                if not acct: raise ValueError("Account row missing")
                current_cash = acct['cash']
                
                pos_rows = conn.execute("SELECT ticker, qty, avg_price FROM positions").fetchall()
                est_mkt_val = 0.0
                for p in pos_rows:
                    cached = self.price_cache.get(p['ticker'])
                    px = cached[0] if (cached and time.time() - cached[1] < 300) else p['avg_price']
                    est_mkt_val += p['qty'] * px
                    
                est_equity = current_cash + est_mkt_val
                
                if side == "BUY":
                    exec_price = round(raw_price * (1 + self.slippage), price_prec)
                    gross_val = round(exec_price * qty, 2)
                    comm = round(gross_val * self.comm_pct, 2)
                    net_val = round(gross_val + comm, 2) 
                    cash_flow = -net_val                 
                    
                    if net_val <= 0: raise ValueError("Zero Value Trade")
                    if current_cash < net_val: 
                        raise ValueError(f"Insufficient Funds: Need {net_val}, Have {current_cash}")
                        
                    if net_val > (est_equity * 0.30):
                        raise ValueError(f"BROKER GUARDRAIL: Order ({net_val:.2f}) > 30% of est. equity ({est_equity:.2f}). Fat finger prevented.")
                        
                    if (est_mkt_val + net_val) > (est_equity * self.max_gross_exposure):
                        raise ValueError(f"BROKER GUARDRAIL: Order breaches {self.max_gross_exposure*100}% Max Gross Exposure limit.")
                        
                    baseline_eq = self.get_daily_baseline(est_equity)
                    current_dd = (baseline_eq - est_equity) / baseline_eq
                    
                    if baseline_eq > 0 and current_dd > self.broker_dd_limit:
                        raise FiduciaryHaltException(f"System in {current_dd*100:.2f}% daily drawdown. Fiduciary limit breached.")
                    
                    conn.execute("UPDATE account SET cash = round(cash - ?, 2) WHERE id=1", (net_val,))
                    
                    pos = conn.execute("SELECT qty, avg_price, stop_loss, initial_sl FROM positions WHERE ticker=?", (ticker,)).fetchone()
                    if pos:
                        new_qty = round(pos['qty'] + qty, qty_prec)
                        current_cost = pos['qty'] * pos['avg_price']
                        new_avg = round((current_cost + net_val) / new_qty, price_prec)
                        safe_stop = max(float(pos['stop_loss']), float(stop_loss))
                        
                        conn.execute("UPDATE positions SET qty=?, avg_price=?, stop_loss=? WHERE ticker=?", (new_qty, new_avg, safe_stop, ticker))
                    else:
                        conn.execute("INSERT INTO positions (ticker, qty, avg_price, stop_loss, initial_sl) VALUES (?, ?, ?, ?, ?)", 
                                     (ticker, qty, round(net_val/qty, price_prec), stop_loss, stop_loss))
                    
                    pnl = 0.0
                    print(f"✅ BUY {qty} {ticker} @ {exec_price}")

                elif side == "SELL":
                    exec_price = round(raw_price * (1 - self.slippage), price_prec)
                    gross_val = round(exec_price * qty, 2)
                    comm = round(gross_val * self.comm_pct, 2)
                    net_val = round(gross_val - comm, 2) 
                    cash_flow = net_val                  
                    
                    pos = conn.execute("SELECT qty, avg_price FROM positions WHERE ticker=?", (ticker,)).fetchone()
                    if not pos or pos['qty'] < qty: 
                        raise ValueError(f"Insufficient Shares: Have {pos['qty'] if pos else 0}")
                    
                    cost_basis_sold = round(pos['avg_price'] * qty, 2)
                    pnl = round(net_val - cost_basis_sold, 2)
                    
                    conn.execute("UPDATE account SET cash = round(cash + ?, 2), realized_pnl = round(realized_pnl + ?, 2) WHERE id=1", (net_val, pnl))
                    
                    new_qty = round(pos['qty'] - qty, qty_prec)
                    if new_qty <= 1e-9:
                        conn.execute("DELETE FROM positions WHERE ticker=?", (ticker,))
                    else:
                        conn.execute("UPDATE positions SET qty=? WHERE ticker=?", (new_qty, ticker))
                        
                    print(f"✅ SELL {qty} {ticker} @ {exec_price} | PnL: {pnl}")

                clean_reason = str(reason).replace(",", ";")[:50]
                conn.execute("""
                    INSERT INTO trades (trade_id, date, ticker, action, qty, exec_price, gross_val, comm, net_val, cash_flow, realized_pnl, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (trade_id, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), ticker, side, qty, exec_price, gross_val, comm, net_val, cash_flow, pnl, clean_reason))
                
                conn.commit()

        except FiduciaryHaltException as fhe:
            logger.critical(f"🚨 Fiduciary Guardrail Triggered: {fhe}")
            self.halt_broker(str(fhe), active_conn=None) 
            return False
            
        except ConnectionError as ce:
            logger.critical(f"🚨 FATAL NETWORK ERROR: {ce}")
            raise 
            
        except Exception as e:
            print(f"⚠️ Trade Rolled Back: {e}")
            return False

        self._log_equity()
        return True

    def _log_equity(self):
        try:
            with self._get_conn() as conn:
                acct = conn.execute("SELECT cash FROM account WHERE id=1").fetchone()
                positions = conn.execute("SELECT ticker, qty, avg_price FROM positions").fetchall()
                
            total_pos_val = 0.0
            method = "Live"
            
            for p in positions:
                ticker = p['ticker']
                cached = self.price_cache.get(ticker)
                
                if cached and (time.time() - cached[1] < 300):
                    price = cached[0]
                else:
                    try:
                        price = self.get_live_price(ticker, force_refresh=True)
                    except ConnectionError:
                        price = None
                
                if not price: 
                    price = p['avg_price']
                    method = "Mixed"
                
                total_pos_val += (price * p['qty'])
                
            total_equity = round(acct['cash'] + total_pos_val, 2)
            
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO equity_log (date, total_equity, cash, positions_val, est_method)
                    VALUES (?, ?, ?, ?, ?)
                """, (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), total_equity, acct['cash'], total_pos_val, method))
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ Equity Log Failed: {e}")

    def export_csv(self):
        with self._get_conn() as conn:
            pd.read_sql("SELECT * FROM trades", conn).to_csv("paper_trades.csv", index=False)
            pd.read_sql("SELECT * FROM equity_log", conn).to_csv("paper_equity.csv", index=False)
            print("✅ Data exported to CSV.")
