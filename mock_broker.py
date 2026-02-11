import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime
import time
import uuid
import config 

class MockBroker:
    """
    The Refined Standard.
    A forensic-grade, ACID-compliant SQLite broker.
    
    FINAL REFINEMENT:
    - Explicit Transaction Management: Replaced silent 'return False' with 'raise ValueError'.
      This forces all failures into the exception handler for explicit rollback logging.
    - IEEE 754 Mitigation: Aggressive application-layer rounding on every write.
    - Architecture: SQLite WAL Mode for concurrency.
    """
    def __init__(self, db_path="paper.db", initial_capital=100000.0):
        self.db_path = db_path
        self.price_cache = {} 
        self.cache_ttl = getattr(config, "PRICE_CACHE_SEC", 10)
        
        # 1. Config Sanity
        self.slippage = float(getattr(config, "SLIPPAGE_PCT", 0.0005))
        self.comm_pct = float(getattr(config, "COMMISSION_PCT", 0.001))
        
        if self.slippage < 0 or self.slippage > 0.10: raise ValueError("Unrealistic Slippage")
        if self.comm_pct < 0 or self.comm_pct > 0.05: raise ValueError("Unrealistic Commission")

        self._init_db(initial_capital)

    def _get_conn(self):
        """Returns connection with WAL mode and strong durability."""
        conn = sqlite3.connect(self.db_path, timeout=30) 
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL") 
        conn.execute("PRAGMA synchronous = NORMAL") 
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self, initial_capital):
        """Creates Schema with Semantic Auditing."""
        with self._get_conn() as conn:
            # 1. Account (Strict Solvency)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL CHECK (cash >= 0),
                    realized_pnl REAL DEFAULT 0.0
                )
            """)
            # Self-Heal: Ensure Row 1 exists
            conn.execute("INSERT OR IGNORE INTO account (id, cash) VALUES (1, ?)", (initial_capital,))

            # 2. Positions (Strict Quantity)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    ticker TEXT PRIMARY KEY,
                    qty REAL NOT NULL CHECK (qty > 0),
                    avg_price REAL NOT NULL CHECK (avg_price >= 0)
                )
            """)

            # 3. Trade Log (Semantic Cash Flow)
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

            # 4. Equity History
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
            
            conn.commit()

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
        except:
            pass
        return None

    def get_cash(self):
        """Fetches cash with Self-Healing."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT cash FROM account WHERE id=1").fetchone()
            if row:
                return row['cash']
            else:
                print("⚠️ CRITICAL: Account row missing! Restoring to 0.0 safety state.")
                conn.execute("INSERT INTO account (id, cash) VALUES (1, 0.0)")
                conn.commit()
                return 0.0

    def get_positions(self):
        with self._get_conn() as conn:
            rows = conn.execute("SELECT ticker, qty, avg_price FROM positions").fetchall()
            return {r['ticker']: {'qty': r['qty'], 'avg_price': r['avg_price']} for r in rows}

    def execute_order(self, ticker, qty, side, reason):
        # --- PHASE 1: PREP ---
        ticker = str(ticker).strip().upper()
        side = str(side).strip().upper()
        is_crypto = ticker in getattr(config, "CRYPTO_TICKERS", [])
        
        price_prec = 8 if is_crypto else 2
        qty_prec = 8 if is_crypto else 0
        
        if qty <= 0: return False
        if not is_crypto and not isinstance(qty, int): return False
        if side not in {"BUY", "SELL"}: return False

        # 1. Fetch Price (Network)
        fetch_start = time.time() # Defined before use
        raw_price = self.get_live_price(ticker)
        if not raw_price or raw_price <= 0: return False

        # --- PHASE 2: TRANSACTION (ACID) ---
        trade_id = str(uuid.uuid4())[:8]
        
        try:
            with self._get_conn() as conn:
                conn.execute("BEGIN IMMEDIATE")
                
                # Stale Price Guard
                if time.time() - fetch_start > 5:
                    print(f"⚠️ Price stale for {ticker}, re-fetching...")
                    raw_price = self.get_live_price(ticker, force_refresh=True)
                    if not raw_price: raise ValueError("Could not refresh price")

                acct = conn.execute("SELECT cash FROM account WHERE id=1").fetchone()
                if not acct: raise ValueError("Account row missing")
                current_cash = acct['cash']
                
                if side == "BUY":
                    exec_price = round(raw_price * (1 + self.slippage), price_prec)
                    gross_val = round(exec_price * qty, 2)
                    comm = round(gross_val * self.comm_pct, 2)
                    net_val = round(gross_val + comm, 2) # Total Cost
                    cash_flow = -net_val                 # Negative Flow
                    
                    if net_val <= 0: raise ValueError("Zero Value Trade")
                    if current_cash < net_val: 
                        raise ValueError(f"Insufficient Funds: Need {net_val}, Have {current_cash}")
                    
                    # Update Cash
                    conn.execute("UPDATE account SET cash = round(cash - ?, 2) WHERE id=1", (net_val,))
                    
                    # Upsert Position
                    pos = conn.execute("SELECT qty, avg_price FROM positions WHERE ticker=?", (ticker,)).fetchone()
                    if pos:
                        new_qty = round(pos['qty'] + qty, qty_prec)
                        current_cost = pos['qty'] * pos['avg_price']
                        new_avg = round((current_cost + net_val) / new_qty, price_prec)
                        conn.execute("UPDATE positions SET qty=?, avg_price=? WHERE ticker=?", (new_qty, new_avg, ticker))
                    else:
                        conn.execute("INSERT INTO positions (ticker, qty, avg_price) VALUES (?, ?, ?)", 
                                     (ticker, qty, round(net_val/qty, price_prec)))
                    
                    pnl = 0.0
                    print(f"✅ BUY {qty} {ticker} @ {exec_price}")

                elif side == "SELL":
                    exec_price = round(raw_price * (1 - self.slippage), price_prec)
                    gross_val = round(exec_price * qty, 2)
                    comm = round(gross_val * self.comm_pct, 2)
                    net_val = round(gross_val - comm, 2) # Total Proceeds
                    cash_flow = net_val                  # Positive Flow
                    
                    pos = conn.execute("SELECT qty, avg_price FROM positions WHERE ticker=?", (ticker,)).fetchone()
                    if not pos or pos['qty'] < qty: 
                        raise ValueError(f"Insufficient Shares: Have {pos['qty'] if pos else 0}")
                    
                    cost_basis_sold = round(pos['avg_price'] * qty, 2)
                    pnl = round(net_val - cost_basis_sold, 2)
                    
                    # Update Cash
                    conn.execute("UPDATE account SET cash = round(cash + ?, 2), realized_pnl = round(realized_pnl + ?, 2) WHERE id=1", (net_val, pnl))
                    
                    # Update Position
                    new_qty = round(pos['qty'] - qty, qty_prec)
                    if new_qty <= 1e-9:
                        conn.execute("DELETE FROM positions WHERE ticker=?", (ticker,))
                    else:
                        conn.execute("UPDATE positions SET qty=? WHERE ticker=?", (new_qty, ticker))
                        
                    print(f"✅ SELL {qty} {ticker} @ {exec_price} | PnL: {pnl}")

                # 3. Log Trade (Semantic)
                clean_reason = str(reason).replace(",", ";")[:50]
                conn.execute("""
                    INSERT INTO trades (trade_id, date, ticker, action, qty, exec_price, gross_val, comm, net_val, cash_flow, realized_pnl, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (trade_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ticker, side, qty, exec_price, gross_val, comm, net_val, cash_flow, pnl, clean_reason))
                
                conn.commit()

        except Exception as e:
            # Explicit Rollback happens automatically here as context manager exits with error
            # But the print log is now guaranteed for ALL failure paths
            print(f"⚠️ Trade Rolled Back: {e}")
            return False

        # --- PHASE 3: EQUITY LOG ---
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
                    price = self.get_live_price(ticker, force_refresh=True)
                
                if not price: 
                    price = p['avg_price']
                    method = "Mixed"
                
                total_pos_val += (price * p['qty'])
                
            total_equity = round(acct['cash'] + total_pos_val, 2)
            
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO equity_log (date, total_equity, cash, positions_val, est_method)
                    VALUES (?, ?, ?, ?, ?)
                """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), total_equity, acct['cash'], total_pos_val, method))
                conn.commit()
                
        except Exception as e:
            print(f"⚠️ Equity Log Failed: {e}")

    def export_csv(self):
        """Exports DB to CSV for auditing tools."""
        with self._get_conn() as conn:
            pd.read_sql("SELECT * FROM trades", conn).to_csv("paper_trades.csv", index=False)
            pd.read_sql("SELECT * FROM equity_log", conn).to_csv("paper_equity.csv", index=False)
            print("✅ Data exported to CSV.")
