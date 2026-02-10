import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
import config
from logic import LogicEngine
from risk import PortfolioGovernor
from data import DataGuard

# Setup module-level logger
logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
    logger.addHandler(logging.NullHandler())

class BacktestEngine:
    """
    The Time Machine (Fiduciary-Platinum Standard).
    
    Responsibility:
    1. Replay historical data bar-by-bar (Strict Causality).
    2. Simulate Execution (Slippage, Commission, Liquidity, Gap-Aware Sizing).
    3. Enforce Risk Limits via PortfolioGovernor.
    4. Track Performance (Double-Entry Bookkeeping, Fiduciary Worst-Case).
    
    Philosophy:
    - Decisions are made at Close[i].
    - Execution happens at Open[i+1].
    - Sizing happens at Open[i+1] (Pre-Market Logic) to eliminate Gap Risk.
    - Stops are prioritized: Gap > Signal > Intraday.
    
    ASSUMPTIONS:
    - No Partial Fills: Binary liquidity based on volume limits.
    - Stops always fill: Infinite liquidity assumed at the stop price (slippage applied).
    - Single-Ticker Focus: Portfolio effects like correlation are not modeled here.
    - Daily Bars: Intraday stop precision is approximated by Low/High.
    """
    
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital # Cash on hand
        self.equity_curve = []
        self.trade_log = []
        self.active_positions = [] # List of dicts
        
        # Instantiate Core Modules
        self.risk_gov = PortfolioGovernor()
        
        # Configurable Costs & Constraints
        self.commission_pct = getattr(config, "COMMISSION_PCT", 0.001) # 0.1% per side
        self.slippage_pct = getattr(config, "SLIPPAGE_PCT", 0.0005)    # 0.05% per side
        self.max_pos_size_pct = 0.20 # Max equity per trade
        self.max_pct_volume = 0.01   # Max 1% of daily volume (Liquidity)

    def run(self, ticker, start_date=None, end_date=None):
        """
        Runs the simulation for a single ticker.
        """
        logger.info(f"Starting Backtest for {ticker}...")
        
        # 1. Fetch Full History
        df = DataGuard.fetch_data(ticker, period="5y", interval="1d")
        
        if df.empty:
            logger.error(f"Backtest aborted: No data for {ticker}")
            return None

        # Filter Date Range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        if len(df) < 100:
            logger.error(f"Backtest aborted: Insufficient history ({len(df)} bars).")
            return None

        # State Variables
        pending_entry = None # {ticker, sl_price_target, reason}
        
        # 2. The Simulation Loop
        # 'i' represents the CURRENT DAY.
        # Execution at Open[i], Logic at Close[i].
        
        for i in range(50, len(df)):
            # Market Data for THIS Bar (Execution Context)
            date = df.index[i]
            open_price = df["Open"].iloc[i]
            high_price = df["High"].iloc[i]
            low_price = df["Low"].iloc[i]
            close_price = df["Close"].iloc[i]
            volume = df["Volume"].iloc[i]
            
            # Liquidity Reference (Volume Collapse Aware)
            # Use min(prior, current) to capture events where volume evaporates.
            prior_volume = df["Volume"].iloc[i-1] if i > 0 else volume
            effective_volume = min(prior_volume, volume)
            
            # ------------------------------------------------------------------
            # PHASE 1: EXECUTION (Morning)
            # 1. Check Stops on Existing Positions
            # 2. Execute Pending Entries from Yesterday
            # ------------------------------------------------------------------
            
            # A. Manage Existing Positions
            for pos in list(self.active_positions): 
                pending_signal = pos.get('pending_exit_signal') 
                
                # Check 1: GAP RISK (Stop Hit at Open)
                # If Open is below SL, we gap down -> Fill at Open.
                if pos['side'] == 'long':
                    if open_price <= pos['current_sl']:
                        reason = "Stop Loss (Gap)"
                        if pending_signal: reason += " [Signal Overridden]"
                        # Separate Fill Price from Valuation Price
                        self._close_position(pos, date, open_price, reason, valuation_price=open_price)
                        continue
                
                # Check 2: PENDING SIGNAL (Market on Open)
                if pending_signal:
                    self._close_position(pos, date, open_price, pending_signal, valuation_price=open_price)
                    continue

                # Check 3: INTRADAY STOP (Volatility Risk)
                if pos['side'] == 'long':
                    if low_price <= pos['current_sl']:
                        # Intraday hit -> Fill at SL.
                        self._close_position(pos, date, pos['current_sl'], "Stop Loss (Intraday)", valuation_price=pos['current_sl'])
                        continue

            # B. Execute Pending Entry (Market on Open)
            if pending_entry and pending_entry['ticker'] == ticker:
                
                is_active = any(p['ticker'] == ticker for p in self.active_positions)
                
                if not is_active:
                    # Dynamic Sizing & Gap Validation
                    entry_price = open_price
                    sl_target = pending_entry['sl_price_target'] 
                    
                    risk_per_share = entry_price - sl_target
                    
                    # Gap Inversion Check
                    if risk_per_share <= 0:
                        logger.info(f"Entry invalidated by gap at {date}. Open ({entry_price}) < SL ({sl_target}). Skipping.")
                        pending_entry = None
                        continue
                    
                    # Estimate Equity for Sizing (Mark-to-Market at Open)
                    est_equity = self.current_cash + sum(open_price * p['qty'] for p in self.active_positions)
                    
                    risk_amt = est_equity * 0.01 # 1% Risk
                    raw_qty = int(risk_amt / risk_per_share)
                    
                    # Liquidity Constraint
                    max_vol_qty = int(effective_volume * self.max_pct_volume)
                    
                    # Max Equity Constraint
                    max_eq_qty = int((est_equity * self.max_pos_size_pct) / entry_price)
                    
                    qty = min(raw_qty, max_vol_qty, max_eq_qty)
                    
                    if qty > 0:
                        success = self._open_position(ticker, date, entry_price, qty, sl_target)
                        # Only clear if execution attempted/succeeded
                        if success:
                            pending_entry = None
                        else:
                            # Cash failure or similar -> Clear to avoid stale entry
                            pending_entry = None
                    else:
                        pending_entry = None # Cancel due to zero size
                else:
                     pending_entry = None # Already active

            # ------------------------------------------------------------------
            # PHASE 2: MARK-TO-MARKET (Afternoon)
            # Update Portfolio Value & Check Intraday Ruin
            # ------------------------------------------------------------------
            
            # [FIX #2] Fiduciary Intraday Worst-Case (Per-Position Commission)
            # Equity = Cash + Sum(Position Liquidation Value)
            # Liquidation Value = (Low * (1-Slippage) * Qty) - Commission
            # Commission calculated PER TRADE, not on aggregate.
            
            worst_liq_equity = self.current_cash
            for p in self.active_positions:
                liq_price = low_price * (1 - self.slippage_pct)
                liq_proceeds = liq_price * p['qty']
                liq_comm = liq_proceeds * self.commission_pct
                worst_liq_equity += (liq_proceeds - liq_comm)
            
            if worst_liq_equity <= 0:
                logger.critical(f"INTRADAY BANKRUPTCY at {date}. Fire-Sale Equity: {worst_liq_equity:.2f}")
                break

            # Standard Close Equity
            current_market_val = sum(close_price * p['qty'] for p in self.active_positions)
            total_equity = self.current_cash + current_market_val

            self.equity_curve.append({
                "date": date,
                "equity": total_equity,
                "cash": self.current_cash,
                "worst_equity": worst_liq_equity, 
                "positions": len(self.active_positions)
            })

            # ------------------------------------------------------------------
            # PHASE 3: DECISION (Evening)
            # Run Logic on Close data. Actions queue for NEXT morning.
            # ------------------------------------------------------------------
            
            history_slice = df.iloc[:i+1]
            
            # [FIX #4] Structural Break Safeguard for ATR
            # If a massive gap occurs (Limit Down/Earnings), standard ATR lags.
            # We enforce that the volatility measure respects today's True Range explicitly.
            atr = LogicEngine._calculate_atr(history_slice)
            if not atr or not np.isfinite(atr): 
                atr = close_price * 0.02
            else:
                # Instant adaptation to regime shock
                current_tr = max(high_price - low_price, abs(high_price - close_price), abs(low_price - close_price))
                atr = max(atr, current_tr)
            
            # A. Update Trailing Stops
            for pos in self.active_positions:
                if high_price > pos.get('highest_high', 0):
                    pos['highest_high'] = high_price

                logic_pos_state = {
                    "entry": pos['entry'],
                    "current_sl": pos['current_sl'],
                    "initial_sl": pos['initial_sl'],
                    "entry_date": pos['entry_date'],
                    "highest_high": pos['highest_high']
                }

                decision = LogicEngine.check_exit_conditions(history_slice, logic_pos_state)
                
                if decision['action'] == "EXIT":
                    pos['pending_exit_signal'] = decision['reason']
                    
                elif decision['action'] == "UPDATE_SL":
                    new_sl = decision['new_sl']
                    
                    if pos['side'] == 'long':
                        if new_sl > pos['current_sl']:
                             # Volatility-Aware Clamp
                             vol_buffer = max(atr * 0.25, close_price * 0.001)
                             safe_new_sl = min(new_sl, close_price - vol_buffer)
                             
                             if safe_new_sl > pos['current_sl']:
                                 pos['current_sl'] = safe_new_sl
                                 pos['sl_source'] = "Strategy"

            # B. Risk Governance
            # Note: We continue to use Close-based heat for the Governor to maintain API consistency
            # with Risk.py. Ideally, 'worst_liq_equity' would drive this in a V2 Risk Engine.
            gov_positions = [{
                "entry": p['entry'], 
                "initial_sl": p['initial_sl'], 
                "current_sl": p['current_sl']
            } for p in self.active_positions]
            
            risk_mult = self.risk_gov.get_risk_multiplier(positions=gov_positions)
            
            if risk_mult < 1.0:
                for pos in self.active_positions:
                    if pos['side'] == 'long':
                        dist = close_price - pos['current_sl']
                        min_dist = close_price * 0.005 
                        
                        if dist > min_dist:
                            tightened_dist = dist * risk_mult
                            tightened_dist = max(tightened_dist, min_dist)
                            new_sl_risk = close_price - tightened_dist
                            
                            if new_sl_risk > pos['current_sl']:
                                pos['current_sl'] = new_sl_risk
                                pos['sl_source'] = "RiskGovernor"

            # C. Generate New Entry Signal
            is_active = any(p['ticker'] == ticker for p in self.active_positions)
            
            if not is_active and not pending_entry:
                recent_high = df['High'].iloc[i-20:i].max() 
                
                if close_price > recent_high:
                    sl_level = close_price - (atr * 2.0)
                    
                    pending_entry = {
                        "ticker": ticker,
                        "sl_price_target": sl_level,
                        "reason": "Breakout"
                    }

        # Force Close at End
        if self.active_positions:
            last_close = df["Close"].iloc[-1]
            last_date = df.index[-1]
            for pos in list(self.active_positions):
                self._close_position(pos, last_date, last_close, "End of Backtest", valuation_price=last_close)

        if not self.equity_curve:
            return pd.DataFrame()
            
        final_eq = self.equity_curve[-1]['equity']
        logger.info(f"Backtest Complete. Final Equity: {final_eq:.2f}")
        return pd.DataFrame(self.equity_curve).set_index("date")

    def _open_position(self, ticker, date, price, qty, sl):
        """Executes a Long Entry."""
        filled_price = price * (1 + self.slippage_pct)
        cost = filled_price * qty
        comm = cost * self.commission_pct
        
        if self.current_cash < (cost + comm):
            logger.warning("Insufficient cash for entry. Skipping.")
            return False

        self.current_cash -= (cost + comm)
        
        pos = {
            "ticker": ticker,
            "entry": filled_price,
            "qty": qty,
            "initial_sl": sl,
            "current_sl": sl,
            "entry_date": date,
            "side": "long",
            "highest_high": filled_price,
            "entry_comm": comm,
            "pending_exit_signal": None,
            "sl_source": "Initial"
        }
        self.active_positions.append(pos)
        
        # True Equity = Cash + Market Value (at fill price)
        current_mkt_val = sum(p['qty'] * filled_price for p in self.active_positions)
        current_equity = self.current_cash + current_mkt_val
        
        self.trade_log.append({
            "date": date,
            "ticker": ticker,
            "action": "BUY",
            "price": filled_price,
            "qty": qty,
            "commission": comm,
            "reason": "Entry",
            "equity": current_equity
        })
        return True

    def _close_position(self, pos, date, price, reason, valuation_price=None):
        """Executes a Long Exit."""
        if valuation_price is None:
            valuation_price = price

        filled_price = price * (1 - self.slippage_pct)
        proceeds = filled_price * pos['qty']
        comm = proceeds * self.commission_pct
        
        self.current_cash += (proceeds - comm)
        
        entry_cost = pos['entry'] * pos['qty']
        gross_pnl = proceeds - entry_cost
        entry_comm = pos.get('entry_comm', 0.0)
        net_pnl = gross_pnl - comm - entry_comm
        
        # [FIX #1] Clear the pending signal to prevent object-reuse bugs
        pos['pending_exit_signal'] = None
        
        # True Remaining Value
        remaining_mkt_val = sum(p['qty'] * valuation_price for p in self.active_positions if p is not pos)
        current_equity = self.current_cash + remaining_mkt_val
        
        self.trade_log.append({
            "date": date,
            "ticker": pos['ticker'],
            "action": "SELL",
            "price": filled_price,
            "qty": pos['qty'],
            "commission": comm,
            "pnl": net_pnl,
            "reason": reason,
            "equity": current_equity
        })
        
        self.active_positions.remove(pos)
                          
