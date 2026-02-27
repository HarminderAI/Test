import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import logging
import os

logger = logging.getLogger("RegimeEngine")
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

class RegimeEngine:
    """
    Proactive Market State Classifier
    ---------------------------------
    Classifies the macro environment based on Trend Structure and Relative Volatility.
    Maps the detected regime to a deterministic risk scalar.
    """

    def __init__(
        self,
        index_ticker="^NSEI",  # NIFTY 50 (Adjust to SPY/QQQ if trading US Equities)
        db_path="paper.db",
        trend_fast=50,
        trend_slow=200,
        vol_short_window=20,
        vol_long_window=252,
        vol_spike_threshold=1.25 # Short vol > 25% above long vol = "Volatile"
    ):
        self.index_ticker = index_ticker
        self.db_path = db_path
        self.trend_fast = trend_fast
        self.trend_slow = trend_slow
        self.vol_short_window = vol_short_window
        self.vol_long_window = vol_long_window
        self.vol_spike_threshold = vol_spike_threshold

        # Fiduciary Regime Matrix (Multiplier Caps)
        self.regime_risk_map = {
            "BULL_QUIET": 1.0,     # Perfect conditions (Full Size)
            "BULL_VOLATILE": 0.8,  # Uptrend but choppy (Slight reduction)
            "CHOP_QUIET": 0.7,     # Sideways, low energy (Reduced size)
            "CHOP_VOLATILE": 0.5,  # Sideways, high energy (Whipsaw danger)
            "BEAR_QUIET": 0.4,     # Slow bleed (Heavy reduction)
            "BEAR_VOLATILE": 0.2   # Market Crash (Near-Halt / Survival mode)
        }

    # ==========================================================
    # DATA INGESTION
    # ==========================================================

    def _fetch_index_data(self):
        """Fetches 1.5 years of broad market data to calculate 252-day baseline."""
        try:
            # We need at least 252 days + 20 days + buffer
            df = yf.download(self.index_ticker, period="2y", interval="1d", progress=False)
            if df.empty:
                raise ValueError(f"Empty dataframe returned for {self.index_ticker}")
            
            # Standardization
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.columns = [c.capitalize() for c in df.columns]
            return df['Close'].dropna()
            
        except Exception as e:
            logger.error(f"Failed to fetch macro index {self.index_ticker}: {e}")
            return pd.Series(dtype=float)

    # ==========================================================
    # CORE CLASSIFICATION
    # ==========================================================

    def _classify_trend(self, close, fast_sma, slow_sma):
        """
        Structural Trend Definition:
        BULL: Price > Slow AND Fast > Slow
        BEAR: Price < Slow AND Fast < Slow
        CHOP: Contradictory states (e.g. Price < Slow but Fast > Slow)
        """
        if close > slow_sma and fast_sma > slow_sma:
            return "BULL"
        elif close < slow_sma and fast_sma < slow_sma:
            return "BEAR"
        else:
            return "CHOP"

    def _classify_volatility(self, current_vol, baseline_vol):
        """
        Volatility State Definition:
        VOLATILE: Short-term vol is significantly higher than long-term historical baseline.
        QUIET: Short-term vol is normal or compressed.
        """
        if current_vol > (baseline_vol * self.vol_spike_threshold):
            return "VOLATILE"
        return "QUIET"

    def evaluate(self):
        logger.info(f"Scanning macro environment via {self.index_ticker}...")
        
        close_series = self._fetch_index_data()
        
        if len(close_series) < self.trend_slow:
            logger.warning("Insufficient macro data for Regime Classification. Defaulting to Neutral (0.7x).")
            self._store_regime("UNKNOWN", 0.7)
            return 0.7

        # 1. Trend Calculations
        current_close = float(close_series.iloc[-1])
        fast_sma = float(close_series.rolling(self.trend_fast).mean().iloc[-1])
        slow_sma = float(close_series.rolling(self.trend_slow).mean().iloc[-1])
        
        # 2. Volatility Calculations (Annualized StdDev of Log Returns)
        log_rets = np.log(close_series / close_series.shift(1)).dropna()
        
        short_vol = float(log_rets.rolling(self.vol_short_window).std().iloc[-1] * np.sqrt(252))
        long_vol = float(log_rets.rolling(self.vol_long_window).std().iloc[-1] * np.sqrt(252))

        # 3. Classify
        trend_state = self._classify_trend(current_close, fast_sma, slow_sma)
        vol_state = self._classify_volatility(short_vol, long_vol)
        
        regime_label = f"{trend_state}_{vol_state}"
        risk_mult = self.regime_risk_map.get(regime_label, 0.5)

        logger.info("--- MACRO REGIME METRICS ---")
        logger.info(f"Trend: Close={current_close:,.0f} | 50SMA={fast_sma:,.0f} | 200SMA={slow_sma:,.0f}")
        logger.info(f"Volatility: 20d={short_vol:.1%} | 252d Baseline={long_vol:.1%}")
        logger.info(f"🎯 State Detected: {regime_label}")
        logger.info(f"⚙️ Regime Engine applying {risk_mult:.2f}x capital allocation.")

        # 4. Persist State
        self._store_regime(regime_label, risk_mult)
        return risk_mult

    # ==========================================================
    # STATE WRITES
    # ==========================================================

    def _store_regime(self, label: str, mult: float):
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                timestamp = datetime.now(timezone.utc).isoformat()
                # Store the string label for the telemetry/dashboard
                conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('macro_regime', ?, ?)", (label, timestamp))
                # Store the mathematical multiplier for the Risk Governor
                conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('regime_risk_mult', ?, ?)", (str(mult), timestamp))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to write regime state to database: {e}")

if __name__ == "__main__":
    engine = RegimeEngine()
    engine.evaluate()
