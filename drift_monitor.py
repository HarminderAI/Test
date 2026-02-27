import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import scipy.stats as stats
import logging
import os

logger = logging.getLogger("DriftMonitor")
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

class AlphaDriftMonitor:
    """
    Alpha Integrity & Statistical Drift Layer
    -----------------------------------------
    Compares the live trading distribution against the backtest baseline.
    Detects structural regime shifts, alpha decay, and edge evaporation.
    """

    def __init__(
        self,
        live_db_path="paper.db",
        backtest_db_path="backtest.db",
        min_live_trades=30,
        ks_pvalue_threshold=0.05,       # 95% confidence that distributions differ
        mean_decay_threshold=0.50,      # Live mean < 50% of Backtest mean
        tail_decay_threshold=0.50,      # Frequency of fat tail winners drops by 50%
        wasserstein_limit=0.02,         # Earth Mover's Distance threshold
        risk_floor=0.2,                 # Maximum penalty
        risk_ceiling=1.0
    ):
        self.live_db = live_db_path
        self.bt_db = backtest_db_path
        self.min_trades = min_live_trades
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.mean_decay_threshold = mean_decay_threshold
        self.tail_decay_threshold = tail_decay_threshold
        self.wasserstein_limit = wasserstein_limit
        self.risk_floor = risk_floor
        self.risk_ceiling = risk_ceiling

    # ==========================================================
    # DATA INGESTION
    # ==========================================================

    def _fetch_normalized_returns(self, db_path, limit=None):
        """
        Fetches trades and calculates Normalized Return (proxy for R-Multiple).
        Normalized Return = Realized PnL / Gross Trade Value
        """
        if not os.path.exists(db_path):
            return np.array([])
            
        try:
            with sqlite3.connect(db_path) as conn:
                query = """
                    SELECT realized_pnl, gross_val 
                    FROM trades 
                    WHERE realized_pnl IS NOT NULL 
                    ORDER BY date DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                    
                df = pd.read_sql(query, conn)
                
            if df.empty: return np.array([])
            
            # Prevent division by zero
            safe_gross = df['gross_val'].replace(0, 1.0)
            norm_returns = (df['realized_pnl'] / safe_gross).values
            return norm_returns
            
        except Exception as e:
            logger.warning(f"Failed to fetch data from {db_path}: {e}")
            return np.array([])

    # ==========================================================
    # STATISTICAL ENGINE
    # ==========================================================

    def evaluate(self):
        # 1. Load Distributions
        bt_returns = self._fetch_normalized_returns(self.bt_db)
        if len(bt_returns) < 50:
            logger.info("Insufficient backtest baseline data. Drift Monitor bypassed.")
            self._store_risk_multiplier(1.0)
            return 1.0

        live_returns = self._fetch_normalized_returns(self.live_db, limit=100)
        if len(live_returns) < self.min_trades:
            logger.info(f"Accumulating live trades ({len(live_returns)}/{self.min_trades}). Drift Monitor warming up.")
            self._store_risk_multiplier(1.0)
            return 1.0

        # 2. Compute Distribution Shapes
        bt_mean = np.mean(bt_returns)
        live_mean = np.mean(live_returns)
        
        bt_win_rate = np.mean(bt_returns > 0)
        live_win_rate = np.mean(live_returns > 0)

        # Dynamic Tail Definition: The 90th percentile of backtest winners
        bt_winners = bt_returns[bt_returns > 0]
        if len(bt_winners) > 0:
            fat_tail_threshold = np.percentile(bt_winners, 90)
            bt_tail_prob = np.mean(bt_returns > fat_tail_threshold)
            live_tail_prob = np.mean(live_returns > fat_tail_threshold)
        else:
            bt_tail_prob, live_tail_prob = 0, 0

        # 3. Formal Statistical Distance Tests
        # Kolmogorov-Smirnov Test (Are they from the same continuous distribution?)
        ks_stat, ks_pvalue = stats.ks_2samp(bt_returns, live_returns)
        
        # Wasserstein Distance (Earth Mover's Distance - How much 'work' to transform Live into Backtest?)
        wasserstein_dist = stats.wasserstein_distance(bt_returns, live_returns)

        logger.info("--- ALPHA DRIFT METRICS ---")
        logger.info(f"Mean Ret:  BT={bt_mean:.4f} | Live={live_mean:.4f}")
        logger.info(f"Win Rate:  BT={bt_win_rate:.1%} | Live={live_win_rate:.1%}")
        logger.info(f"Fat Tails: BT={bt_tail_prob:.1%} | Live={live_tail_prob:.1%}")
        logger.info(f"Distances: KS_p={ks_pvalue:.4f} | Wasserstein={wasserstein_dist:.4f}")

        # ======================================================
        # DECISION ENGINE (The Fiduciary Logic)
        # ======================================================
        
        risk_mult = 1.0
        decay_flags = []

        # A. Shape Divergence (Structural Break)
        if ks_pvalue < self.ks_pvalue_threshold:
            decay_flags.append(f"KS Structural Break (p={ks_pvalue:.3f})")
            risk_mult *= 0.6
            
        # B. Edge Evaporation (Mean Collapse)
        if bt_mean > 0 and live_mean < (bt_mean * self.mean_decay_threshold):
            decay_flags.append("Mean Edge Collapse")
            risk_mult *= 0.7
            
        # C. Tail Disappearance (The Alpha is gone)
        if bt_tail_prob > 0 and live_tail_prob < (bt_tail_prob * self.tail_decay_threshold):
            decay_flags.append("Right Tail Disappearance")
            risk_mult *= 0.5

        risk_mult = max(self.risk_floor, min(self.risk_ceiling, risk_mult))

        if decay_flags:
            logger.warning(f"🚨 ALPHA DECAY DETECTED: {', '.join(decay_flags)}")
            logger.warning(f"⚙️ Drift Monitor applying {risk_mult:.2f}x penalty to capital allocation.")
        else:
            logger.info("✅ Alpha signature is statistically intact.")

        # D. Hard Halt Condition (Fatal Strategy Death)
        if ks_pvalue < 0.01 and live_mean < 0 and bt_mean > 0:
            self._trigger_halt("FATAL: Complete Inverse Regime Break. Live mean is negative with 99% statistical divergence.")

        self._store_risk_multiplier(risk_mult)
        return risk_mult

    # ==========================================================
    # STATE WRITES
    # ==========================================================

    def _store_risk_multiplier(self, value):
        try:
            with sqlite3.connect(self.live_db, timeout=5) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('drift_risk_mult', ?, ?)",
                    (str(value), datetime.now(timezone.utc).isoformat())
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to write drift state: {e}")

    def _trigger_halt(self, reason):
        try:
            with sqlite3.connect(self.live_db, timeout=5) as conn:
                conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('halted', 'TRUE', ?)", (datetime.now(timezone.utc).isoformat(),))
                conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('halt_reason', ?, ?)", (str(reason), datetime.now(timezone.utc).isoformat()))
                conn.commit()
            logger.critical(f"🔒 DRIFT MONITOR HALT: {reason}")
        except Exception as e:
            logger.critical(f"Failed to persist Halt State: {e}")

if __name__ == "__main__":
    monitor = AlphaDriftMonitor()
    monitor.evaluate()
