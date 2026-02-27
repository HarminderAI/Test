import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import math
import logging

logger = logging.getLogger("PerformanceController")
# Ensure it outputs to console if run standalone
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

class PerformanceController:
    """
    Statistical Oversight Layer
    --------------------------------
    Converts historical trade data into adaptive risk signals using EVT and PSR.
    """

    def __init__(
        self,
        db_path="paper.db",
        window_trades=75,
        min_trades_required=40,
        sharpe_threshold=0.5,
        psr_threshold=0.6,
        dd_slope_threshold=-0.002,
        risk_floor=0.3,
        risk_ceiling=1.0
    ):
        self.db_path = db_path
        self.window = window_trades
        self.min_trades = min_trades_required
        self.sharpe_threshold = sharpe_threshold
        self.psr_threshold = psr_threshold
        self.dd_slope_threshold = dd_slope_threshold
        self.risk_floor = risk_floor
        self.risk_ceiling = risk_ceiling

    # ==========================================================
    # DATABASE ACCESS
    # ==========================================================

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_recent_trades(self):
        with self._get_conn() as conn:
            df = pd.read_sql(
                f"""
                SELECT date, realized_pnl, trade_value 
                FROM trades
                WHERE realized_pnl IS NOT NULL
                ORDER BY date DESC
                LIMIT {self.window}
                """,
                conn,
                parse_dates=["date"]
            )
        return df.sort_values("date")

    def _load_equity_curve(self):
        with self._get_conn() as conn:
            df = pd.read_sql(
                """
                SELECT date, total_equity 
                FROM equity_log
                ORDER BY date ASC
                """,
                conn,
                parse_dates=["date"]
            )
        return df

    # ==========================================================
    # CORE METRICS
    # ==========================================================

    @staticmethod
    def _compute_sharpe(returns):
        if len(returns) < 2:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(len(returns))

    @staticmethod
    def _compute_psr(sharpe, skew, kurtosis, benchmark=0.0, n=1):
        """
        Probabilistic Sharpe Ratio (Strict Bailey & López de Prado)
        Accounts for non-normal skew and kurtosis.
        """
        if n < 3: return 0.0
        
        # Safe moment clipping to prevent math domain errors
        k_safe = max(0.0, kurtosis) 
        
        numerator = (sharpe - benchmark) * math.sqrt(n - 1)
        var_hac = 1 - (skew * sharpe) + (((k_safe + 2) / 4) * (sharpe**2))
        var_safe = max(var_hac, 1e-12)
        
        denominator = math.sqrt(var_safe)
        z = numerator / denominator
        
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    @staticmethod
    def _compute_expectancy(pnl_series):
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]
        win_rate = len(wins) / len(pnl_series) if len(pnl_series) else 0
        avg_win = wins.mean() if len(wins) else 0
        avg_loss = losses.mean() if len(losses) else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        return expectancy, win_rate

    @staticmethod
    def _compute_drawdown_slope(equity_df):
        if len(equity_df) < 20:
            return 0.0
        # Look at the last 20 periods for acute DD slope
        recent_equity = equity_df["total_equity"].values[-20:]
        x = np.arange(len(recent_equity))
        
        # Prevent Polyfit RankWarning on flat lines
        if np.std(recent_equity) < 1e-8:
            return 0.0
            
        slope = np.polyfit(x, recent_equity, 1)[0]
        mean_eq = recent_equity.mean()
        
        return slope / mean_eq if mean_eq > 0 else 0.0

    # ==========================================================
    # ADAPTIVE LOGIC
    # ==========================================================

    def evaluate(self):
        trades = self._load_recent_trades()

        if len(trades) < self.min_trades:
            logger.info("Not enough trades for statistical oversight. Defaulting to full capacity.")
            self._store_risk_multiplier(1.0)
            return 1.0

        pnl = trades["realized_pnl"]
        # 🚨 PATCH: Safe DataFrame column access
        if "trade_value" in trades.columns:
            gross_val = trades["trade_value"]
        else:
            gross_val = pnl.abs()
            
        safe_gross = gross_val.replace(0, 1.0)
        returns = pnl / safe_gross
        
        # Normalize returns by gross trade value to prevent dollar-inflation distortion
        gross_val = trades.get("trade_value", trades["realized_pnl"].abs())
        safe_gross = gross_val.replace(0, 1.0)
        returns = pnl / safe_gross

        sharpe = self._compute_sharpe(returns)
        skew = returns.skew()
        kurtosis = returns.kurtosis()
        
        if pd.isna(skew): skew = 0.0
        if pd.isna(kurtosis): kurtosis = 0.0

        psr = self._compute_psr(sharpe, skew, kurtosis, n=len(returns))
        expectancy, win_rate = self._compute_expectancy(pnl)

        equity_curve = self._load_equity_curve()
        dd_slope = self._compute_drawdown_slope(equity_curve)

        logger.info(
            f"Stat Oversight | Sharpe: {sharpe:.2f} | PSR: {psr:.2%} | "
            f"Exp: {expectancy:.2f} | WR: {win_rate:.2%} | DD Slope: {dd_slope:.5f}"
        )

        # ======================================================
        # DECISION ENGINE
        # ======================================================

        risk_mult = 1.0

        if psr < self.psr_threshold:
            logger.warning(f"📉 PSR below threshold ({psr:.2%} < {self.psr_threshold:.2%}). Penalizing risk.")
            risk_mult *= 0.5

        if sharpe < self.sharpe_threshold:
            logger.warning(f"📉 Sharpe below threshold ({sharpe:.2f} < {self.sharpe_threshold:.2f}). Penalizing risk.")
            risk_mult *= 0.7

        if dd_slope < self.dd_slope_threshold:
            logger.warning(f"📉 Acute Drawdown Slope detected ({dd_slope:.5f}). Penalizing risk.")
            risk_mult *= 0.5

        # Clamp between floor and ceiling
        risk_mult = max(self.risk_floor, min(self.risk_ceiling, risk_mult))

        # Persist adaptive risk multiplier
        self._store_risk_multiplier(risk_mult)
        logger.info(f"⚙️ Adaptive Risk Multiplier set to: {risk_mult:.2f}x")

        # Hard halt condition: Mathematical failure of strategy
        if psr < 0.20 and sharpe < 0:
            self._trigger_halt(f"Statistical Breakdown: Negative Sharpe ({sharpe:.2f}) & Catastrophic PSR ({psr:.2%})")

        return risk_mult

    # ==========================================================
    # STATE WRITES
    # ==========================================================

    def _store_risk_multiplier(self, value):
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO broker_status (key, value, timestamp)
                VALUES ('adaptive_risk_mult', ?, ?)
                """,
                (str(value), datetime.now(timezone.utc).isoformat())
            )
            conn.commit()

    def _trigger_halt(self, reason):
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO broker_status (key, value, timestamp)
                VALUES ('halted', 'TRUE', ?)
                """,
                (datetime.now(timezone.utc).isoformat(),)
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO broker_status (key, value, timestamp)
                VALUES ('halt_reason', ?, ?)
                """,
                (reason, datetime.now(timezone.utc).isoformat())
            )
            conn.commit()
        logger.critical(f"🚨 PERFORMANCE HALT: {reason}")

if __name__ == "__main__":
    pc = PerformanceController()
    pc.evaluate()
