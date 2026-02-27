import sqlite3
import logging
from datetime import datetime, timezone

logger = logging.getLogger("HealthScore")
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

class StrategyHealthScore:
    """
    The Risk Committee Aggregator
    -----------------------------
    Ingests all independent risk signals (Regime, Performance, Drift, Exposure)
    and combines them into a single weighted Health Score (0-100).
    Maps the score to a definitive, discrete Master Risk Multiplier.
    """

    def __init__(self, db_path="paper.db"):
        self.db_path = db_path
        
        # Institutional Weighting Matrix
        # Macro and Alpha Integrity matter most. Recent performance and exposure are secondary.
        self.weights = {
            'regime': 0.35,      # Macro Environment
            'drift': 0.30,       # Statistical Alpha Decay
            'performance': 0.20, # Rolling Sharpe / PSR
            'exposure': 0.15     # Correlation / Concentration
        }
        
        # Allocation Mapping (Health Score -> Final Capital Multiplier)
        self.tier_map = [
            (85, 1.0),  # 85-100 : Green  (Full Allocation)
            (70, 0.8),  # 70-84  : Yellow (Mild Stress, 20% cut)
            (50, 0.5),  # 50-69  : Orange (High Stress, Half Size)
            (30, 0.2),  # 30-49  : Red    (Severe Stress, Survival Sizing)
            (0,  0.0)   # 0-29   : Black  (System Mathematically Broken, Halt)
        ]

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def _fetch_multiplier(self, key, default=1.0):
        """Safely fetches a multiplier from broker_status. Defaults to 1.0 if missing."""
        try:
            with self._get_conn() as conn:
                row = conn.execute("SELECT value FROM broker_status WHERE key=?", (key,)).fetchone()
                if row: return float(row['value'])
        except Exception:
            pass
        return default

    def evaluate(self):
        logger.info("Convening Risk Committee for final Fiduciary Allocation...")

        # 1. Gather all independent votes
        m_regime = self._fetch_multiplier('regime_risk_mult', 1.0)
        m_drift = self._fetch_multiplier('drift_risk_mult', 1.0)
        m_perf = self._fetch_multiplier('adaptive_risk_mult', 1.0)
        m_exp = self._fetch_multiplier('exposure_risk_mult', 1.0)

        # 2. Calculate the weighted composite score (0 to 100)
        # Assuming each multiplier is bound between 0.0 and 1.0
        raw_score = (
            (m_regime * self.weights['regime']) +
            (m_drift * self.weights['drift']) +
            (m_perf * self.weights['performance']) +
            (m_exp * self.weights['exposure'])
        )
        
        health_score = round(raw_score * 100, 1)

        # 3. Map to discrete risk tiers
        master_mult = 1.0
        tier_color = "🟢"
        
        for threshold, multiplier in self.tier_map:
            if health_score >= threshold:
                master_mult = multiplier
                if multiplier == 0.8: tier_color = "🟡"
                elif multiplier == 0.5: tier_color = "🟠"
                elif multiplier <= 0.2: tier_color = "🔴"
                break
        
        logger.info("--- 🏛️ COMMITTEE VOTES ---")
        logger.info(f"Macro Regime : {m_regime:.2f}x (Weight: {self.weights['regime']*100}%)")
        logger.info(f"Alpha Drift  : {m_drift:.2f}x (Weight: {self.weights['drift']*100}%)")
        logger.info(f"Performance  : {m_perf:.2f}x (Weight: {self.weights['performance']*100}%)")
        logger.info(f"Sys Exposure : {m_exp:.2f}x (Weight: {self.weights['exposure']*100}%)")
        logger.info("-" * 25)
        logger.info(f"{tier_color} HEALTH SCORE : {health_score}/100")
        logger.info(f"⚖️ MASTER ALLOCATION: {master_mult:.2f}x")

        # 4. Enforce Hard Halt if mathematically broken
        if master_mult == 0.0:
            self._trigger_halt(f"Strategy Health Score collapsed to {health_score}/100. Fiduciary Halt engaged.")

        # 5. Persist State
        self._store_master_state(health_score, master_mult)
        return master_mult

    def _store_master_state(self, score, mult):
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            with self._get_conn() as conn:
                conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('health_score', ?, ?)", (str(score), timestamp))
                conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('master_risk_mult', ?, ?)", (str(mult), timestamp))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to write Master State: {e}")

    def _trigger_halt(self, reason):
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            with self._get_conn() as conn:
                conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('halted', 'TRUE', ?)", (timestamp,))
                conn.execute("INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('halt_reason', ?, ?)", (str(reason), timestamp))
                conn.commit()
            logger.critical(f"💀 COMMITTEE HALT: {reason}")
        except Exception as e:
            logger.critical(f"Failed to persist Halt State: {e}")

if __name__ == "__main__":
    committee = StrategyHealthScore()
    committee.evaluate()
