import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import logging
import os

logger = logging.getLogger("ExposureController")
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

class ExposureController:
    """
    Systemic Risk Watchdog
    ----------------------
    Monitors active positions for dangerous concentration:
    1. Sector Concentration (Max % of equity in one sector)
    2. Correlation Clustering (Average pairwise correlation)
    3. Portfolio Beta (Leverage to the broader market)
    """

    def __init__(
        self,
        db_path="paper.db",
        index_ticker="^NSEI",         # Benchmark for Beta calculation
        max_sector_exposure=0.40,     # Max 40% of Equity in a single sector
        max_avg_correlation=0.65,     # If assets are > 65% correlated, penalize
        max_portfolio_beta=1.5,       # If portfolio moves 1.5x the market, penalize
        history_window=60,            # Days of history to calculate correlation
        risk_floor=0.2,
        risk_ceiling=1.0
    ):
        self.db_path = db_path
        self.index_ticker = index_ticker
        self.max_sector_exposure = max_sector_exposure
        self.max_avg_correlation = max_avg_correlation
        self.max_portfolio_beta = max_portfolio_beta
        self.history_window = history_window
        self.risk_floor = risk_floor
        self.risk_ceiling = risk_ceiling

    # ==========================================================
    # DATA INGESTION
    # ==========================================================

    def _get_active_positions(self):
        """Fetches current positions and total equity from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                pos_df = pd.read_sql("SELECT ticker, qty, avg_price FROM positions", conn)
                acct = conn.execute("SELECT cash FROM account WHERE id=1").fetchone()
                cash = acct[0] if acct else 0.0
                
            if pos_df.empty:
                return pos_df, cash, cash
                
            pos_df['mkt_val'] = pos_df['qty'] * pos_df['avg_price']
            total_equity = cash + pos_df['mkt_val'].sum()
            pos_df['weight'] = pos_df['mkt_val'] / total_equity
            
            return pos_df, cash, total_equity
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return pd.DataFrame(), 0.0, 0.0

    def _fetch_histories_and_sectors(self, tickers):
        """Fetches historical daily returns and sector tags for the correlation matrix."""
        returns_dict = {}
        sectors = {}
        
        # Add benchmark to the fetch list for Beta calc
        fetch_list = tickers + [self.index_ticker]
        
        logger.info(f"Fetching {self.history_window}d history for correlation analysis...")
        
        try:
            data = yf.download(fetch_list, period=f"{self.history_window * 2}d", interval="1d", progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                close_data = data['Close']
            else:
                close_data = pd.DataFrame({fetch_list[0]: data['Close']})

            # Calculate daily returns
            rets = close_data.pct_change().dropna()
            
            # Fetch Sector Info (Fail gracefully if Yahoo blocks the request)
            for tkr in tickers:
                returns_dict[tkr] = rets[tkr] if tkr in rets.columns else pd.Series(dtype=float)
                try:
                    info = yf.Ticker(tkr).info
                    sectors[tkr] = info.get('sector', 'Unknown')
                except:
                    sectors[tkr] = 'Unknown'
                    
            benchmark_rets = rets[self.index_ticker] if self.index_ticker in rets.columns else pd.Series(dtype=float)
            
            return pd.DataFrame(returns_dict), benchmark_rets, sectors
            
        except Exception as e:
            logger.error(f"Failed to fetch correlation data: {e}")
            return pd.DataFrame(), pd.Series(dtype=float), {}

    # ==========================================================
    # CORE CLASSIFICATION
    # ==========================================================

    def evaluate(self):
        pos_df, cash, total_equity = self._get_active_positions()
        
        # If 0 or 1 positions, there is no correlation risk
        if len(pos_df) < 2:
            logger.info("Portfolio has < 2 positions. Systemic exposure risk is minimal.")
            self._store_risk_multiplier(1.0)
            return 1.0

        tickers = pos_df['ticker'].tolist()
        returns_df, bm_returns, sector_map = self._fetch_histories_and_sectors(tickers)
        
        if returns_df.empty:
            logger.warning("Could not calculate correlation matrices. Defaulting to 1.0x.")
            self._store_risk_multiplier(1.0)
            return 1.0

        pos_df['sector'] = pos_df['ticker'].map(sector_map)

        # ------------------------------------------------------
        # 1. Sector Concentration Check
        # ------------------------------------------------------
        sector_weights = pos_df.groupby('sector')['weight'].sum()
        max_sector = sector_weights.idxmax()
        max_sec_weight = sector_weights.max()
        
        # ------------------------------------------------------
        # 2. Correlation Clustering Check
        # ------------------------------------------------------
        corr_matrix = returns_df.corr()
        # Extract the upper triangle of the correlation matrix to get unique pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        avg_correlation = upper_tri.mean().mean()
        if pd.isna(avg_correlation): avg_correlation = 0.0
        
        # ------------------------------------------------------
        # 3. Portfolio Beta Check
        # ------------------------------------------------------
        portfolio_beta = 0.0
        if not bm_returns.empty and len(bm_returns) > 10:
            bm_var = bm_returns.var()
            if bm_var > 0:
                for tkr in tickers:
                    if tkr in returns_df.columns:
                        cov = returns_df[tkr].cov(bm_returns)
                        asset_beta = cov / bm_var
                        weight = pos_df.loc[pos_df['ticker'] == tkr, 'weight'].values[0]
                        portfolio_beta += (asset_beta * weight)

        logger.info("--- PORTFOLIO EXPOSURE METRICS ---")
        logger.info(f"Top Sector:   {max_sector} ({max_sec_weight:.1%})")
        logger.info(f"Avg Pair Corr: {avg_correlation:.3f}")
        logger.info(f"Port. Beta:   {portfolio_beta:.2f}")

        # ======================================================
        # DECISION ENGINE (The Fiduciary Logic)
        # ======================================================
        risk_mult = 1.0
        flags = []

        # A. Over-concentration in one sector
        if max_sec_weight > self.max_sector_exposure:
            penalty = max(0.5, 1.0 - (max_sec_weight - self.max_sector_exposure))
            risk_mult *= penalty
            flags.append(f"Sector Risk ({max_sector})")

        # B. Highly correlated assets (If avg correlation is 0.8, it's virtually 1 asset)
        if avg_correlation > self.max_avg_correlation:
            risk_mult *= 0.6
            flags.append("Correlation Clustering")

        # C. Aggressive Beta (Leveraged to the market)
        if portfolio_beta > self.max_portfolio_beta:
            risk_mult *= 0.7
            flags.append(f"High Beta ({portfolio_beta:.1f})")

        risk_mult = max(self.risk_floor, min(self.risk_ceiling, risk_mult))

        if flags:
            logger.warning(f"🚨 SYSTEMIC EXPOSURE DETECTED: {', '.join(flags)}")
            logger.warning(f"⚙️ Exposure Controller applying {risk_mult:.2f}x penalty to force diversification.")
        else:
            logger.info("✅ Portfolio exposure is diversified and statistically stable.")

        self._store_risk_multiplier(risk_mult)
        return risk_mult

    # ==========================================================
    # STATE WRITES
    # ==========================================================

    def _store_risk_multiplier(self, value):
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO broker_status (key, value, timestamp) VALUES ('exposure_risk_mult', ?, ?)",
                    (str(value), datetime.now(timezone.utc).isoformat())
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to write exposure state: {e}")

if __name__ == "__main__":
    controller = ExposureController()
    controller.evaluate()
