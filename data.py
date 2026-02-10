import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta, timezone
import config

# Setup module-level logger with safe NullHandler injection
logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
    logger.addHandler(logging.NullHandler())

# Immutable Config Snapshot
# Freezes configuration at module load time to ensure consistency.
# Logs a warning so operators know runtime config changes won't apply here.
_CONFIG = {
    "ALLOW_NEGATIVE_PRICES": getattr(config, "ALLOW_NEGATIVE_PRICES", False),
    "ALLOW_ZERO_VOLUME": getattr(config, "ALLOW_ZERO_VOLUME", False),
    "DROP_INCOMPLETE_INTRADAY_BAR": getattr(config, "DROP_INCOMPLETE_INTRADAY_BAR", True),
    "MIN_HISTORY_BARS": getattr(config, "MIN_HISTORY_BARS", 60),
    "OHLC_CACHE_TTL_DAILY": getattr(config, "OHLC_CACHE_TTL_DAILY", 60),
    "OHLC_CACHE_TTL_INTRADAY": getattr(config, "OHLC_CACHE_TTL_INTRADAY", 10),
    "PRICE_CACHE_TTL_SEC": getattr(config, "PRICE_CACHE_TTL_SEC", 60),
    "CACHE_OHLC": getattr(config, "CACHE_OHLC", False)
}
logger.info(f"DataGuard initialized with immutable config snapshot: {_CONFIG}")

class DataGuard:
    """
    The Data Sanitizer.
    
    Responsibility:
    1. Fetch market data (OHLCV) with robust retries & backoff.
    2. Enforce schema (Open, High, Low, Close, Volume) and Numeric Types.
    3. Enforce time monotonicity (Sort, Dedup, Heal).
    4. Enforce physics (High >= Low, Configurable Negative Prices).
    5. Handle missing data (Fail-Closed on Price NaNs).
    6. Caching: Manual, failure-aware caching (never caches errors).
    """
    
    # [FIX #1] Manual Cache Storage
    # Format: {(ticker, period, interval): (dataframe, expiry_timestamp)}
    _OHLC_CACHE = {}
    _PRICE_CACHE = {}

    @staticmethod
    def fetch_data(ticker, period="2y", interval="1d", retries=3):
        """
        Public entry point for data fetching. 
        Supports failure-aware caching.
        """
        use_cache = _CONFIG["CACHE_OHLC"]
        
        # Cache Key does NOT include retries (effort doesn't change data)
        cache_key = (ticker, period, interval)
        now = time.time()
        
        if use_cache:
            # Check Cache
            if cache_key in DataGuard._OHLC_CACHE:
                cached_df, expiry = DataGuard._OHLC_CACHE[cache_key]
                if now < expiry:
                    return cached_df.copy() # Return copy to prevent mutation bugs
                else:
                    del DataGuard._OHLC_CACHE[cache_key] # Expired

        # Perform Fetch
        df = DataGuard._fetch_data_internal(ticker, period, interval, retries)
        
        # [FIX #1] Cache Success Only
        # Only cache if we got valid data. Never cache an empty DataFrame (failure).
        if use_cache and not df.empty:
            if interval == "1d":
                ttl = _CONFIG["OHLC_CACHE_TTL_DAILY"]
            else:
                ttl = _CONFIG["OHLC_CACHE_TTL_INTRADAY"]
            
            DataGuard._OHLC_CACHE[cache_key] = (df, now + ttl)
            
        return df

    @staticmethod
    def _fetch_data_internal(ticker, period, interval, retries):
        """
        Internal robust fetcher with exponential backoff and strict validation.
        """
        df = pd.DataFrame()
        attempt = 0
        last_error = None
        
        allow_negative = _CONFIG["ALLOW_NEGATIVE_PRICES"]
        allow_zero_vol = _CONFIG["ALLOW_ZERO_VOLUME"]
        drop_intraday = _CONFIG["DROP_INCOMPLETE_INTRADAY_BAR"]
        
        if allow_negative:
             logger.debug(f"ALLOW_NEGATIVE_PRICES=True for {ticker} — Negative prices enabled.")
        
        while attempt < retries:
            try:
                # yfinance prints errors to stdout; we suppress/ignore them.
                df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
                
                if not df.empty:
                    break
                else:
                    last_error = "Empty DataFrame returned by yfinance (Symbol delisted? Rate Limit?)"
                    
            except Exception as e:
                last_error = str(e)
                sleep_time = (2 ** attempt) + np.random.uniform(0, 1)
                logger.warning(f"Attempt {attempt+1}/{retries} failed for {ticker}: {e}. Retrying in {sleep_time:.2f}s...")
                time.sleep(min(sleep_time, 30))
            
            attempt += 1
            
        # Critical Failure Check
        if df.empty:
            logger.error(f"CRITICAL: No data found for {ticker} after {retries} attempts. Reason: {last_error}")
            return pd.DataFrame() 

        # [FIX #7] Data Completeness Sanity Check
        # If we asked for 2 years of daily data (~500 bars) and got 5, something is wrong.
        if period == "2y" and interval == "1d":
            expected_min = 250 * 0.5 
            if len(df) < expected_min:
                logger.warning(f"Data completeness warning for {ticker}: Requested 2y, got {len(df)} rows. (Potential partial history)")

        # Standardization
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Strict Column Mapping
        column_map = {
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
            "adj close": "Adj Close"
        }
        df.columns = [column_map.get(c.lower(), c) for c in df.columns]
        
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        
        # Check for missing columns
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"CRITICAL: Data schema invalid for {ticker}. Missing: {missing}")
            return pd.DataFrame()

        # Log unexpected columns
        unknown_cols = set(df.columns) - set(required_cols)
        if unknown_cols:
            logger.debug(f"Ignored extra columns for {ticker}: {unknown_cols}")

        df = df[required_cols].copy()
        
        # Numeric Coercion
        for c in required_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
        # Pre-Physics NaN Detection
        if df[required_cols].isnull().all(axis=1).any():
             logger.critical(f"CRITICAL: Entire OHLC rows are NaN for {ticker}. Feed corruption.")
             return pd.DataFrame()

        # Time Index Integrity
        df = df.sort_index()
        if df.index.duplicated().any():
            logger.warning(f"Duplicate timestamps found in {ticker}. Keeping last.")
            df = df[~df.index.duplicated(keep="last")]
            
        # [FIX #6] Strict Monotonicity Check
        # If deduplication failed to fix monotonicity, the feed is garbage.
        if not df.index.is_monotonic_increasing:
             logger.critical(f"CRITICAL: Time index corruption (Irreparable) for {ticker}. Discarding.")
             return pd.DataFrame()
        
        # Initial Row Count
        initial_len = len(df)
        dropped_physics = 0

        # Physics Validation
        bad_physics_mask = (df["High"] < df["Low"]) | \
                           (df["Open"] > df["High"]) | (df["Close"] > df["High"]) | \
                           (df["Open"] < df["Low"]) | (df["Close"] < df["Low"])

        if not allow_negative:
            zero_price_mask = (df[["Open","High","Low","Close"]] <= 0).any(axis=1)
            bad_physics_mask = bad_physics_mask | zero_price_mask
        
        neg_vol_mask = (df["Volume"] < 0)
        combined_bad_mask = bad_physics_mask | neg_vol_mask
        
        if combined_bad_mask.any():
            dropped_physics = combined_bad_mask.sum()
            if dropped_physics == initial_len:
                logger.critical(f"CRITICAL: All {initial_len} candles failed physics for {ticker}. Data feed corrupted.")
                return pd.DataFrame()
                
            logger.warning(f"Dropped {dropped_physics} rows with IMPOSSIBLE physics for {ticker}")
            df = df[~combined_bad_mask]

        # Fail-Closed on Price NaNs
        price_cols = ["Open", "High", "Low", "Close"]
        if df[price_cols].isnull().any().any():
            nan_count = df[price_cols].isnull().sum().sum()
            logger.error(f"CRITICAL: Found {nan_count} Price NaNs in {ticker}. Integrity compromised. Discarding.")
            return pd.DataFrame()

        # [FIX #4] Pure Volume Semantics
        if allow_zero_vol:
            # Accept zero volumes. Fill NaNs with 0.
            df["Volume"] = df["Volume"].fillna(0).astype(float)
        else:
            # Reject zero volumes.
            if df["Volume"].isnull().any():
                logger.error(f"CRITICAL: Volume NaNs detected in {ticker} and ALLOW_ZERO_VOLUME=False. Discarding.")
                return pd.DataFrame()
            
            if (df["Volume"] == 0).any():
                zero_count = (df["Volume"] == 0).sum()
                zero_ratio = zero_count / len(df)
                
                # If too many zeros, reject dataset.
                if zero_ratio > 0.10: 
                    logger.critical(f"CRITICAL: {zero_ratio:.1%} of rows have Zero Volume. Quality too low for {ticker}. Dataset Rejected.")
                    return pd.DataFrame()
                
                # Revert to dropping rows (No Epsilon Injection).
                # Epsilon injection is mathematically unsafe for VWAP/OBV.
                # If the user forbids zero volume, they must accept gaps.
                logger.warning(f"Dropped {zero_count} rows with Zero Volume (ALLOW_ZERO_VOLUME=False).")
                df = df[df["Volume"] != 0]

        # Post-Filter Sanity Check
        if not allow_negative and (df[price_cols] <= 0).any().any():
             logger.critical(f"CRITICAL: Negative prices detected in {ticker} after filtering. Logic error. Discarding.")
             return pd.DataFrame()

        # Safe Timezone Handling
        if interval == "1d" and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # [FIX #3] Robust Live-Bar Dropping (Time Delta Heuristic)
        if drop_intraday and not df.empty:
            last_ts = df.index[-1]
            
            # Normalize to UTC
            if last_ts.tzinfo is None:
                last_ts_utc = last_ts.replace(tzinfo=timezone.utc)
            else:
                last_ts_utc = last_ts.astimezone(timezone.utc)
                
            now_utc = datetime.now(timezone.utc)
            
            # Calculate Age
            age = max(now_utc - last_ts_utc, timedelta(0))
            
            # Thresholds:
            # Intraday: Drop if < 4 hours old (likely forming)
            # Daily: Drop if < 20 hours old (likely today's forming bar)
            # This handles holidays/half-days better than strict date matching.
            threshold = timedelta(hours=20) if interval == "1d" else timedelta(hours=4)
            
            if age < threshold:
                df = df.iloc[:-1]

        # Minimum Length Check
        min_bars = _CONFIG["MIN_HISTORY_BARS"]
        if len(df) < min_bars:
            logger.error(
                f"Insufficient usable history for {ticker}. "
                f"Got {len(df)} bars (Required: {min_bars}). "
                f"Initial fetch: {initial_len}, Dropped Physics: {dropped_physics}."
            )
            return pd.DataFrame()

        return df

    @staticmethod
    def get_latest_price(ticker):
        """
        Public entry point for price fetching with Manual Cache.
        """
        ttl_sec = _CONFIG["PRICE_CACHE_TTL_SEC"]
        now = time.time()
        
        # Check Cache
        if ticker in DataGuard._PRICE_CACHE:
            price, expiry = DataGuard._PRICE_CACHE[ticker]
            if now < expiry:
                return price
            else:
                del DataGuard._PRICE_CACHE[ticker]
        
        # Fetch
        price = DataGuard._fetch_price_internal(ticker)
        
        # Cache Success Only
        if price is not None:
            DataGuard._PRICE_CACHE[ticker] = (price, now + ttl_sec)
            
        return price

    @staticmethod
    def _fetch_price_internal(ticker):
        """Internal price fetcher."""
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            
            price = (
                info.get("last_price") or 
                info.get("lastPrice") or 
                info.get("regularMarketPrice")
            )
            
            # Stale Price Detection
            ts = info.get("lastTradeTime") or info.get("regularMarketTime")
            if ts:
                age_sec = datetime.now(timezone.utc).timestamp() - ts
                if age_sec > 14400: # 4 hours
                     logger.warning(f"Fast price for {ticker} is STALE ({age_sec/3600:.1f}h old). Forcing fallback.")
                     price = None 
            
            allow_negative = _CONFIG["ALLOW_NEGATIVE_PRICES"]
            if price and np.isfinite(price):
                if allow_negative or price > 0:
                    return float(price)
                
            # Fallback 1: history(period='1d')
            hist = t.history(period="1d")
            if not hist.empty:
                last_close = hist["Close"].iloc[-1]
                last_ts = hist.index[-1]
                
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
                else:
                    last_ts = last_ts.astimezone(timezone.utc)
                    
                age = (datetime.now(timezone.utc) - last_ts).total_seconds()
                if age > 86400: # 24 hours
                     logger.warning(f"Fallback history price for {ticker} is STALE ({age/3600:.1f}h old).")
                
                return float(last_close)
                
        except Exception as e:
            logger.warning(f"Fast price fetch failed for {ticker}: {e}")
            
        # Fallback 2: Full Guarded Fetch
        logger.info(f"Falling back to full fetch for {ticker} price...")
        df = DataGuard.fetch_data(ticker, period="5d", interval="1d")
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
      
