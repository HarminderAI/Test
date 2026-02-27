"""
THE CONSTITUTION (Config.py)
----------------------------
This file controls the global behavior of the trading system.
Changes here affect Data Validation, Risk Management, Logic Execution, and Backtesting.

PHILOSOPHY:
- Fail-Closed: When in doubt, do not trade.
- Fiduciary Standard: Assume slippage, assume commissions, assume gaps.
- Explicit Intent: No magic numbers in the code; all constants live here.
"""

# ==============================================================================
# 1. DATA GUARD (data.py)
# Controls how market data is fetched, sanitized, and validated.
# ==============================================================================

# Minimum number of bars required to trust a dataset.
# Default: 60 (Enough for 20-period indicators + warmup)
MIN_HISTORY_BARS = 60

# ------------------------------------------------------------------------------
# SAFETY SWITCHES (Modify with extreme caution)
# ------------------------------------------------------------------------------

# Allow prices <= 0?
# Default: False (Strictly forbidden for Equities/Crypto).
# Set to True ONLY for specific Futures contracts (e.g., Crude Oil 2020).
ALLOW_NEGATIVE_PRICES = False

# Allow Volume = 0?
# Default: False (Strict Equity Mode).
# - If False: Rejects datasets with zero-volume rows (protects against bad feeds).
# - If True: Fills zero volume with 0 (Necessary for FOREX / INDICES / CRYPTO).
ALLOW_ZERO_VOLUME = False

# Drop the last bar if it's "Live" (incomplete)?
# Default: True (Prevents "Repainting" bias).
# - If True: We only trade completely closed candles.
# - If False: We use the forming candle (High risk of signal disappearing).
DROP_INCOMPLETE_INTRADAY_BAR = True

# ------------------------------------------------------------------------------
# CACHING (Performance vs. Freshness)
# ------------------------------------------------------------------------------

# Enable caching for OHLC data?
# Default: False (Always fetch fresh data for safety).
# Set to True for high-frequency loops to avoid rate limits.
CACHE_OHLC = False

# Time-To-Live (TTL) for caches (in seconds).
# Daily data is stable; Intraday data needs frequent refresh.
OHLC_CACHE_TTL_DAILY = 60       # 1 minute
OHLC_CACHE_TTL_INTRADAY = 10    # 10 seconds
PRICE_CACHE_TTL_SEC = 30        # Real-time price check cache

# ==============================================================================
# 2. RISK GOVERNOR (risk.py)
# Controls position sizing, stop losses, and portfolio heat.
# ==============================================================================

# Maximum % of Account Equity to risk on a SINGLE trade.
# Default: 0.01 (1%).
# This is "R" - the amount you lose if your stop is hit.
RISK_PER_TRADE_PCT = 0.01

# Maximum % of Account Equity allowed in Total Open Risk (Heat).
# Default: 0.06 (6%).
# If total risk exceeds this, new trades are blocked and stops are tightened.
MAX_PORTFOLIO_HEAT_PCT = 0.06

# Maximum Drawdown Tolerance (Soft Limit).
# Default: 0.10 (10%).
# If drawdown exceeds this, the Governor enters "DEFCON 1" (Halts buying).
MAX_DRAWDOWN_LIMIT_PCT = 0.10

# ==============================================================================
# 3. LOGIC ENGINE (logic.py)
# Controls technical indicators and signal generation parameters.
# ==============================================================================

# Lookback period for ATR (Average True Range) volatility calculation.
ATR_PERIOD = 14

# Lookback period for Donchian Channels (Breakout Logic).
DONCHIAN_PERIOD = 20

# Stop Loss Distance Multiplier (in units of ATR).
# Default: 2.0 (Wide enough to breathe, tight enough to survive).
ATR_STOP_MULTIPLIER = 2.0

# ==============================================================================
# 4. BACKTEST SIMULATION (backtest.py)
# Controls the "Time Machine" physics.
# ==============================================================================

# Commission per trade (as a decimal percentage of trade value).
# Default: 0.001 (0.1% or 10 bps).
# E.g., Binance Spot = 0.1%, Interactive Brokers Pro ≈ 0.05%
COMMISSION_PCT = 0.001

# Slippage assumption (as a decimal percentage of price).
# Default: 0.0005 (0.05% or 5 bps).
# Represents the cost of crossing the spread + market impact.
SLIPPAGE_PCT = 0.0005

# Liquidity Constraint.
# Maximum % of a day's volume we are allowed to "be".
# Default: 0.01 (1%).
# Prevents the backtest from buying 100% of a penny stock's daily volume.
MAX_PCT_VOLUME = 0.01

# ==============================================================================
# 5. ASSET CLASSES (Context-Aware overrides)
# ==============================================================================

# List of Crypto Tickers (Used to trigger warnings or override volume checks).
CRYPTO_TICKERS = {
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "BNB-USD",
    "XRP-USD"
}

# List of Forex Tickers (Used to allow Zero Volume).
FOREX_TICKERS = {
    "EURUSD=X",
    "GBPUSD=X",
    "JPY=X"
}

# ==============================================================================
# 6. LOGIC & TRAILING STOP THRESHOLDS (Required by logic.py)
# ==============================================================================
SL_PRICE_BUFFER = 0.995        # 0.5% buffer for long stops
SL_PRICE_BUFFER_SHORT = 1.005  # 0.5% buffer for short stops
MIN_ATR_PCT = 0.015            # Minimum volatility to take a trade (1.5%)
MAX_ATR_PCT = 0.08             # Maximum volatility to take a trade (8%)
MIN_RISK_MULT = 0.3            # Maximum tightening allowed by Risk Governor
EMA_MID = 50                   # Core structural trend line
WEEKLY_CONFIRM_MODE = False    # Require weekly trend alignment?
RESPECT_TIGHTEN_MODE = True    # Allow risk.py to override standard logic?
MOMENTUM_R_THRESH = 2.0        # R-Multiple required to trigger momentum exit
VOL_DRY_RATIO = 0.5            # Volume drop required to exit on momentum decay
RSI_EXIT = 70                  # RSI overbought threshold
ATR_TRAIL_MULT = 3.0           # Standard trailing stop distance
TIGHTEN_MULT = 1.5             # Tightened trailing stop distance (Emergency)
PARTIAL_BOOK_R = 3.0           # R-Multiple to automatically take partial profits
