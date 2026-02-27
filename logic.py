import pandas as pd
import pandas_ta as ta
import numpy as np
import config

# ==========================================================
# 0. CONFIG VALIDATION
# ==========================================================
def _validate_config():
    """ Enforce Safe Configuration Bounds at Runtime """
    try:
        assert config.SL_PRICE_BUFFER < 1.0, "SL_PRICE_BUFFER must be < 1.0"
        assert config.SL_PRICE_BUFFER_SHORT > 1.0, "SL_PRICE_BUFFER_SHORT must be > 1.0"
        assert config.MAX_ATR_PCT > 0, "MAX_ATR_PCT must be > 0"
        assert 0 < config.MIN_RISK_MULT <= 1.0, "MIN_RISK_MULT must be in (0, 1.0]"
    except AssertionError as e:
        raise RuntimeError(f"CRITICAL CONFIG ERROR: {e}")

_validate_config()

# ==========================================================
# 1. IMMUTABLE & STRICT HELPERS
# ==========================================================

def validate_position(position):
    """ Universal Asset Support (Negative Prices / Spreads) """
    pos = position.copy()
    
    required = ["entry", "initial_sl"]
    for k in required:
        if k not in pos: raise ValueError(f"Position missing '{k}'")
            
    try:
        entry = float(pos["entry"])
        init_sl = float(pos["initial_sl"])
        
        # Detect Direction
        if entry > init_sl:
            pos["is_long"] = True
            unit_risk = entry - init_sl
        elif entry < init_sl:
            pos["is_long"] = False
            unit_risk = init_sl - entry
        else:
            raise ValueError("Entry equals Initial SL (Infinite Risk)")
            
        # Absolute Magnitude for Negative-Priced Assets
        # Uses abs(entry) to scale risk floor.
        min_allowed_risk = max(abs(entry) * 1e-5, 1e-8)
        if unit_risk < min_allowed_risk:
            raise ValueError(f"Unit Risk ({unit_risk}) below safe floor. Invalid stop.")
            
        pos["entry"] = entry
        pos["initial_sl"] = init_sl
        pos["unit_risk"] = unit_risk
        
        if "current_sl" in pos:
            pos["current_sl"] = float(pos["current_sl"])
            
    except ValueError as e:
        raise ValueError(f"Position Data Error: {e}")
        
    return pos

def safe_resample_weekly(df_daily):
    """ Timezone-Agnostic Resampling """
    df = df_daily.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # Resample W-FRI
    df_w = df.resample('W-FRI', closed='right', label='right').apply({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    })
    
    # Drop future-dated weekly bars
    if not df_w.empty and not df.empty:
        if df_w.index[-1] > (df.index[-1] + pd.Timedelta(days=1)):
             df_w = df_w.iloc[:-1]
                 
    return df_w

# ==========================================================
# 2. CORE EXIT LOGIC
# ==========================================================

def evaluate_exit(df, position, risk_mult=1.0):
    
    # --- A. INPUT GUARDS ---
    pos = validate_position(position)
    is_long = pos["is_long"]
    unit_risk = pos["unit_risk"]
    
    risk_mult = max(min(risk_mult, 1.0), config.MIN_RISK_MULT)
    
    # --- B. STATE UNPACKING ---
    df = df.copy()
    close = df["Close"]
    current_price = close.iloc[-1]
    
    entry = pos["entry"]
    current_sl = pos.get("current_sl", pos["initial_sl"])
    partial_taken = pos.get("partial_taken", False)
    
    # R-Multiple Calculation
    if is_long:
        r_multiple = (current_price - entry) / unit_risk
    else:
        r_multiple = (entry - current_price) / unit_risk
        
    # Diagnostic Cap Flagging
    r_capped = False
    if r_multiple > 100.0:
        r_multiple = 100.0
        r_capped = True
    elif r_multiple < -100.0:
        r_multiple = -100.0
        r_capped = True

    # --- C. PRIORITY 1: HARD EXITS ---
    
    stop_hit = False
    is_gap = False
    
    # [FIX #1] Absolute Micro-Scalar Tolerance
    # Handles Price=0 scenarios by enforcing a 1e-10 floor.
    # Tighter of (1% Risk) or (0.1% Price).
    base_eps = min(unit_risk * 0.01, abs(current_price) * 0.001)
    eps = max(base_eps, abs(current_price) * 1e-6, 1e-10)
    
    if is_long:
        if df["Open"].iloc[-1] < (current_sl - eps): 
            stop_hit = True; is_gap = True
        elif df["Low"].iloc[-1] <= current_sl:
            stop_hit = True; is_gap = False
    else:
        if df["Open"].iloc[-1] > (current_sl + eps): 
            stop_hit = True; is_gap = True
        elif df["High"].iloc[-1] >= current_sl:
            stop_hit = True; is_gap = False
            
    if stop_hit:
        slippage_dist = abs(current_price - current_sl)
        slippage_r = slippage_dist / unit_risk
        reason = f"Stop Loss Hit ({current_sl})"
        reason += f" [GAP: {slippage_r:.2f}R]" if is_gap else " [INTRABAR WICK]"
        return _validate_output({
            "action": "EXIT", "reason": reason, "new_sl": None, 
            "set_partial_taken": partial_taken, "r_capped": r_capped
        }, is_long, current_sl)

    if len(df) < 60:
         return _validate_output({
            "action": "HOLD", "reason": "Initializing", "new_sl": current_sl, 
            "set_partial_taken": partial_taken, "r_capped": r_capped
        }, is_long, current_sl)

    # --- D. INDICATORS & FAILURE MODES ---
    
    tighten_mode = False
    data_failure = False
    
    ema50_d = ta.ema(close, config.EMA_MID).iloc[-1]
    rsi_d   = ta.rsi(close, 14).iloc[-1]
    
    # Critical Indicator Failure
    if np.isnan(ema50_d) or np.isnan(rsi_d):
        data_failure = True
        tighten_mode = True 

    # ATR Logic (With Fail-Safe)
    atr_val = ta.atr(df["High"], df["Low"], close, 14).iloc[-1]
    
    # Fail-Safe Volatility
    if np.isnan(atr_val) or atr_val <= 0 or data_failure:
        dist_to_stop = abs(current_price - current_sl)
        abs_price = abs(current_price)
        
        if data_failure:
            # Emergency Mode: Tighten fast, never expand.
            safe_dist = abs_price * 0.001 
            atr_val = max(min(dist_to_stop, abs_price * config.MIN_ATR_PCT), safe_dist)
            atr_val = min(atr_val, dist_to_stop) 
        else:
            min_vol = abs_price * config.MIN_ATR_PCT
            max_vol = abs_price * config.MAX_ATR_PCT
            atr_val = min(max(dist_to_stop, min_vol), max_vol)

    # Volume (Safe Median)
    vol_avg = df["Volume"].rolling(20).median().iloc[-1]
    
    # [POLICY] Missing volume defaults to neutral (1.0).
    if np.isnan(vol_avg) or vol_avg <= 0:
        vol_ratio = 1.0 
    else:
        vol_ratio = df["Volume"].iloc[-1] / vol_avg

    # Weekly Context
    trend_valid = False
    if not data_failure: 
        # [POLICY] Weekly trend is ignored if Daily data is corrupted.
        df_w = safe_resample_weekly(df)
        if len(df_w) >= 25:
            ema20_w_series = ta.ema(df_w["Close"], 20)
            
            # [FIX #2] Relative "Swiss Cheese" Data Check
            # If >20% of the history is NaN, the trend is unreliable.
            if ema20_w_series.isna().mean() > 0.2:
                trend_valid = False 
            else:
                ema20_w = ema20_w_series.iloc[-1]
                if not np.isnan(ema20_w):
                    trend_valid = (df_w["Close"].iloc[-1] > ema20_w) if is_long else (df_w["Close"].iloc[-1] < ema20_w)

    # --- E. LOGIC GATES ---
    
    exit_signal = False
    exit_reason = ""

    if not data_failure:
        
        # 1. Structure
        structure_broken = (current_price < ema50_d) if is_long else (current_price > ema50_d)
        
        if structure_broken:
            if config.WEEKLY_CONFIRM_MODE and trend_valid:
                tighten_mode = True
                exit_reason = "Daily Broken / Weekly Valid (Tightening)"
            else:
                exit_signal = True
                exit_reason = "Trend Broken (Structure)"

        # 2. Momentum
        can_exit_mom = not (config.RESPECT_TIGHTEN_MODE and tighten_mode)
        if can_exit_mom and r_multiple > config.MOMENTUM_R_THRESH and vol_ratio < config.VOL_DRY_RATIO:
            mom_fail = (rsi_d < config.RSI_EXIT) if is_long else (rsi_d > (100 - config.RSI_EXIT))
            if mom_fail:
                exit_signal = True
                exit_reason = "Momentum Decay"

    else:
        exit_reason = "Data Failure (Emergency Tightening)"

    if exit_signal:
        return _validate_output({
            "action": "EXIT", "reason": exit_reason, "new_sl": None, 
            "set_partial_taken": partial_taken, "r_capped": r_capped
        }, is_long, current_sl)

    # --- F. TRAILING STOP (RATCHET) ---
    
    base_mult = config.ATR_TRAIL_MULT
    effective_mult = base_mult * risk_mult
    if tighten_mode:
        effective_mult = min(effective_mult, config.TIGHTEN_MULT)

    new_sl = current_sl
    
    if is_long:
        raw_trail = current_price - (effective_mult * atr_val)
        new_sl = max(current_sl, raw_trail) # Ratchet UP
        
        # Dead Man's Switch (Stagnation Escape)
        if data_failure:
            new_sl = max(new_sl, current_sl + (unit_risk * 0.1))
            new_sl = min(new_sl, current_price) # Intermediate Clamp
            
        new_sl = min(new_sl, current_price * config.SL_PRICE_BUFFER) # Final Clamp
    else:
        raw_trail = current_price + (effective_mult * atr_val)
        new_sl = min(current_sl, raw_trail) # Ratchet DOWN
        
        # Dead Man's Switch
        if data_failure:
            new_sl = min(new_sl, current_sl - (unit_risk * 0.1))
            new_sl = max(new_sl, current_price) # Intermediate Clamp
            
        new_sl = max(new_sl, current_price * config.SL_PRICE_BUFFER_SHORT) # Final Clamp

    new_sl = round(new_sl, 4)

    # --- G. PROFIT TAKING ---

    if r_multiple >= config.PARTIAL_BOOK_R and not partial_taken:
        be_sl = max(new_sl, entry) if is_long else min(new_sl, entry)
        return _validate_output({
            "action": "PARTIAL_EXIT", "reason": f"Target Hit (+{r_multiple:.2f}R)",
            "new_sl": be_sl, "set_partial_taken": True, "r_capped": r_capped
        }, is_long, current_sl)

    return _validate_output({
        "action": "HOLD",
        "reason": exit_reason if exit_reason else "Trend Intact",
        "new_sl": new_sl,
        "set_partial_taken": partial_taken, "r_capped": r_capped
    }, is_long, current_sl)

# ==========================================================
# 3. FINAL OUTPUT SANITATION
# ==========================================================

def _validate_output(result, is_long, old_sl):
    """ The Last Line of Defense. """
    # [FIX #3] Schema Guarantee
    result.setdefault("r_capped", False)
    result.setdefault("was_corrected", False)
    
    if result["action"] in ["HOLD", "PARTIAL_EXIT"]:
        new_sl = result["new_sl"]
        
        # 1. Finite Check
        if new_sl is None or not np.isfinite(new_sl):
            raise RuntimeError(f"Engine Fault: New SL is {new_sl}")
            
        # 2. Regression Check (The Ratchet Assertion)
        if is_long and new_sl < old_sl:
             if (old_sl - new_sl) > (abs(old_sl) * 0.0001):
                 raise RuntimeError(f"SL Regression (Long): {old_sl} -> {new_sl}")
             result["new_sl"] = old_sl 
             result["was_corrected"] = True
             
        if not is_long and new_sl > old_sl:
             if (new_sl - old_sl) > (abs(old_sl) * 0.0001):
                 raise RuntimeError(f"SL Regression (Short): {old_sl} -> {new_sl}")
             result["new_sl"] = old_sl
             result["was_corrected"] = True

    if result["action"] == "EXIT":
        if not result["reason"]:
            result["reason"] = "Unknown Exit"
            
    return result

# ==========================================================
# 4. BOT INTERFACE (The Engine Wrapper)
# ==========================================================

class LogicEngine:
    @staticmethod
    def get_atr(df):
        return ta.atr(df["High"], df["Low"], df["Close"], config.ATR_PERIOD).iloc[-1]
        
    @staticmethod
    def _calculate_atr(df):
        return ta.atr(df["High"], df["Low"], df["Close"], config.ATR_PERIOD).iloc[-1]

    @staticmethod
    def check_exit_conditions(df, pos_state, risk_mult=1.0):
        # Wraps the core engine logic to provide the bot with a standardized dictionary response
        return evaluate_exit(df, pos_state, risk_mult=risk_mult)
