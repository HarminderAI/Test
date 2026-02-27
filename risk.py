import numpy as np
import logging
import config

# Setup module-level logger with safe NullHandler injection
logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
    logger.addHandler(logging.NullHandler())

class PortfolioGovernor:
    """
    The Risk Manager.
    
    Responsibility:
    1. Calculate 'Policy Heat' (Total Capped Risk) across all active positions.
    2. Provide a 'governor_multiplier' to tighten stops when risk is high.
    
    Philosophy:
    - Risk Basis: ENTRY vs Current SL (Accounting Risk).
    - Winners do not subsidize losers. Locked profit counts as 0 Risk.
    - Fail-Closed: Fatal data corruption triggers immediate MAX RISK.
    - Policy Cap: Limits Single-Position Risk impact.
    - Invariant Checks: Enforces strict SL progression with micro-tolerance.
    
    CRITICAL NOTE:
    The heat calculated here is the SUM of capped position risks. The total itself is 
    INTENTIONALLY UNCAPPED to ensure telemetry captures the full extent of overleverage.
    Do NOT use this metric for Margin Calls or Value-at-Risk (VaR).
    """
    
    def __init__(self):
        self.max_heat = getattr(config, "MAX_PORTFOLIO_HEAT", 6.0) 
        self.min_mult = getattr(config, "MIN_RISK_MULT", 0.3)
        self.max_single_r = getattr(config, "MAX_SINGLE_POSITION_R", 5.0)
        
        # [FIX #1] Configurable Invariant Enforcement
        # Default to True (Strict Safety). Set False if running scale-in strategies.
        self.enforce_invariants = getattr(config, "ENFORCE_SL_INVARIANTS", True)
        
        # [FIX #3] Configurable Raw Risk Sanity Check
        # Set False for Futures/Options/Crypto where gaps > 25R are possible valid states.
        self.enforce_raw_sanity = getattr(config, "ENFORCE_RAW_RISK_SANITY", True)

        # Config Validation
        if not np.isfinite(self.max_heat) or self.max_heat <= 0:
            raise ValueError(f"CRITICAL: Invalid MAX_PORTFOLIO_HEAT ({self.max_heat})")
        if not np.isfinite(self.min_mult) or not (0 < self.min_mult <= 1.0):
            raise ValueError(f"CRITICAL: Invalid MIN_RISK_MULT ({self.min_mult})")
        if not np.isfinite(self.max_single_r) or self.max_single_r <= 0:
            raise ValueError(f"CRITICAL: Invalid MAX_SINGLE_POSITION_R ({self.max_single_r})")

    def calculate_policy_heat(self, positions):
        """
        Calculates 'Policy Heat': The sum of Capped Open R-Risk for all positions.
        Returns: Total Open R (e.g., 4.5 means 4.5R is currently at risk).
        """
        # Generator Safety
        try:
            safe_positions = list(positions) if positions is not None else []
        except TypeError:
            logger.error("PortfolioGovernor received non-iterable 'positions' argument")
            return 0.0

        if not safe_positions:
            return 0.0
            
        total_r = 0.0
        skipped_soft = 0
        skipped_fatal = 0
        total_count = len(safe_positions)
        
        for i, pos in enumerate(safe_positions):
            # 1. Schema Validation (Soft Skip)
            if not isinstance(pos, dict) or not all(k in pos for k in ["entry", "initial_sl", "current_sl"]):
                skipped_soft += 1
                continue
            
            ticker = pos.get("ticker", f"IDX_{i}")
                
            try:
                entry = float(pos["entry"])
                init_sl = float(pos["initial_sl"])
                curr_sl = float(pos["current_sl"])
                # Monotonic Baseline SL
                baseline_sl = float(pos.get("baseline_sl", init_sl))
            except (ValueError, TypeError):
                skipped_soft += 1
                continue

            # 2. NaN / Inf Guard (Fatal Skip - Data Corruption)
            if not np.isfinite(entry) or not np.isfinite(init_sl) or not np.isfinite(curr_sl):
                logger.error(f"FATAL: Non-finite values detected in position {ticker}")
                skipped_fatal += 1
                continue

            # 3. Strict Direction & Unit Risk
            if entry > init_sl:
                is_long = True
                unit_risk = entry - init_sl
            elif entry < init_sl:
                is_long = False
                unit_risk = init_sl - entry
            else:
                # Infinite Risk (Entry == SL) - Fatal Logic Error
                logger.error(f"FATAL: Zero-width stop detected in {ticker}")
                skipped_fatal += 1
                continue

            # Strict Baseline Validation
            if is_long and baseline_sl > entry:
                logger.critical(f"FATAL: Long Baseline SL ({baseline_sl}) > Entry ({entry}) for {ticker}. Logic Corrupted.")
                skipped_fatal += 1
                continue
            
            if not is_long and baseline_sl < entry:
                logger.critical(f"FATAL: Short Baseline SL ({baseline_sl}) < Entry ({entry}) for {ticker}. Logic Corrupted.")
                skipped_fatal += 1
                continue

            # Micro-Price Tolerant Invariant Check
            # Use Hybrid Tolerance: 0.01% OR 1e-8 (whichever is larger)
            tol = max(abs(baseline_sl) * 1e-4, 1e-8)
            
            violation = False
            if is_long and curr_sl < (baseline_sl - tol):
                violation = True
            elif not is_long and curr_sl > (baseline_sl + tol):
                violation = True
                
            if violation:
                msg = f"INTEGRITY FAILURE: {ticker} SL ({curr_sl}) worse than Baseline ({baseline_sl})."
                if self.enforce_invariants:
                    logger.critical(f"{msg} Forcing MAX RISK.")
                    return float(self.max_heat)
                else:
                    logger.warning(f"{msg} Allowing due to config override.")
                    # [FIX #1] Baseline Reset
                    # If we allow the violation, we must treat the new SL as the new reality
                    # to prevent "ghost" risk calculations in future checks.
                    # Note: This affects local logic only, as we don't mutate the 'pos' dict upstream.
                    baseline_sl = curr_sl

            # Pathological Geometry Guard
            if abs(entry) < 1e-12 and unit_risk < 1e-9:
                skipped_soft += 1
                continue

            # Valid Micro-Risk Support
            if unit_risk <= 0.0:
                skipped_fatal += 1 # Mathematical impossibility
                continue
                
            # Precision Floor (for division safety only)
            safe_unit_risk = max(unit_risk, 1e-15)
            
            # 4. Calculate Open Risk
            if is_long:
                raw_risk = max(0.0, entry - curr_sl)
            else:
                raw_risk = max(0.0, curr_sl - entry)

            # [FIX #3] Configurable Raw Risk Explosion Guard
            # For gap-prone assets (futures/crypto), >25R might be real.
            # For equity/forex, it's likely a bug.
            if raw_risk > (safe_unit_risk * 25.0):
                 msg = f"FATAL: Raw Risk ({raw_risk:.6f}) > 25x Unit Risk for {ticker}."
                 if self.enforce_raw_sanity:
                     logger.critical(f"{msg} Forcing MAX RISK.")
                     return float(self.max_heat)
                 else:
                     logger.warning(f"{msg} Allowing due to config override.")

            # Calculate R-Multiple
            current_r = raw_risk / safe_unit_risk
                
            if not np.isfinite(current_r):
                current_r = self.max_single_r

            # Upstream Misuse Circuit Breaker (Dollar vs R-Multiple)
            if current_r > 20.0 and self.enforce_raw_sanity:
                logger.critical(f"FATAL: Position {ticker} has {current_r:.1f}R risk. Dollar amounts detected? Forcing MAX RISK.")
                return float(self.max_heat)

            # Symmetric Telemetry Clamp
            current_r = min(current_r, 20.0)
            
            # 5. Conservative Heat Calculation
            effective_risk = current_r 
            
            # Policy Cap (Configurable)
            if effective_risk > self.max_single_r:
                logger.warning(f"Position {ticker} capped at Policy Max ({self.max_single_r}R). Real Risk: {effective_risk:.2f}R")
                effective_risk = self.max_single_r
            
            total_r += effective_risk

        # [FIX #2] Strict Fail-Closed for Fatal Errors
        # If ANY position has fatal data corruption (NaNs, Logic Errors), the entire state is untrustworthy.
        # We do NOT return max(total_r, max_heat) because total_r is comprised of potentially corrupt partial sums.
        # We fail closed to the defined Maximum Emergency state.
        if skipped_fatal > 0:
             logger.critical(f"FATAL: {skipped_fatal} positions have fatal data corruption. Forcing MAX RISK.")
             return float(self.max_heat)

        valid_count = total_count - (skipped_soft + skipped_fatal)
        
        if total_count >= 1:
            corruption_ratio = (skipped_soft + skipped_fatal) / total_count
            
            # Soft Corruption (Missing Keys) -> Threshold 50%
            if corruption_ratio >= 0.5:
                 logger.error(f"CRITICAL: {corruption_ratio:.1%} positions invalid/skipped. Forcing MAX RISK.")
                 return max(float(total_r), float(self.max_heat))
                 
            # Small Sample Size Instability
            if total_count >= 3 and valid_count < 2:
                 logger.error(f"CRITICAL: Only {valid_count} valid positions. Sample too small. Forcing MAX RISK.")
                 return max(float(total_r), float(self.max_heat))

        if skipped_soft > 0:
            logger.warning(f"PortfolioGovernor skipped {skipped_soft}/{total_count} positions (Soft Errors).")

        # Global Sanity Limit (Physical Impossibility Check)
        if total_r > (self.max_heat * 10.0):
            logger.critical(f"PHYSICAL LIMIT WARNING: Total Policy Heat {total_r:.1f}R exceeds 10x Max Heat.")

        return float(total_r)

    def get_risk_multiplier(self, current_heat=None, positions=None):
        # 🚨 THE DUAL-LAYER WIRING: Read both adaptive states
        perf_mult = 1.0
        drift_mult = 1.0
        
        try:
            with sqlite3.connect("paper.db", timeout=5) as conn:
                # 🚨 PATCH: Check if table exists before querying to prevent noisy logs on initial boot
                table_exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='broker_status'").fetchone()
                if table_exists:
                    # 1. Read Performance Controller (Sharpe/Expectancy decay)
                    row_perf = conn.execute("SELECT value FROM broker_status WHERE key='adaptive_risk_mult'").fetchone()
                    if row_perf: perf_mult = float(row_perf[0])
                    
                    # 2. Read Drift Monitor (Distribution/Shape decay)
                    row_drift = conn.execute("SELECT value FROM broker_status WHERE key='drift_risk_mult'").fetchone()
                    if row_drift: drift_mult = float(row_drift[0])
        except Exception as e:
            # 🚨 PATCH: Log the exception subtly for debugging
            logger.debug(f"Could not read dynamic risk multipliers: {e}")
        
        """
        Returns a multiplier (min_mult to 1.0) based on Portfolio Heat.
        """
        if current_heat is None:
            if positions is None:
                raise ValueError("Must provide either 'current_heat' or 'positions'")
            current_heat = self.calculate_policy_heat(positions)
            
        # Utilization Logic
        raw_utilization = current_heat / self.max_heat
        
        # Telemetry
        if raw_utilization > 3.0:
            logger.critical(f"EXTREME OVERLEVERAGE: Utilization {raw_utilization:.1%}")
        elif raw_utilization > 1.0:
            logger.warning(f"Portfolio Overloaded: Utilization {raw_utilization:.1%}")
            
        # Cap utilization for calculation stability
        calc_utilization = min(raw_utilization, 3.0) 
        
        # Two-Stage Tightening Curve
        if calc_utilization <= 0.5:
            mult = 1.0
        elif calc_utilization <= 1.0:
            # Stage 1: Linear Tightening (1.0 -> 0.5)
            slope = (0.5 - 1.0) / (1.0 - 0.5) 
            mult = 1.0 + slope * (calc_utilization - 0.5)
        else:
            # Stage 2: Super-Emergency (0.5 -> min_mult)
            slope = (self.min_mult - 0.5) / (2.0 - 1.0)
            mult = 0.5 + slope * (calc_utilization - 1.0)
            
        mult = max(mult, self.min_mult)
        
        return float(mult)
          
