import numpy as np
import logging
import sqlite3
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
        
        self.enforce_invariants = getattr(config, "ENFORCE_SL_INVARIANTS", True)
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
            if not isinstance(pos, dict) or not all(k in pos for k in ["entry", "initial_sl", "current_sl"]):
                skipped_soft += 1
                continue
            
            ticker = pos.get("ticker", f"IDX_{i}")
                
            try:
                entry = float(pos["entry"])
                init_sl = float(pos["initial_sl"])
                curr_sl = float(pos["current_sl"])
                baseline_sl = float(pos.get("baseline_sl", init_sl))
            except (ValueError, TypeError):
                skipped_soft += 1
                continue

            if not np.isfinite(entry) or not np.isfinite(init_sl) or not np.isfinite(curr_sl):
                logger.error(f"FATAL: Non-finite values detected in position {ticker}")
                skipped_fatal += 1
                continue

            if entry > init_sl:
                is_long = True
                unit_risk = entry - init_sl
            elif entry < init_sl:
                is_long = False
                unit_risk = init_sl - entry
            else:
                logger.error(f"FATAL: Zero-width stop detected in {ticker}")
                skipped_fatal += 1
                continue

            if is_long and baseline_sl > entry:
                logger.critical(f"FATAL: Long Baseline SL ({baseline_sl}) > Entry ({entry}) for {ticker}. Logic Corrupted.")
                skipped_fatal += 1
                continue
            
            if not is_long and baseline_sl < entry:
                logger.critical(f"FATAL: Short Baseline SL ({baseline_sl}) < Entry ({entry}) for {ticker}. Logic Corrupted.")
                skipped_fatal += 1
                continue

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
                    baseline_sl = curr_sl

            if abs(entry) < 1e-12 and unit_risk < 1e-9:
                skipped_soft += 1
                continue

            if unit_risk <= 0.0:
                skipped_fatal += 1 
                continue
                
            safe_unit_risk = max(unit_risk, 1e-15)
            
            if is_long:
                raw_risk = max(0.0, entry - curr_sl)
            else:
                raw_risk = max(0.0, curr_sl - entry)

            if raw_risk > (safe_unit_risk * 25.0):
                 msg = f"FATAL: Raw Risk ({raw_risk:.6f}) > 25x Unit Risk for {ticker}."
                 if self.enforce_raw_sanity:
                     logger.critical(f"{msg} Forcing MAX RISK.")
                     return float(self.max_heat)
                 else:
                     logger.warning(f"{msg} Allowing due to config override.")

            current_r = raw_risk / safe_unit_risk
                
            if not np.isfinite(current_r):
                current_r = self.max_single_r

            if current_r > 20.0 and self.enforce_raw_sanity:
                logger.critical(f"FATAL: Position {ticker} has {current_r:.1f}R risk. Dollar amounts detected? Forcing MAX RISK.")
                return float(self.max_heat)

            current_r = min(current_r, 20.0)
            
            effective_risk = current_r 
            
            if effective_risk > self.max_single_r:
                logger.warning(f"Position {ticker} capped at Policy Max ({self.max_single_r}R). Real Risk: {effective_risk:.2f}R")
                effective_risk = self.max_single_r
            
            total_r += effective_risk

        if skipped_fatal > 0:
             logger.critical(f"FATAL: {skipped_fatal} positions have fatal data corruption. Forcing MAX RISK.")
             return float(self.max_heat)

        valid_count = total_count - (skipped_soft + skipped_fatal)
        
        if total_count >= 1:
            corruption_ratio = (skipped_soft + skipped_fatal) / total_count
            
            if corruption_ratio >= 0.5:
                 logger.error(f"CRITICAL: {corruption_ratio:.1%} positions invalid/skipped. Forcing MAX RISK.")
                 return max(float(total_r), float(self.max_heat))
                 
            if total_count >= 3 and valid_count < 2:
                 logger.error(f"CRITICAL: Only {valid_count} valid positions. Sample too small. Forcing MAX RISK.")
                 return max(float(total_r), float(self.max_heat))

        if skipped_soft > 0:
            logger.warning(f"PortfolioGovernor skipped {skipped_soft}/{total_count} positions (Soft Errors).")

        if total_r > (self.max_heat * 10.0):
            logger.critical(f"PHYSICAL LIMIT WARNING: Total Policy Heat {total_r:.1f}R exceeds 10x Max Heat.")

        return float(total_r)

    def get_risk_multiplier(self, current_heat=None, positions=None):
        """
        Returns a multiplier (min_mult to 1.0) based on Portfolio Heat AND the Risk Committee.
        """
        if current_heat is None:
            if positions is None:
                raise ValueError("Must provide either 'current_heat' or 'positions'")
            current_heat = self.calculate_policy_heat(positions)
            
        # 1. Utilization Logic (Mechanical Heat)
        raw_utilization = current_heat / self.max_heat
        
        if raw_utilization > 3.0:
            logger.critical(f"EXTREME OVERLEVERAGE: Utilization {raw_utilization:.1%}")
        elif raw_utilization > 1.0:
            logger.warning(f"Portfolio Overloaded: Utilization {raw_utilization:.1%}")
            
        calc_utilization = min(raw_utilization, 3.0) 
        
        # Two-Stage Tightening Curve
        if calc_utilization <= 0.5:
            mult = 1.0
        elif calc_utilization <= 1.0:
            slope = (0.5 - 1.0) / (1.0 - 0.5) 
            mult = 1.0 + slope * (calc_utilization - 0.5)
        else:
            slope = (self.min_mult - 0.5) / (2.0 - 1.0)
            mult = 0.5 + slope * (calc_utilization - 1.0)
            
        mult = max(mult, self.min_mult)

        # 🚨 THE FINAL WIRING: Read only the Committee's Master Multiplier
        master_mult = 1.0
        
        try:
            with sqlite3.connect("paper.db", timeout=5) as conn:
                # Safe check for fresh boots
                table_exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='broker_status'").fetchone()
                if table_exists:
                    row = conn.execute("SELECT value FROM broker_status WHERE key='master_risk_mult'").fetchone()
                    if row: master_mult = float(row[0])
        except Exception as e:
            logger.debug(f"Could not read Risk Committee override: {e}")

        # Apply the Committee's penalty to the base mechanical heat multiplier
        final_mult = mult * master_mult
        final_mult = max(self.min_mult, min(1.0, final_mult)) # Clamp to physical bounds
        
        if master_mult < 1.0:
            logger.info(f"Risk Committee capped capital at {master_mult:.2f}x. Final Mechanical Override: {final_mult:.2f}x")
            
        return float(final_mult)
