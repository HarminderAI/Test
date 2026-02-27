"""
APEX EPOCH QUANT ENGINE
================================================================================
DOMAIN ARCHITECTURE AXIOMS:
1. Risk & Tail Metrics (Sharpe, VaR, Beta): Strictly evaluated in the Simple/Arithmetic 
   return domain to preserve cross-sectional aggregation and cash-flow risk premiums.
2. Execution & Robustness (Kelly, Monte Carlo): Strictly evaluated in the Log/Geometric 
   return domain to accurately model continuous compounding, survival, and gap-risk.
3. Inference: DSR/PSR map dependent (HAC) variance into an IID EVT framework. 
   Kelly utilizes an extreme survival-first penalty stack.
4. Resampling: Block Bootstrapping preserves autocorrelation but sacrifices block-boundary 
   continuity. EVT Peaks-Over-Threshold (POT) and DFA are recommended for ultimate tail/memory inference.
================================================================================
"""

import sqlite3
import logging
import math
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from notifier import QuantNotifier
from typing import Tuple, Dict, Optional, Union, Any, get_origin, get_args

# --- CAPABILITY TRACKING ---
try:
    import seaborn as sns 
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import statsmodels.api as sm
    import statsmodels.tsa.stattools as stattools
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# --- CONFIGURATION & ISOLATED LOGGING ---
logger = logging.getLogger("QuantAudit")
logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

DEFAULT_MIN_PERIODS = 126 
Z_SCORE_95 = -1.6448536269514722
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_BATCH_SIZE = 250
MAX_MC_BATCH_SIZE = 500
MAX_MATRIX_ELEMENTS = 10_000_000
MAX_KURTOSIS_CLIP = 10.0 


# 🏛️ ARCHITECTURE 1: Immutable Domain Dataclasses (Strict SRP)
@dataclass(frozen=True)
class MetaMetrics:
    schema_version: str = "14.0.0" 
    timestamp_utc: str = ""
    is_out_of_sample: bool = False
    alpha_inference_type: str = "Bypassed"  
    mc_inference_type: str = "Bypassed"
    has_scipy: bool = HAS_SCIPY
    has_statsmodels: bool = HAS_STATSMODELS
    seed_used: int = 42
    n_samples: int = 0
    dsr_trials: int = 1 
    mc_block_size_used: int = 0
    ruin_threshold_pct: float = np.nan 
    compute_time_sec: float = 0.0

@dataclass(frozen=True)
class ReturnMetrics:
    total_return_pct: float = np.nan
    cagr_pct: float = np.nan
    ann_arith_mean_pct: float = np.nan 
    volatility_drag_pct: float = np.nan 
    net_profit: float = np.nan
    max_drawdown_pct: float = np.nan
    max_dd_duration_bars: int = 0         
    time_under_water_pct: float = np.nan  
    total_drawdown_periods: int = 0       
    drawdown_div_score: float = np.nan    
    recovery_factor: float = np.nan
    ulcer_index: float = np.nan

@dataclass(frozen=True)
class DistributionMetrics:
    gross_volatility_pct: float = np.nan 
    excess_volatility_pct: float = np.nan
    skewness: float = np.nan
    kurtosis: float = np.nan 
    autocorr_lag1: float = np.nan
    hurst_exponent: float = np.nan 
    effective_sample_size: float = np.nan 
    positive_months_pct: float = np.nan

@dataclass(frozen=True)
class TailMetrics:
    downside_semivol_pct: float = np.nan
    raw_cvar_95_pct: float = np.nan
    cf_var_95_pct: float = np.nan 
    cdar_95_pct: float = np.nan                  
    worst_5d_loss_pct: float = np.nan
    worst_10d_loss_pct: float = np.nan
    tail_ratio: float = np.nan
    gain_to_pain: float = np.nan
    omega_ratio: float = np.nan
    expected_shortfall_ratio: float = np.nan

@dataclass(frozen=True)
class SharpeMetrics:
    sharpe_ratio: float = np.nan
    skew_adj_sharpe_ratio: float = np.nan
    lo_adj_sharpe_ratio: float = np.nan  
    prob_sharpe_ratio_pct: float = np.nan
    deflated_sharpe_ratio_pct: float = np.nan 
    sortino_ratio: float = np.nan
    calmar_ratio: float = np.nan
    mar_ratio: float = np.nan
    rolling_sharpe_std: float = np.nan    
    rolling_sharpe_under_zero_pct: float = np.nan 
    rolling_sharpe_worst_1y: float = np.nan        

@dataclass(frozen=True)
class ExecutionMetrics:
    total_closed_trades: int = 0
    win_rate_pct: float = np.nan
    profit_factor: float = np.nan
    avg_win: float = np.nan
    avg_loss: float = np.nan
    expectancy: float = np.nan
    max_cons_losses: int = 0
    kelly_fraction_pct: float = np.nan
    bayesian_kelly_pct: float = np.nan    
    half_kelly_pct: float = np.nan
    avg_gross_exposure_pct: float = np.nan
    annual_turnover_times: float = np.nan  

@dataclass(frozen=True)
class AlphaMetrics:
    beta: float = np.nan
    downside_beta: float = np.nan         
    beta_drift_volatility: float = np.nan 
    jensens_alpha_pct: float = np.nan
    jensens_alpha_pvalue: float = np.nan
    bootstrap_alpha_pvalue: float = np.nan 
    bootstrap_alpha_ci_lower: float = np.nan 
    bootstrap_alpha_ci_upper: float = np.nan 
    rsquared: float = np.nan
    information_ratio: float = np.nan
    tracking_error: float = np.nan

@dataclass(frozen=True)
class RobustnessMetrics:
    sharpe_95_ci_lower: float = np.nan
    sharpe_95_ci_upper: float = np.nan
    mc_median_cagr_pct: float = np.nan
    mc_median_max_dd_pct: float = np.nan
    mc_ret_p05: float = np.nan
    mc_ret_p95: float = np.nan
    prob_of_drawdown_breach_pct: float = np.nan 

@dataclass(frozen=True)
class QuantReport:
    meta: MetaMetrics
    returns: ReturnMetrics
    distribution: DistributionMetrics
    tail: TailMetrics
    sharpe: SharpeMetrics
    execution: ExecutionMetrics
    alpha: AlphaMetrics
    robustness: RobustnessMetrics

    def __post_init__(self):
        pass

    def validate(self, min_periods: int, strict: bool = False) -> None:
        if self.meta.n_samples > min_periods and np.isnan(self.sharpe.sharpe_ratio):
            logger.warning("[Validation] Sharpe Ratio evaluated to NaN. Inspect input return array.")
            
        if strict:
            if self.returns.max_drawdown_pct < -1e-8:
                raise ValueError(f"Strict Validation Failed: Max drawdown is negative ({self.returns.max_drawdown_pct}%). Must be positive magnitude.")
            if pd.notna(self.sharpe.sharpe_ratio) and abs(self.sharpe.sharpe_ratio) > 20:
                raise ValueError(f"Strict Validation Failed: Sharpe ratio anomaly (|SR| > 20): {self.sharpe.sharpe_ratio}")
            
            for metric_group in [self.returns, self.distribution, self.tail, self.sharpe, self.alpha]:
                for field_name, value in asdict(metric_group).items():
                    if isinstance(value, float) and np.isinf(value):
                        raise ValueError(f"Strict Validation Failed: Metric '{field_name}' evaluated to infinity.")

    def to_dict(self) -> dict:
        return asdict(self)


# 🧮 ARCHITECTURE 2: Central Sanitization & Construction Layer
class DataSanitizer:
    @staticmethod
    def clean_series(s: pd.Series) -> pd.Series:
        return s.replace([np.inf, -np.inf], np.nan).dropna()

def default_for_field(f) -> Any:
    t = f.type
    origin = get_origin(t)
    if origin is Union:
        args = get_args(t)
        if float in args: return np.nan
        if int in args: return 0
    if t is float or getattr(t, '__name__', str(t)) == 'float': return np.nan
    if t is int or getattr(t, '__name__', str(t)) == 'int': return 0
    if t is str or getattr(t, '__name__', str(t)) == 'str': return ""
    return None

def safe_build(cls: type, data: Dict[str, Any], strict: bool = False) -> Any:
    expected_keys = {f.name for f in fields(cls)}
    if strict:
        missing = expected_keys - set(data.keys())
        if missing: 
            raise ValueError(f"Strict Construction Error: Missing fields for {cls.__name__}: {missing}")
        extra = set(data.keys()) - expected_keys
        if extra:
            raise ValueError(f"Strict Construction Error: Unexpected fields for {cls.__name__}: {extra}")
            
    kwargs = {}
    for f in fields(cls):
        kwargs[f.name] = data.get(f.name, default_for_field(f))
    return cls(**kwargs)

class DataLoader:
    @staticmethod
    def fetch(db_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not db_path.exists(): raise FileNotFoundError(f"Missing DB: {db_path}")
        uri_path = f"file:{db_path}?mode=ro"
        trades, equity = pd.DataFrame(), pd.DataFrame()
        
        try:
            with sqlite3.connect(uri_path, uri=True) as conn:
                try:
                    trades = pd.read_sql("SELECT * FROM trades", conn)
                    if not trades.empty:
                        trades['date'] = pd.to_datetime(trades['date'], errors='coerce')
                        trades['realized_pnl'] = pd.to_numeric(trades['realized_pnl'], errors='coerce')
                        if 'trade_value' in trades.columns:
                            trades['trade_value'] = pd.to_numeric(trades['trade_value'], errors='coerce')
                        if 'gross_exposure' in trades.columns:
                            trades['gross_exposure'] = pd.to_numeric(trades['gross_exposure'], errors='coerce')
                        trades = trades.dropna(subset=['date', 'realized_pnl'])
                except (sqlite3.Error, ValueError) as e:
                    logger.warning(f"Error reading trades: {e}.")
                    
                try:
                    equity = pd.read_sql("SELECT * FROM equity_log", conn)
                    if not equity.empty:
                        equity['date'] = pd.to_datetime(equity['date'], errors='coerce')
                        equity['total_equity'] = pd.to_numeric(equity['total_equity'], errors='coerce')
                        equity = equity.dropna(subset=['date', 'total_equity'])
                        
                        if not equity['date'].is_monotonic_increasing:
                            logger.info("[Data] Non-monotonic dates detected. Enforcing sort order.")
                            equity = equity.sort_values('date')
                        equity = equity.drop_duplicates(subset=['date']).reset_index(drop=True)
                        
                        returns = equity['total_equity'].pct_change()
                        if (returns < -0.99).any():
                            logger.warning("[Data] Catastrophic loss (>99% drop) detected in equity curve.")
                            
                        if (equity['total_equity'] <= 0).any():
                            raise ValueError("[Data] Strategy bankruptcy (equity <= 0) detected. Execution aborted.")
                            
                except (sqlite3.Error, ValueError) as e:
                    logger.error(f"Error reading equity: {e}")
                    equity = pd.DataFrame()
        except sqlite3.OperationalError as e:
             logger.error(f"Failed to DB connection: {e}")
             
        return trades, equity


# 🧮 ARCHITECTURE 3: Pure Functional Mathematical Sub-Modules
class ReturnModel:
    @staticmethod
    def apply(safe_rets: pd.Series, eq: pd.Series, daily_dd: pd.Series, hwm: pd.Series, trading_days: int, min_years_cagr: float) -> ReturnMetrics:
        start_eq, end_eq = float(eq.iloc[0]), float(eq.iloc[-1])
        total_return_pct = float(((end_eq - start_eq) / start_eq) * 100) if start_eq > 1e-8 else np.nan
        net_profit = float(end_eq - start_eq)
        
        years = max(float(len(eq) / trading_days), 1e-9)
        
        cagr_pct = np.nan
        if years >= min_years_cagr and start_eq > 1e-8 and end_eq > 0: 
            cagr_pct = float((((end_eq / start_eq) ** (1/years)) - 1) * 100)
            
        ann_arith_mean_pct = float(safe_rets.mean() * trading_days * 100)
        
        volatility_drag_pct = np.nan
        if pd.notna(cagr_pct) and pd.notna(ann_arith_mean_pct):
            volatility_drag_pct = float(ann_arith_mean_pct - cagr_pct)

        max_drawdown_pct = abs(float(min(daily_dd.min() * 100, 0.0)))
        max_dd_abs = max_drawdown_pct
        recovery_factor = float(total_return_pct / max_dd_abs) if max_dd_abs > 1e-8 else np.nan
        ulcer_index = float(np.sqrt((daily_dd**2).mean()) * 100)

        underwater = eq < hwm
        time_under_water_pct = float((underwater.sum() / len(underwater)) * 100) if len(underwater) > 0 else 0.0
        
        is_uw = underwater.astype(int)
        dd_groups = (is_uw != is_uw.shift()).cumsum()
        
        total_drawdown_periods = 0
        drawdown_div_score = np.nan
        max_dd_duration_bars = 0
        
        if is_uw.any():
            dd_durations = is_uw.groupby(dd_groups).sum()
            dd_durations = dd_durations[dd_durations > 0]
            max_dd_duration_bars = int(dd_durations.max()) if not dd_durations.empty else 0
            
            underwater_groups = dd_groups[underwater]
            dd_mags = (eq[underwater] - hwm[underwater]) / np.where(hwm[underwater] > 1e-8, hwm[underwater], np.nan)
            dd_mags_min = dd_mags.groupby(underwater_groups).min().abs()
            dd_mags_filtered = dd_mags_min[dd_mags_min > 1e-6]
            
            total_drawdown_periods = int(len(dd_mags_filtered))
            if not dd_mags_filtered.empty and dd_mags_filtered.sum() > 1e-8:
                dd_probs = dd_mags_filtered / dd_mags_filtered.sum()
                sum_sq = float((dd_probs**2).sum())
                n_probs = len(dd_probs)
                drawdown_div_score = float((1.0 - sum_sq) * (n_probs / (n_probs - 1))) if n_probs > 1 else 0.0
            
        return ReturnMetrics(
            total_return_pct=total_return_pct, cagr_pct=cagr_pct, ann_arith_mean_pct=ann_arith_mean_pct, 
            volatility_drag_pct=volatility_drag_pct, net_profit=net_profit,
            max_drawdown_pct=max_drawdown_pct, max_dd_duration_bars=max_dd_duration_bars,
            time_under_water_pct=time_under_water_pct, total_drawdown_periods=total_drawdown_periods,
            drawdown_div_score=drawdown_div_score, recovery_factor=recovery_factor, ulcer_index=ulcer_index
        )


class DistributionModel:
    @staticmethod
    def _compute_hurst(rets: pd.Series) -> float:
        if len(rets) < 500: return np.nan
        log_rets = np.log1p(np.clip(rets.values, -0.999999, None))
        log_rets = log_rets - log_rets.mean() 
        prices = np.cumsum(log_rets)
        lags = range(2, min(20, len(prices) // 4))
        try:
            tau = np.array([np.std(prices[lag:] - prices[:-lag]) for lag in lags], dtype=float)
            valid = tau > 1e-12
            if valid.sum() < 2: return np.nan
            poly = np.polyfit(np.log(np.array(lags)[valid]), np.log(tau[valid]), 1)
            return float(poly[0])
        except Exception as e: 
            logger.warning(f"[Distribution] Hurst R/S proxy failed: {e}")
            return np.nan

    @staticmethod
    def apply(rets: pd.Series, excess: pd.Series, std_excess: float, trading_days: int) -> Tuple[Dict[str, float], float, float]:
        dist = {f.name: np.nan for f in fields(DistributionMetrics)}
        dist['gross_volatility_pct'] = float(rets.std(ddof=1) * np.sqrt(trading_days) * 100)
        dist['excess_volatility_pct'] = float(std_excess * np.sqrt(trading_days) * 100)
        dist['skewness'] = float(rets.skew())
        dist['kurtosis'] = float(rets.kurtosis())  
        dist['hurst_exponent'] = DistributionModel._compute_hurst(rets)
        
        q_sharpe = max(1, int(4 * ((len(rets) / 100) ** (2/9))))
        if HAS_STATSMODELS:
            acf_vals = stattools.acf(excess.values, nlags=q_sharpe, fft=True, adjusted=True)
            rhos = [acf_vals[k] * (1 - k/(q_sharpe+1)) for k in range(1, q_sharpe + 1)]
            dist['autocorr_lag1'] = float(acf_vals[1]) if len(acf_vals) > 1 else 0.0
        else:
            vals_np = excess.values
            rhos = []
            for k in range(1, q_sharpe + 1):
                if len(vals_np) > k:
                    v1, v2 = vals_np[:-k], vals_np[k:]
                    corr = np.corrcoef(v1, v2)[0, 1] if np.std(v1) > 1e-12 and np.std(v2) > 1e-12 else 0.0
                    rhos.append(corr * (1 - k/(q_sharpe+1)))
                    if k == 1: dist['autocorr_lag1'] = float(corr)
                    
        penalty_sq = max(1.0, 1 + 2 * np.sum(rhos))
        # FIX: Mathematical ESS evaluation entirely dictating downward variance and confidence thresholds
        n_eff = len(rets) / penalty_sq
        if n_eff < 30.0:
            logger.warning(f"[Statistical] Effective Sample Size critically low ({n_eff:.1f}). Inference unreliable.")
            
        dist['effective_sample_size'] = float(n_eff)
        
        return dist, penalty_sq, n_eff


class TailRiskModel:
    @staticmethod
    def apply(safe_rets: pd.Series, excess: pd.Series, daily_dd: pd.Series, min_periods: int, trading_days: int, dist: Dict[str, float]) -> Dict[str, float]:
        tail = {f.name: np.nan for f in fields(TailMetrics)}
        downside_abs = np.minimum(0, safe_rets)
        tail['downside_semivol_pct'] = float(np.sqrt((downside_abs**2).mean()) * np.sqrt(trading_days) * 100)
        
        tail_len_cvar = max(5, int(np.ceil(0.05 * len(safe_rets))))
        tail_rets = safe_rets.nsmallest(tail_len_cvar)
        tail['raw_cvar_95_pct'] = abs(float(tail_rets.mean() * 100))
        
        g_vol = dist.get('gross_volatility_pct', np.nan)
        if pd.notna(g_vol) and g_vol > 1e-8:
            tail['expected_shortfall_ratio'] = float(tail['raw_cvar_95_pct'] / g_vol)
            
        if len(safe_rets) < 200:
            logger.debug("[TailRisk] Sample N < 200: Cornish-Fisher VaR bypassed to prevent severe polynomial instability.")
        else:
            z = Z_SCORE_95
            s = dist.get('skewness', 0.0)
            k_ex = dist.get('kurtosis', 0.0)
            
            if abs(s) > 3.0 or k_ex > 10.0:
                logger.debug(f"[TailRisk] Extreme structural moments (Skew: {s:.2f}, Kurt: {k_ex:.2f}) exceed expansion limits. CF-VaR bypassed.")
                tail['cf_var_95_pct'] = np.nan
            else:
                k_clip = np.clip(k_ex, -2.0, MAX_KURTOSIS_CLIP)
                s_clip = np.clip(s, -5.0, 5.0)
                z_cf = z + (1/6)*(z**2 - 1)*s_clip + (1/24)*(z**3 - 3*z)*k_clip - (1/36)*(2*z**3 - 5*z)*(s_clip**2)
                z_cf = np.clip(z_cf, -5.0, 5.0) 
                cf_var_daily = safe_rets.mean() + z_cf * safe_rets.std(ddof=1)
                tail['cf_var_95_pct'] = abs(float(cf_var_daily * 100))
        
        log_rets = np.log1p(safe_rets)
        tail['worst_5d_loss_pct'] = float(np.expm1(log_rets.rolling(5, min_periods=5).sum()).min() * 100)
        tail['worst_10d_loss_pct'] = float(np.expm1(log_rets.rolling(10, min_periods=10).sum()).min() * 100)
        
        pos_sum = float(safe_rets[safe_rets > 0].sum())
        neg_sum = abs(float(safe_rets[safe_rets < 0].sum()))
        tail['gain_to_pain'] = float(pos_sum / neg_sum) if neg_sum > 1e-8 else np.nan
        
        excess_gains = float(excess[excess > 0].sum())
        excess_losses = abs(float(excess[excess < 0].sum()))
        tail['omega_ratio'] = float(excess_gains / excess_losses) if excess_losses > 1e-8 else np.nan
        
        tail_len_tr = max(5, int(0.01 * len(safe_rets))) 
        neg_tail_rets = safe_rets.nsmallest(tail_len_tr)
        pos_tail_rets = safe_rets.nlargest(tail_len_tr)
        
        if len(neg_tail_rets) >= tail_len_tr and len(pos_tail_rets) >= tail_len_tr:
            mean_neg_tail = abs(float(neg_tail_rets.mean()))
            tail['tail_ratio'] = float(pos_tail_rets.mean() / mean_neg_tail) if mean_neg_tail > 1e-8 else np.nan
            
        if len(daily_dd) >= min_periods:
            tail_len_cdar = max(5, int(0.05 * len(daily_dd)))
            tail_dd_cond = daily_dd.nsmallest(tail_len_cdar)
            if len(tail_dd_cond) > 0:
                tail['cdar_95_pct'] = abs(float(tail_dd_cond.mean() * 100))
                
        return tail


class SharpeModel:
    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def calc_dsr(n_trials: int, variance: float, sr_daily: float) -> Tuple[float, float]:
        variance = max(variance, 1e-12)
        if n_trials <= 1: 
            return 0.0, float(SharpeModel._norm_cdf(sr_daily / math.sqrt(variance)) * 100)
            
        if HAS_SCIPY:
            import scipy.stats as stats
            euler_mascheroni = 0.5772156649
            z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
            z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
            sr_0 = math.sqrt(variance) * ((1 - euler_mascheroni) * z1 + euler_mascheroni * z2)
        else:
            if n_trials < 50:
                logger.debug("[Sharpe] DSR asymptotic approximation unreliable for n_trials < 50 without SciPy.")
            z_approx = math.sqrt(2 * math.log(n_trials)) - (math.log(math.log(n_trials)) + math.log(4 * math.pi)) / (2 * math.sqrt(2 * math.log(n_trials)))
            sr_0 = math.sqrt(variance) * z_approx 
            
        dsr_t_stat = (sr_daily - sr_0) / math.sqrt(variance)
        return sr_0, float(SharpeModel._norm_cdf(dsr_t_stat) * 100)

    @staticmethod
    def apply(dist: Dict[str, float], ret_m: ReturnMetrics, mean_excess: float, std_excess: float, excess: pd.Series, 
              n_eff: float, trading_days: int, n_trials: int) -> Dict[str, float]:
        
        sharpe = {f.name: np.nan for f in fields(SharpeMetrics)}
        if std_excess < 1e-10:
            logger.warning("[Sharpe] Excess return variance critically low. Risk-adjusted metrics unstable.")
            return sharpe
            
        downside_target = np.minimum(0, excess)
        downside_rms = np.sqrt((downside_target**2).mean())
        if downside_rms > 1e-8: sharpe['sortino_ratio'] = float((mean_excess / downside_rms) * np.sqrt(trading_days))

        max_dd_abs = abs(ret_m.max_drawdown_pct) / 100
        if pd.notna(ret_m.cagr_pct) and max_dd_abs > 1e-8:
            sharpe['calmar_ratio'] = sharpe['mar_ratio'] = float(ret_m.cagr_pct / max_dd_abs)
            
        sr_daily = mean_excess / std_excess
        sr = float(sr_daily * np.sqrt(trading_days))
        sharpe['sharpe_ratio'] = sr
        
        k_ex = dist.get('kurtosis', 0.0)
        s = dist.get('skewness', 0.0)
        sr_adj_daily = sr_daily * (1 + (s / 6) * sr_daily - (k_ex / 24) * (sr_daily ** 2))
        sharpe['skew_adj_sharpe_ratio'] = float(sr_adj_daily * np.sqrt(trading_days))
        
        penalty = math.sqrt(len(excess) / n_eff) if n_eff > 1e-8 else 1.0
        sharpe['lo_adj_sharpe_ratio'] = float(sr / penalty)
        
        if n_eff > 30 and pd.notna(s) and pd.notna(k_ex):
            if k_ex > MAX_KURTOSIS_CLIP:
                logger.debug(f"[Sharpe] High kurtosis ({k_ex:.2f}) detected. Analytical Sharpe variance approximation degrades.")
                
            df_denom = max(1.0, n_eff - 1.0)
            k_safe = min(k_ex, MAX_KURTOSIS_CLIP)
            
            # FIX: Mathematical purity strictly mapping IID EVT logic scaled solely by effective degrees of freedom
            sr_var_hac = (1 - s * sr_daily + ((k_safe + 2) / 4) * (sr_daily**2)) / df_denom
            sr_var_hac = max(sr_var_hac, 1e-12)
            
            t_stat = sr_daily / math.sqrt(sr_var_hac)
            sharpe['prob_sharpe_ratio_pct'] = float(SharpeModel._norm_cdf(t_stat) * 100)
            
            variance = max(sr_var_hac, 1e-12)
            _, sharpe['deflated_sharpe_ratio_pct'] = SharpeModel.calc_dsr(n_trials, variance, sr_daily)
                    
        return sharpe


class RollingStabilityModel:
    @staticmethod
    def apply(excess: pd.Series, trading_days: int) -> Dict[str, float]:
        roll = {}
        roll_win = int(max(30, min(trading_days // 2, len(excess) // 3)))
        roll_win = min(roll_win, len(excess))
        
        if len(excess) >= roll_win and roll_win >= 30:
            roll_min = max(1, roll_win // 4)
            roll_mean = excess.rolling(roll_win, min_periods=roll_min).mean()
            roll_std = excess.rolling(roll_win, min_periods=roll_min).std(ddof=1)
            
            roll_sharpe = (roll_mean / np.where(roll_std > 1e-12, roll_std, np.nan)) * np.sqrt(trading_days)
            
            roll['rolling_sharpe_std'] = float(roll_sharpe.std(ddof=1))
            roll['rolling_sharpe_under_zero_pct'] = float(np.mean(roll_sharpe < 0) * 100)
            roll['rolling_sharpe_worst_1y'] = float(np.nanmin(roll_sharpe)) if np.isfinite(roll_sharpe).any() else np.nan
            
        return roll


class KellyModel:
    @staticmethod
    def apply(excess: pd.Series, skew: float, kurtosis: float, 
              n_eff: float, prior_years: float, trading_days: int, conservative: bool) -> Dict[str, float]:
        
        kelly = {f.name: np.nan for f in fields(ExecutionMetrics) if 'kelly' in f.name}
        if n_eff < 2: return kelly
        
        log_ex = np.log1p(np.clip(excess.values, -0.999999, None))
        mean_log = float(log_ex.mean())
        var_log = float(log_ex.var(ddof=1))
        
        if var_log < 1e-12: return kelly
        
        if conservative:
            skew_penalty = max(0.0, -skew / 6.0)
            sr_log = mean_log / math.sqrt(var_log)
            finite_sample_penalty = 1.0 + (2.0 / n_eff) + ((sr_log**2) / n_eff)
            est_uncert_penalty = 1.0 + (1.0 / math.sqrt(n_eff))
            
            if kurtosis > 5.0:
                est_uncert_penalty += (kurtosis / 10.0)
                
            adj_var = var_log * (1.0 + skew_penalty + est_uncert_penalty) * finite_sample_penalty
        else:
            adj_var = var_log
            
        raw_kelly = mean_log / max(adj_var, 1e-10)
        clipped_kelly = np.clip(raw_kelly, -1.0, 1.0)
        
        shrinkage = n_eff / (n_eff + (prior_years * trading_days)) if conservative else 1.0
        bayesian_k = clipped_kelly * shrinkage
        clipped_bayesian = np.clip(bayesian_k, -1.0, 1.0)
        
        kelly['kelly_fraction_pct'] = float(clipped_kelly * 100)
        kelly['bayesian_kelly_pct'] = float(clipped_bayesian * 100)
        kelly['half_kelly_pct'] = float((clipped_bayesian / 2.0) * 100)
        return kelly


class TradeModel:
    @staticmethod
    def apply(trades: pd.DataFrame, avg_equity: float, years: float) -> Dict[str, float]:
        exec_m = {f.name: np.nan for f in fields(ExecutionMetrics) if 'kelly' not in f.name}
        exec_m['total_closed_trades'] = 0
        exec_m['max_cons_losses'] = 0
        
        if trades.empty: return exec_m
        
        if 'trade_value' in trades.columns and years > 0 and avg_equity > 0:
            total_turnover = trades['trade_value'].abs().sum()
            exec_m['annual_turnover_times'] = float((total_turnover / avg_equity) / years)
            
        if 'gross_exposure' in trades.columns:
            exec_m['avg_gross_exposure_pct'] = float(trades['gross_exposure'].mean() * 100)

        if 'realized_pnl' not in trades.columns: return exec_m

        sells = trades[trades['realized_pnl'].notna()].copy()
        if sells.empty: return exec_m
        
        sells = sells.sort_values('date').reset_index(drop=True)
        wins, losses = sells[sells['realized_pnl'] > 0], sells[sells['realized_pnl'] < 0]
        breakevens = sells[sells['realized_pnl'] == 0]
        active_trades = len(wins) + len(losses) + len(breakevens)
        
        win_rate = len(wins) / active_trades if active_trades > 0 else 0.0
        exec_m['win_rate_pct'] = float(win_rate * 100)
        exec_m['total_closed_trades'] = int(len(sells))
        
        win_sum = float(wins['realized_pnl'].sum())
        loss_sum = abs(float(losses['realized_pnl'].sum()))
        exec_m['avg_win'] = float(wins['realized_pnl'].mean()) if not wins.empty else 0.0
        exec_m['avg_loss'] = abs(float(losses['realized_pnl'].mean())) if not losses.empty else 0.0
        exec_m['expectancy'] = float((win_rate * exec_m['avg_win']) - ((1 - win_rate) * exec_m['avg_loss']))

        is_loss = sells['realized_pnl'] < 0
        groups = (is_loss != is_loss.shift(fill_value=False)).cumsum()
        exec_m['max_cons_losses'] = int(is_loss.groupby(groups).sum().max())
        exec_m['profit_factor'] = float(win_sum / loss_sum) if loss_sum > 1e-8 else np.nan
        
        return exec_m


class AlphaModel:
    @staticmethod
    def apply(rets: pd.Series, benchmark_df: Optional[pd.DataFrame], 
              rf_series: pd.Series, trading_days: int, rng: np.random.Generator, min_periods: int) -> Tuple[Dict[str, float], str]:
        
        alpha_dict = {f.name: np.nan for f in fields(AlphaMetrics)}
        inference_type = "Bypassed"
        
        if benchmark_df is None or benchmark_df.empty or len(rets) < min_periods: 
            return alpha_dict, inference_type
            
        if 'close' not in benchmark_df.columns:
            logger.warning("[Alpha] Benchmark data missing 'close' column. Bypassing execution.")
            return alpha_dict, "Bypassed (Missing Close Col)"
        
        bm = benchmark_df.copy()
        if 'date' in bm.columns:
            bm['date'] = pd.to_datetime(bm['date'])
            bm = bm.set_index('date')
            
        bm_rets = bm['close'].pct_change().dropna()
        if bm_rets.index.tz is not None: bm_rets.index = bm_rets.index.tz_localize(None)
        
        df = pd.DataFrame({'strat': rets, 'bm': bm_rets, 'rf': rf_series}).dropna()
        if len(df) < min_periods: return alpha_dict, "Bypassed"

        overlap_ratio = len(df) / len(rets) if len(rets) > 0 else 0.0
        if overlap_ratio < 0.80:
            logger.warning(f"[Alpha] Benchmark alignment critically low ({overlap_ratio*100:.1f}%). Alpha bypassed to prevent beta distortion.")
            return alpha_dict, "Bypassed (Low Overlap)"
        elif overlap_ratio < 0.95:
            logger.debug(f"[Alpha] Benchmark alignment overlap is {overlap_ratio*100:.1f}%. Mild distortion possible.")

        excess_strat = df['strat'] - df['rf']
        excess_bm = df['bm'] - df['rf']
        excess_strat_vals = excess_strat.values
        excess_bm_vals = excess_bm.values
        
        down_mask = excess_bm_vals < 0
        if down_mask.sum() >= max(30, min_periods // 4):
            cov_down = np.cov(excess_strat_vals[down_mask], excess_bm_vals[down_mask])[0, 1]
            var_down = np.var(excess_bm_vals[down_mask], ddof=1)
            alpha_dict['downside_beta'] = float(cov_down / var_down) if var_down > 1e-12 else np.nan

        roll_win = max(1, trading_days // 2)
        roll_min = max(1, trading_days // 4)
        roll_cov = excess_strat.rolling(roll_win, min_periods=roll_min).cov(excess_bm)
        roll_var = excess_bm.rolling(roll_win, min_periods=roll_min).var()
        roll_beta = (roll_cov / roll_var).dropna()
        if not roll_beta.empty:
            alpha_dict['beta_drift_volatility'] = float(roll_beta.std(ddof=1))

        used_statsmodels = False
        if HAS_STATSMODELS:
            try:
                X = sm.add_constant(excess_bm)
                rlm_model = sm.RLM(excess_strat, X, M=sm.robust.norms.HuberT()).fit()
                alpha_daily = rlm_model.params.iloc[0]
                alpha_dict['beta'] = float(rlm_model.params.iloc[1])
                
                maxlags = int(4 * ((len(df) / 100) ** (2/9)))
                maxlags = min(maxlags, trading_days // 4) 
                maxlags = max(1, maxlags)
                
                ols_hac_model = sm.OLS(excess_strat, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
                alpha_dict['jensens_alpha_pvalue'] = float(ols_hac_model.pvalues.iloc[0])
                alpha_dict['rsquared'] = float(ols_hac_model.rsquared)
                alpha_dict['jensens_alpha_pct'] = float(alpha_daily * trading_days * 100)
                
                n_boot = BOOTSTRAP_ITERATIONS
                block_size = max(1, int(1.5 * (len(df) ** (1/3))))
                n_blocks = int(np.ceil(len(df) / block_size))
                offsets = np.arange(block_size)
                
                boot_alphas = np.full(n_boot, np.nan)
                
                batch_size = max(1, min(BOOTSTRAP_BATCH_SIZE, MAX_MATRIX_ELEMENTS // max(1, len(df))))
                
                for i in range(0, n_boot, batch_size):
                    current_batch = min(batch_size, n_boot - i)
                    starts = rng.integers(0, len(df), size=(current_batch, n_blocks))
                    
                    step_indices = (starts[:, :, None] + offsets[None, None, :]).reshape(current_batch, -1)[:, :len(df)] % len(df)
                    
                    strat_boot = excess_strat_vals[step_indices]
                    bm_boot = excess_bm_vals[step_indices]
                    
                    strat_mean_1d = strat_boot.mean(axis=1)
                    bm_mean_1d = bm_boot.mean(axis=1)
                    
                    # FIX: Mathematical honesty bounding bootstrap variance strictly to N block-concatenated length
                    df_denom = max(1, len(df) - 1)
                    covs = np.sum((strat_boot - strat_mean_1d[:, None]) * (bm_boot - bm_mean_1d[:, None]), axis=1) / df_denom
                    vars_bm = np.var(bm_boot, axis=1, ddof=1)
                    
                    valid_mask = vars_bm > 1e-12
                    boot_alphas_batch = np.full(current_batch, np.nan)
                    
                    if valid_mask.any():
                        boot_betas = covs[valid_mask] / vars_bm[valid_mask]
                        boot_alphas_batch[valid_mask] = strat_mean_1d[valid_mask] - boot_betas * bm_mean_1d[valid_mask]
                    
                    boot_alphas[i : i + current_batch] = boot_alphas_batch
                
                valid_alphas = boot_alphas[~np.isnan(boot_alphas)]
                
                if len(valid_alphas) < n_boot * 0.5:
                    logger.warning(f"[Alpha] Bootstrap beta inherently unstable: {n_boot - len(valid_alphas)} paths dropped.")
                
                if len(valid_alphas) > 0:
                    count_less = np.sum(valid_alphas <= 0.0)
                    count_greater = np.sum(valid_alphas >= 0.0)
                    valid_boot_n = len(valid_alphas)
                    p_less = (count_less + 1) / (valid_boot_n + 1)
                    p_greater = (count_greater + 1) / (valid_boot_n + 1)
                    alpha_dict['bootstrap_alpha_pvalue'] = float(min(1.0, 2.0 * min(p_less, p_greater)))
                    
                    alpha_dict['bootstrap_alpha_ci_lower'] = float(np.nanpercentile(valid_alphas, 2.5) * trading_days * 100)
                    alpha_dict['bootstrap_alpha_ci_upper'] = float(np.nanpercentile(valid_alphas, 97.5) * trading_days * 100)
                
                used_statsmodels = True
                inference_type = "Robust HAC"

            except (ValueError, np.linalg.LinAlgError) as e:
                logger.debug(f"[Alpha] Regression failed: {e}")

        if not used_statsmodels:
            inference_type = "Fallback (Winsorized OLS)"
            if np.var(excess_bm_vals) > 1e-12:
                x_clip = np.clip(excess_bm_vals, np.percentile(excess_bm_vals, 1), np.percentile(excess_bm_vals, 99))
                y_clip = np.clip(excess_strat_vals, np.percentile(excess_strat_vals, 1), np.percentile(excess_strat_vals, 99))
                
                slope, intercept = np.polyfit(x_clip, y_clip, deg=1)
                alpha_dict['beta'] = float(slope)
                preds = intercept + slope * excess_bm_vals
                ss_res = np.sum((excess_strat_vals - preds) ** 2)
                ss_tot = np.sum((excess_strat_vals - np.mean(excess_strat_vals)) ** 2)
                alpha_dict['rsquared'] = float(1 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0
                alpha_dict['jensens_alpha_pct'] = float(intercept * trading_days * 100)
            else:
                logger.warning("[Alpha] Benchmark variance too low for OLS fallback. Parameters defaulted to NaN.")
                
        active_returns = df['strat'] - df['bm']
        te_daily = float(active_returns.std(ddof=1))
        te_annual = float(te_daily * np.sqrt(trading_days))
        alpha_dict['tracking_error'] = te_annual
        
        mean_active_daily = float(active_returns.mean())
        if te_daily > 1e-8:
            alpha_dict['information_ratio'] = float((mean_active_daily * trading_days) / te_annual)
            
        return alpha_dict, inference_type


class RobustnessModel:
    @staticmethod
    def apply(rets: pd.Series, rf_series: pd.Series, start_equity: float, trading_days: int, 
              n_sims: int, rng: np.random.Generator, ruin_threshold: float, mc_block_size: Optional[int] = None) -> Tuple[Dict[str, float], int, str]:
        
        rob = {f.name: np.nan for f in fields(RobustnessMetrics)}
        vals = np.clip(rets.values, -0.999999, None)
        rf_vals = rf_series.values
        
        if not rets.index.equals(rf_series.index):
            raise ValueError("[Robustness] RF series time-index misalignment in MC sampling. Arrays must be strictly parallel.")
            
        if len(vals) < 100: return rob, 0, "Bypassed"
        
        if mc_block_size is None or mc_block_size < 1:
            block_size = max(1, int(1.5 * (len(vals) ** (1/3))))
        else:
            block_size = int(mc_block_size)
            
        n_blocks = int(np.ceil(len(vals) / block_size))
        
        log_rets = np.log1p(vals)
        offsets = np.arange(block_size)
        years_mc = len(vals) / trading_days
        
        batch_size = max(1, min(MAX_MC_BATCH_SIZE, MAX_MATRIX_ELEMENTS // max(1, len(vals))))
            
        dds_min = np.empty(n_sims)
        sharpes = np.empty(n_sims)
        cagrs = np.full(n_sims, np.nan) 
        
        for i in range(0, n_sims, batch_size):
            current_batch = min(batch_size, n_sims - i)
            starts = rng.integers(0, len(vals), size=(current_batch, n_blocks))
            
            step_indices = (starts[:, :, None] + offsets[None, None, :]).reshape(current_batch, -1)[:, :len(vals)] % len(vals)
            
            log_samples = log_rets[step_indices]
            rf_samples = rf_vals[step_indices]
            
            log_curves = np.log(start_equity) + np.cumsum(log_samples, axis=1)
            curves = np.exp(log_curves)
            hwms = np.maximum.accumulate(curves, axis=1)
            hwms_safe = np.maximum(hwms, 1e-12)
            
            dds = (curves - hwms_safe) / hwms_safe
            dds_min[i : i + current_batch] = dds.min(axis=1)
            
            samples = np.expm1(log_samples)
            excess_samples = samples - rf_samples
            means = excess_samples.mean(axis=1)
            stds = excess_samples.std(axis=1, ddof=1)
            
            batch_sharpes = (means / np.where(stds > 1e-12, stds, np.nan)) * np.sqrt(trading_days)
            sharpes[i : i + current_batch] = batch_sharpes
            
            if years_mc >= 1.0:
                batch_cagrs = ((curves[:, -1] / start_equity) ** (1.0 / years_mc)) - 1
                cagrs[i : i + current_batch] = batch_cagrs
            
        rob['mc_median_max_dd_pct'] = float(np.median(dds_min) * 100)
        rob['prob_of_drawdown_breach_pct'] = float(np.mean(dds_min <= ruin_threshold) * 100)
        
        if np.isfinite(sharpes).any():
            rob['sharpe_95_ci_lower'] = float(np.nanpercentile(sharpes, 2.5))
            rob['sharpe_95_ci_upper'] = float(np.nanpercentile(sharpes, 97.5))
        
        if years_mc >= 1.0 and np.isfinite(cagrs).any():
            rob['mc_median_cagr_pct'] = float(np.nanmedian(cagrs) * 100)
            rob['mc_ret_p05'] = float(np.nanpercentile(cagrs, 5) * 100)
            rob['mc_ret_p95'] = float(np.nanpercentile(cagrs, 95) * 100)
            
        return rob, block_size, "Paired Block Bootstrap"


# ⚙️ ARCHITECTURE 4: The Orchestrator
class QuantEngine:
    def __init__(self, db_path: Optional[str] = None, benchmark_df: Optional[pd.DataFrame] = None, annualized_risk_free_rate: Union[float, pd.Series] = 0.05, 
                 trading_days: int = 252, calendar: str = "business", seed: int = 42, n_trials: int = 1, 
                 mc_sims: int = 2000, mc_block_size: Optional[int] = None, ruin_threshold: float = -0.50,
                 enable_mc: bool = True, enable_alpha: bool = True, min_periods: int = DEFAULT_MIN_PERIODS,
                 kelly_prior_years: float = 1.0, conservative_kelly: bool = True, min_years_cagr: float = 1.0,
                 strict_validation: bool = False, is_out_of_sample: bool = False):
        
        if not isinstance(trading_days, int) or trading_days <= 0:
            raise ValueError(f"trading_days must be a positive integer. Received: {trading_days}")
        if ruin_threshold >= 0:
            raise ValueError(f"ruin_threshold must be strictly negative (e.g., -0.50). Received: {ruin_threshold}")
        if not isinstance(min_periods, int) or min_periods < 2:
            raise ValueError(f"min_periods must be >= 2. Received: {min_periods}")
            
        logger.debug(f"Initialized Quant Engine (Pandas {pd.__version__}, NumPy {np.__version__})")
        
        self.db_path = Path(db_path).resolve() if db_path else None
        self.benchmark_df = benchmark_df
        self.annual_rf = annualized_risk_free_rate 
        
        if isinstance(self.annual_rf, (int, float)):
            if abs(self.annual_rf) > 1.0:
                raise ValueError(f"Scalar RF ({self.annual_rf}) exceeds 1.0. Input must be an annualized decimal (e.g. 0.05).")
            if 0.0 < abs(self.annual_rf) < 0.01 and trading_days >= 250:
                logger.warning(f"Scalar RF ({self.annual_rf}) is < 1%. Verify this is an annualized figure and not a daily rate.")
                
        if isinstance(self.annual_rf, pd.Series):
            if self.annual_rf.abs().max() > 1.0:
                raise ValueError("RF series absolute max exceeds 1.0. Ensure input is strictly annualized and represented as a decimal.")
            if self.annual_rf.abs().max() < 0.01 and self.annual_rf.abs().mean() > 0 and trading_days >= 250:
                logger.warning("RF series max < 1%. Verify input is annualized. Daily rates will be destructively double-compounded.")
            
        self.trading_days = trading_days
        self.calendar = calendar.lower()
        self.n_trials = max(1, n_trials)
        self.min_periods = min_periods
        self.enable_mc = enable_mc
        self.enable_alpha = enable_alpha
        self.mc_sims = max(100, mc_sims)
        self.mc_block_size = mc_block_size
        self.ruin_threshold = ruin_threshold
        self.kelly_prior_years = kelly_prior_years
        self.conservative_kelly = conservative_kelly
        self.min_years_cagr = min_years_cagr
        self.strict_validation = strict_validation
        self.is_out_of_sample = is_out_of_sample
        self.seed = seed
        
        self.equity: pd.DataFrame = pd.DataFrame()
        self.trades: pd.DataFrame = pd.DataFrame()

    def load_data(self) -> None:
        if self.db_path:
            self.trades, self.equity = DataLoader.fetch(self.db_path)

    def _prepare_data(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, float, float]:
        eq = self.equity.copy()
        if len(eq) >= 3:
            inferred = pd.infer_freq(eq['date'].drop_duplicates().sort_values().iloc[:200])
            if inferred and any(inferred.upper().endswith(x) for x in ['T', 'MIN', 'H', 'S']):
                logger.info("[Data] Intraday data detected. Aggregating strictly to daily closing equity.")
                eq['date'] = eq['date'].dt.normalize()
                eq = eq.groupby('date').last().reset_index()

        daily_eq = eq.set_index('date')
        daily_eq = daily_eq[~daily_eq.index.duplicated(keep='last')].sort_index().dropna()
        if len(daily_eq) < 2: 
            return pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series(), 0.0, 0.0
        
        start_equity = float(daily_eq['total_equity'].iloc[0])
        if start_equity <= 1e-8:
            logger.warning("[Data] Degenerate starting equity detected. Bypassing analytics.")
            return pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series(), 0.0, 0.0
            
        daily_returns = DataSanitizer.clean_series(daily_eq['total_equity'].pct_change())
        if daily_returns.index.tz is not None:
            daily_returns.index = daily_returns.index.tz_localize(None)
            
        hwm_daily = daily_eq['total_equity'].cummax()
        daily_dd = (daily_eq['total_equity'] - hwm_daily) / np.where(hwm_daily > 1e-8, hwm_daily, np.nan)
        
        if isinstance(self.annual_rf, pd.Series):
            rf_clean = self.annual_rf.copy()
            if rf_clean.index.tz is not None:
                rf_clean.index = rf_clean.index.tz_localize(None)
            rf_aligned = rf_clean.reindex(daily_returns.index)
            if rf_aligned.isna().mean() > 0.2:
                raise ValueError("[Data] Risk-Free series exceeds 20% NaN baseline. Fatal alignment risk; execution aborted.")
            rf_aligned = rf_aligned.ffill().bfill().fillna(0.0)
            rf_series = (1 + rf_aligned) ** (1 / self.trading_days) - 1
        else:
            daily_rf_scalar = ((1 + self.annual_rf) ** (1 / self.trading_days)) - 1
            rf_series = pd.Series(daily_rf_scalar, index=daily_returns.index)

        avg_equity = daily_eq['total_equity'].mean()
        years = len(daily_eq) / self.trading_days
        excess_returns = daily_returns - rf_series
        
        return daily_returns, excess_returns, daily_eq['total_equity'], daily_dd, hwm_daily, rf_series, avg_equity, years

    def _compute_core_metrics(self, daily_returns: pd.Series, excess: pd.Series, total_equity: pd.Series, daily_dd: pd.Series, 
                              hwm_daily: pd.Series, avg_equity: float, years: float) -> Tuple[Dict, Dict, Dict, Dict, Dict, float]:
        
        logger.info("[Core] Computing Return, Distribution, Tail, Sharpe, and Execution domains...")
        safe_rets = np.clip(daily_returns, -0.999999, None)
        ret_m = ReturnModel.apply(safe_rets, total_equity, daily_dd, hwm_daily, self.trading_days, self.min_years_cagr)
        
        dist_dict, tail_dict, sharpe_dict, exec_dict = {}, {}, {}, {}
        n_eff = float(len(daily_returns))
        
        if len(daily_returns) >= 2:
            mean_excess = float(excess.mean())
            std_excess = float(excess.std(ddof=1))
            
            dist_dict, penalty_sq, n_eff = DistributionModel.apply(daily_returns, excess, std_excess, self.trading_days)
            
            rule = pd.offsets.MonthEnd()
            monthly_returns = daily_returns.groupby(pd.Grouper(freq=rule)).apply(
                lambda x: (1 + x).prod() - 1 if not x.empty else np.nan
            ).dropna()
            dist_dict['positive_months_pct'] = float((monthly_returns > 0).sum() / len(monthly_returns) * 100) if len(monthly_returns) >= 3 else np.nan
            
            exec_dict = TradeModel.apply(self.trades, avg_equity, years)
            kurtosis = dist_dict.get('kurtosis', 0.0)
            
            kelly_dict = KellyModel.apply(
                excess=excess, skew=dist_dict.get('skewness', 0.0), kurtosis=kurtosis,
                n_eff=n_eff, prior_years=self.kelly_prior_years, 
                trading_days=self.trading_days, conservative=self.conservative_kelly
            )
            exec_dict.update(kelly_dict)
            
            if len(daily_returns) >= self.min_periods:
                tail_dict = TailRiskModel.apply(safe_rets, excess, daily_dd, self.min_periods, self.trading_days, dist_dict)
                sharpe_dict = SharpeModel.apply(dist_dict, ret_m, mean_excess, std_excess, excess, n_eff, self.trading_days, self.n_trials)
                roll_dict = RollingStabilityModel.apply(excess, self.trading_days)
                sharpe_dict.update(roll_dict)

        return asdict(ret_m), dist_dict, tail_dict, sharpe_dict, exec_dict, n_eff

    def run_analytics(self) -> Optional[QuantReport]:
        start_time = time.perf_counter()
        
        self.alpha_rng = np.random.default_rng(self.seed)
        self.mc_rng = np.random.default_rng(self.seed + 1)
        
        daily_returns, excess, total_equity, daily_dd, hwm_daily, rf_series, avg_equity, years = self._prepare_data()
        if len(daily_returns) < 2: return None
        n_samples = len(daily_returns)
        
        ret_dict, dist_dict, tail_dict, sharpe_dict, exec_dict, _ = self._compute_core_metrics(
            daily_returns, excess, total_equity, daily_dd, hwm_daily, avg_equity, years
        )
        
        dist_m = safe_build(DistributionMetrics, dist_dict, self.strict_validation)
        tail_m = safe_build(TailMetrics, tail_dict, self.strict_validation)
        sharpe_m = safe_build(SharpeMetrics, sharpe_dict, self.strict_validation)
        exec_m = safe_build(ExecutionMetrics, exec_dict, self.strict_validation)
        ret_m = safe_build(ReturnMetrics, ret_dict, self.strict_validation) 
        
        alpha_data = {}
        alpha_inf = "Bypassed"
        if self.enable_alpha:
            logger.info("[Alpha] Constructing multi-factor regression arrays...")
            alpha_data, alpha_inf = AlphaModel.apply(
                daily_returns, self.benchmark_df, 
                rf_series, self.trading_days, self.alpha_rng, self.min_periods
            )
        alpha_m = safe_build(AlphaMetrics, alpha_data, self.strict_validation)
            
        rob_data = {}
        block_size_used, rob_inf = 0, "Bypassed"
        if self.enable_mc and len(daily_returns) >= 2:
            logger.info("[Robustness] Executing Vectorized Paired-Block Monte Carlo (C-space)...")
            start_equity = float(total_equity.iloc[0])
            rob_data, block_size_used, rob_inf = RobustnessModel.apply(
                daily_returns, rf_series, start_equity, self.trading_days, 
                self.mc_sims, self.mc_rng, self.ruin_threshold, self.mc_block_size
            )
        rob_m = safe_build(RobustnessMetrics, rob_data, self.strict_validation)

        meta = MetaMetrics(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            is_out_of_sample=self.is_out_of_sample,
            seed_used=self.seed, dsr_trials=self.n_trials, n_samples=n_samples, 
            ruin_threshold_pct=self.ruin_threshold*100,
            alpha_inference_type=alpha_inf, mc_inference_type=rob_inf,
            mc_block_size_used=block_size_used, compute_time_sec=float(time.perf_counter() - start_time)
        )
            
        report = QuantReport(meta=meta, returns=ret_m, distribution=dist_m, tail=tail_m, sharpe=sharpe_m, execution=exec_m, alpha=alpha_m, robustness=rob_m)
        report.validate(self.min_periods, self.strict_validation)
        
        return report

class QuantReporter:
    def __init__(self, currency: str = "₹"): self.currency = currency

    def _format_pval(self, p: float) -> str:
        if pd.isna(p): return "N/A"
        if p < 0.01: return f"{p:.3f} ***"
        if p < 0.05: return f"{p:.3f} **"
        if p < 0.10: return f"{p:.3f} *"
        return f"{p:.3f}"

    def print_console(self, report: QuantReport) -> None:
        def sf(val, fmt=",.2f", prefix="", suffix="") -> str:
            if pd.isna(val): return "N/A"
            return f"{prefix}{val:{fmt}}{suffix}"

        meta, ret, dist, tail, sharpe, exe, alpha, rob = report.meta, report.returns, report.distribution, report.tail, report.sharpe, report.execution, report.alpha, report.robustness

        print("\n" + "="*85)
        print("🏛️ THE APEX EPOCH (INSTITUTIONAL QUANT)".center(85))
        
        sample_tag = "[OUT-OF-SAMPLE EVALUATION]" if meta.is_out_of_sample else "[IN-SAMPLE FULL HISTORY]"
        print(sample_tag.center(85))
        
        print("="*85)
        print(f"💰 Net Profit:     {sf(ret.net_profit, prefix=self.currency)} ({sf(ret.total_return_pct, suffix='%')})")
        print(f"📈 CAGR:           {sf(ret.cagr_pct, suffix='%')} | Ann. Arith Mean: {sf(ret.ann_arith_mean_pct, suffix='%')} | Vol Drag: {sf(ret.volatility_drag_pct, suffix='%')}")
        print(f"📉 Max Drawdown:   {sf(ret.max_drawdown_pct, suffix='%')} ({ret.max_dd_duration_bars} Bars) | Div Score: {sf(ret.drawdown_div_score, fmt='.4f')} (N={ret.total_drawdown_periods})")
        print(f"🌊 Time Under Water: {sf(ret.time_under_water_pct, suffix='%')} | Ulcer Index: {sf(ret.ulcer_index, fmt='.2f')}")
        print("-" * 85)
        print("📊 RISK DECOMPOSITION".center(85))
        print(f"⚖️ Sharpe Ratio:   {sf(sharpe.sharpe_ratio)} | Deflated SR Prob: {sf(sharpe.deflated_sharpe_ratio_pct, suffix='%')} | PSR: {sf(sharpe.prob_sharpe_ratio_pct, suffix='%')}")
        print(f"📉 Volatility:     {sf(dist.gross_volatility_pct, suffix='%')} (Gross) | Semi-Vol (Down): {sf(tail.downside_semivol_pct, suffix='%')} | Sortino: {sf(sharpe.sortino_ratio)}")
        print(f"📉 Tail Risk CVaR: {sf(tail.raw_cvar_95_pct, suffix='%')} | CF-VaR (95%): {sf(tail.cf_var_95_pct, suffix='%')} | CDaR: {sf(tail.cdar_95_pct, suffix='%')}")
        print(f"📐 Skewness:       {sf(dist.skewness)} | Kurtosis: {sf(dist.kurtosis)} | ES Ratio: {sf(tail.expected_shortfall_ratio, fmt='.2f')}")
        print("-" * 85)
        print("🛡️ REGIME STABILITY & DIAGNOSTICS".center(85))
        print(f"⚖️ Rolling Sharpe Std: {sf(sharpe.rolling_sharpe_std, fmt='.3f')} | Worst 1Y Sharpe: {sf(sharpe.rolling_sharpe_worst_1y, fmt='.2f')} | % Time < 0: {sf(sharpe.rolling_sharpe_under_zero_pct, suffix='%')}")
        print(f"🧬 Hurst Exponent: {sf(dist.hurst_exponent, fmt='.3f')} | Autocorr (L1): {sf(dist.autocorr_lag1, fmt='.3f')} | Eff. N: {sf(dist.effective_sample_size, fmt='.1f')}")
        print("-" * 85)
        print("⚔️ TRADE EXECUTION & CAPACITY".center(85))
        print(f"🏆 Win Rate:       {sf(exe.win_rate_pct, suffix='%')} ({exe.total_closed_trades} Trades) | Profit Factor: {sf(exe.profit_factor)}")
        print(f"🧠 Ann. Kelly:     {sf(exe.kelly_fraction_pct, suffix='%')} | Bayes Kelly: {sf(exe.bayesian_kelly_pct, suffix='%')} | Turnover: {sf(exe.annual_turnover_times)}x/yr")
        print("-" * 85)
        print("🛡️ ROBUSTNESS & ALPHA".center(85))
        print(f"🎲 Sharpe 95% CI:  [{sf(rob.sharpe_95_ci_lower, fmt='.2f')} - {sf(rob.sharpe_95_ci_upper, fmt='.2f')}] | MC Median CAGR: {sf(rob.mc_median_cagr_pct, suffix='%')}")
        
        deps = f"[Scipy: {'✅' if meta.has_scipy else '❌'} | Statsmodels: {'✅' if meta.has_statsmodels else '❌'}]"
        print(f"💣 Prob DD Breach: {sf(rob.prob_of_drawdown_breach_pct, suffix='%')} (<{sf(meta.ruin_threshold_pct, fmt='.0f')}% DD)")
        
        boot_pval_str = f"(boot p={self._format_pval(alpha.bootstrap_alpha_pvalue)}, 2-sided)" if pd.notna(alpha.bootstrap_alpha_pvalue) else ""
        boot_ci_str = f"[{sf(alpha.bootstrap_alpha_ci_lower, fmt='.2f')}% - {sf(alpha.bootstrap_alpha_ci_upper, fmt='.2f')}%]" if pd.notna(alpha.bootstrap_alpha_ci_lower) else ""
        
        print(f"📈 Alpha (Jensen): {sf(alpha.jensens_alpha_pct, suffix='%')} {boot_ci_str} {boot_pval_str} | Beta: {sf(alpha.beta)} | Downside Beta: {sf(alpha.downside_beta)}")
        print(f"📊 Info Ratio:     {sf(alpha.information_ratio)} | Beta Drift Vol: {sf(alpha.beta_drift_volatility, fmt='.4f')}")
        print("-" * 85)
        print("⚙️ ENGINE CONFIG & REPRODUCIBILITY".center(85))
        print(f"Alpha Inference: {meta.alpha_inference_type} | MC Inference: {meta.mc_inference_type}")
        print(f"Seed: {meta.seed_used} | Samples: {meta.n_samples} | DSR Trials: {meta.dsr_trials} | MC Block: {meta.mc_block_size_used}")
        print(f"Compute Time: {meta.compute_time_sec:.3f}s | Schema v{meta.schema_version} | UTC: {meta.timestamp_utc[:19]} | {deps}")
        print("="*85 + "\n")

def generate_institutional_audit(db_path: str = "paper.db", output_dir: str = "reports", currency: str = "₹", annualized_risk_free_rate: float = 0.05, n_trials: int = 1, mc_sims: int = 2000, mc_block_size: Optional[int] = None, ruin_threshold: float = -0.50, min_periods: int = DEFAULT_MIN_PERIODS) -> None:
    logger.info("Initiating The Apex Epoch Pipeline...")
    engine = QuantEngine(db_path=db_path, annualized_risk_free_rate=annualized_risk_free_rate, seed=42, n_trials=n_trials, mc_sims=mc_sims, mc_block_size=mc_block_size, ruin_threshold=ruin_threshold, min_periods=min_periods, strict_validation=True)
    
    try:
        engine.load_data()
        report = engine.run_analytics()
        if report:
            QuantReporter(currency).print_console(report)
            logger.info(f"Pipeline Execution Successful in {report.meta.compute_time_sec:.3f}s.")
            # 🚨 INJECT NOTIFIER HERE
            notifier = QuantNotifier()
            notifier.broadcast(report)
        else: logger.warning("Pipeline exited cleanly due to insufficient data.")
    except Exception as e: logger.error(f"Critical Pipeline Failure: {e}", exc_info=True)


# ==============================================================================
# 🧬 SYNTHETIC DATA GENERATOR: FRACTIONAL BROWNIAN MOTION (fBM) + STUDENT-T
# ==============================================================================

class SyntheticDataGenerator:
    @staticmethod
    def generate_fgn_fft(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
        if H == 0.5:
            return rng.standard_normal(n)
            
        k = np.arange(1, n + 1)
        r = np.zeros(n + 1)
        r[0] = 1.0 + 1e-8
        r[1:] = 0.5 * ((k + 1)**(2*H) - 2*k**(2*H) + (k - 1)**(2*H))
            
        r_ext = np.concatenate([r, r[len(r)-2:0:-1]])
        eigenvalues = np.fft.fft(r_ext).real
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        
        rnd = rng.standard_normal(len(eigenvalues)) + 1j * rng.standard_normal(len(eigenvalues))
        fgn = np.fft.ifft(rnd * np.sqrt(eigenvalues)).real
        return fgn[:n]
        
    @staticmethod
    def apply_student_t_copula(fgn: np.ndarray, df: float) -> np.ndarray:
        """Injects heavy Student-t tails into the fGn preserving Hurst autocorrelation via PIT."""
        if not HAS_SCIPY or df >= 50: return fgn
        import scipy.stats as stats
        
        fgn_std = (fgn - fgn.mean()) / (fgn.std() + 1e-12)
        u = stats.norm.cdf(fgn_std)
        u = np.clip(u, 1e-15, 1.0 - 1e-15)
        
        t_sim = stats.t.ppf(u, df=df)
        return t_sim / (t_sim.std() + 1e-12)

    @staticmethod
    def create_synthetic_db(db_path: str = "paper.db", n_days: int = 2520, H_strat: float = 0.70, 
                            H_bm: float = 0.50, vol_ann: float = 0.15, drift_ann: float = 0.10, 
                            t_df: float = 4.0, rho: float = 0.60, start_equity: float = 100_000.0, seed: int = 42):
        
        rng = np.random.default_rng(seed)
        logger.info(f"Generating synthetic fBM. N={n_days}, Strat_H={H_strat}, BM_H={H_bm}, T-Dist_DF={t_df}, Rho={rho}")
        
        strat_fgn = SyntheticDataGenerator.generate_fgn_fft(n_days, H_strat, rng)
        bm_fgn = SyntheticDataGenerator.generate_fgn_fft(n_days, H_bm, rng)
        
        strat_fgn = SyntheticDataGenerator.apply_student_t_copula(strat_fgn, t_df)
        bm_fgn = SyntheticDataGenerator.apply_student_t_copula(bm_fgn, t_df)
        
        strat_fgn = rho * bm_fgn + math.sqrt(max(0.0, 1.0 - rho**2)) * strat_fgn
        strat_fgn = (strat_fgn - strat_fgn.mean()) / (strat_fgn.std() + 1e-12)
        
        daily_vol = vol_ann / math.sqrt(252)
        daily_drift = drift_ann / 252
        
        strat_log_rets = (strat_fgn * daily_vol) + daily_drift
        bm_log_rets = (bm_fgn * daily_vol) + daily_drift
        
        strat_curves = start_equity * np.exp(np.cumsum(strat_log_rets))
        bm_curves = start_equity * np.exp(np.cumsum(bm_log_rets))
        
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
        
        equity_df = pd.DataFrame({'date': dates, 'total_equity': strat_curves})
        bm_df = pd.DataFrame({'date': dates, 'close': bm_curves})
        
        trade_dates = rng.choice(dates, size=int(n_days * 0.1), replace=False)
        realized_pnl = rng.laplace(loc=50, scale=300, size=len(trade_dates))
        trades_df = pd.DataFrame({
            'date': sorted(trade_dates),
            'realized_pnl': realized_pnl,
            'trade_value': np.abs(realized_pnl) * rng.uniform(5, 20, size=len(trade_dates)),
            'gross_exposure': rng.uniform(0.5, 1.5, size=len(trade_dates))
        })
        
        db_file = Path(db_path)
        if db_file.exists(): db_file.unlink()
            
        with sqlite3.connect(db_file) as conn:
            equity_df.to_sql('equity_log', conn, index=False)
            trades_df.to_sql('trades', conn, index=False)
            
        logger.info(f"Synthetic database '{db_path}' created.")
        return bm_df

if __name__ == "__main__":
    db_name = "synthetic_fbm.db"
    
    synthetic_benchmark = SyntheticDataGenerator.create_synthetic_db(
        db_path=db_name, n_days=3000, H_strat=0.75, H_bm=0.50, vol_ann=0.20, drift_ann=0.15, t_df=4.0, rho=0.60
    )
    
    engine = QuantEngine(
        db_path=db_name, benchmark_df=synthetic_benchmark, annualized_risk_free_rate=0.04, 
        seed=1337, n_trials=10, mc_sims=3000, ruin_threshold=-0.30, strict_validation=True
    )
    
    try:
        engine.load_data()
        report = engine.run_analytics()
        if report:
            QuantReporter(currency="$").print_console(report)
    except Exception as e:
        logger.error(f"Execution Failed: {e}", exc_info=True)
