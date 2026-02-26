#!/usr/bin/env python3
"""
APEX EPOCH: BREAK-GLASS PROTOCOL (MANUAL OVERRIDE)
================================================================================
Purpose: Provides the Risk Committee and Lead PMs direct CLI access to the 
         Quant Engine, bypassing automated CI/CD schedules. 
         Allows dynamic injection of stress-test parameters without altering 
         production source code or environment variables.
================================================================================
"""

import argparse
import logging
import sys
import time

# Import the core mathematical engine and telemetry relay
try:
    from generate_report import QuantEngine, QuantReporter, DEFAULT_MIN_PERIODS
    from notifier import QuantNotifier
except ImportError as e:
    print(f"CRITICAL: Failed to import Apex Epoch Core Modules. {e}")
    sys.exit(1)

def setup_override_logger():
    logger = logging.getLogger("QuantOverride")
    logger.setLevel(logging.DEBUG) # Manual overrides get debug verbosity by default
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - [MANUAL OVERRIDE] - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    return logger

def main():
    logger = setup_override_logger()
    
    parser = argparse.ArgumentParser(
        description="Apex Epoch: Institutional Quant Engine Manual Override",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data & Core Params
    parser.add_argument("--db", type=str, default="paper.db", help="Target SQLite database path.")
    parser.add_argument("--currency", type=str, default="₹", help="Reporting currency symbol.")
    parser.add_argument("--oos", action="store_true", help="Flag report as Out-Of-Sample.")
    
    # Stress Testing & Mathematical Overrides
    parser.add_argument("--rf", type=float, default=0.05, help="Annualized Risk-Free Rate (e.g., 0.0525 for 5.25%).")
    parser.add_argument("--ruin", type=float, default=-0.50, help="Terminal Ruin Threshold (Negative float, e.g., -0.40).")
    parser.add_argument("--mc-sims", type=int, default=2000, help="Number of Monte Carlo paths to generate.")
    parser.add_argument("--min-periods", type=int, default=DEFAULT_MIN_PERIODS, help="Minimum days required to evaluate tail risk.")
    
    # Execution & Safety Overrides
    parser.add_argument("--disable-strict", action="store_true", help="WARNING: Disables strict schema and anomaly validation.")
    parser.add_argument("--notify", action="store_true", help="Force telemetry dispatch to Slack/Telegram.")
    parser.add_argument("--silent", action="store_true", help="Suppress console report printing (useful if only notifying).")

    args = parser.parse_args()

    logger.warning(f"INITIATING MANUAL OVERRIDE PROTOCOL. Target DB: {args.db}")
    if args.disable_strict:
        logger.warning("STRICT VALIDATION DISABLED. NaN propagation and bounds failures may occur silently.")

    # 1. Instantiate the Engine with CLI Overrides
    engine = QuantEngine(
        db_path=args.db,
        benchmark_df=None,  # Benchmarks usually injected programmatically, left None for pure absolute return evaluation
        annualized_risk_free_rate=args.rf,
        seed=int(time.time()), # Force true randomness on manual overrides, bypassing deterministic CI seeds
        mc_sims=args.mc_sims,
        ruin_threshold=args.ruin,
        min_periods=args.min_periods,
        strict_validation=not args.disable_strict,
        is_out_of_sample=args.oos
    )

    # 2. Execute Analytics
    try:
        engine.load_data()
        report = engine.run_analytics()
        
        if not report:
            logger.error("Execution aborted. Engine returned Null Report (insufficient data or mathematical collapse).")
            sys.exit(1)

        # 3. Present Results
        if not args.silent:
            QuantReporter(args.currency).print_console(report)
            
        logger.info(f"Override execution completed in {report.meta.compute_time_sec:.3f} seconds.")

        # 4. Telemetry Routing
        if args.notify:
            logger.info("Dispatching override telemetry to communications grid...")
            notifier = QuantNotifier()
            notifier.broadcast(report)
            
    except Exception as e:
        logger.error(f"FATAL EXCEPTION DURING OVERRIDE EXECUTION: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
