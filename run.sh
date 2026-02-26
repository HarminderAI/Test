#!/usr/bin/env bash

# ==============================================================================
# APEX EPOCH QUANT ENGINE - EXECUTION ORCHESTRATOR
# ==============================================================================
# Enforce strict error handling:
# -e: Exit immediately if a command exits with a non-zero status.
# -u: Treat unset variables as an error when substituting.
# -o pipefail: The return value of a pipeline is the status of the last command 
#              to exit with a non-zero status.
set -euo pipefail

# --- CONFIGURATION ---
PYTHON_EXEC="python3"
VENV_DIR="venv"
DB_FILE="paper.db"
GENERATOR_SCRIPT="create_paper_db.py"
REPORT_SCRIPT="generate_report.py"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXEC_LOG="${LOG_DIR}/exec_${TIMESTAMP}.log"

# --- COLOR CODES ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- LOGGING FUNCTIONS ---
log_info() { echo -e "${GREEN}[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1${NC}" | tee -a "$EXEC_LOG"; }
log_warn() { echo -e "${YELLOW}[WARN] $(date +'%Y-%m-%d %H:%M:%S') - $1${NC}" | tee -a "$EXEC_LOG"; }
log_err()  { echo -e "${RED}[ERROR] $(date +'%Y-%m-%d %H:%M:%S') - $1${NC}" | tee -a "$EXEC_LOG"; }

# --- ERROR TRAP ---
trap 'log_err "Pipeline failed at line $LINENO. Exit code: $?"; exit 1' ERR

# ==============================================================================

# 1. Initialization
mkdir -p "$LOG_DIR"
log_info "Initializing Citadel Core Pipeline Orchestrator..."

# 2. Environment Management
if [ -d "$VENV_DIR" ]; then
    log_info "Activating isolated virtual environment..."
    # source command works in bash, dot operator is strictly POSIX
    . "${VENV_DIR}/bin/activate"
else
    log_warn "No virtual environment found at './${VENV_DIR}'. Executing on global Python."
    # Optional: Automatically create venv and install requirements here
fi

# Check for Python availability
if ! command -v $PYTHON_EXEC &> /dev/null; then
    log_err "Python executable '$PYTHON_EXEC' not found. Aborting."
    exit 1
fi

# 3. Data Dependency Resolution
if [ ! -f "$DB_FILE" ]; then
    log_warn "Target database '$DB_FILE' missing. Triggering Synthetic Generator..."
    if [ -f "$GENERATOR_SCRIPT" ]; then
        $PYTHON_EXEC "$GENERATOR_SCRIPT" 2>&1 | tee -a "$EXEC_LOG"
        log_info "Synthetic database generated successfully."
    else
        log_err "Generator script '$GENERATOR_SCRIPT' not found. Cannot resolve data dependency."
        exit 1
    fi
else
    log_info "Database '$DB_FILE' verified. Dependency satisfied."
fi

# 4. Engine Execution
log_info "Dispatching workload to Quant Engine ($REPORT_SCRIPT)..."
echo "--------------------------------------------------------------------------------" | tee -a "$EXEC_LOG"

# Execute the core engine, capturing output to both console and audit log
$PYTHON_EXEC "$REPORT_SCRIPT" 2>&1 | tee -a "$EXEC_LOG"

echo "--------------------------------------------------------------------------------" | tee -a "$EXEC_LOG"
log_info "Pipeline execution completed gracefully."

# 5. Cleanup
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate
    log_info "Virtual environment deactivated."
fi

exit 0
