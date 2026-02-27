#!/usr/bin/env python3
import sqlite3
import os
import shutil
import logging
from datetime import datetime, timezone
from notifier import QuantNotifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HealthCheck")

def run_diagnostics(db_path="paper.db"):
    status = {
        "status": "🟢 ONLINE",
        "disk_usage_pct": 0.0,
        "db_size_mb": 0.0,
        "is_halted": False,
        "halt_reason": "None",
        "open_positions": 0,
        "available_cash": 0.0
    }

    # 1. Check Disk Space (Prevent SQLite "Disk Full" corruption)
    total, used, free = shutil.disk_usage("/")
    status["disk_usage_pct"] = (used / total) * 100
    if status["disk_usage_pct"] > 90:
        status["status"] = "🔴 WARNING: DISK NEARLY FULL"

    # 2. Check Database Integrity & Size
    if os.path.exists(db_path):
        status["db_size_mb"] = os.path.getsize(db_path) / (1024 * 1024)
        try:
            with sqlite3.connect(db_path, timeout=5) as conn:
                conn.row_factory = sqlite3.Row
                
                # Check positions & cash
                acct = conn.execute("SELECT cash FROM account WHERE id=1").fetchone()
                status["available_cash"] = acct['cash'] if acct else 0.0
                
                pos_count = conn.execute("SELECT COUNT(*) as cnt FROM positions").fetchone()
                status["open_positions"] = pos_count['cnt'] if pos_count else 0

                # Check Halt State
                halt_row = conn.execute("SELECT value FROM broker_status WHERE key='halted'").fetchone()
                if halt_row and halt_row['value'] == 'TRUE':
                    status["is_halted"] = True
                    status["status"] = "🔴 HALTED"
                    reason_row = conn.execute("SELECT value FROM broker_status WHERE key='halt_reason'").fetchone()
                    status["halt_reason"] = reason_row['value'] if reason_row else "Unknown"
                    
        except Exception as e:
            status["status"] = f"🔴 DB CORRUPTION DETECTED: {e}"
    else:
        status["status"] = "🟡 WAITING FOR DB INITIALIZATION"

    return status

def broadcast_heartbeat():
    logger.info("Running pre-market system diagnostics...")
    stats = run_diagnostics()
    
    msg = (
        f"🩺 *APEX EPOCH HEARTBEAT*\n"
        f"⏱ `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`\n"
        f"\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\n"
        f"• Status: `{stats['status']}`\n"
        f"• Broker Halted: `{stats['is_halted']}`\n"
        f"• Open Positions: `{stats['open_positions']}`\n"
        f"• Available Cash: `${stats['available_cash']:,.2f}`\n"
        f"• DB Size: `{stats['db_size_mb']:.2f} MB`\n"
        f"• Disk Usage: `{stats['disk_usage_pct']:.1f}%`\n"
    )
    
    if stats['is_halted']:
        msg += f"\n⚠️ *HALT REASON*: `{stats['halt_reason']}`"
        
    reserved = ['_', '[', ']', '(', ')', '~', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in reserved:
        if char != '*' and char != '`' and char != '\\': # don't escape our markdown
             msg = msg.replace(char, f"\\{char}")

    # Directly ping the notifier
    notifier = QuantNotifier()
    try:
        url = f"https://api.telegram.org/bot{notifier.telegram_token}/sendMessage"
        payload = {"chat_id": notifier.telegram_chat_id, "text": msg, "parse_mode": "MarkdownV2"}
        notifier.session.post(url, json=payload, timeout=10)
        logger.info("Heartbeat dispatched.")
    except Exception as e:
        logger.error(f"Failed to dispatch heartbeat: {e}")

if __name__ == "__main__":
    broadcast_heartbeat()
