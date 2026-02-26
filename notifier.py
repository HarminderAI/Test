import os
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any, Optional

logger = logging.getLogger("QuantNotifier")
logger.addHandler(logging.NullHandler())

class QuantNotifier:
    """
    Institutional Telemetry Relay.
    Features: 
    - Exponential backoff for transient network failures.
    - Graceful degradation (failures log errors but do not crash the pipeline).
    - Environment-variable strict secret management.
    """
    def __init__(self):
        # 🚨 FIX: Strict Secret Isolation. Never hardcode credentials.
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        
        # 🚨 FIX: Network Resilience Layer (Exponential Backoff)
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _format_telegram_message(self, report_dict: Dict[str, Any]) -> str:
        """Constructs a heavily structured, MarkdownV2 compliant executive summary."""
        meta = report_dict.get('meta', {})
        ret = report_dict.get('returns', {})
        sharpe = report_dict.get('sharpe', {})
        rob = report_dict.get('robustness', {})
        
        # Helper to format floats safely
        def sf(val, fmt=",.2f", suffix=""):
            if val is None or (isinstance(val, float) and math.isnan(val)): return "N/A"
            return f"{val:{fmt}}{suffix}"

        # MarkdownV2 requires strict escaping of certain characters
        msg = (
            f"🏛 *APEX EPOCH EXECUTED*\n"
            f"⏱ `{meta.get('timestamp_utc', 'Unknown')[:19]} UTC`\n"
            f"\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\n"
            f"📈 *Returns & Drawdown*\n"
            f"• CAGR: `{sf(ret.get('cagr_pct'), suffix='%')}`\n"
            f"• Max DD: `{sf(ret.get('max_drawdown_pct'), suffix='%')}`\n"
            f"• Net Profit: `{sf(ret.get('net_profit'))}`\n\n"
            f"⚖️ *Risk Adjusted*\n"
            f"• Sharpe: `{sf(sharpe.get('sharpe_ratio'))}`\n"
            f"• Prob Sharpe \\(PSR\\): `{sf(sharpe.get('prob_sharpe_ratio_pct'), suffix='%')}`\n"
            f"• Deflated Sharpe: `{sf(sharpe.get('deflated_sharpe_ratio_pct'), suffix='%')}`\n\n"
            f"🛡️ *Robustness \\(MC Bootstrap\\)*\n"
            f"• Prob of Ruin: `{sf(rob.get('prob_of_drawdown_breach_pct'), suffix='%')}`\n"
            f"• MC Median CAGR: `{sf(rob.get('mc_median_cagr_pct'), suffix='%')}`\n"
            f"• 5% Tail Return: `{sf(rob.get('mc_ret_p05'), suffix='%')}`\n"
            f"\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\\-\n"
            f"⏱ Compute Time: `{sf(meta.get('compute_time_sec'), fmt='.3f')}s`"
        )
        # Escape reserved characters for MarkdownV2 (except those used for formatting)
        reserved = ['_', '[', ']', '(', ')', '~', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in reserved:
            msg = msg.replace(char, f"\\{char}")
        return msg

    def send_telegram(self, report_dict: Dict[str, Any]) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            logger.debug("Telegram credentials not found in ENV. Skipping Telegram alert.")
            return

        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        text = self._format_telegram_message(report_dict)
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": text,
            "parse_mode": "MarkdownV2"
        }

        try:
            # 🚨 FIX: Strict timeout preventing thread hanging
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("Successfully dispatched telemetry to Telegram.")
        except Exception as e:
            logger.error(f"Failed to dispatch Telegram alert: {e}")

    def send_slack(self, report_dict: Dict[str, Any]) -> None:
        if not self.slack_webhook_url:
            logger.debug("Slack webhook not found in ENV. Skipping Slack alert.")
            return

        # Simple text fallback for Slack. For production, use Slack Block Kit.
        ret = report_dict.get('returns', {})
        sharpe = report_dict.get('sharpe', {})
        
        text = (
            f"🏛 *Apex Epoch Execution*\n"
            f"CAGR: {ret.get('cagr_pct', 'N/A'):.2f}% | Max DD: {ret.get('max_drawdown_pct', 'N/A'):.2f}%\n"
            f"Sharpe: {sharpe.get('sharpe_ratio', 'N/A'):.2f} | PSR: {sharpe.get('prob_sharpe_ratio_pct', 'N/A'):.2f}%"
        )

        try:
            response = self.session.post(self.slack_webhook_url, json={"text": text}, timeout=10)
            response.raise_for_status()
            logger.info("Successfully dispatched telemetry to Slack.")
        except Exception as e:
            logger.error(f"Failed to dispatch Slack alert: {e}")

    def broadcast(self, report: Any) -> None:
        """Primary interface. Accepts the QuantReport object, extracts dict, and broadcasts."""
        if not report:
            logger.warning("Received null report. Aborting broadcast.")
            return
            
        try:
            # Check if it's our dataclass and convert to dict safely
            report_dict = report.to_dict() if hasattr(report, 'to_dict') else report
            self.send_telegram(report_dict)
            self.send_slack(report_dict)
        except Exception as e:
            logger.error(f"Fatal error in broadcast routing: {e}")
