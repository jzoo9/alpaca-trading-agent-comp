"""Local read-only monitoring dashboard for the trading ledger.

Serves a single-page dashboard (portfolio equity, P&L, open positions, risk
gate utilization, and a live activity feed of trades/decisions) by reading
directly from the SQLite ledger (`ledger.py`) that every cycle already writes
to. Deliberately dependency-free (stdlib `http.server` only) and read-only --
it never touches Alpaca, the LLM, or order placement, so it's safe to leave
running continuously alongside the daemon during the competition.

    python -m alpaca_quant_agent.dashboard              # http://localhost:8787
    python -m alpaca_quant_agent.dashboard --port 9000
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from alpaca_quant_agent import control, universe

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "dashboard_static"


def _load_raw_config() -> dict[str, Any]:
    """Reads config.yaml directly -- deliberately independent of
    config.load_config(), which requires Alpaca/Featherless credentials the
    dashboard has no need for (it never calls out to either)."""
    path = REPO_ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _db_path() -> str:
    return os.environ.get("AGENT_DB_PATH", "./data/agent.db")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _dte(expiration: str | None, today: date) -> int | None:
    if not expiration:
        return None
    try:
        return (datetime.strptime(expiration, "%Y-%m-%d").date() - today).days
    except ValueError:
        return None


def build_snapshot() -> dict[str, Any]:
    config = _load_raw_config()
    limits = config["risk_gates"]
    starting_balance = float(config["account"]["starting_balance"])
    db_path = _db_path()
    today = date.today()

    halt_state = control.get_halt_state(db_path)
    snapshot: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "has_data": False,
        "starting_balance": starting_balance,
        "universe_size": len(universe.ALL_ENTRIES),
        "universe_symbols": list(universe.SYMBOLS),
        "risk_limits": limits,
        "manual_halt": {
            "halted": halt_state.halted,
            "reason": halt_state.reason,
            "set_at": halt_state.set_at,
        },
    }

    if not Path(db_path).exists():
        return snapshot

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if not _table_exists(conn, "equity_snapshots"):
            return snapshot

        latest = conn.execute(
            "SELECT * FROM equity_snapshots ORDER BY snapshot_date DESC LIMIT 1"
        ).fetchone()

        curve = conn.execute(
            "SELECT snapshot_date, equity FROM equity_snapshots ORDER BY snapshot_date"
        ).fetchall()
        snapshot["equity_curve"] = [
            {"date": r["snapshot_date"], "equity": r["equity"]} for r in curve
        ]

        if latest is None:
            return snapshot

        equity = float(latest["equity"])
        equity_peak = float(latest["equity_peak"])
        total_pnl = equity - starting_balance
        total_pnl_pct = total_pnl / starting_balance if starting_balance else 0.0
        drawdown_pct = (equity - equity_peak) / equity_peak if equity_peak else 0.0

        # --- open positions, reconstructed the same way risk gates do ---
        open_rows = conn.execute(
            """SELECT t.position_group, t.symbol, t.sleeve, t.strategy_type, t.contracts,
                      t.credit_or_debit AS credit_received, t.max_loss, t.net_delta, t.net_vega,
                      t.expiration, t.days_to_earnings, t.created_at AS opened_at
               FROM trades t
               WHERE t.action = 'open'
                 AND NOT EXISTS (
                     SELECT 1 FROM trades c
                     WHERE c.action = 'close' AND c.position_group = t.position_group
                 )
               ORDER BY t.created_at DESC"""
        ).fetchall()

        open_positions = []
        portfolio_heat = 0.0
        net_delta = 0.0
        net_vega = 0.0
        sleeve_b_heat = 0.0
        bucket_heat: dict[str, float] = {}
        for r in open_rows:
            max_loss = float(r["max_loss"] or 0.0)
            delta = float(r["net_delta"] or 0.0)
            vega = float(r["net_vega"] or 0.0)
            portfolio_heat += max_loss
            net_delta += delta
            net_vega += vega
            if r["sleeve"] == "B":
                sleeve_b_heat += max_loss
            bucket = universe.bucket_for(r["symbol"])
            bucket_heat[bucket] = bucket_heat.get(bucket, 0.0) + max_loss
            open_positions.append({
                "position_group": r["position_group"],
                "symbol": r["symbol"],
                "sleeve": r["sleeve"],
                "strategy_type": r["strategy_type"],
                "contracts": r["contracts"],
                "credit_received": r["credit_received"],
                "max_loss": max_loss,
                "net_delta": delta,
                "net_vega": vega,
                "expiration": r["expiration"],
                "dte": _dte(r["expiration"], today),
                "days_to_earnings": r["days_to_earnings"],
                "opened_at": r["opened_at"],
            })

        bucket_caps = limits.get("max_bucket_heat_pct")
        snapshot["bucket_heat"] = sorted(
            (
                {
                    "bucket": b,
                    "heat": h,
                    "pct": (h / equity) if equity else 0.0,
                    "cap_pct": bucket_caps,
                }
                for b, h in bucket_heat.items()
                if h > 0
            ),
            key=lambda x: x["heat"],
            reverse=True,
        )

        # --- realized P&L from closed positions (match open <-> close by position_group) ---
        closed_rows = conn.execute(
            """SELECT o.position_group, o.symbol, o.sleeve, o.contracts,
                      o.credit_or_debit AS open_credit, c.credit_or_debit AS close_debit,
                      o.created_at AS opened_at, c.created_at AS closed_at
               FROM trades o
               JOIN trades c ON c.position_group = o.position_group AND c.action = 'close'
               WHERE o.action = 'open'
               ORDER BY c.created_at DESC"""
        ).fetchall()

        closed_trades = []
        wins = 0
        total_realized = 0.0
        for r in closed_rows:
            contracts = r["contracts"] or 1
            realized = (float(r["open_credit"] or 0.0) - float(r["close_debit"] or 0.0)) * contracts
            total_realized += realized
            wins += 1 if realized > 0 else 0
            closed_trades.append({
                "symbol": r["symbol"],
                "sleeve": r["sleeve"],
                "contracts": contracts,
                "opened_at": r["opened_at"],
                "closed_at": r["closed_at"],
                "realized_pnl": realized,
            })

        closed_count = len(closed_trades)
        snapshot["closed_stats"] = {
            "count": closed_count,
            "wins": wins,
            "losses": closed_count - wins,
            "win_rate": (wins / closed_count) if closed_count else None,
            "total_realized_pnl": total_realized,
            "avg_pnl": (total_realized / closed_count) if closed_count else None,
        }
        snapshot["closed_trades"] = closed_trades[:15]

        # --- recent activity feed ---
        recent_trades = conn.execute(
            """SELECT created_at, symbol, sleeve, strategy_type, action, contracts,
                      credit_or_debit, rationale, order_id
               FROM trades ORDER BY created_at DESC LIMIT 25"""
        ).fetchall()
        snapshot["recent_trades"] = [dict(r) for r in recent_trades]

        recent_decisions = conn.execute(
            """SELECT created_at, symbol, decision, detail
               FROM decisions ORDER BY created_at DESC LIMIT 40"""
        ).fetchall()
        snapshot["recent_decisions"] = [dict(r) for r in recent_decisions]

        snapshot.update({
            "has_data": True,
            "equity": equity,
            "equity_peak": equity_peak,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "today_pnl_pct": float(latest["daily_pnl_pct"]),
            "drawdown_pct": drawdown_pct,
            "open_position_count": len(open_positions),
            "open_positions": open_positions,
            "portfolio_heat": portfolio_heat,
            "portfolio_heat_pct": (portfolio_heat / equity) if equity else 0.0,
            "net_delta": net_delta,
            "net_delta_pct": (net_delta / equity) if equity else 0.0,
            "net_vega": net_vega,
            "net_vega_pct": (net_vega / equity) if equity else 0.0,
            "sleeve_b_heat": sleeve_b_heat,
            "sleeve_b_heat_pct": (sleeve_b_heat / equity) if equity else 0.0,
        })
        return snapshot
    finally:
        conn.close()


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler API
        if self.path in ("/", "/index.html"):
            html_path = STATIC_DIR / "index.html"
            self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/data":
            try:
                data = build_snapshot()
                self._send(200, json.dumps(data).encode("utf-8"), "application/json")
            except Exception as exc:  # noqa: BLE001 -- surface errors to the UI, don't crash the server
                self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler API
        # These endpoints place real (paper) orders or change what the daemon
        # will do next cycle -- unlike everything under GET, which is
        # read-only. They lazily import the full trading stack (Alpaca/MCP
        # client, OpenAI client) since plain viewing needs none of it.
        try:
            body = self._read_json_body()
        except (json.JSONDecodeError, ValueError) as exc:
            self._send(400, json.dumps({"error": f"invalid JSON body: {exc}"}).encode(), "application/json")
            return

        try:
            if self.path == "/api/kill-switch":
                result = self._handle_kill_switch(body)
            elif self.path == "/api/run-cycle":
                result = self._handle_run_cycle(body)
            elif self.path == "/api/close-position":
                result = self._handle_close_position(body)
            else:
                self._send(404, json.dumps({"error": "not found"}).encode(), "application/json")
                return
            self._send(200, json.dumps(result).encode("utf-8"), "application/json")
        except Exception as exc:  # noqa: BLE001 -- surface to the UI as an error, never crash the server
            self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")

    def _handle_kill_switch(self, body: dict[str, Any]) -> dict[str, Any]:
        halted = bool(body.get("halted"))
        reason = str(body.get("reason") or ("manual pause via dashboard" if halted else ""))
        state = control.set_halt_state(_db_path(), halted=halted, reason=reason)
        return {"halted": state.halted, "reason": state.reason, "set_at": state.set_at}

    def _handle_run_cycle(self, body: dict[str, Any]) -> dict[str, Any]:
        import asyncio
        from alpaca_quant_agent.config import load_config
        from alpaca_quant_agent.cycle import run_one_cycle

        dry_run = bool(body.get("dry_run", True))
        config = load_config()
        summary = asyncio.run(run_one_cycle(config, dry_run=dry_run))
        return {"summary": summary, "dry_run": dry_run}

    def _handle_close_position(self, body: dict[str, Any]) -> dict[str, Any]:
        import asyncio
        from alpaca_quant_agent.config import load_config
        from alpaca_quant_agent.cycle import close_position_manually

        position_group = body.get("position_group")
        if not position_group:
            raise ValueError("position_group is required")
        dry_run = bool(body.get("dry_run", True))
        config = load_config()
        summary = asyncio.run(close_position_manually(config, position_group, dry_run=dry_run))
        return {"summary": summary, "dry_run": dry_run}


def run(host: str = "127.0.0.1", port: int = 8787) -> None:
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://{host}:{port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local read-only dashboard for the trading ledger")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
