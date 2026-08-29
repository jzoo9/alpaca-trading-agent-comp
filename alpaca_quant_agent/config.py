"""Loads .env credentials and config.yaml strategy/risk constants into one object."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Credentials:
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_paper: bool
    featherless_api_key: str
    featherless_model: str


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    creds: Credentials
    db_path: str

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def load_config(config_path: Path | None = None, env_path: Path | None = None) -> Config:
    load_dotenv(env_path or REPO_ROOT / ".env")

    def _require(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(
                f"Missing required environment variable {name}. "
                f"Copy .env.example to .env and fill it in."
            )
        return value

    creds = Credentials(
        alpaca_api_key=_require("ALPACA_API_KEY"),
        alpaca_secret_key=_require("ALPACA_SECRET_KEY"),
        alpaca_paper=os.environ.get("ALPACA_PAPER_TRADE", "true").lower() == "true",
        featherless_api_key=_require("FEATHERLESS_API_KEY"),
        featherless_model=os.environ.get("FEATHERLESS_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
    )

    path = config_path or REPO_ROOT / "config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    db_path = os.environ.get("AGENT_DB_PATH", "./data/agent.db")

    return Config(raw=raw, creds=creds, db_path=db_path)
