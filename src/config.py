"""Configuration loader for database and LLM settings."""

from __future__ import annotations

import configparser
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.environ.get("DIAGHINT_CONFIG_DIR", ROOT_DIR / "config"))


def _read_conf(filename: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(CONFIG_DIR / filename, encoding="utf-8")
    return parser


def _get(parser: configparser.ConfigParser, section: str, key: str, default: str = "") -> str:
    env_key = f"DIAGHINT_{section}_{key}".upper()
    if env_key in os.environ:
        return os.environ[env_key]
    return parser.get(section, key, fallback=default)


_db = _read_conf("db.conf")
_llm = _read_conf("llm.conf")
_llm_section = _llm.get("option", "apply", fallback="openai_compatible")


class Config:
    DB_HOST = _get(_db, "postgres", "host", "localhost")
    DB_PORT = _get(_db, "postgres", "port", "5432")
    DB_USER = _get(_db, "postgres", "username", "postgres")
    DB_PASSWORD = _get(_db, "postgres", "password", "")
    DB_DATABASE = _get(_db, "postgres", "database", "")

    LLM_API = _get(_llm, _llm_section, "api", "https://api.openai.com/v1")
    LLM_KEY = _get(_llm, _llm_section, "api_key", "")
    LLM_MODEL = _get(_llm, _llm_section, "model", "gpt-4o-mini")
