"""Load .env and Hugging Face cache paths before transformers is imported.

transformers 4.31 reads HF_HOME / TRANSFORMERS_CACHE at import time, so this
module must run first in every entry script.
"""

from __future__ import annotations

import os
from pathlib import Path


def _parse_env_file(path: Path) -> None:
    """Minimal KEY=VALUE parser used if python-dotenv is not installed yet."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_runtime_env() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    env_file = repo_root / ".env"
    if env_file.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file, override=False)
        except ImportError:
            _parse_env_file(env_file)

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return

    hf_home_path = Path(hf_home)
    hf_home_path.mkdir(parents=True, exist_ok=True)

    hub_cache = str(hf_home_path / "hub")
    datasets_cache = str(hf_home_path / "datasets")
    os.environ.setdefault("HF_HUB_CACHE", hub_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hub_cache)
    os.environ.setdefault("HF_DATASETS_CACHE", datasets_cache)
