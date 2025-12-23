"""Assemble and write meta data to metadata.json."""

from __future__ import annotations

import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from telco_churn.io.atomic import atomic_write_json


def _safe_cfg_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return {"cfg_repr": repr(cfg)}

def assemble_metadata_payload(
    *,
    run_id: str,
    model_type: str,
    best_params: Dict[str, Any],
    cfg: Any = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "run_id": run_id,
        "model_type": model_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_params": best_params,
        "cfg": _safe_cfg_dict(cfg),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import sklearn
        meta["sklearn_version"] = sklearn.__version__
    except Exception:
        pass

    try:
        import optuna
        meta["optuna_version"] = optuna.__version__
    except Exception:
        pass

    return meta

def write_metadata_json(bundle_dir: Path, payload: Dict[str, Any]) -> Path:
    path = bundle_dir / "metadata.json"
    atomic_write_json(path, payload)
    return path