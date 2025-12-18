from __future__ import annotations

import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write_text(path, text)


def _safe_cfg_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return {"cfg_repr": repr(cfg)}


def create_bundle_overwrite(
    *,
    model: Any,
    model_name: str,
    best_params: Dict[str, Any],
    cv_summary: Dict[str, Any],
    holdout_metrics: Dict[str, Any],
    primary_metric: str,
    direction: str,
    threshold: float = 0.5,
    cfg: Any = None,
    feature_spec: Optional[Dict[str, Any]] = None,
    out_dir: Union[str, Path] = "artifacts",
    clean_dir: bool = False,
) -> Path:
    bundle_dir = Path(out_dir) / model_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if clean_dir:
        for p in bundle_dir.iterdir():
            if p.is_file():
                p.unlink()

    model_path = bundle_dir / "model.joblib"
    tmp_model = model_path.with_suffix(".joblib.tmp")
    joblib.dump(model, tmp_model)
    tmp_model.replace(model_path)

    metrics_payload = {
        "primary_metric": primary_metric,
        "direction": direction,
        "threshold": float(threshold),
        "cv": cv_summary,
        "holdout": holdout_metrics,
    }
    _atomic_write_json(bundle_dir / "metrics.json", metrics_payload)

    meta: Dict[str, Any] = {
        "model_name": model_name,
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

    _atomic_write_json(bundle_dir / "metadata.json", meta)

    return bundle_dir
