from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class ModelArtifact:
    run_id: str
    model_type: str
    model: Any
    threshold: float = 0.5
    feature_spec: Optional[Dict[str, Any]] = None
