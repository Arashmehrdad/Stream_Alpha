"""Small local work-directory helpers for Windows-safe training runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


STREAMALPHA_LOCAL_TRAINING_TEMP_ROOT_ENV = "STREAMALPHA_LOCAL_TRAINING_TEMP_ROOT"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOCAL_TRAINING_TEMP_ROOT = _REPO_ROOT / "artifacts" / "tmp" / "autogluon"


def resolve_local_training_temp_root() -> Path:
    """Return the authoritative local temp root for training-time work dirs."""
    override = os.environ.get(STREAMALPHA_LOCAL_TRAINING_TEMP_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_LOCAL_TRAINING_TEMP_ROOT.resolve()


def create_local_training_work_dir(*, prefix: str) -> Path:
    """Create one scoped temp directory under the authoritative local temp root."""
    temp_root = resolve_local_training_temp_root()
    temp_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(temp_root)))
