from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
UPLOAD_DATA_DIR = ROOT / "data" / "uploads"
ARTIFACT_DIR = ROOT / "models" / "quam_artifacts"

UPLOAD_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory app state for the current server session.
DATASETS: Dict[str, Dict[str, Any]] = {}
TRAIN_RUNS: Dict[str, Dict[str, Any]] = {}
MODELS: Dict[str, Dict[str, Any]] = {}
